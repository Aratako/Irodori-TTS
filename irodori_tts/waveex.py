"""
WaveEx: Wavelet-guided extrapolation for flow-matching ODE sampling.

Implements the training-free, plug-in inference acceleration from
"WaveEx: Accelerating Flow Matching-based Speech Generation via
Wavelet-guided Extrapolation" (Liu et al., AAAI 2026).

At selected step indices the model is evaluated as usual to obtain a velocity
(ODE step). At the remaining indices the next latent state is predicted from a
sliding window of past latents via:

  1. Wavelet decomposition (DWT) along the step axis → low / high bands.
  2. Independent Taylor extrapolation of each band by one new coefficient.
  3. Inverse wavelet transform (IDWT) to map back to the latent space; the
     last reconstructed sample is taken as the predicted next latent.

For very short history the implementation falls back to plain Taylor
extrapolation on the raw latents, which preserves correctness during the
warm-up phase.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import torch


# ---------------------------------------------------------------------------
# Wavelet filter bank (orthogonal). Coefficients taken from PyWavelets.
# ---------------------------------------------------------------------------

_HAAR_DEC_LO = [
    0.7071067811865476,
    0.7071067811865476,
]

_DB2_DEC_LO = [
    -0.12940952255092145,
    0.22414386804185735,
    0.836516303737469,
    0.48296291314469025,
]

_DB4_DEC_LO = [
    -0.010597401784997278,
    0.032883011666982945,
    0.030841381835986965,
    -0.18703481171888114,
    -0.02798376941698385,
    0.6308807679295904,
    0.7148465705525415,
    0.23037781330885523,
]

_SYM4_DEC_LO = [
    -0.07576571478927333,
    -0.02963552764599851,
    0.49761866763201545,
    0.8037387518059161,
    0.29785779560527736,
    -0.09921954357684722,
    -0.012603967262037833,
    0.0322231006040427,
]

_SYM6_DEC_LO = [
    0.015404109327027373,
    0.0034907120842174702,
    -0.11799011114819057,
    -0.048311742585633,
    0.4910559419267466,
    0.787641141030194,
    0.3379294217276218,
    -0.07263752278646252,
    -0.021060292512300564,
    0.04472490177066578,
    0.0017677118642428036,
    -0.007800708325034148,
]


def _build_filter_bank(dec_lo: Sequence[float]) -> dict[str, torch.Tensor]:
    """
    Build the orthogonal QMF filter bank from a decomposition lowpass filter.

    Conventions match PyWavelets:
      dec_hi[k] = (-1)^k * dec_lo[N-1-k]
      rec_lo    = reverse(dec_lo)
      rec_hi    = reverse(dec_hi)
    """
    h = torch.tensor(list(dec_lo), dtype=torch.float64)
    n = h.numel()
    signs = torch.tensor([(-1.0) ** k for k in range(n)], dtype=torch.float64)
    dec_hi = signs * h.flip(0)
    rec_lo = h.flip(0)
    rec_hi = dec_hi.flip(0)
    return {
        "dec_lo": h,
        "dec_hi": dec_hi,
        "rec_lo": rec_lo,
        "rec_hi": rec_hi,
    }


_WAVELET_BANKS: dict[str, dict[str, torch.Tensor]] = {
    "haar": _build_filter_bank(_HAAR_DEC_LO),
    "db1": _build_filter_bank(_HAAR_DEC_LO),
    "db2": _build_filter_bank(_DB2_DEC_LO),
    "db4": _build_filter_bank(_DB4_DEC_LO),
    "sym4": _build_filter_bank(_SYM4_DEC_LO),
    "sym6": _build_filter_bank(_SYM6_DEC_LO),
}


def available_wavelets() -> list[str]:
    return sorted(_WAVELET_BANKS.keys())


def _get_filter_bank(name: str, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    key = str(name).strip().lower()
    if key not in _WAVELET_BANKS:
        raise ValueError(
            f"Unsupported wavelet={name!r}. Expected one of: {available_wavelets()}"
        )
    bank = _WAVELET_BANKS[key]
    return {k: v.to(device=device, dtype=dtype) for k, v in bank.items()}


# ---------------------------------------------------------------------------
# 1-D DWT / IDWT along the leading (step) axis with periodization.
# ---------------------------------------------------------------------------


_DWT_MATRIX_CACHE: dict[tuple[str, int, str, str], torch.Tensor] = {}


def _build_dwt_matrix(
    *,
    wavelet: str,
    T: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build the orthogonal DWT analysis matrix W of shape (T, T) for periodic
    boundary. The first T/2 rows hold the lowpass projections and the last T/2
    rows hold the highpass projections, so y = W @ x packs (cA, cD) along the
    leading axis.

    For orthogonal wavelets W is unitary, so the IDWT matrix is W.transpose().
    """
    if T < 2 or T % 2 != 0:
        raise ValueError(f"DWT matrix requires even T>=2, got T={T}.")
    cache_key = (str(wavelet).lower(), int(T), str(device), str(dtype))
    cached = _DWT_MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    bank = _get_filter_bank(wavelet, device=device, dtype=dtype)
    h = bank["dec_lo"]
    g = bank["dec_hi"]
    L = int(h.numel())
    W = torch.zeros(T, T, device=device, dtype=dtype)
    half = T // 2
    for n in range(half):
        for k in range(L):
            col = (2 * n + k) % T
            W[n, col] = W[n, col] + h[k]
            W[half + n, col] = W[half + n, col] + g[k]
    _DWT_MATRIX_CACHE[cache_key] = W
    return W


def _dwt1d(x: torch.Tensor, wavelet: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single-level DWT along dim 0 with periodization.

    Args:
      x: (T, *F) tensor with even T.
      wavelet: wavelet name (e.g., 'sym6', 'haar').

    Returns:
      (cA, cD) each of shape (T/2, *F).
    """
    if x.ndim < 1:
        raise ValueError("DWT requires at least a 1-D tensor.")
    T = int(x.shape[0])
    if T < 2 or T % 2 != 0:
        raise ValueError(f"DWT requires even T>=2 along step axis, got T={T}.")
    W = _build_dwt_matrix(wavelet=wavelet, T=T, device=x.device, dtype=x.dtype)
    rest_shape = x.shape[1:]
    flat = x.reshape(T, -1)
    y = W @ flat  # (T, prod(rest))
    half = T // 2
    cA = y[:half].reshape(half, *rest_shape)
    cD = y[half:].reshape(half, *rest_shape)
    return cA, cD


def _idwt1d(cA: torch.Tensor, cD: torch.Tensor, wavelet: str) -> torch.Tensor:
    """
    Single-level IDWT along dim 0 with periodization.

    Args:
      cA, cD: (T_c, *F) approximation and detail coefficients.
      wavelet: wavelet name (e.g., 'sym6', 'haar').

    Returns:
      Reconstruction of shape (2*T_c, *F).
    """
    if cA.shape != cD.shape:
        raise ValueError(f"DWT band shape mismatch: cA={tuple(cA.shape)} cD={tuple(cD.shape)}")
    T_c = int(cA.shape[0])
    if T_c == 0:
        return torch.zeros(0, *cA.shape[1:], device=cA.device, dtype=cA.dtype)
    T = 2 * T_c
    W = _build_dwt_matrix(wavelet=wavelet, T=T, device=cA.device, dtype=cA.dtype)
    rest_shape = cA.shape[1:]
    y = torch.cat([cA.reshape(T_c, -1), cD.reshape(T_c, -1)], dim=0)  # (T, prod(rest))
    x = W.transpose(0, 1) @ y  # (T, prod(rest))
    return x.reshape(T, *rest_shape)


# ---------------------------------------------------------------------------
# Public WaveEx config and runtime.
# ---------------------------------------------------------------------------


@dataclass
class WaveExConfig:
    """
    Configuration for a WaveEx-accelerated sampling run.

    `ode_step_indices` lists the step indices (0-based, in [0, num_steps)) at
    which the model is evaluated normally. All other indices are predicted via
    wavelet-guided extrapolation.

    By default the schedule follows the operating point that won an internal
    sweep on the duration-control checkpoint (40 NFE → 6 ODE steps at indices
    [0, 1, 2, 5, 10, 20]; haar + 1st-order direct Taylor on a 2-frame buffer)
    and is rescaled to the actual num_steps when the sampler calls
    `resolve_ode_step_indices`.
    """

    enabled: bool = False
    ode_step_indices: tuple[int, ...] | None = None
    wavelet: str = "haar"
    taylor_order: int = 1
    history_size: int = 2
    high_freq_mode: str = "extrapolate"

    def __post_init__(self) -> None:
        if self.taylor_order not in (1, 2):
            raise ValueError(
                f"taylor_order must be 1 or 2, got {self.taylor_order}."
            )
        if self.high_freq_mode not in {"extrapolate", "freeze", "zero"}:
            raise ValueError(
                "high_freq_mode must be one of: extrapolate, freeze, zero; "
                f"got {self.high_freq_mode!r}."
            )
        if self.history_size < 2:
            raise ValueError(f"history_size must be >= 2, got {self.history_size}.")
        if self.wavelet.lower() not in _WAVELET_BANKS:
            raise ValueError(
                f"Unsupported wavelet={self.wavelet!r}. "
                f"Expected one of: {available_wavelets()}"
            )

    def resolve_ode_step_indices(self, num_steps: int) -> set[int]:
        """
        Return the set of step indices at which a full ODE evaluation must run.

        If `ode_step_indices` is not set explicitly, derive a schedule from
        the calibrated default (40 NFE: ODE at [0, 1, 2, 5, 10, 20]) by linear
        rescaling to the requested `num_steps`. Step 0 is always kept so the
        buffer can warm up.
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}.")
        if self.ode_step_indices is not None:
            indices = {int(i) for i in self.ode_step_indices if 0 <= int(i) < num_steps}
            indices.add(0)
            return indices

        # Calibrated default: 6 ODE steps out of 40 NFE.
        paper_default = (0, 1, 2, 5, 10, 20)
        paper_total = 40
        if num_steps == paper_total:
            return set(paper_default)
        scale = num_steps / paper_total
        scaled = {0}
        for idx in paper_default:
            mapped = int(round(idx * scale))
            if 0 <= mapped < num_steps:
                scaled.add(mapped)
        return scaled


class WaveExBuffer:
    """
    Sliding window of past latent states with wavelet-guided extrapolation.
    """

    def __init__(self, cfg: WaveExConfig) -> None:
        self.cfg = cfg
        self._history: deque[torch.Tensor] = deque(maxlen=int(cfg.history_size))

    def reset(self) -> None:
        self._history.clear()

    def push(self, x: torch.Tensor) -> None:
        # Detach is unnecessary inside torch.inference_mode but keeps the
        # buffer self-contained for any caller that disables it.
        self._history.append(x.detach())

    def __len__(self) -> int:
        return len(self._history)

    def predict_next(self) -> torch.Tensor:
        """
        Predict the next latent state from the current buffer using the
        configured wavelet + Taylor extrapolation. Falls back to direct
        Taylor extrapolation when the buffer is too short for a meaningful
        wavelet decomposition.
        """
        if len(self._history) < 2:
            raise RuntimeError(
                "WaveExBuffer.predict_next requires at least 2 past states; "
                "extrapolation cannot be the very first step."
            )

        order = int(self.cfg.taylor_order)
        history = list(self._history)
        if len(history) >= 4 and len(history) >= order + 1:
            stack = torch.stack(history, dim=0)  # (T, *)
            return self._wavelet_predict(stack)
        return self._taylor_predict_direct()

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _wavelet_predict(self, stack: torch.Tensor) -> torch.Tensor:
        # Use highest available power-of-two prefix so DWT/IDWT round trips cleanly.
        T = int(stack.shape[0])
        if T % 2 == 1:
            stack = stack[1:]  # drop the oldest sample to keep T even
            T -= 1

        # Internal computations in float32 for numerical stability of small
        # wavelet filters when the model runs in bf16/fp16.
        compute_dtype = torch.float32
        original_dtype = stack.dtype
        x = stack.to(compute_dtype)

        cA, cD = _dwt1d(x, self.cfg.wavelet)

        cA_next = _taylor_extrapolate(cA, order=int(self.cfg.taylor_order))
        if self.cfg.high_freq_mode == "extrapolate":
            cD_next = _taylor_extrapolate(cD, order=int(self.cfg.taylor_order))
        elif self.cfg.high_freq_mode == "freeze":
            cD_next = torch.cat([cD, cD[-1:]], dim=0)
        else:  # zero
            cD_next = torch.cat([cD, torch.zeros_like(cD[-1:])], dim=0)

        rec = _idwt1d(cA_next, cD_next, self.cfg.wavelet)
        # Extending each band by 1 coefficient adds 2 new time samples to the
        # reconstruction (one even + one odd). rec[-2] is the immediate next
        # state x_{n+1}; rec[-1] would be x_{n+2}.
        return rec[-2].to(original_dtype)

    def _taylor_predict_direct(self) -> torch.Tensor:
        history = list(self._history)
        order = min(int(self.cfg.taylor_order), len(history) - 1)
        x_n = history[-1]
        if order >= 1 and len(history) >= 2:
            d1 = history[-1] - history[-2]
        else:
            return x_n
        x_next = x_n + d1
        if order >= 2 and len(history) >= 3:
            d2 = history[-1] - 2 * history[-2] + history[-3]
            x_next = x_next + 0.5 * d2
        return x_next


def _taylor_extrapolate(c: torch.Tensor, *, order: int) -> torch.Tensor:
    """
    Append one Taylor-extrapolated sample at the end of `c` along dim 0.

    For first-order with uniform spacing:
      c_{n+1} = 2 c_n - c_{n-1}
    For second-order with uniform spacing:
      c_{n+1} = 3 c_n - 3 c_{n-1} + c_{n-2}
    """
    if c.shape[0] < 2:
        raise ValueError(f"Taylor extrapolation needs >=2 samples, got {c.shape[0]}.")
    if order >= 2 and c.shape[0] >= 3:
        new = 3.0 * c[-1] - 3.0 * c[-2] + c[-3]
    else:
        new = 2.0 * c[-1] - c[-2]
    return torch.cat([c, new.unsqueeze(0)], dim=0)


def parse_ode_step_indices(spec: str | None, *, num_steps: int) -> tuple[int, ...] | None:
    """
    Parse a CLI specification of explicit ODE step indices.

    Accepts:
      * None / "" / "auto" — return None (sampler uses the paper default)
      * "0,2,4,6,8,14"     — comma-separated step indices
    """
    if spec is None:
        return None
    text = str(spec).strip()
    if text == "" or text.lower() in {"auto", "default"}:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if not parts:
        return None
    indices: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(
                f"Invalid ODE step index {part!r} in --waveex-ode-steps={spec!r}."
            ) from exc
        if value < 0 or value >= num_steps:
            raise ValueError(
                f"--waveex-ode-steps index {value} is out of range [0, {num_steps})."
            )
        indices.append(value)
    return tuple(sorted(set(indices)))


__all__ = [
    "WaveExBuffer",
    "WaveExConfig",
    "available_wavelets",
    "parse_ode_step_indices",
]

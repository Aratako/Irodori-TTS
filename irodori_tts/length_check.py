"""Two-pass synthesis that trims an overlong duration prediction (``--length-check on``).

The text has no end marker, so when the predicted duration is longer than the text needs, the model
reads the text at its usual pace and then fills the remaining frames: it says words that are not in
the text, or goes back and reads part of it again. Short sentences suffer most, because the duration
predictor adds a fixed amount for the leading BOS (about 0.85 s) and, for short declarative sentences,
often for the final "。" (1-1.7 s).

In the v4.1 Small DiT, heads 2 and 4 of layer 0 act as a pointer to the text token being read: for
each latent frame, the text position with the largest attention score follows the reading (correlation
0.99). From the pointer of the last sampling step of a first pass we estimate where the reading ends,
and when the first pass is clearly longer, synthesize again with the same seed at that length.

* ``reading_end`` fits a monotonic reading path to the pointer by dynamic programming (DTW-like), so
  that a momentary jump to the end of the text (common during pauses, and towards a sentence-final
  "。" in the middle of a text) is not taken as the end. Each frame costs the (capped) distance between
  the pointer and the path, skipping tokens costs extra, and frames pointing at the final punctuation
  are only weak evidence.
* ``estimated_frames`` = frame where the path reaches the last word + the duration the predictor gave
  to that word + a short tail, times a margin. Adding the last word's own duration matters for long
  final tokens (e.g. 「ければなりません」): a fixed tail made such sentences much too fast. The estimate
  is never shorter than ``FLOOR`` times the predicted duration of the words (BOS and final punctuation
  excluded), which bounds what any pointer glitch can do.
* The second pass is used only when the estimate is at least ``RESYNTH_SHRINK`` shorter than the
  first pass. Re-synthesis changes the reading a little (for better or worse), so a small excess is
  better left alone. The first pass is stopped at its last DiT call, before decoding, when it will be
  re-synthesized. If the pointer of the second pass does not reach the end, it is retried once a
  little longer, and otherwise the first-pass length is used.

The pointer heads are a property of the trained weights, so this only runs on RF models with the
v4.1 Small architecture (12 layers, 20 heads, width 1280, including fine-tunes); other models are
synthesized once as usual. It also needs the independent CFG mode (the default), which makes one DiT
call per sampling step with the conditional row first; joint and alternating CFG make a separate
unconditional call, so the call count no longer marks the last step.

Measured with v4.1 Small (FP32, 40 steps, no reference, Haas-ja-v1.0 + jacer kana CER):
488 takes of 61 sentences that often get extra speech at the end: speech after the text (5+ kana)
49 -> 0, kana CER 12.86% -> 4.54%. JSUT basic5000 every 10th sentence: see the pull request.
"""
from __future__ import annotations

import dataclasses
import math
import secrets

import numpy as np
import torch

POINTER_LAYER = 0
POINTER_HEADS = (2, 4)
# Tail after reaching the last word, when the predictor gives no per-token durations.
TAIL_SECONDS = 0.5
# Tail after the last word's predicted duration.
LAST_WORD_TAIL_SECONDS = 0.3
# Never estimate less than this times the predicted duration of the words (about 9 kana/s).
FLOOR = 0.8
MARGIN = 1.1
# Re-synthesize only when the estimate is at least this much shorter than the first pass.
RESYNTH_SHRINK = 0.10
# When a too-short second pass is retried longer, give up once it is within this of the first pass.
MIN_SHRINK = 0.05
# Cost per frame of the "finished reading" state (same as being one token off).
ABSORB_COST = 1.0
# Cap of the per-frame distance cost, in tokens. Without it a long jump of the pointer to the last
# word (0.4 s or more) outweighed ending there, and the estimate could drop below half.
DIST_CAP = 3.0
# Cost per token skipped by the path (advancing by one token is free).
SKIP_COST = 2.0
# A final token made only of these is punctuation; frames pointing at it are weak evidence.
FINAL_PUNCT = "。．.！!？?…‥、,〜~」』）)"
PUNCT_COST = 0.25
RETRIES = 1
RETRY_GROWTH = 1.25
# Sampling steps when the request leaves num_steps unset. Must match the runtime's default for RF
# models (InferenceRuntime.synthesize_batch).
DEFAULT_RF_STEPS = 40


def supported(model_cfg) -> bool:
    """Same architecture as the model whose pointer heads were checked."""
    return (getattr(model_cfg, "flow_parameterization", "rf_velocity") != "meanflow"
            and model_cfg.num_layers == 12 and model_cfg.num_heads == 20
            and model_cfg.model_dim == 1280)


class PointerRecorder:
    """Pre-hook on the layer-0 attention. For the first conditional row, records which text position
    each latent frame attends to most (argmax over text keys, larger of the two heads). Overwritten on
    every call, so after sampling it holds the pointer of the last step. ``on_call(self)`` runs after
    every call; raising from it stops the synthesis (``_Stop``)."""

    def __init__(self, model, on_call=None):
        self.attention = model.blocks[POINTER_LAYER].attention
        self.pointer: torch.Tensor | None = None
        self.text_len = 0
        self.calls = 0
        self.on_call = on_call
        self.handle = self.attention.register_forward_pre_hook(self._hook, with_kwargs=True)

    def close(self) -> None:
        self.handle.remove()

    @torch.no_grad()
    def _hook(self, mod, args, kwargs):
        x = kwargs["x"][:1]
        seq_len = x.shape[1]
        q = mod.wq(x).reshape(1, seq_len, mod.heads, mod.head_dim)
        q = mod._apply_rotary_half(mod.q_norm(q), kwargs["freqs_cis"][:seq_len])
        context_kv = kwargs.get("context_kv")
        if context_kv is not None:
            k_text = context_kv[0][:1]
        else:
            text = kwargs["text_context"][:1]
            k_text = mod.k_norm(mod.wk_text(text).reshape(1, text.shape[1], mod.heads, mod.head_dim))
        mask = kwargs.get("text_mask")
        valid = (mask[0].bool() if mask is not None
                 else torch.ones(k_text.shape[1], dtype=torch.bool, device=x.device))
        heads = list(POINTER_HEADS)
        # argmax over the text keys only, so the softmax denominator is not needed
        logits = torch.einsum("bshd,bthd->bhst", q[:, :, heads].float(), k_text[:, :, heads].float())
        logits = logits.masked_fill(~valid[None, None, None, :], float("-inf"))
        self.pointer = logits.argmax(-1).amax(1)[0]  # (S,)
        self.text_len = int(valid.sum())
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self)


def ends_with_punct(text: str | None) -> bool:
    return bool(text) and text.rstrip()[-1:] in FINAL_PUNCT


def is_punct_token(token: str | None) -> bool:
    """The token (without the "▁" space marker) is punctuation only."""
    t = (token or "").replace("▁", "")
    return bool(t) and all(ch in FINAL_PUNCT for ch in t)


def last_token_is_punct(text: str | None, tokens: dict | None = None) -> bool:
    """Whether the last token of the synthesized text is punctuation.

    The runtime normalizes the text before tokenizing it (outer brackets are removed, 〜 becomes ー,
    ...), so the last character of the original text can be wrong: 「今日はいい天気です」 ends with
    「す」, not with a closing bracket. Uses the last token kept by the runtime (``duration_tokens``),
    or else the normalized text.
    """
    if tokens is not None and tokens.get("last_token") is not None:
        return is_punct_token(tokens["last_token"])
    if not text:
        return False
    from .text_normalization import normalize_text

    return ends_with_punct(normalize_text(text).strip())


def single_call_per_step(request) -> bool:
    """Independent CFG (the default) makes one DiT call per step, conditional row first. Joint and
    alternating CFG also make an unconditional call per step, which would be counted as a step and
    overwrite the pointer (with an empty text in joint mode)."""
    mode = getattr(request, "cfg_guidance_mode", None) or "independent"
    return str(mode).strip().lower() == "independent"


def reading_end(pointer, text_len: int, final_punct: bool = False) -> int | None:
    """First frame where a monotonic reading path fitted to the pointer reaches the second-to-last
    token, or None if it never does.

    States: the token being read (0..L-1, never going back) and "finished" (-1), which can only be
    entered from token L-2 or later and costs ABSORB_COST per frame whatever the pointer says. A frame
    costs min(|pointer - state|, DIST_CAP); moving k tokens ahead costs SKIP_COST * (k - 1) (the path
    may start at token 0 for free). With ``final_punct`` the last token is punctuation and a frame
    pointing at it costs PUNCT_COST for states before the last word and nothing after.
    O(L x frames), a few ms for a long sentence.
    """
    if isinstance(pointer, torch.Tensor):
        pointer = pointer.detach().cpu().numpy()
    last = text_len - 2
    p = np.clip(np.asarray(pointer, dtype=np.float32), 0, text_len - 1)
    steps = len(p)
    if text_len < 3 or steps == 0:
        return None
    pos = np.arange(text_len, dtype=np.float32)
    flat = np.where(pos >= last, 0.0, PUNCT_COST).astype(np.float32)

    def row(v):
        return flat if final_punct and v == text_len - 1 else np.minimum(np.abs(v - pos), DIST_CAP)

    cost = np.empty((steps, text_len), dtype=np.float32)
    done = np.empty(steps, dtype=np.float32)
    cost[0] = row(p[0]) + SKIP_COST * pos
    done[0] = np.inf
    # via[s]: cheapest way to arrive at s from two or more tokens back
    via = np.empty(text_len, dtype=np.float32)
    via[0] = np.inf
    for t in range(1, steps):
        prev = cost[t - 1]
        via[1:] = np.minimum.accumulate(prev - SKIP_COST * pos)[:-1] + SKIP_COST * (pos[1:] - 1)
        cost[t] = row(p[t]) + np.minimum(prev, via)
        done[t] = ABSORB_COST + min(done[t - 1], prev[last:].min())
    # Trace back. The path never goes back, so the part at L-2 or later is one run at the end.
    state = -1 if done[-1] <= cost[-1].min() else int(cost[-1].argmin())
    end = None
    for t in range(steps - 1, -1, -1):
        if state != -1 and state < last:
            break
        end = t
        if t == 0:
            break
        if state == -1:
            tail = cost[t - 1, last:]
            state = -1 if done[t - 1] <= tail.min() else last + int(tail.argmin())
        else:
            back = cost[t - 1, :state + 1].copy()
            if state >= 1:
                back[:state] += SKIP_COST * (state - 1 - pos[:state])
            state = int(back.argmin())
    return end


def duration_tokens(runtime, text_len: int) -> dict | None:
    """Per-token duration prediction left by the runtime (InferenceRuntime._duration_tokens), or None
    when it is missing or does not match the pointer's text."""
    tokens = getattr(runtime, "last_duration_tokens", None)
    if not tokens or len(tokens.get("frames") or ()) != text_len:
        return None
    return tokens


def estimated_frames(pointer, text_len: int, frames_per_step: int, frames_per_second: float,
                     text: str | None = None, tokens: dict | None = None) -> int | None:
    """Latent frames needed to finish reading (with margin), or None if the reading never ends.
    With ``tokens`` (duration_tokens): reach of the last word + its predicted frames (the last two
    tokens when the last token is not punctuation, see ``last_token_is_punct``) +
    LAST_WORD_TAIL_SECONDS, at least FLOOR times the predicted frames of the words.
    Without: reach + TAIL_SECONDS."""
    punct = last_token_is_punct(text, tokens)
    end = reading_end(pointer, text_len, punct)
    if end is None:
        return None
    end_frames = (end + 1) * frames_per_step
    if tokens is None:
        return int(math.ceil((end_frames + TAIL_SECONDS * frames_per_second) * MARGIN))
    tf = tokens["frames"]
    remain = tf[text_len - 2] if punct else tf[text_len - 2] + tf[text_len - 1]
    frames = int(math.ceil((end_frames + remain + LAST_WORD_TAIL_SECONDS * frames_per_second) * MARGIN))
    return max(frames, int(math.ceil(FLOOR * tokens["content_frames"])))


class _Stop(Exception):
    """Raised from a pointer callback to stop a synthesis."""

    def __init__(self, frames=None, first_frames=None):
        super().__init__("length_check stop")
        self.frames, self.first_frames = frames, first_frames


def synthesize(runtime, request, log_fn=None):
    """Two-pass synthesis. Returns (result, info). Without a second pass the first result is returned.

    Skipped (one normal pass) when ``request.seconds`` is given, the model is not supported, the CFG
    mode is not independent (``single_call_per_step``), or more than one candidate is requested (the
    pointer is read from the first candidate only, and a length chosen for it could cut the others).
    The seed is fixed before the first pass (random if unset) so the second pass starts from the same
    noise. When the last DiT call of the first pass shows that it is too long, the first pass stops
    there without decoding. If the second pass does not finish reading (checked at its last call,
    before decoding), it is retried once RETRY_GROWTH longer, else the first-pass length is used.
    """
    def log(message: str) -> None:
        if log_fn is not None:
            log_fn(message)

    info = {"length_check": "skipped"}
    if (request.seconds is not None or not supported(runtime.model_cfg)
            or not single_call_per_step(request)
            or int(getattr(request, "num_candidates", 1) or 1) != 1):
        return runtime.synthesize(request, log_fn=log_fn), info
    sample_rate = int(runtime.codec.sample_rate)
    hop = int(runtime.codec.model.hop_length)
    patch = int(runtime.model_cfg.latent_patch_size)
    fps = sample_rate / hop

    def shrink_to(recorder):
        """(frames to re-synthesize or None, first-pass frames, estimate)."""
        first_frames = int(recorder.pointer.shape[0]) * patch
        estimate = estimated_frames(recorder.pointer, recorder.text_len, patch, fps, request.text,
                                    duration_tokens(runtime, recorder.text_len))
        if estimate is None or estimate >= first_frames * (1 - RESYNTH_SHRINK):
            return None, first_frames, estimate
        return estimate, first_frames, estimate

    # The last DiT call of the first pass is call number `steps` (one call per step, see
    # single_call_per_step). An unset num_steps must resolve to the runtime's default, or the check
    # never fires there and an overlong first pass is decoded only to be thrown away.
    steps = int(request.num_steps) if request.num_steps else DEFAULT_RF_STEPS

    def at_last_call(recorder):
        if recorder.calls == steps:
            frames, first_frames, _ = shrink_to(recorder)
            if frames is not None:
                raise _Stop(frames, first_frames)

    seed = int(request.seed) if request.seed is not None else int(secrets.randbits(63))
    first_request = dataclasses.replace(request, seed=seed)
    recorder = PointerRecorder(runtime.model, at_last_call)
    first = None
    try:
        first = runtime.synthesize(first_request, log_fn=log_fn)
    except _Stop as stop:
        frames, first_frames = stop.frames, stop.first_frames
    finally:
        recorder.close()
    if first is not None:
        if recorder.pointer is None:
            return first, info
        frames, first_frames, estimate = shrink_to(recorder)
        info = {"length_check": "kept", "first_frames": first_frames, "estimated_frames": estimate}
        if frames is None:
            log(f"[length_check] kept {first_frames} frames (estimate {estimate})")
            return first, info
    info = {"length_check": "resynthesized", "first_frames": first_frames, "estimated_frames": frames}
    log(f"[length_check] {first_frames} -> {frames} frames "
        f"({first_frames / fps:.2f}s -> {frames / fps:.2f}s)")
    last_call = recorder.calls
    # Decided from the first pass's tokens (the runtime drops them at the start of each synthesis).
    punct = last_token_is_punct(request.text, getattr(runtime, "last_duration_tokens", None))

    def verify(rec):
        if rec.calls == last_call and reading_end(rec.pointer, rec.text_len, punct) is None:
            raise _Stop()

    second = None
    for attempt in range(1 + RETRIES):
        checker = PointerRecorder(runtime.model, verify)
        try:
            second = runtime.synthesize(dataclasses.replace(request, seconds=frames / fps, seed=seed),
                                        log_fn=log_fn)
            break
        except _Stop:
            pass
        finally:
            checker.close()
        longer = int(math.ceil(frames * RETRY_GROWTH))
        log(f"[length_check] {frames} frames did not reach the end"
            + (f"; retrying with {longer}" if attempt < RETRIES and longer < first_frames * (1 - MIN_SHRINK) else ""))
        if longer >= first_frames * (1 - MIN_SHRINK):
            break
        frames = longer
        info["estimated_frames"] = frames
    if second is None:
        info["length_check"] = "fallback"
        if first is None:
            first = runtime.synthesize(first_request, log_fn=log_fn)
        return first, info
    if attempt:
        info["retries"] = attempt
    return second, info

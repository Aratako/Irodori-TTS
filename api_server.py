#!/usr/bin/env python3
"""
FastAPI server for Irodori-TTS — multi-worker, async.

Endpoint:
  POST /synthesize
    - text              : desired synthesis text
    - reference_audio   : audio file (wav/mp3/flac/…) for voice cloning (optional;
                          omit to run in no-reference / voice-design mode)
    - reference_text    : transcript of reference audio (reserved; not used by model)
    - seconds           : output duration in seconds   (default: 30.0)
    - num_steps         : diffusion steps              (default: 40)
    - cfg_scale_text    : text CFG scale               (default: 3.0)
    - cfg_scale_speaker : speaker CFG scale            (default: 5.0)
    - seed              : integer seed; omit=random    (default: None)

  Returns: audio/wav (mono, codec sample-rate)

  GET /health  →  {"status":"ok","pool_size":N,"available":K}

Usage (single GPU, 1 worker):
  python api_server.py --checkpoint Aratako/Irodori-TTS-500M-v2 --devices cuda:0

Usage (two GPUs, 2 workers):
  python api_server.py --checkpoint Aratako/Irodori-TTS-500M-v2 --devices cuda:0,cuda:1

Usage (one GPU, 2 workers — both share the same device):
  python api_server.py --checkpoint Aratako/Irodori-TTS-500M-v2 --devices cuda:0 --num-workers 2
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import tempfile
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    resolve_runtime_device,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime pool
# ---------------------------------------------------------------------------

class RuntimePool:
    """Async pool of InferenceRuntime workers."""

    def __init__(self, runtimes: list[InferenceRuntime]) -> None:
        if not runtimes:
            raise ValueError("RuntimePool requires at least one runtime.")
        self._queue: asyncio.Queue[InferenceRuntime] = asyncio.Queue()
        for rt in runtimes:
            self._queue.put_nowait(rt)
        self._size = len(runtimes)

    @property
    def size(self) -> int:
        return self._size

    @property
    def available(self) -> int:
        return self._queue.qsize()

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[InferenceRuntime]:
        runtime = await self._queue.get()
        try:
            yield runtime
        finally:
            self._queue.put_nowait(runtime)

    def unload_all(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait().unload()
            except asyncio.QueueEmpty:
                break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint(checkpoint: str) -> str:
    checkpoint = checkpoint.strip()
    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"} or Path(checkpoint).exists():
        return checkpoint
    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
    logger.info("downloaded checkpoint: hf://%s → %s", checkpoint, resolved)
    return resolved


def _build_runtime(
    checkpoint: str,
    device: str,
    precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
) -> InferenceRuntime:
    key = RuntimeKey(
        checkpoint=checkpoint,
        model_device=device,
        model_precision=precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
    )
    logger.info("loading worker on %s (%s) …", device, precision)
    rt = InferenceRuntime.from_key(key)
    logger.info("worker ready on %s.", device)
    return rt


def _audio_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    # torchcodec backend (used in newer torchaudio) cannot write to BytesIO —
    # it requires a real file path. Write to a temp file and read back.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torchaudio.save(tmp_path, audio.float(), sample_rate)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Irodori-TTS API")


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI, args: argparse.Namespace) -> AsyncIterator[None]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    num_workers = args.num_workers
    worker_devices = [devices[i % len(devices)] for i in range(num_workers)]
    codec_precision = args.codec_precision or args.precision

    runtimes = [
        _build_runtime(
            checkpoint=checkpoint,
            device=dev,
            precision=args.precision,
            codec_device=dev,
            codec_precision=codec_precision,
            enable_watermark=args.enable_watermark,
        )
        for dev in worker_devices
    ]

    pool = RuntimePool(runtimes)
    executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="tts-worker")
    logger.info("pool ready: %d worker(s) on %s", num_workers, worker_devices)

    app.state.pool = pool
    app.state.executor = executor

    yield

    executor.shutdown(wait=True)
    pool.unload_all()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/synthesize",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def synthesize(
    request: Request,
    text: Annotated[str, Form()],
    reference_audio: Annotated[UploadFile | None, File()] = None,
    reference_text: Annotated[str | None, Form()] = None,
    seconds: Annotated[float, Form()] = 30.0,
    num_steps: Annotated[int, Form()] = 40,
    cfg_scale_text: Annotated[float, Form()] = 3.0,
    cfg_scale_speaker: Annotated[float, Form()] = 5.0,
    seed: Annotated[int | None, Form()] = None,
) -> Response:
    _ = reference_text  # accepted for API compatibility; not used by current model

    pool: RuntimePool = request.app.state.pool
    executor: ThreadPoolExecutor = request.app.state.executor

    if not text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty.")
    if seconds <= 0:
        raise HTTPException(status_code=422, detail="seconds must be > 0.")
    if num_steps < 1:
        raise HTTPException(status_code=422, detail="num_steps must be >= 1.")

    # Read upload bytes before handing off to the thread.
    no_ref = reference_audio is None
    audio_bytes: bytes = b""
    ref_suffix = ".wav"
    if not no_ref:
        audio_bytes = await reference_audio.read()  # type: ignore[union-attr]
        ref_suffix = Path(reference_audio.filename or "ref.wav").suffix or ".wav"  # type: ignore[union-attr]

    def _run_synthesis(runtime: InferenceRuntime) -> bytes:
        ref_wav_path: str | None = None
        if not no_ref:
            with tempfile.NamedTemporaryFile(suffix=ref_suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                ref_wav_path = tmp.name
        try:
            req = SamplingRequest(
                text=text,
                ref_wav=ref_wav_path,
                no_ref=no_ref,
                seconds=seconds,
                num_steps=num_steps,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
                seed=seed,
            )
            result = runtime.synthesize(req, log_fn=logger.debug)
        finally:
            if ref_wav_path is not None:
                Path(ref_wav_path).unlink(missing_ok=True)
        return _audio_to_wav_bytes(result.audio, result.sample_rate)

    loop = asyncio.get_running_loop()
    try:
        async with pool.acquire() as runtime:
            wav_bytes = await loop.run_in_executor(executor, _run_synthesis, runtime)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@app.get("/health")
def health(request: Request) -> dict:
    pool: RuntimePool | None = getattr(request.app.state, "pool", None)
    if pool is None:
        return {"status": "starting", "pool_size": 0, "available": 0}
    return {"status": "ok", "pool_size": pool.size, "available": pool.available}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Irodori-TTS FastAPI server.")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Local .pt/.safetensors path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--devices", default="cuda",
        help=(
            "Comma-separated list of devices for workers "
            "(e.g. 'cuda:0,cuda:1'). Cycled if fewer than --num-workers. "
            "Default: cuda"
        ),
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel inference workers. Defaults to number of devices.",
    )
    parser.add_argument(
        "--precision", default="fp32", choices=["fp32", "bf16"],
        help="Model precision (default: fp32).",
    )
    parser.add_argument(
        "--codec-precision", default=None, choices=["fp32", "bf16"],
        help="Codec precision (defaults to --precision).",
    )
    parser.add_argument("--enable-watermark", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        raise SystemExit("--devices must not be empty.")
    for dev in devices:
        resolve_runtime_device(dev)

    if args.num_workers is None:
        args.num_workers = len(devices)
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be >= 1.")

    # Bind the lifespan with startup args via a wrapper.
    @contextlib.asynccontextmanager
    async def lifespan(a: FastAPI) -> AsyncIterator[None]:
        async with _lifespan(a, args):
            yield

    app.router.lifespan_context = lifespan

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()

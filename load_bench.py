import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    ModelConfig,
    _load_checkpoint_for_inference,
    download_hf_checkpoint,
    is_torchao_quantized_state_dict,
    merge_dataclass_overrides,
)
from irodori_tts.model import TextToLatentRFDiT

CHECKPOINT_REPO = os.environ.get("IRODORI_BENCH_CHECKPOINT", "Aratako/Irodori-TTS-v4-Small")


def _sysinfo() -> str:
    try:
        free = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=5).stdout
        free_line = [line_ for line_ in free.splitlines() if line_.startswith("Mem:")][0]
        swap_line = [line_ for line_ in free.splitlines() if line_.startswith("Swap:")][0]
    except Exception as exc:  # noqa: BLE001
        free_line = swap_line = f"ERR {exc}"
    try:
        pids = subprocess.run(
            ["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=5
        ).stdout
        gpu_procs = [
            line_ for line_ in pids.splitlines() if line_.strip() and line_[0].isdigit()
        ]
    except Exception as exc:  # noqa: BLE001
        gpu_procs = [f"ERR {exc}"]
    return f"{free_line} | {swap_line} | gpu_procs={gpu_procs}"


def main() -> None:
    print(f"[sysinfo-before] {_sysinfo()}", flush=True)

    print(f"[cfg] checkpoint={CHECKPOINT_REPO}", flush=True)
    ckpt_path = Path(download_hf_checkpoint(CHECKPOINT_REPO))
    t0 = time.perf_counter()
    model_state, model_cfg_dict, train_cfg, text_encoder_config = _load_checkpoint_for_inference(
        ckpt_path
    )
    print(f"[t] load_checkpoint: {time.perf_counter() - t0:.3f}s", flush=True)
    quantized_model = is_torchao_quantized_state_dict(model_state)
    print(f"[cfg] quantized_model={quantized_model}", flush=True)

    model_cfg = merge_dataclass_overrides(ModelConfig(), model_cfg_dict, section="x")
    t0 = time.perf_counter()
    model = TextToLatentRFDiT(
        model_cfg,
        pretrained_backbone_config=text_encoder_config,
        load_pretrained_backbone_weights=not model_cfg.use_pretrained_text_encoder,
    )
    print(f"[t] construct: {time.perf_counter() - t0:.3f}s", flush=True)

    assign_env = os.environ.get("IRODORI_BENCH_ASSIGN")
    if assign_env is None:
        assign = model_cfg.use_pretrained_text_encoder or quantized_model
    else:
        assign = assign_env == "1"
    print(f"[cfg] assign={assign}", flush=True)

    pre_move_state = os.environ.get("IRODORI_BENCH_PREMOVE_STATE") == "1"
    clone_before_cuda = os.environ.get("IRODORI_BENCH_CLONE_BEFORE_CUDA") == "1"
    if pre_move_state:
        if clone_before_cuda:
            t0 = time.perf_counter()
            model_state = {k: v.clone() for k, v in model_state.items()}
            print(f"[t] clone state_dict tensors on CPU: {time.perf_counter() - t0:.3f}s", flush=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model_state = {k: v.to("cuda") for k, v in model_state.items()}
        torch.cuda.synchronize()
        print(f"[t] pre-move state_dict tensors to cuda (one by one): {time.perf_counter() - t0:.3f}s", flush=True)

    t0 = time.perf_counter()
    model.load_state_dict(model_state, assign=assign)
    print(f"[t] load_state_dict: {time.perf_counter() - t0:.3f}s", flush=True)

    print(f"[sysinfo-just-before-to-cuda] {_sysinfo()}", flush=True)
    t0 = time.perf_counter()
    model = model.to("cuda")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[t] model.to(cuda): {elapsed:.3f}s", flush=True)
    print(f"[RESULT] to_cuda_seconds={elapsed:.3f}", flush=True)

    # 動作確認: 全パラメータが本当にcudaに乗っているか
    devices = {p.device.type for p in model.parameters()}
    print(f"[check] param devices after all steps = {devices}", flush=True)


if __name__ == "__main__":
    main()

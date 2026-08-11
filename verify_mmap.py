import time
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from irodori_tts.inference_runtime import (
    ModelConfig,
    _load_checkpoint_for_inference,
    merge_dataclass_overrides,
)
from irodori_tts.model import TextToLatentRFDiT

ckpt_path = Path(hf_hub_download("Aratako/Irodori-TTS-v4-Small", "model.safetensors"))
model_state, model_cfg_dict, train_cfg, text_encoder_config = _load_checkpoint_for_inference(ckpt_path)

# state_dict内のtensorがmmapされているか確認
sample_key = next(iter(model_state.keys()))
t = model_state[sample_key]
print(f"sample tensor: key={sample_key}, is_contiguous={t.is_contiguous()}, "
      f"data_ptr={t.data_ptr()}, storage_size={t.untyped_storage().size()}, "
      f"is_pinned={t.is_pinned()}")

model_cfg = merge_dataclass_overrides(ModelConfig(), model_cfg_dict, section="x")
model = TextToLatentRFDiT(
    model_cfg,
    pretrained_backbone_config=text_encoder_config,
    load_pretrained_backbone_weights=not model_cfg.use_pretrained_text_encoder,
)
model.load_state_dict(model_state, assign=model_cfg.use_pretrained_text_encoder)

# state_dictロード直後のパラメータ(mmap由来のはず)の状態を確認
p = next(model.parameters())
print(f"model param after load_state_dict: is_contiguous={p.is_contiguous()}, "
      f"is_pinned={p.is_pinned()}, requires_grad={p.requires_grad}")

# --- テストA: そのまま.to(cuda) (mmap由来のまま) ---
torch.cuda.synchronize()
t0 = time.perf_counter()
model_a = model.to("cuda")
torch.cuda.synchronize()
elapsed_a = time.perf_counter() - t0
print(f"[A] そのまま model.to(cuda): {elapsed_a:.3f}s")

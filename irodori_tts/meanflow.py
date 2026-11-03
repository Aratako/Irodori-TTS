from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .model import TextToLatentRFDiT
from .rf import draw_initial_noise


@dataclass(frozen=True)
class EncodedConditions:
    text_state: torch.Tensor
    text_mask: torch.Tensor
    speaker_state: torch.Tensor | None
    speaker_mask: torch.Tensor | None
    caption_state: torch.Tensor | None
    caption_mask: torch.Tensor | None


def sample_meanflow_interval(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    logit_mean: float = 0.4,
    logit_std: float = 1.0,
    anchor_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample Irodori-time intervals r <= t and formula-anchor indicators."""
    first = torch.sigmoid(
        torch.randn(batch_size, device=device, dtype=dtype) * float(logit_std) + float(logit_mean)
    )
    second = torch.sigmoid(
        torch.randn(batch_size, device=device, dtype=dtype) * float(logit_std) + float(logit_mean)
    )
    r = torch.minimum(first, second)
    t = torch.maximum(first, second)
    anchor_mask = torch.rand(batch_size, device=device) < float(anchor_prob)
    r = torch.where(anchor_mask, t, r)
    return r, t, t - r, anchor_mask


def apply_condition_profile(
    conditions: EncodedConditions,
    *,
    text_drop: torch.Tensor,
    speaker_drop: torch.Tensor | None,
    caption_drop: torch.Tensor | None,
) -> EncodedConditions:
    """Apply one shared condition-availability profile to encoded conditions."""
    text_mask = conditions.text_mask.clone()
    text_mask[text_drop.to(device=text_mask.device, dtype=torch.bool)] = False

    speaker_mask = conditions.speaker_mask
    if speaker_mask is not None and speaker_drop is not None:
        speaker_mask = speaker_mask.clone()
        speaker_mask[speaker_drop.to(device=speaker_mask.device, dtype=torch.bool)] = False

    caption_mask = conditions.caption_mask
    if caption_mask is not None and caption_drop is not None:
        caption_mask = caption_mask.clone()
        caption_mask[caption_drop.to(device=caption_mask.device, dtype=torch.bool)] = False

    return EncodedConditions(
        text_state=conditions.text_state,
        text_mask=text_mask,
        speaker_state=conditions.speaker_state,
        speaker_mask=speaker_mask,
        caption_state=conditions.caption_state,
        caption_mask=caption_mask,
    )


def drop_condition(
    conditions: EncodedConditions,
    name: str,
) -> EncodedConditions:
    """Drop one condition from an already-profiled condition bundle."""
    text_mask = conditions.text_mask
    speaker_mask = conditions.speaker_mask
    caption_mask = conditions.caption_mask
    if name == "text":
        text_mask = torch.zeros_like(text_mask)
    elif name == "speaker":
        if speaker_mask is not None:
            speaker_mask = torch.zeros_like(speaker_mask)
    elif name == "caption":
        if caption_mask is not None:
            caption_mask = torch.zeros_like(caption_mask)
    else:
        raise ValueError(f"Unknown condition name: {name!r}")
    return EncodedConditions(
        text_state=conditions.text_state,
        text_mask=text_mask,
        speaker_state=conditions.speaker_state,
        speaker_mask=speaker_mask,
        caption_state=conditions.caption_state,
        caption_mask=caption_mask,
    )


def index_conditions(
    conditions: EncodedConditions,
    index: torch.Tensor,
) -> EncodedConditions:
    def take(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.index_select(0, index.to(device=value.device))

    return EncodedConditions(
        text_state=take(conditions.text_state),
        text_mask=take(conditions.text_mask),
        speaker_state=take(conditions.speaker_state),
        speaker_mask=take(conditions.speaker_mask),
        caption_state=take(conditions.caption_state),
        caption_mask=take(conditions.caption_mask),
    )


def conditions_to_device(
    conditions: EncodedConditions,
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> EncodedConditions:
    def move(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.to(device=device, non_blocking=non_blocking)

    return EncodedConditions(
        text_state=move(conditions.text_state),
        text_mask=move(conditions.text_mask),
        speaker_state=move(conditions.speaker_state),
        speaker_mask=move(conditions.speaker_mask),
        caption_state=move(conditions.caption_state),
        caption_mask=move(conditions.caption_mask),
    )


ContextKVCache = list[tuple[torch.Tensor, ...]]


def _gather_context_kv_cache(
    cache: ContextKVCache,
    index: torch.Tensor,
    *,
    reuse_source: bool,
) -> ContextKVCache:
    # nonzero() returns indices in ascending order. A single all-active branch
    # can therefore consume the original cache without allocating a copy.
    if reuse_source:
        return cache
    return [
        tuple(value.index_select(0, index.to(device=value.device)) for value in layer)
        for layer in cache
    ]


def _pack_dropped_condition_branches(
    conditions: EncodedConditions,
    tasks: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> tuple[EncodedConditions, torch.Tensor]:
    """Gather a branch group once, then apply its per-branch dropped masks."""
    if not tasks:
        raise ValueError("At least one condition-drop task is required.")
    combined_index = tasks[0][1] if len(tasks) == 1 else torch.cat([task[1] for task in tasks])
    if len(tasks) == 1 and combined_index.numel() == conditions.text_state.shape[0]:
        return drop_condition(conditions, tasks[0][0]), combined_index

    packed = index_conditions(conditions, combined_index)
    text_mask = packed.text_mask
    speaker_mask = packed.speaker_mask
    caption_mask = packed.caption_mask
    start = 0
    for name, active_index, _ in tasks:
        end = start + active_index.numel()
        if name == "text":
            text_mask[start:end] = False
        elif name == "speaker" and speaker_mask is not None:
            speaker_mask[start:end] = False
        elif name == "caption" and caption_mask is not None:
            caption_mask[start:end] = False
        else:
            if name not in {"speaker", "caption"}:
                raise ValueError(f"Unknown condition name: {name!r}")
        start = end
    return (
        EncodedConditions(
            text_state=packed.text_state,
            text_mask=text_mask,
            speaker_state=packed.speaker_state,
            speaker_mask=speaker_mask,
            caption_state=packed.caption_state,
            caption_mask=caption_mask,
        ),
        combined_index,
    )


def _pack_base_and_dropped_condition_branches(
    conditions: EncodedConditions,
    tasks: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> tuple[EncodedConditions, torch.Tensor]:
    """Pack the full base batch followed by active dropped-condition branches."""
    if not tasks:
        raise ValueError("At least one condition-drop task is required.")
    batch_size = conditions.text_state.shape[0]
    base_index = torch.arange(batch_size, device=conditions.text_state.device)
    combined_index = torch.cat([base_index, *(task[1] for task in tasks)])
    packed = index_conditions(conditions, combined_index)
    text_mask = packed.text_mask
    speaker_mask = packed.speaker_mask
    caption_mask = packed.caption_mask
    start = batch_size
    for name, active_index, _ in tasks:
        end = start + active_index.numel()
        if name == "text":
            text_mask[start:end] = False
        elif name == "speaker" and speaker_mask is not None:
            speaker_mask[start:end] = False
        elif name == "caption" and caption_mask is not None:
            caption_mask[start:end] = False
        else:
            if name not in {"speaker", "caption"}:
                raise ValueError(f"Unknown condition name: {name!r}")
        start = end
    return (
        EncodedConditions(
            text_state=packed.text_state,
            text_mask=text_mask,
            speaker_state=packed.speaker_state,
            speaker_mask=speaker_mask,
            caption_state=packed.caption_state,
            caption_mask=caption_mask,
        ),
        combined_index,
    )


def _condition_is_active(mask: torch.Tensor | None, batch_size: int, device) -> torch.Tensor:
    if mask is None:
        return torch.zeros(batch_size, device=device, dtype=torch.bool)
    return mask.any(dim=1)


@torch.no_grad()
def compute_teacher_meanflow_target(
    *,
    teacher: TextToLatentRFDiT,
    x_t: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    conditions: EncodedConditions,
    latent_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    teacher_steps: int = 40,
    cfg_text_scale: float = 3.0,
    cfg_speaker_scale: float = 5.0,
    cfg_caption_scale: float = 4.0,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    sample_chunk_size: int = 8,
    branch_batch_size: int = 1,
    fuse_base_branch: bool = False,
) -> torch.Tensor:
    """Build an FP32 CFG-fused average-velocity target with online Euler rollout."""
    if teacher_steps <= 0:
        raise ValueError(f"teacher_steps must be positive, got {teacher_steps}")
    if str(teacher.cfg.flow_parameterization).strip().lower() != "rf_velocity":
        raise ValueError("MeanFlow targets require an RF velocity teacher.")
    if x_t.ndim != 3 or latent_mask.shape != x_t.shape[:2]:
        raise ValueError("x_t must be (B,S,D) and latent_mask must be (B,S).")
    if valid_mask.shape != latent_mask.shape:
        raise ValueError("valid_mask must match latent_mask.")
    if sample_chunk_size <= 0:
        raise ValueError("sample_chunk_size must be positive.")
    if not 1 <= branch_batch_size <= 3:
        raise ValueError("branch_batch_size must be in [1, 3].")

    batch_size = x_t.shape[0]
    effective_chunk_size = sample_chunk_size
    target = torch.zeros_like(x_t, dtype=torch.float32)
    teacher_forward = getattr(
        teacher,
        "_meanflow_compiled_forward",
        teacher.forward_with_encoded_conditions,
    )

    with torch.autocast(device_type=x_t.device.type, enabled=False):
        def evaluate(
            bundle: EncodedConditions,
            z_in: torch.Tensor,
            tau: torch.Tensor,
            mask: torch.Tensor,
            cache: ContextKVCache,
        ) -> torch.Tensor:
            return teacher_forward(
                x_t=z_in,
                t=tau,
                text_state=bundle.text_state.float(),
                text_mask=bundle.text_mask,
                speaker_state=(
                    None if bundle.speaker_state is None else bundle.speaker_state.float()
                ),
                speaker_mask=bundle.speaker_mask,
                caption_state=(
                    None if bundle.caption_state is None else bundle.caption_state.float()
                ),
                caption_mask=bundle.caption_mask,
                latent_mask=mask,
                context_kv_cache=cache,
            ).float()

        for chunk_start in range(0, batch_size, effective_chunk_size):
            chunk_end = min(chunk_start + effective_chunk_size, batch_size)
            chunk_index = torch.arange(chunk_start, chunk_end, device=x_t.device)
            chunk_conditions = index_conditions(conditions, chunk_index)
            chunk_latent_mask = latent_mask.index_select(0, chunk_index)
            chunk_valid_mask = valid_mask.index_select(0, chunk_index)
            z = x_t.index_select(0, chunk_index).float()
            chunk_t = t.index_select(0, chunk_index).float()
            chunk_r = r.index_select(0, chunk_index).float()
            step = (chunk_r - chunk_t) / float(teacher_steps)
            update_mask = chunk_valid_mask[:, :, None].float()
            chunk_size = chunk_end - chunk_start

            # Condition states are identical across CFG branches; only their masks
            # differ. Cache one FP32 projection per sample chunk and share it across
            # all teacher substeps and active drop branches.
            context_kv_cache = teacher.build_context_kv_cache(
                text_state=chunk_conditions.text_state.float(),
                speaker_state=(
                    None
                    if chunk_conditions.speaker_state is None
                    else chunk_conditions.speaker_state.float()
                ),
                caption_state=(
                    None
                    if chunk_conditions.caption_state is None
                    else chunk_conditions.caption_state.float()
                ),
            )
            active_by_name = {
                "text": _condition_is_active(
                    chunk_conditions.text_mask, chunk_size, x_t.device
                ),
                "speaker": _condition_is_active(
                    chunk_conditions.speaker_mask, chunk_size, x_t.device
                ),
                "caption": _condition_is_active(
                    chunk_conditions.caption_mask, chunk_size, x_t.device
                ),
            }
            scale_by_name = {
                "text": float(cfg_text_scale),
                "speaker": float(cfg_speaker_scale),
                "caption": float(cfg_caption_scale),
            }

            for step_index in range(teacher_steps):
                tau = chunk_t + step * float(step_index)
                cfg_active = (tau >= float(cfg_min_t)) & (tau <= float(cfg_max_t))
                tasks: list[tuple[str, torch.Tensor, torch.Tensor]] = []
                for name in ("text", "speaker", "caption"):
                    weight = (
                        cfg_active.float()
                        * active_by_name[name].float()
                        * scale_by_name[name]
                    )
                    active_index = torch.nonzero(weight, as_tuple=False).squeeze(-1)
                    if active_index.numel() == 0:
                        continue
                    tasks.append(
                        (
                            name,
                            active_index,
                            weight.index_select(0, active_index),
                        )
                    )

                first_task = 0
                if fuse_base_branch and tasks:
                    task_group = tasks[:branch_batch_size]
                    group_sizes = [task[1].numel() for task in task_group]
                    group_conditions, group_index = (
                        _pack_base_and_dropped_condition_branches(
                            chunk_conditions,
                            task_group,
                        )
                    )
                    packed_velocity = evaluate(
                        group_conditions,
                        z.index_select(0, group_index),
                        tau.index_select(0, group_index),
                        chunk_latent_mask.index_select(0, group_index),
                        _gather_context_kv_cache(
                            context_kv_cache,
                            group_index,
                            reuse_source=False,
                        ),
                    )
                    v_base, *dropped_group = packed_velocity.split(
                        [chunk_size, *group_sizes], dim=0
                    )
                    guided = v_base.clone()
                    for task, velocity in zip(task_group, dropped_group, strict=True):
                        active_index, active_weight = task[1], task[2]
                        contribution = active_weight[:, None, None] * (
                            v_base.index_select(0, active_index) - velocity
                        )
                        guided.index_add_(0, active_index, contribution)
                    first_task = len(task_group)
                else:
                    v_base = evaluate(
                        chunk_conditions,
                        z,
                        tau,
                        chunk_latent_mask,
                        context_kv_cache,
                    )
                    guided = v_base.clone()

                for task_start in range(first_task, len(tasks), branch_batch_size):
                    task_group = tasks[task_start : task_start + branch_batch_size]
                    group_sizes = [task[1].numel() for task in task_group]
                    group_conditions, group_index = _pack_dropped_condition_branches(
                        chunk_conditions,
                        task_group,
                    )
                    reuse_chunk = len(task_group) == 1 and group_index.numel() == chunk_size
                    dropped_velocity = evaluate(
                        group_conditions,
                        z if reuse_chunk else z.index_select(0, group_index),
                        tau if reuse_chunk else tau.index_select(0, group_index),
                        (
                            chunk_latent_mask
                            if reuse_chunk
                            else chunk_latent_mask.index_select(0, group_index)
                        ),
                        _gather_context_kv_cache(
                            context_kv_cache,
                            group_index,
                            reuse_source=reuse_chunk,
                        ),
                    )
                    for task, velocity in zip(
                        task_group,
                        dropped_velocity.split(group_sizes, dim=0),
                        strict=True,
                    ):
                        active_index, active_weight = task[1], task[2]
                        contribution = active_weight[:, None, None] * (
                            v_base.index_select(0, active_index) - velocity
                        )
                        guided.index_add_(0, active_index, contribution)

                z = z + guided * step[:, None, None] * update_mask

            delta = (chunk_t - chunk_r).clamp_min(1e-8)
            chunk_target = (x_t.index_select(0, chunk_index).float() - z) / delta[
                :, None, None
            ]
            target.index_copy_(0, chunk_index, chunk_target * update_mask)

    return target


def adaptive_meanflow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    power: float = 0.5,
    eps: float = 0.001,
) -> torch.Tensor:
    """Per-utterance adaptive weighted mean-velocity regression loss."""
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    with torch.autocast(device_type=pred.device.type, enabled=False):
        squared = (pred.float() - target.float()).square().mean(dim=-1)
        weight = valid_mask.float()
        per_sample = (squared * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
        adaptive = (per_sample.detach() + float(eps)).pow(-float(power))
        return (adaptive * per_sample).mean()


@torch.inference_mode()
def sample_euler_meanflow(
    *,
    model: TextToLatentRFDiT,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor | None,
    ref_mask: torch.Tensor | None,
    sequence_length: int,
    caption_input_ids: torch.Tensor | None = None,
    caption_mask: torch.Tensor | None = None,
    speaker_state_override: torch.Tensor | None = None,
    speaker_mask_override: torch.Tensor | None = None,
    speaker_uncond_mode: str = "mask",
    num_steps: int = 4,
    seed: int = 0,
    latent_mask: torch.Tensor | None = None,
    initial_noise: torch.Tensor | None = None,
    step_callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Linear MeanFlow sampler from Irodori time 1 to 0 (default: 4 NFE).

    step_callback, when given, is called as (completed_steps, num_steps) before each step and
    once after the last one. An exception it raises aborts sampling.
    """
    if str(model.cfg.flow_parameterization).strip().lower() != "meanflow":
        raise ValueError("sample_euler_meanflow requires a MeanFlow model.")
    if num_steps <= 0:
        raise ValueError(f"MeanFlow num_steps must be positive, got {num_steps}.")

    device = model.device
    dtype = model.dtype
    batch_size = text_input_ids.shape[0]
    if initial_noise is None:
        x_t = draw_initial_noise(
            seed,
            (batch_size, sequence_length, model.cfg.patched_latent_dim),
            device=device,
            dtype=dtype,
        )[0]
    else:
        x_t = initial_noise.to(device=device, dtype=dtype)

    encoded = EncodedConditions(
        *model.encode_conditions(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_input_ids,
            caption_mask=caption_mask,
            speaker_state_override=speaker_state_override,
            speaker_mask_override=speaker_mask_override,
            speaker_uncond_mode=speaker_uncond_mode,
        )
    )
    context_kv_cache = model.build_context_kv_cache(
        text_state=encoded.text_state,
        speaker_state=encoded.speaker_state,
        caption_state=encoded.caption_state,
    )
    schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    for index in range(num_steps):
        if step_callback is not None:
            step_callback(index, num_steps)
        t_value = schedule[index]
        next_value = schedule[index + 1]
        t_vec = t_value.expand(batch_size)
        delta = (t_value - next_value).expand(batch_size)
        velocity = model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t_vec,
            delta_t=delta,
            text_state=encoded.text_state,
            text_mask=encoded.text_mask,
            speaker_state=encoded.speaker_state,
            speaker_mask=encoded.speaker_mask,
            caption_state=encoded.caption_state,
            caption_mask=encoded.caption_mask,
            latent_mask=latent_mask,
            context_kv_cache=context_kv_cache,
        )
        x_t = x_t + velocity * (next_value - t_value)
    if step_callback is not None:
        step_callback(num_steps, num_steps)
    return x_t

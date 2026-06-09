from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from core.utils import move_images_to_device


def stable_softmax(logits: torch.Tensor, dim: int = 1, eps: float = 1e-7) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=dim)
    probs = probs.clamp(min=eps, max=1.0)
    return probs / probs.sum(dim=dim, keepdim=True)


def jsd_consistency_loss(
    logits_clean: torch.Tensor,
    logits_augmix_1: torch.Tensor,
    logits_augmix_2: torch.Tensor,
) -> torch.Tensor:
    p_clean = stable_softmax(logits_clean)
    p_augmix_1 = stable_softmax(logits_augmix_1)
    p_augmix_2 = stable_softmax(logits_augmix_2)

    log_p_mixture = torch.clamp(
        (p_clean + p_augmix_1 + p_augmix_2) / 3.0,
        min=1e-7,
        max=1.0,
    ).log()

    return (
        F.kl_div(log_p_mixture, p_clean, reduction="batchmean")
        + F.kl_div(log_p_mixture, p_augmix_1, reduction="batchmean")
        + F.kl_div(log_p_mixture, p_augmix_2, reduction="batchmean")
    ) / 3.0


def compute_augmix_loss(
    model: nn.Module,
    batch: tuple[object, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
    jsd_weight: float = 12.0,
    channels_last: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    inputs, targets = batch
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
        raise RuntimeError("AugMix batches must contain clean, augmix_1, and augmix_2 views.")

    targets = targets.to(device, non_blocking=True)
    clean, augmix_1, augmix_2 = [
        move_images_to_device(view, device, channels_last=channels_last)
        for view in inputs
    ]

    logits_all = model(torch.cat([clean, augmix_1, augmix_2], dim=0))
    logits_clean, logits_augmix_1, logits_augmix_2 = torch.chunk(logits_all, 3, dim=0)

    ce_loss = criterion(logits_clean, targets)
    jsd_loss = jsd_consistency_loss(logits_clean, logits_augmix_1, logits_augmix_2)
    loss = ce_loss + jsd_weight * jsd_loss

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite AugMix loss: "
            f"ce={ce_loss.item()}, jsd={jsd_loss.item()}, total={loss.item()}, "
            f"logits_min={logits_all.min().item()}, logits_max={logits_all.max().item()}"
        )

    metrics = {
        "ce_loss": float(ce_loss.detach().cpu()),
        "jsd_loss": float(jsd_loss.detach().cpu()),
        "total_loss": float(loss.detach().cpu()),
    }
    return loss, logits_clean, targets, metrics

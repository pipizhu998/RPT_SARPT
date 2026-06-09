from __future__ import annotations

import csv
import json
import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_torch_runtime(
    device: torch.device,
    deterministic: bool = False,
    cudnn_benchmark: bool = True,
    tf32: bool = True,
) -> None:
    if device.type != "cuda":
        return

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def cuda_mixed_precision_enabled(device: torch.device, requested: bool) -> bool:
    return requested and device.type == "cuda"


def channels_last_enabled(device: torch.device, requested: bool) -> bool:
    return requested and device.type == "cuda"


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    active = cuda_mixed_precision_enabled(device, enabled)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=active)
    return torch.cuda.amp.GradScaler(enabled=active)


def move_images_to_device(
    inputs: torch.Tensor,
    device: torch.device,
    channels_last: bool = False,
) -> torch.Tensor:
    if channels_last and inputs.ndim == 4:
        return inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
    return inputs.to(device, non_blocking=True)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def progress_total(
    dataloader: torch.utils.data.DataLoader,
    max_batches: int | None = None,
) -> int | None:
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    if total_batches is None or max_batches is None:
        return total_batches if max_batches is None else max_batches
    return min(total_batches, max_batches)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    progress_desc: str | None = None,
    show_progress: bool = False,
    mixed_precision: bool = False,
    channels_last: bool = False,
    progress_update_interval: int = 10,
    batch_metrics_callback: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_batches = progress_total(dataloader, max_batches)
    iterator = islice(dataloader, max_batches) if max_batches is not None else dataloader
    progress_bar = None

    if show_progress:
        progress_bar = tqdm(
            iterator,
            total=total_batches,
            desc=progress_desc or "Eval",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = progress_bar

    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs = move_images_to_device(inputs, device, channels_last=channels_last)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device, enabled=mixed_precision):
                logits = model(inputs)
                loss = criterion(logits, targets)

            batch_size = inputs.size(0)
            batch_loss = loss.item()
            batch_correct = (logits.argmax(dim=1) == targets).sum().item()
            total_loss += batch_loss * batch_size
            total_correct += batch_correct
            total_examples += batch_size
            if batch_metrics_callback is not None:
                batch_metrics_callback(
                    {
                        "step": float(batch_idx + 1),
                        "batch_examples": float(batch_size),
                        "batch_loss": batch_loss,
                        "batch_accuracy": batch_correct / batch_size,
                        "batch_correct": float(batch_correct),
                        "cumulative_examples": float(total_examples),
                        "cumulative_loss": total_loss / total_examples,
                        "cumulative_accuracy": total_correct / total_examples,
                        "cumulative_correct": float(total_correct),
                    }
                )
            should_update = (
                progress_update_interval <= 1
                or (batch_idx + 1) % progress_update_interval == 0
                or (total_batches is not None and batch_idx + 1 >= total_batches)
            )
            if progress_bar is not None and should_update:
                progress_bar.set_postfix(
                    loss=f"{total_loss / total_examples:.4f}",
                    acc=f"{total_correct / total_examples:.4f}",
                )

    if progress_bar is not None:
        progress_bar.close()

    if total_examples == 0:
        raise RuntimeError("No evaluation examples were processed.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": float(total_examples),
    }


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

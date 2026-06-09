from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from itertools import islice

import torch
from torch import nn
from tqdm.auto import tqdm

from core.utils import progress_total

from .eata import setup_eata


def evaluate_model_eata(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    steps: int = 1,
    lr: float = 2.5e-4,
    optimizer_name: str = "sgd",
    weight_decay: float = 0.0,
    episodic: bool = False,
    e_margin: float = math.log(10) * 0.4,
    d_margin: float = 0.05,
    fisher_dataloader: torch.utils.data.DataLoader | None = None,
    fisher_size: int = 0,
    fisher_alpha: float = 2000.0,
    fisher_clip_by_norm: float | None = None,
    max_batches: int | None = None,
    progress_desc: str | None = None,
    show_progress: bool = False,
    batch_metrics_callback: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    model = deepcopy(model).to(device)
    eata_model, _param_names = setup_eata(
        model=model,
        optimizer_name=optimizer_name,
        lr=lr,
        steps=steps,
        weight_decay=weight_decay,
        episodic=episodic,
        e_margin=e_margin,
        d_margin=d_margin,
        fisher_dataloader=fisher_dataloader,
        fisher_size=fisher_size,
        fisher_alpha=fisher_alpha,
        fisher_clip_by_norm=fisher_clip_by_norm,
        device=device,
    )

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    iterator = islice(dataloader, max_batches) if max_batches is not None else dataloader
    progress_bar = None

    if show_progress:
        progress_bar = tqdm(
            iterator,
            total=progress_total(dataloader, max_batches),
            desc=progress_desc or "EATA",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = progress_bar

    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = eata_model(inputs).detach()
        eval_loss = criterion(logits, targets)

        batch_size = inputs.size(0)
        batch_loss = eval_loss.item()
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
        if progress_bar is not None:
            progress_bar.set_postfix(
                loss=f"{total_loss / total_examples:.4f}",
                acc=f"{total_correct / total_examples:.4f}",
                updates=eata_model.num_samples_update_2,
            )

    if progress_bar is not None:
        progress_bar.close()

    if total_examples == 0:
        raise RuntimeError("No evaluation examples were processed.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": float(total_examples),
        "eata_reliable_examples": float(eata_model.num_samples_update_1),
        "eata_selected_examples": float(eata_model.num_samples_update_2),
    }

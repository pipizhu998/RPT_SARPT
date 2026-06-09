from __future__ import annotations

import shutil
from collections.abc import Callable
from copy import deepcopy
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from core.utils import progress_total

from .methods import (
    AugMixBatchAugmenter,
    AugMixTTAParams,
    jsd_consistency_loss_per_sample,
    setup_rpt,
    split_adapt_inputs,
)


AUGMIX_CACHE_SAVE_MARGIN_BYTES = 512 * 1024 * 1024
AUGMIX_CACHE_SAVE_OVERHEAD_FACTOR = 1.05


class PrecomputedAugMixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        clean: torch.Tensor,
        augmix_1: torch.Tensor,
        augmix_2: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if not (len(clean) == len(augmix_1) == len(augmix_2) == len(targets)):
            raise ValueError("Precomputed AugMix tensors must have matching lengths.")
        self.clean = clean
        self.augmix_1 = augmix_1
        self.augmix_2 = augmix_2
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        return (
            (self.clean[index], self.augmix_1[index], self.augmix_2[index]),
            self.targets[index],
        )


def make_precomputed_augmix_loader(
    clean: torch.Tensor,
    augmix_1: torch.Tensor,
    augmix_2: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    dataset = PrecomputedAugMixDataset(
        clean=clean,
        augmix_1=augmix_1,
        augmix_2=augmix_2,
        targets=targets,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def move_adapt_inputs_to_device(
    inputs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=True)
    return tuple(view.to(device, non_blocking=True) for view in inputs)


def adapt_inputs_batch_size(
    inputs: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> int:
    if isinstance(inputs, torch.Tensor):
        return inputs.size(0)
    return inputs[0].size(0)


def augmix_cache_path(cache_dir: str | Path | None, cache_key: str | None) -> Path | None:
    if cache_dir is None or cache_key is None:
        return None
    if isinstance(cache_dir, str) and cache_dir.strip().lower() in {"", "none", "null"}:
        return None
    safe_key = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in cache_key)
    return Path(cache_dir) / f"{safe_key}.pt"


def load_precomputed_augmix_cache(
    path: Path,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid AugMix cache payload: {path}")
    required = {"clean", "augmix_1", "augmix_2", "targets"}
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"AugMix cache {path} is missing keys: {missing}")
    return make_precomputed_augmix_loader(
        clean=payload["clean"].contiguous(),
        augmix_1=payload["augmix_1"].contiguous(),
        augmix_2=payload["augmix_2"].contiguous(),
        targets=payload["targets"].contiguous(),
        batch_size=batch_size,
    )


def tensor_payload_nbytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.nelement() * tensor.element_size() for tensor in tensors)


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def ensure_augmix_cache_save_space(
    path: Path,
    *tensors: torch.Tensor,
) -> None:
    payload_bytes = tensor_payload_nbytes(*tensors)
    estimated_bytes = int(payload_bytes * AUGMIX_CACHE_SAVE_OVERHEAD_FACTOR)
    needed_bytes = estimated_bytes + AUGMIX_CACHE_SAVE_MARGIN_BYTES
    free_bytes = shutil.disk_usage(path.parent).free
    if free_bytes < needed_bytes:
        raise RuntimeError(
            "not enough free space for AugMix cache save "
            f"(need about {format_bytes(needed_bytes)} including margin, "
            f"have {format_bytes(free_bytes)})"
        )


def save_precomputed_augmix_cache(
    path: Path,
    clean: torch.Tensor,
    augmix_1: torch.Tensor,
    augmix_2: torch.Tensor,
    targets: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_augmix_cache_save_space(path, clean, augmix_1, augmix_2, targets)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(
            {
                "version": 1,
                "metadata": metadata or {},
                "clean": clean,
                "augmix_1": augmix_1,
                "augmix_2": augmix_2,
                "targets": targets,
            },
            temp_path,
        )
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def precompute_augmix_loader(
    dataloader: torch.utils.data.DataLoader,
    augmenter: AugMixBatchAugmenter,
    max_batches: int | None = None,
    progress_desc: str | None = None,
    show_progress: bool = False,
    cache_dir: str | Path | None = None,
    cache_key: str | None = None,
    cache_metadata: dict[str, Any] | None = None,
    rebuild_cache: bool = False,
) -> torch.utils.data.DataLoader:
    batch_size = dataloader.batch_size or len(dataloader.dataset)
    cache_file = augmix_cache_path(cache_dir, cache_key)
    if cache_file is not None and cache_file.exists() and not rebuild_cache:
        try:
            cached_loader = load_precomputed_augmix_cache(
                cache_file,
                batch_size=batch_size,
            )
        except Exception as exc:
            message = f"Ignoring unreadable AugMix cache {cache_file}: {exc}"
            if show_progress:
                tqdm.write(message)
            else:
                print(message)
        else:
            if show_progress:
                tqdm.write(f"Using AugMix cache: {cache_file}")
            return cached_loader

    iterator = islice(dataloader, max_batches) if max_batches is not None else dataloader
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            iterator,
            total=progress_total(dataloader, max_batches),
            desc=progress_desc or "Precompute AugMix",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = progress_bar

    clean_batches: list[torch.Tensor] = []
    augmix_1_batches: list[torch.Tensor] = []
    augmix_2_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    for inputs, targets in iterator:
        inputs = inputs.detach().cpu()
        clean_batches.append(inputs)
        augmix_1_batches.append(augmenter(inputs).cpu())
        augmix_2_batches.append(augmenter(inputs).cpu())
        target_batches.append(targets.detach().cpu())

    if progress_bar is not None:
        progress_bar.close()

    if not clean_batches:
        raise RuntimeError("No examples were available for AugMix precomputation.")

    clean = torch.cat(clean_batches, dim=0).contiguous()
    augmix_1 = torch.cat(augmix_1_batches, dim=0).contiguous()
    augmix_2 = torch.cat(augmix_2_batches, dim=0).contiguous()
    targets = torch.cat(target_batches, dim=0).contiguous()

    if cache_file is not None:
        try:
            save_precomputed_augmix_cache(
                cache_file,
                clean=clean,
                augmix_1=augmix_1,
                augmix_2=augmix_2,
                targets=targets,
                metadata=cache_metadata,
            )
        except Exception as exc:
            message = (
                f"Could not save AugMix cache {cache_file}; "
                f"continuing without cache: {exc}"
            )
            if show_progress:
                tqdm.write(message)
            else:
                print(message)
        else:
            if show_progress:
                tqdm.write(f"Saved AugMix cache: {cache_file}")

    return make_precomputed_augmix_loader(
        clean=clean,
        augmix_1=augmix_1,
        augmix_2=augmix_2,
        targets=targets,
        batch_size=batch_size,
    )


def evaluate_model_rpt(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    steps: int = 1,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    episodic: bool = False,
    jsd_weight: float = 0.1,
    normalization_dataset: str | None = None,
    augmix_severity: int = 3,
    augmix_width: int = 3,
    augmix_depth: int = -1,
    augmix_alpha: float = 1.0,
    augmix_all_ops: bool = True,
    source_anchor_weight: float = 0.0,
    precompute_augmix: bool = True,
    augmix_cache_dir: str | Path | None = None,
    augmix_cache_key: str | None = None,
    augmix_cache_metadata: dict[str, Any] | None = None,
    rebuild_augmix_cache: bool = False,
    max_batches: int | None = None,
    progress_desc: str | None = None,
    show_progress: bool = False,
    batch_metrics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    model = deepcopy(model).to(device)
    should_precompute_augmix = precompute_augmix and jsd_weight > 0.0 and steps == 1
    if should_precompute_augmix:
        if normalization_dataset is None:
            raise ValueError("RPT requires normalization_dataset when JSD weight is positive.")
        precompute_augmenter = AugMixBatchAugmenter.from_dataset(
            normalization_dataset,
            params=AugMixTTAParams(
                severity=augmix_severity,
                width=augmix_width,
                depth=augmix_depth,
                alpha=augmix_alpha,
                all_ops=augmix_all_ops,
            ),
        )
        dataloader = precompute_augmix_loader(
            dataloader=dataloader,
            augmenter=precompute_augmenter,
            max_batches=max_batches,
            progress_desc=f"{progress_desc or 'RPT'} AugMix cache",
            show_progress=show_progress,
            cache_dir=augmix_cache_dir,
            cache_key=augmix_cache_key,
            cache_metadata=augmix_cache_metadata,
            rebuild_cache=rebuild_augmix_cache,
        )
        max_batches = None

    rpt_model, _param_names = setup_rpt(
        model=model,
        optimizer_name=optimizer_name,
        lr=lr,
        steps=steps,
        weight_decay=weight_decay,
        episodic=episodic,
        jsd_weight=jsd_weight,
        normalization_dataset=normalization_dataset,
        augmix_severity=augmix_severity,
        augmix_width=augmix_width,
        augmix_depth=augmix_depth,
        augmix_alpha=augmix_alpha,
        augmix_all_ops=augmix_all_ops,
        source_anchor_weight=source_anchor_weight,
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
            desc=progress_desc or "RPT",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = progress_bar

    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs = move_adapt_inputs_to_device(inputs, device)
        targets = targets.to(device, non_blocking=True)

        logits = rpt_model(inputs).detach()
        eval_loss = criterion(logits, targets)

        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)
        batch_size = adapt_inputs_batch_size(inputs)
        batch_loss = eval_loss.item()
        batch_correct = (preds == targets).sum().item()
        total_loss += batch_loss * batch_size
        total_correct += batch_correct
        total_examples += batch_size
        if batch_metrics_callback is not None:
            callback_metrics: dict[str, Any] = {
                "step": float(batch_idx + 1),
                "batch_examples": float(batch_size),
                "batch_loss": batch_loss,
                "batch_accuracy": batch_correct / batch_size,
                "batch_correct": float(batch_correct),
                "cumulative_examples": float(total_examples),
                "cumulative_loss": total_loss / total_examples,
                "cumulative_accuracy": total_correct / total_examples,
                "cumulative_correct": float(total_correct),
                "targets": targets.detach().cpu(),
                "preds": preds.detach().cpu(),
                "probs": probs.detach().cpu(),
            }
            _inputs_clean, inputs_augmix_1, inputs_augmix_2 = split_adapt_inputs(inputs)
            if inputs_augmix_1 is not None and inputs_augmix_2 is not None:
                with torch.no_grad():
                    logits_augmix_1 = rpt_model.model(inputs_augmix_1).detach()
                    logits_augmix_2 = rpt_model.model(inputs_augmix_2).detach()
                    jsd_values = jsd_consistency_loss_per_sample(
                        logits,
                        logits_augmix_1,
                        logits_augmix_2,
                    )
                callback_metrics.update(
                    {
                        "aug1_probs": logits_augmix_1.softmax(dim=1).detach().cpu(),
                        "aug2_probs": logits_augmix_2.softmax(dim=1).detach().cpu(),
                        "aug1_preds": logits_augmix_1.argmax(dim=1).detach().cpu(),
                        "aug2_preds": logits_augmix_2.argmax(dim=1).detach().cpu(),
                        "jsd": jsd_values.detach().cpu(),
                    }
                )
            batch_metrics_callback(callback_metrics)
        if progress_bar is not None:
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

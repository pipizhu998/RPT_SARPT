from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from itertools import islice

import torch
from torch import nn
from tqdm.auto import tqdm

from core.utils import autocast_context, move_images_to_device, progress_total


BatchNorm = nn.modules.batchnorm._BatchNorm


@dataclass
class BatchNormState:
    was_training: bool
    track_running_stats: bool
    running_mean: torch.Tensor | None
    running_var: torch.Tensor | None
    num_batches_tracked: torch.Tensor | None


@dataclass
class PerBatchAdaBNState:
    model_was_training: bool
    batch_norm_states: list[BatchNormState]


def batch_norm_modules(model: nn.Module) -> list[BatchNorm]:
    return [module for module in model.modules() if isinstance(module, BatchNorm)]


def batch_inputs(batch: object) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


class AdaBN(nn.Module):
    """Adaptive BatchNorm wrapper for target-domain BN-stat adaptation."""

    def __init__(
        self,
        model: nn.Module,
        reset_stats: bool = True,
        momentum: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.reset_stats = reset_stats
        self.momentum = momentum

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def adapt(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        max_batches: int | None = None,
        progress_desc: str | None = None,
        show_progress: bool = False,
    ) -> None:
        adapt_batch_norm(
            model=self.model,
            dataloader=dataloader,
            device=device,
            reset_stats=self.reset_stats,
            momentum=self.momentum,
            max_batches=max_batches,
            progress_desc=progress_desc,
            show_progress=show_progress,
        )


def adapt_batch_norm(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    reset_stats: bool = True,
    momentum: float | None = None,
    max_batches: int | None = None,
    progress_desc: str | None = None,
    show_progress: bool = False,
) -> None:
    """Adapt BN running statistics with target-domain batches.

    This mutates the supplied model in place. Call it on a freshly loaded
    checkpoint or on a deepcopy when evaluating independent target domains.

    This preserves the project's existing AdaBN mode: freeze learned weights,
    optionally reset BN running stats, put only BN modules in training mode,
    stream target inputs through the model, then evaluate with the adapted
    running statistics.
    """
    bn_modules = batch_norm_modules(model)
    if not bn_modules:
        raise RuntimeError("AdaBN requires at least one BatchNorm layer.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    old_momentums = [module.momentum for module in bn_modules]
    model.eval()
    for module in bn_modules:
        if reset_stats:
            module.reset_running_stats()
        module.momentum = momentum
        module.train()

    iterator = islice(dataloader, max_batches) if max_batches is not None else dataloader
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            iterator,
            total=progress_total(dataloader, max_batches),
            desc=progress_desc or "AdaBN",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = progress_bar

    with torch.no_grad():
        for batch in iterator:
            inputs = batch_inputs(batch)
            model(inputs.to(device, non_blocking=True))

    if progress_bar is not None:
        progress_bar.close()

    for module, old_momentum in zip(bn_modules, old_momentums):
        module.momentum = old_momentum
    model.eval()


def clone_optional_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.detach().clone()


def configure_per_batch_adabn(model: nn.Module) -> PerBatchAdaBNState:
    """Freeze weights and make BatchNorm layers use current-batch stats only."""
    bn_modules = batch_norm_modules(model)
    if not bn_modules:
        raise RuntimeError("AdaBN requires at least one BatchNorm layer.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    state = PerBatchAdaBNState(
        model_was_training=model.training,
        batch_norm_states=[],
    )
    for module in bn_modules:
        state.batch_norm_states.append(
            BatchNormState(
                was_training=module.training,
                track_running_stats=module.track_running_stats,
                running_mean=clone_optional_tensor(module.running_mean),
                running_var=clone_optional_tensor(module.running_var),
                num_batches_tracked=clone_optional_tensor(module.num_batches_tracked),
            )
        )

    model.eval()
    for module in bn_modules:
        module.train()
        module.track_running_stats = False
    return state


def restore_optional_tensor(
    module: nn.Module,
    name: str,
    value: torch.Tensor | None,
) -> None:
    current = getattr(module, name)
    if value is None:
        if current is not None:
            setattr(module, name, None)
        return
    if current is None:
        setattr(module, name, value.detach().clone())
        return
    current.copy_(value.to(current.device))


def restore_per_batch_adabn(
    model: nn.Module,
    state: PerBatchAdaBNState,
) -> None:
    bn_modules = batch_norm_modules(model)
    if len(bn_modules) != len(state.batch_norm_states):
        raise RuntimeError("BatchNorm module count changed during AdaBN evaluation.")

    for module, bn_state in zip(bn_modules, state.batch_norm_states):
        module.track_running_stats = bn_state.track_running_stats
        restore_optional_tensor(module, "running_mean", bn_state.running_mean)
        restore_optional_tensor(module, "running_var", bn_state.running_var)
        restore_optional_tensor(
            module,
            "num_batches_tracked",
            bn_state.num_batches_tracked,
        )
        module.train(bn_state.was_training)
    model.train(state.model_was_training)


def evaluate_model_per_batch_adabn(
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
    """Evaluate with stateless per-batch AdaBN statistics."""
    adabn_state = configure_per_batch_adabn(model)
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
            desc=progress_desc or "AdaBN-batch",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = progress_bar

    try:
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
    finally:
        if progress_bar is not None:
            progress_bar.close()
        restore_per_batch_adabn(model, adabn_state)

    if total_examples == 0:
        raise RuntimeError("No evaluation examples were processed.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": float(total_examples),
    }


class BatchNormStatHook:
    """Accumulate input activation statistics for BatchNorm layers."""

    def __init__(self) -> None:
        self.bn_stats: dict[str, dict[str, torch.Tensor | int]] = {}

    def __call__(
        self,
        module: BatchNorm,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
        name: str,
    ) -> None:
        if name not in self.bn_stats:
            self.bn_stats[name] = {"mean": 0, "var": 0, "count": 0}

        activation = inputs[0].detach()
        if activation.ndim < 2:
            raise ValueError(f"BatchNorm input for {name} must have at least 2 dimensions.")

        reduction_dims = [0, *range(2, activation.ndim)]
        mean = activation.mean(dim=reduction_dims)
        var = activation.var(dim=reduction_dims, unbiased=False)
        batch_size = activation.size(0)
        stats = self.bn_stats[name]

        if isinstance(stats["mean"], int):
            stats["mean"] = torch.zeros_like(mean)
            stats["var"] = torch.zeros_like(var)

        stats["mean"] = stats["mean"] + mean * batch_size
        stats["var"] = stats["var"] + var * batch_size
        stats["count"] = int(stats["count"]) + batch_size


def compute_bn_stats(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    progress_desc: str | None = None,
    show_progress: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute target-domain BatchNorm input statistics with hooks."""
    hook = BatchNormStatHook()
    handles = []
    original_training = model.training
    bn_modules = batch_norm_modules(model)
    if not bn_modules:
        raise RuntimeError("AdaBN requires at least one BatchNorm layer.")

    for name, module in model.named_modules():
        if isinstance(module, BatchNorm):
            handles.append(module.register_forward_hook(partial(hook, name=name)))

    iterator = islice(dataloader, max_batches) if max_batches is not None else dataloader
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            iterator,
            total=progress_total(dataloader, max_batches),
            desc=progress_desc or "AdaBN stats",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = progress_bar

    try:
        model.eval()
        with torch.no_grad():
            for batch in iterator:
                inputs = batch_inputs(batch)
                model(inputs.to(device, non_blocking=True))
    finally:
        if progress_bar is not None:
            progress_bar.close()
        for handle in handles:
            handle.remove()
        model.train(original_training)

    final_stats: dict[str, dict[str, torch.Tensor]] = {}
    for name, stats in hook.bn_stats.items():
        count = int(stats["count"])
        if count <= 0:
            continue
        mean = stats["mean"]
        var = stats["var"]
        if isinstance(mean, int) or isinstance(var, int):
            continue
        final_stats[name] = {"mean": mean / count, "var": var / count}
    return final_stats


def replace_bn_stats(
    model: nn.Module,
    bn_stats: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Replace model BN running statistics with precomputed target stats."""
    with torch.no_grad():
        for name, module in model.named_modules():
            if name not in bn_stats or not isinstance(module, BatchNorm):
                continue
            if module.running_mean is None or module.running_var is None:
                raise RuntimeError(f"BatchNorm layer {name} does not track running stats.")

            computed_mean = bn_stats[name]["mean"]
            computed_var = bn_stats[name]["var"]
            if computed_mean.shape != module.running_mean.shape:
                raise ValueError(
                    f"Shape mismatch for {name}.running_mean: "
                    f"expected {tuple(module.running_mean.shape)}, got {tuple(computed_mean.shape)}"
                )
            if computed_var.shape != module.running_var.shape:
                raise ValueError(
                    f"Shape mismatch for {name}.running_var: "
                    f"expected {tuple(module.running_var.shape)}, got {tuple(computed_var.shape)}"
                )

            module.running_mean.copy_(computed_mean.to(module.running_mean.device))
            module.running_var.copy_(computed_var.to(module.running_var.device))

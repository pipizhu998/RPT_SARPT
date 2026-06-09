from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from core.models import build_model
from core.training_method_interface import TrainingMethodInterface
from core.utils import (
    autocast_context,
    channels_last_enabled,
    configure_torch_runtime,
    cuda_mixed_precision_enabled,
    evaluate_model,
    make_grad_scaler,
    progress_total,
    resolve_device,
    save_csv,
    save_json,
    set_seed,
    timestamp,
)


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    interface: TrainingMethodInterface,
    max_batches: int | None = None,
    progress_desc: str = "Train",
    mixed_precision: bool = False,
    channels_last: bool = False,
    grad_scaler=None,
    progress_update_interval: int = 10,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_jsd_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_batches = progress_total(dataloader, max_batches)
    iterator = islice(dataloader, max_batches) if max_batches is not None else dataloader
    progress_bar = tqdm(
        iterator,
        total=total_batches,
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, enabled=mixed_precision):
            step = interface.compute_loss(
                model,
                batch,
                criterion,
                device,
                channels_last=channels_last,
            )
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.scale(step.loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            step.loss.backward()
            optimizer.step()

        batch_size = step.targets.size(0)
        total_loss += step.loss.item() * batch_size
        total_ce_loss += step.metrics["ce_loss"] * batch_size
        total_jsd_loss += step.metrics["jsd_loss"] * batch_size
        total_correct += (step.logits.argmax(dim=1) == step.targets).sum().item()
        total_examples += batch_size
        should_update = (
            progress_update_interval <= 1
            or (batch_idx + 1) % progress_update_interval == 0
            or (total_batches is not None and batch_idx + 1 >= total_batches)
        )
        if should_update:
            progress_bar.set_postfix(
                loss=f"{total_loss / total_examples:.4f}",
                err=f"{1.0 - total_correct / total_examples:.4f}",
            )

    progress_bar.close()

    if total_examples == 0:
        raise RuntimeError("No training examples were processed.")

    accuracy = total_correct / total_examples
    return {
        "loss": total_loss / total_examples,
        "ce_loss": total_ce_loss / total_examples,
        "jsd_loss": total_jsd_loss / total_examples,
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "examples": float(total_examples),
    }


def run_training(interface: TrainingMethodInterface) -> Path:
    config = interface.config
    set_seed(config.seed)
    device = resolve_device(config.device)
    configure_torch_runtime(
        device,
        deterministic=config.deterministic,
        cudnn_benchmark=config.cudnn_benchmark,
        tf32=config.tf32,
    )
    mixed_precision = cuda_mixed_precision_enabled(device, config.mixed_precision)
    channels_last = channels_last_enabled(device, config.channels_last)

    loaders = interface.build_loaders()
    model = build_model(config.model).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=config.optimizer,
        lr=interface.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    experiment_dir = interface.experiment_dir()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = interface.resolved_config() | {
        "resolved_device": str(device),
        "mixed_precision_active": mixed_precision,
        "channels_last_active": channels_last,
    }
    save_json(resolved_config, experiment_dir / "config.json")
    save_json(interface.command_manifest(), experiment_dir / "commands.json")

    best_val_acc = -1.0
    history: list[dict[str, Any]] = []
    grad_scaler = make_grad_scaler(device, mixed_precision)

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=loaders.train,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            interface=interface,
            max_batches=config.max_train_batches,
            progress_desc=f"Train {epoch:03d}/{config.epochs:03d} {config.method}",
            mixed_precision=mixed_precision,
            channels_last=channels_last,
            grad_scaler=grad_scaler,
            progress_update_interval=config.progress_update_interval,
        )
        val_metrics = evaluate_model(
            model=model,
            dataloader=loaders.val,
            criterion=criterion,
            device=device,
            max_batches=config.max_val_batches,
            progress_desc=f"Val   {epoch:03d}/{config.epochs:03d}",
            show_progress=True,
            mixed_precision=mixed_precision,
            channels_last=channels_last,
            progress_update_interval=config.progress_update_interval,
        )
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_jsd_loss": train_metrics["jsd_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_error_rate": train_metrics["error_rate"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_error_rate": 1.0 - val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        save_csv(history, experiment_dir / "history.csv")
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train loss {record['train_loss']:.4f} | train err {record['train_error_rate']:.4f} | "
            f"val loss {record['val_loss']:.4f} | val err {record['val_error_rate']:.4f}"
        )

        checkpoint = {
            "model_name": config.model,
            "model_state": model.state_dict(),
            "config": resolved_config,
            "epoch": epoch,
            "val_accuracy": val_metrics["accuracy"],
            "val_error_rate": 1.0 - val_metrics["accuracy"],
        }
        torch.save(checkpoint, experiment_dir / "latest.pt")
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(checkpoint, experiment_dir / "best.pt")

    best_checkpoint = torch.load(experiment_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = evaluate_model(
        model=model,
        dataloader=loaders.test,
        criterion=criterion,
        device=device,
        max_batches=config.max_val_batches,
        progress_desc="Test clean",
        show_progress=True,
        mixed_precision=mixed_precision,
        channels_last=channels_last,
        progress_update_interval=config.progress_update_interval,
    )

    summary = {
        "experiment_name": interface.experiment_name(),
        "method": config.method,
        "timestamp": timestamp(),
        "best_val_accuracy": best_val_acc,
        "best_val_error_rate": 1.0 - best_val_acc,
        "test_accuracy_clean": test_metrics["accuracy"],
        "test_error_rate_clean": 1.0 - test_metrics["accuracy"],
        "test_loss_clean": test_metrics["loss"],
        "history": history,
    }
    save_json(summary, experiment_dir / "summary.json")
    print(
        f"Finished {config.method}. Best val err {1.0 - best_val_acc:.4f} | "
        f"clean test err {1.0 - test_metrics['accuracy']:.4f}"
    )
    print(f"Artifacts saved to: {experiment_dir.resolve()}")
    print(f"Shift evaluation command: {interface.evaluate_command()}")
    return experiment_dir


def main() -> None:
    parser = TrainingMethodInterface.build_arg_parser(
        "Train clean or AugMix classification experiments."
    )
    run_training(TrainingMethodInterface.from_args(parser.parse_args()))


if __name__ == "__main__":
    main()

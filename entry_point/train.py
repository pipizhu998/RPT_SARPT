from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from data_utils.data import NOISE_TYPES, build_classification_loaders
from core.models import build_model
from core.utils import evaluate_model, progress_total, resolve_device, save_csv, save_json, set_seed, timestamp


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the original clean baseline.")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    parser.add_argument("--dataset-name", choices=["cifar10", "mnist", "svhn"], default="cifar10")
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--train-noise-type", choices=NOISE_TYPES, default="none")
    parser.add_argument("--train-noise-prob", type=float, default=0.0)
    parser.add_argument("--train-noise-min", type=float, default=0.0)
    parser.add_argument("--train-noise-max", type=float, default=0.0)
    return parser.parse_args()


def default_learning_rate(model_name: str, optimizer_name: str) -> float:
    if optimizer_name == "adamw":
        return 3e-4
    if model_name == "resnet18":
        return 0.1
    return 0.05


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


def format_experiment_value(value: object) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace(".", "p").replace("-", "m")


def make_experiment_name(args) -> str:
    if args.experiment_name:
        return args.experiment_name
    suffix = "clean"
    parts = [args.model]
    if args.dataset_name != "cifar10":
        parts.insert(0, args.dataset_name)
    if args.train_noise_type != "none" and args.train_noise_prob > 0.0:
        suffix = f"{args.train_noise_type}_noise_aug"
    parts.extend(
        [
            suffix,
            f"ep{args.epochs}",
            f"bs{args.batch_size}",
            f"opt{args.optimizer}",
            f"lr{format_experiment_value(args.lr)}",
            f"wd{format_experiment_value(args.weight_decay)}",
            f"seed{args.seed}",
        ]
    )
    if args.optimizer == "sgd":
        parts.append(f"mom{format_experiment_value(args.momentum)}")
    if args.scheduler != "none":
        parts.append(f"sched{args.scheduler}")
    if args.label_smoothing > 0.0:
        parts.append(f"ls{format_experiment_value(args.label_smoothing)}")
    if args.val_ratio > 0.0:
        parts.append(f"val{format_experiment_value(args.val_ratio)}")
    if args.max_train_batches is not None:
        parts.append(f"mtb{args.max_train_batches}")
    if args.max_val_batches is not None:
        parts.append(f"mvb{args.max_val_batches}")
    if args.train_noise_type != "none" and args.train_noise_prob > 0.0:
        parts.extend(
            [
                f"np{format_experiment_value(args.train_noise_prob)}",
                f"nmin{format_experiment_value(args.train_noise_min)}",
                f"nmax{format_experiment_value(args.train_noise_max)}",
            ]
        )
    return "_".join(parts)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    progress_desc: str = "Train",
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    iterator = islice(dataloader, max_batches) if max_batches is not None else dataloader
    progress_bar = tqdm(
        iterator,
        total=progress_total(dataloader, max_batches),
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += inputs.size(0)
        progress_bar.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            acc=f"{total_correct / total_examples:.4f}",
        )

    progress_bar.close()

    if total_examples == 0:
        raise RuntimeError("No training examples were processed.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def main() -> None:
    args = get_config()
    args.lr = args.lr if args.lr is not None else default_learning_rate(args.model, args.optimizer)

    set_seed(args.seed)
    device = resolve_device(args.device)

    loaders = build_classification_loaders(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_noise_type=args.train_noise_type,
        train_noise_prob=args.train_noise_prob,
        train_noise_min=args.train_noise_min,
        train_noise_max=args.train_noise_max,
    )

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    experiment_name = make_experiment_name(args)
    experiment_dir = Path(args.out_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"resolved_device": str(device)}
    save_json(config, experiment_dir / "config.json")

    best_val_acc = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=loaders.train,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_train_batches,
            progress_desc=f"Train {epoch:03d}/{args.epochs:03d}",
        )
        val_metrics = evaluate_model(
            model=model,
            dataloader=loaders.val,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
            progress_desc=f"Val   {epoch:03d}/{args.epochs:03d}",
            show_progress=True,
        )
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        save_csv(history, experiment_dir / "history.csv")
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {record['train_loss']:.4f} | train acc {record['train_accuracy']:.4f} | "
            f"val loss {record['val_loss']:.4f} | val acc {record['val_accuracy']:.4f}"
        )

        checkpoint = {
            "model_name": args.model,
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_accuracy": val_metrics["accuracy"],
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
        max_batches=args.max_val_batches,
        progress_desc="Test clean",
        show_progress=True,
    )

    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp(),
        "best_val_accuracy": best_val_acc,
        "test_accuracy_clean": test_metrics["accuracy"],
        "test_loss_clean": test_metrics["loss"],
        "history": history,
    }
    save_json(summary, experiment_dir / "summary.json")
    print(
        f"Finished training. Best val acc {best_val_acc:.4f} | "
        f"clean test acc {test_metrics['accuracy']:.4f}"
    )
    print(f"Artifacts saved to: {experiment_dir.resolve()}")


if __name__ == "__main__":
    main()

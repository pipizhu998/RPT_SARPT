from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from baseline.augmix import (
    build_augmix_cifar10_loaders,
    build_augmix_loaders,
    compute_augmix_loss,
)
from data_utils.data import build_cifar10_loaders, build_classification_loaders
from core.utils import move_images_to_device


@dataclass
class TrainingMethodConfig:
    method: str = "augmix"
    model: str = "resnet18"
    dataset_name: str = "cifar10"
    data_dir: str = "datasets"
    out_dir: str = "outputs/training_methods"
    experiment_name: str | None = None
    epochs: int = 20
    batch_size: int = 128
    num_workers: int = 2
    optimizer: str = "sgd"
    lr: float | None = None
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "cosine"
    label_smoothing: float = 0.0
    val_ratio: float = 0.1
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    download: bool = False
    mixed_precision: bool = True
    channels_last: bool = True
    tf32: bool = True
    cudnn_benchmark: bool = True
    deterministic: bool = False
    progress_update_interval: int = 10
    augmix_severity: int = 3
    augmix_width: int = 3
    augmix_depth: int = -1
    augmix_alpha: float = 1.0
    augmix_jsd_weight: float = 12.0
    augmix_all_ops: bool = True


@dataclass
class TrainingStepResult:
    loss: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor
    metrics: dict[str, float]


class TrainingMethodInterface:
    VALID_METHODS = ("clean", "augmix")

    def __init__(self, config: TrainingMethodConfig) -> None:
        method = config.method.lower()
        if method not in self.VALID_METHODS:
            raise ValueError(f"Unknown method '{config.method}'. Available: {self.VALID_METHODS}")
        config.method = method
        self.config = config

    @staticmethod
    def default_learning_rate(model_name: str, optimizer_name: str) -> float:
        if optimizer_name == "adamw":
            return 3e-4
        if model_name == "resnet18":
            return 0.1
        return 0.05

    @staticmethod
    def format_value(value: object) -> str:
        if isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value)
        return text.replace(".", "p").replace("-", "m")

    @property
    def lr(self) -> float:
        if self.config.lr is not None:
            return self.config.lr
        return self.default_learning_rate(self.config.model, self.config.optimizer)

    def experiment_name(self) -> str:
        if self.config.experiment_name:
            return self.config.experiment_name

        parts = [
            self.config.method,
            self.config.model,
            f"ep{self.config.epochs}",
            f"bs{self.config.batch_size}",
            f"opt{self.config.optimizer}",
            f"lr{self.format_value(self.lr)}",
            f"wd{self.format_value(self.config.weight_decay)}",
            f"seed{self.config.seed}",
        ]
        if self.config.optimizer == "sgd":
            parts.append(f"mom{self.format_value(self.config.momentum)}")
        if self.config.scheduler != "none":
            parts.append(f"sched{self.config.scheduler}")
        if self.config.label_smoothing > 0.0:
            parts.append(f"ls{self.format_value(self.config.label_smoothing)}")
        if self.config.val_ratio > 0.0:
            parts.append(f"val{self.format_value(self.config.val_ratio)}")

        if self.config.method == "augmix":
            parts.extend(
                [
                    f"sev{self.config.augmix_severity}",
                    f"w{self.config.augmix_width}",
                    f"d{self.format_value(self.config.augmix_depth)}",
                    f"a{self.format_value(self.config.augmix_alpha)}",
                    f"jsd{self.format_value(self.config.augmix_jsd_weight)}",
                ]
            )
            if not self.config.augmix_all_ops:
                parts.append("baseops")
        if self.config.max_train_batches is not None:
            parts.append(f"mtb{self.config.max_train_batches}")
        if self.config.max_val_batches is not None:
            parts.append(f"mvb{self.config.max_val_batches}")
        return "_".join(parts)

    def experiment_dir(self) -> Path:
        return Path(self.config.out_dir) / self.config.method / self.experiment_name()

    def resolved_config(self) -> dict[str, Any]:
        payload = asdict(self.config)
        payload["lr"] = self.lr
        payload["experiment_name"] = self.experiment_name()
        payload["experiment_dir"] = str(self.experiment_dir())
        return payload

    def build_loaders(self):
        if self.config.method == "augmix":
            if self.config.dataset_name != "cifar10":
                return build_augmix_loaders(
                    dataset_name=self.config.dataset_name,
                    data_dir=self.config.data_dir,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    val_ratio=self.config.val_ratio,
                    seed=self.config.seed,
                    severity=self.config.augmix_severity,
                    width=self.config.augmix_width,
                    depth=self.config.augmix_depth,
                    alpha=self.config.augmix_alpha,
                    all_ops=self.config.augmix_all_ops,
                    download=self.config.download,
                )
            return build_augmix_cifar10_loaders(
                data_dir=self.config.data_dir,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                val_ratio=self.config.val_ratio,
                seed=self.config.seed,
                severity=self.config.augmix_severity,
                width=self.config.augmix_width,
                depth=self.config.augmix_depth,
                alpha=self.config.augmix_alpha,
                all_ops=self.config.augmix_all_ops,
                download=self.config.download,
            )

        if self.config.dataset_name == "cifar10":
            return build_cifar10_loaders(
                data_dir=self.config.data_dir,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                val_ratio=self.config.val_ratio,
                seed=self.config.seed,
                download=self.config.download,
            )

        return build_classification_loaders(
            dataset_name=self.config.dataset_name,
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            val_ratio=self.config.val_ratio,
            seed=self.config.seed,
            download=self.config.download,
        )

    def compute_loss(
        self,
        model: nn.Module,
        batch: tuple[object, torch.Tensor],
        criterion: nn.Module,
        device: torch.device,
        channels_last: bool = False,
    ) -> TrainingStepResult:
        inputs, targets = batch

        if self.config.method == "augmix":
            loss, logits_clean, targets, metrics = compute_augmix_loss(
                model=model,
                batch=batch,
                criterion=criterion,
                device=device,
                jsd_weight=self.config.augmix_jsd_weight,
                channels_last=channels_last,
            )

            return TrainingStepResult(
                loss=loss,
                logits=logits_clean,
                targets=targets,
                metrics=metrics,
            )

        targets = targets.to(device, non_blocking=True)
        inputs = move_images_to_device(inputs, device, channels_last=channels_last)
        logits = model(inputs)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss: "
                f"loss={loss.item()}, logits_min={logits.min().item()}, logits_max={logits.max().item()}"
            )

        return TrainingStepResult(
            loss=loss,
            logits=logits,
            targets=targets,
            metrics={
                "ce_loss": float(loss.detach().cpu()),
                "jsd_loss": 0.0,
                "total_loss": float(loss.detach().cpu()),
            },
        )

    def train_command(self) -> str:
        command = [
            sys.executable,
            "-m",
            "core.training_method_train",
            "--method",
            self.config.method,
            "--model",
            self.config.model,
            "--dataset-name",
            self.config.dataset_name,
            "--data-dir",
            self.config.data_dir,
            "--out-dir",
            self.config.out_dir,
            "--epochs",
            str(self.config.epochs),
            "--batch-size",
            str(self.config.batch_size),
            "--num-workers",
            str(self.config.num_workers),
            "--optimizer",
            self.config.optimizer,
            "--momentum",
            str(self.config.momentum),
            "--weight-decay",
            str(self.config.weight_decay),
            "--scheduler",
            self.config.scheduler,
            "--label-smoothing",
            str(self.config.label_smoothing),
            "--val-ratio",
            str(self.config.val_ratio),
            "--seed",
            str(self.config.seed),
            "--device",
            self.config.device,
        ]

        if self.config.lr is not None:
            command.extend(["--lr", str(self.config.lr)])
        if self.config.experiment_name is not None:
            command.extend(["--experiment-name", self.config.experiment_name])
        if self.config.max_train_batches is not None:
            command.extend(["--max-train-batches", str(self.config.max_train_batches)])
        if self.config.max_val_batches is not None:
            command.extend(["--max-val-batches", str(self.config.max_val_batches)])
        if self.config.download:
            command.append("--download")
        if not self.config.mixed_precision:
            command.append("--no-mixed-precision")
        if not self.config.channels_last:
            command.append("--no-channels-last")
        if not self.config.tf32:
            command.append("--no-tf32")
        if not self.config.cudnn_benchmark:
            command.append("--no-cudnn-benchmark")
        if self.config.deterministic:
            command.append("--deterministic")
        if self.config.progress_update_interval != 10:
            command.extend(["--progress-update-interval", str(self.config.progress_update_interval)])

        if self.config.method == "augmix":
            command.extend(
                [
                    "--augmix-severity",
                    str(self.config.augmix_severity),
                    "--augmix-width",
                    str(self.config.augmix_width),
                    "--augmix-depth",
                    str(self.config.augmix_depth),
                    "--augmix-alpha",
                    str(self.config.augmix_alpha),
                    "--augmix-jsd-weight",
                    str(self.config.augmix_jsd_weight),
                ]
            )
            if not self.config.augmix_all_ops:
                command.append("--no-augmix-all-ops")

        return subprocess.list2cmdline(command)

    def evaluate_command(self) -> str:
        checkpoint = self.experiment_dir() / "best.pt"
        output = self.experiment_dir() / "best_shift_results.csv"
        if self.config.dataset_name in {"mnist", "svhn"}:
            source_dataset = self.config.dataset_name
            target_dataset = "svhn" if source_dataset == "mnist" else "mnist"
            digitrobust_corruptions = [
                "gaussian_noise",
                "shot_noise",
                "impulse_noise",
                "speckle_noise",
                "motion_blur",
                "defocus_blur",
                "brightness",
                "contrast",
                "rotate",
                "translate",
            ]
            command = [
                sys.executable,
                "-m",
                "entry_point.evaluate",
                "--checkpoint",
                str(checkpoint),
                "--clean-dataset",
                target_dataset,
                "--source-dataset",
                source_dataset,
                "--ood-dataset",
                f"digitrobust_{target_dataset}c",
                "--digitrobust-subset",
                target_dataset,
                "--clean-data-dir",
                "datasets",
                "--data-dir",
                "datasets/DigitRobust",
                "--corruptions",
                *digitrobust_corruptions,
                "--severity-levels",
                "1",
                "--batch-size",
                str(self.config.batch_size),
                "--num-workers",
                str(self.config.num_workers),
                "--out-file",
                str(output),
            ]
            return subprocess.list2cmdline(command)

        command = [
            sys.executable,
            "-m",
            "entry_point.evaluate",
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            "datasets/CIFAR-10-C",
            "--batch-size",
            str(self.config.batch_size),
            "--num-workers",
            str(self.config.num_workers),
            "--out-file",
            str(output),
        ]
        return subprocess.list2cmdline(command)

    def command_manifest(self) -> dict[str, str]:
        return {
            "train": self.train_command(),
            "evaluate": self.evaluate_command(),
        }

    @classmethod
    def build_arg_parser(cls, description: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--method", choices=cls.VALID_METHODS, default="augmix")
        parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
        parser.add_argument("--dataset-name", choices=["cifar10", "mnist", "svhn"], default="cifar10")
        parser.add_argument("--data-dir", default="datasets")
        parser.add_argument("--out-dir", default="outputs/training_methods")
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
        parser.add_argument("--download", action="store_true")
        parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--deterministic", action="store_true")
        parser.add_argument("--progress-update-interval", type=int, default=10)
        parser.add_argument("--augmix-severity", type=int, default=3)
        parser.add_argument("--augmix-width", type=int, default=3)
        parser.add_argument("--augmix-depth", type=int, default=-1)
        parser.add_argument("--augmix-alpha", type=float, default=1.0)
        parser.add_argument("--augmix-jsd-weight", type=float, default=12.0)
        parser.add_argument("--augmix-all-ops", action=argparse.BooleanOptionalAction, default=True)
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingMethodInterface":
        return cls(TrainingMethodConfig(**vars(args)))


def main() -> None:
    parser = TrainingMethodInterface.build_arg_parser(
        "Print reproducible training-method experiment commands."
    )
    interface = TrainingMethodInterface.from_args(parser.parse_args())
    print(json.dumps(interface.command_manifest(), indent=2))


if __name__ == "__main__":
    main()

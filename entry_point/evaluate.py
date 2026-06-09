from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path

import torch
from torch import nn

from data_utils.cifar10c_data import (
    build_cifar10c_loader,
    build_mixed_cifar10c_loader,
    ensure_cifar10c_files,
)
from data_utils.cifar10_1_data import build_cifar10_1_loader
from data_utils.cifar10_1_c_data import (
    build_cifar10_1_c_loader,
    build_mixed_cifar10_1_c_loader,
    ensure_cifar10_1_c_files,
)
from data_utils.data import build_clean_test_loader
from data_utils.digitrobust_data import (
    DIGITROBUST_CORRUPTIONS,
    build_digitrobust_clean_loader,
    build_digitrobust_corrupt_loader,
    build_mixed_digitrobust_loader,
    ensure_digitrobust_files,
)
from data_utils.mnistc_data import (
    build_mixed_mnistc_loader,
    build_mnistc_loader,
    ensure_mnistc_files,
)
from core.models import build_model
from core.test_time_adapt import (
    evaluate_model_cotta,
    evaluate_model_eata,
    evaluate_model_per_batch_adabn,
    evaluate_model_rpt,
    evaluate_model_tent,
)
from core.utils import evaluate_model, resolve_device, save_json


DIGITROBUST_OOD_DATASETS = {
    "digitrobust_mnistc",
    "digitrobust_svhnc",
    "digitrobust_combined",
}

STREAM_HISTORY_FIELDNAMES = [
    "step",
    "row_label",
    "batch_examples",
    "batch_loss",
    "batch_accuracy",
    "batch_correct",
    "cumulative_examples",
    "cumulative_loss",
    "cumulative_accuracy",
    "cumulative_correct",
]

DIGIT_CLASS_COUNT = 10
PROBABILITY_LOG_FIELDNAMES = [
    "image_number",
    "step",
    "row_label",
    "target",
    "pred",
    "correct",
    "confidence",
    "entropy",
    *[f"prob_{digit}" for digit in range(DIGIT_CLASS_COUNT)],
    "aug1_pred",
    "aug2_pred",
    "clean_aug1_agree",
    "clean_aug2_agree",
    "aug1_aug2_agree",
    "jsd",
    *[f"aug1_prob_{digit}" for digit in range(DIGIT_CLASS_COUNT)],
    *[f"aug2_prob_{digit}" for digit in range(DIGIT_CLASS_COUNT)],
]


@dataclass
class EvaluationConfig:
    checkpoint: str
    clean_dataset: str = "mnist"
    source_dataset: str = "svhn"
    ood_dataset: str = "digitrobust_mnistc"
    clean_data_dir: str = "datasets"
    clean_download: bool = False
    clean_shuffle: bool = False
    clean_seed: int = 0
    data_dir: str = "datasets/DigitRobust"
    cifar10_1_data_dir: str = "datasets/CIFAR-10.1"
    cifar10_1_version: str = "v6"
    batch_size: int = 128
    num_workers: int = 2
    device: str = "auto"
    corruptions: list[str] = field(
        default_factory=lambda: list(DIGITROBUST_CORRUPTIONS)
    )
    severity_levels: list[int] = field(default_factory=lambda: [1])
    max_examples_per_condition: int | None = None
    condition_seed: int = 0
    max_batches: int | None = None
    out_file: str | None = None
    evaluation_mode: str = "standard"
    standard_include_clean: bool = True
    mixed_seed: int = 0
    mixed_max_examples: int | None = None
    mixed_include_clean: bool = False
    digitrobust_subset: str | None = None
    digitrobust_subsets: list[str] | None = None
    test_adapt: str = "none"
    adabn_reset_stats: bool = True
    adabn_momentum: float | None = None
    adapt_max_batches: int | None = None
    eata_lr: float = 2.5e-4
    eata_steps: int = 1
    eata_optimizer: str = "sgd"
    eata_weight_decay: float = 0.0
    eata_episodic: bool = False
    eata_e_margin: float = math.log(10) * 0.4
    eata_d_margin: float = 0.05
    eata_fisher_size: int = 2000
    eata_fisher_alpha: float = 2000.0
    eata_fisher_clip_by_norm: float | None = None
    cotta_lr: float = 1e-3
    cotta_steps: int = 1
    cotta_optimizer: str = "adam"
    cotta_weight_decay: float = 0.0
    cotta_episodic: bool = False
    cotta_mt_alpha: float = 0.999
    cotta_rst_m: float = 0.01
    cotta_ap: float = 0.92
    cotta_augmentation_views: int = 32
    cotta_gaussian_std: float = 0.005
    cotta_soft_augmentations: bool = False
    cotta_beta: float = 0.9
    tent_lr: float = 1e-3
    tent_steps: int = 1
    tent_optimizer: str = "adam"
    tent_weight_decay: float = 0.0
    tent_episodic: bool = False
    rpt_lr: float = 1e-3
    rpt_steps: int = 1
    rpt_optimizer: str = "adam"
    rpt_weight_decay: float = 0.0
    rpt_episodic: bool = False
    rpt_jsd_weight: float = 0.1
    rpt_augmix_severity: int = 3
    rpt_augmix_width: int = 3
    rpt_augmix_depth: int = -1
    rpt_augmix_alpha: float = 1.0
    rpt_augmix_all_ops: bool = True
    rpt_source_anchor_weight: float = 0.01
    tent_jsd_hard_gate: str = "none"
    tent_jsd_accept_delta: float = 0.0
    tent_prediction_change_threshold: float | None = None
    tent_batch_jsd_change_threshold: float | None = None
    tent_global_reset_on_reject: bool = False
    tent_reject_use_source_stats: bool = False
    tent_prob_log_file: str | None = None
    rpt_precompute_augmix: bool = True
    rpt_augmix_cache_dir: str | None = "outputs/cache/ra_rpt_augmix"
    rpt_rebuild_augmix_cache: bool = False
    sweep_enabled: bool = False
    sweep_name: str | None = None
    rpt_lr_values: list[float] | None = None
    rpt_jsd_weight_values: list[float] | None = None
    rpt_source_anchor_weight_values: list[float] | None = None


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints on clean and corrupted target datasets."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--clean-dataset",
        choices=["cifar10", "cifar10_1", "mnist", "svhn"],
        default="mnist",
    )
    parser.add_argument("--source-dataset", choices=["cifar10", "mnist", "svhn"], default="svhn")
    parser.add_argument(
        "--ood-dataset",
        choices=[
            "cifar10c",
            "cifar10_1",
            "cifar10_1_c",
            "mnistc",
            "digitrobust_mnistc",
            "digitrobust_svhnc",
            "digitrobust_combined",
        ],
        default="digitrobust_mnistc",
    )
    parser.add_argument("--clean-data-dir", default="datasets")
    parser.add_argument("--clean-download", action="store_true")
    parser.add_argument("--clean-shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-seed", type=int, default=0)
    parser.add_argument("--data-dir", default="datasets/DigitRobust")
    parser.add_argument("--cifar10-1-data-dir", default="datasets/CIFAR-10.1")
    parser.add_argument("--cifar10-1-version", choices=["v4", "v6"], default="v6")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=list(DIGITROBUST_CORRUPTIONS),
    )
    parser.add_argument(
        "--severity-levels",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=[1],
    )
    parser.add_argument("--max-examples-per-condition", type=int, default=None)
    parser.add_argument("--condition-seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--out-file", default=None)
    parser.add_argument(
        "--evaluation-mode",
        choices=["clean", "standard", "mixed"],
        default="standard",
    )
    parser.add_argument(
        "--standard-include-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mixed-seed", type=int, default=0)
    parser.add_argument("--mixed-max-examples", type=int, default=None)
    parser.add_argument("--mixed-include-clean", action="store_true")
    parser.add_argument("--digitrobust-subset", choices=["mnist", "svhn"], default=None)
    parser.add_argument(
        "--digitrobust-subsets",
        nargs="+",
        choices=["mnist", "svhn"],
        default=None,
    )
    parser.add_argument(
        "--test-adapt",
        choices=["none", "adabn", "eata", "cotta", "tent", "rpt", "sarpt"],
        default="none",
    )
    parser.add_argument(
        "--adabn-reset-stats",
        dest="adabn_reset_stats",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--adabn-keep-stats",
        dest="adabn_reset_stats",
        action="store_false",
    )
    parser.add_argument("--adabn-momentum", type=float, default=None)
    parser.add_argument("--adapt-max-batches", type=int, default=None)
    parser.add_argument("--eata-lr", type=float, default=2.5e-4)
    parser.add_argument("--eata-steps", type=int, default=1)
    parser.add_argument("--eata-optimizer", choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--eata-weight-decay", type=float, default=0.0)
    parser.add_argument("--eata-episodic", action="store_true")
    parser.add_argument("--eata-e-margin", type=float, default=math.log(10) * 0.4)
    parser.add_argument("--eata-d-margin", type=float, default=0.05)
    parser.add_argument("--eata-fisher-size", type=int, default=2000)
    parser.add_argument("--eata-fisher-alpha", type=float, default=2000.0)
    parser.add_argument("--eata-fisher-clip-by-norm", type=float, default=None)
    parser.add_argument("--cotta-lr", type=float, default=1e-3)
    parser.add_argument("--cotta-steps", type=int, default=1)
    parser.add_argument("--cotta-optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--cotta-weight-decay", type=float, default=0.0)
    parser.add_argument("--cotta-episodic", action="store_true")
    parser.add_argument("--cotta-mt-alpha", type=float, default=0.999)
    parser.add_argument("--cotta-rst-m", type=float, default=0.01)
    parser.add_argument("--cotta-ap", type=float, default=0.92)
    parser.add_argument("--cotta-augmentation-views", type=int, default=32)
    parser.add_argument("--cotta-gaussian-std", type=float, default=0.005)
    parser.add_argument(
        "--cotta-soft-augmentations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cotta-beta", type=float, default=0.9)
    parser.add_argument("--tent-lr", type=float, default=1e-3)
    parser.add_argument("--tent-steps", type=int, default=1)
    parser.add_argument("--tent-optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--tent-weight-decay", type=float, default=0.0)
    parser.add_argument("--tent-episodic", action="store_true")
    parser.add_argument("--rpt-lr", type=float, default=1e-3)
    parser.add_argument("--rpt-steps", type=int, default=1)
    parser.add_argument("--rpt-optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--rpt-weight-decay", type=float, default=0.0)
    parser.add_argument("--rpt-episodic", action="store_true")
    parser.add_argument("--rpt-jsd-weight", "--tent-jsd-weight", dest="rpt_jsd_weight", type=float, default=0.1)
    parser.add_argument("--rpt-augmix-severity", "--tent-augmix-severity", dest="rpt_augmix_severity", type=int, default=3)
    parser.add_argument("--rpt-augmix-width", "--tent-augmix-width", dest="rpt_augmix_width", type=int, default=3)
    parser.add_argument("--rpt-augmix-depth", "--tent-augmix-depth", dest="rpt_augmix_depth", type=int, default=-1)
    parser.add_argument("--rpt-augmix-alpha", "--tent-augmix-alpha", dest="rpt_augmix_alpha", type=float, default=1.0)
    parser.add_argument(
        "--rpt-augmix-all-ops",
        "--tent-augmix-all-ops",
        dest="rpt_augmix_all_ops",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rpt-source-anchor-weight", "--tent-source-anchor-weight", dest="rpt_source_anchor_weight", type=float, default=0.01)
    parser.add_argument(
        "--tent-jsd-hard-gate",
        choices=["none", "version4"],
        default="none",
    )
    parser.add_argument("--tent-jsd-accept-delta", type=float, default=0.0)
    parser.add_argument("--tent-prediction-change-threshold", type=float, default=None)
    parser.add_argument("--tent-batch-jsd-change-threshold", type=float, default=None)
    parser.add_argument(
        "--tent-global-reset-on-reject",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tent-reject-use-source-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--tent-prob-log-file", default=None)
    parser.add_argument(
        "--rpt-precompute-augmix",
        "--tent-precompute-augmix",
        dest="rpt_precompute_augmix",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rpt-augmix-cache-dir", "--tent-augmix-cache-dir", dest="rpt_augmix_cache_dir", default="outputs/cache/ra_rpt_augmix")
    parser.add_argument("--rpt-rebuild-augmix-cache", "--tent-rebuild-augmix-cache", dest="rpt_rebuild_augmix_cache", action="store_true")
    parser.add_argument("--sweep-enabled", action="store_true")
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--rpt-lr-values", "--tent-lr-values", dest="rpt_lr_values", nargs="+", type=float, default=None)
    parser.add_argument("--rpt-jsd-weight-values", "--tent-jsd-weight-values", dest="rpt_jsd_weight_values", nargs="+", type=float, default=None)
    parser.add_argument(
        "--rpt-source-anchor-weight-values",
        "--tent-source-anchor-weight-values",
        dest="rpt_source_anchor_weight_values",
        nargs="+",
        type=float,
        default=None,
    )
    payload = vars(parser.parse_args())
    return EvaluationConfig(**payload)


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint


def default_digitrobust_subset(args: EvaluationConfig) -> str:
    if args.digitrobust_subset:
        return args.digitrobust_subset
    if args.ood_dataset == "digitrobust_svhnc":
        return "svhn"
    return "mnist"


def digitrobust_subsets(args: EvaluationConfig) -> list[str]:
    if args.digitrobust_subsets:
        return list(args.digitrobust_subsets)
    return [default_digitrobust_subset(args)]


def build_clean_loader(args: EvaluationConfig) -> torch.utils.data.DataLoader:
    if args.ood_dataset == "cifar10_1" or args.clean_dataset == "cifar10_1":
        return build_cifar10_1_loader(
            data_dir=args.cifar10_1_data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            version=args.cifar10_1_version,
            download=args.clean_download,
            normalization_dataset=args.source_dataset,
            shuffle=args.clean_shuffle,
            seed=args.clean_seed,
        )
    if args.ood_dataset in DIGITROBUST_OOD_DATASETS:
        subset = args.digitrobust_subset or args.clean_dataset
        return build_digitrobust_clean_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            subset=subset,
            normalization_dataset=args.source_dataset,
        )
    return build_clean_test_loader(
        dataset_name=args.clean_dataset,
        data_dir=args.clean_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.clean_download,
        normalization_dataset=args.source_dataset,
    )


def build_ood_loader(
    args: EvaluationConfig,
    corruption: str,
    severity: int,
) -> torch.utils.data.DataLoader:
    if args.ood_dataset in DIGITROBUST_OOD_DATASETS:
        subsets = digitrobust_subsets(args)
        if len(subsets) != 1:
            raise ValueError(
                "Standard DigitRobust evaluation supports one subset. "
                "Use evaluation_mode: mixed for multiple digitrobust_subsets."
            )
        subset = subsets[0]
        if severity != 1:
            raise ValueError("DigitRobust corruptions use severity level 1.")
        return build_digitrobust_corrupt_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            subset=subset,
            corruption=corruption,
            normalization_dataset=args.source_dataset,
        )
    if args.ood_dataset == "cifar10c":
        return build_cifar10c_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption=corruption,
            severity=severity,
            max_examples=args.max_examples_per_condition,
            seed=args.condition_seed,
        )
    if args.ood_dataset == "cifar10_1_c":
        return build_cifar10_1_c_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption=corruption,
            severity=severity,
            max_examples=args.max_examples_per_condition,
            seed=args.condition_seed,
        )
    if args.ood_dataset == "mnistc":
        return build_mnistc_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruption=corruption,
            severity=severity,
            normalization_dataset=args.source_dataset,
        )
    raise ValueError(f"Unsupported OOD dataset: {args.ood_dataset}")


def build_mixed_ood_loader(args: EvaluationConfig) -> torch.utils.data.DataLoader:
    if args.ood_dataset in DIGITROBUST_OOD_DATASETS:
        subset = digitrobust_subsets(args)
        return build_mixed_digitrobust_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            subset=subset,
            corruptions=args.corruptions,
            include_clean=args.mixed_include_clean,
            seed=args.mixed_seed,
            max_examples=args.mixed_max_examples,
            normalization_dataset=args.source_dataset,
        )
    if args.ood_dataset == "cifar10c":
        return build_mixed_cifar10c_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruptions=args.corruptions,
            severity_levels=args.severity_levels,
            clean_data_dir=args.clean_data_dir,
            clean_download=args.clean_download,
            include_clean=args.mixed_include_clean,
            seed=args.mixed_seed,
            max_examples=args.mixed_max_examples,
            max_examples_per_condition=args.max_examples_per_condition,
        )
    if args.ood_dataset == "cifar10_1_c":
        return build_mixed_cifar10_1_c_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruptions=args.corruptions,
            severity_levels=args.severity_levels,
            cifar10_1_data_dir=args.cifar10_1_data_dir,
            cifar10_1_version=args.cifar10_1_version,
            clean_download=args.clean_download,
            include_clean=args.mixed_include_clean,
            seed=args.mixed_seed,
            max_examples=args.mixed_max_examples,
            max_examples_per_condition=args.max_examples_per_condition,
        )
    if args.ood_dataset == "mnistc":
        return build_mixed_mnistc_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            corruptions=args.corruptions,
            clean_data_dir=args.clean_data_dir,
            clean_download=args.clean_download,
            include_clean=args.mixed_include_clean,
            seed=args.mixed_seed,
            max_examples=args.mixed_max_examples,
            normalization_dataset=args.source_dataset,
        )
    raise ValueError(f"Unsupported OOD dataset: {args.ood_dataset}")


def build_eata_fisher_loader(args: EvaluationConfig) -> torch.utils.data.DataLoader | None:
    if args.eata_fisher_size <= 0:
        return None
    if args.ood_dataset in DIGITROBUST_OOD_DATASETS:
        return build_digitrobust_clean_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            subset=args.source_dataset,
            split="train",
            normalization_dataset=args.source_dataset,
        )
    return build_clean_test_loader(
        dataset_name=args.source_dataset,
        data_dir=args.clean_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.clean_download,
        normalization_dataset=args.source_dataset,
    )


def build_augmix_cache_metadata(
    args: EvaluationConfig,
    row_label: str,
) -> dict[str, object]:
    metadata = {
        "cache_version": 1,
        "row_label": row_label,
        "ood_dataset": args.ood_dataset,
        "clean_dataset": args.clean_dataset,
        "source_dataset": args.source_dataset,
        "cifar10_1_version": args.cifar10_1_version,
        "data_dir": args.data_dir,
        "cifar10_1_data_dir": args.cifar10_1_data_dir,
        "clean_data_dir": args.clean_data_dir,
        "clean_shuffle": args.clean_shuffle,
        "clean_seed": args.clean_seed,
        "digitrobust_subset": args.digitrobust_subset,
        "evaluation_mode": args.evaluation_mode,
        "corruptions": list(args.corruptions),
        "severity_levels": list(args.severity_levels),
        "max_examples_per_condition": args.max_examples_per_condition,
        "condition_seed": args.condition_seed,
        "mixed_seed": args.mixed_seed,
        "mixed_max_examples": args.mixed_max_examples,
        "mixed_include_clean": args.mixed_include_clean,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "normalization_dataset": args.source_dataset,
        "augmix_severity": args.rpt_augmix_severity,
        "augmix_width": args.rpt_augmix_width,
        "augmix_depth": args.rpt_augmix_depth,
        "augmix_alpha": args.rpt_augmix_alpha,
        "augmix_all_ops": args.rpt_augmix_all_ops,
    }
    if args.digitrobust_subsets:
        metadata["digitrobust_subsets"] = list(args.digitrobust_subsets)
    return metadata


def build_augmix_cache_key(metadata: dict[str, object]) -> str:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    label = str(metadata["row_label"]).replace("@", "_").replace("/", "_")
    dataset = str(metadata["ood_dataset"]).replace("/", "_")
    mode = str(metadata["evaluation_mode"]).replace("/", "_")
    return f"{dataset}_{mode}_{label}_{digest}"


def is_tent_like_method(test_adapt: str) -> bool:
    return test_adapt in {"tent", "rpt", "sarpt"}


def tent_episodic_file_suffix(test_adapt: str, tent_episodic: bool) -> str | None:
    if not is_tent_like_method(test_adapt):
        return None
    return f"{test_adapt}_episodic_{str(bool(tent_episodic)).lower()}"


def eata_episodic_file_suffix(test_adapt: str, eata_episodic: bool) -> str | None:
    if test_adapt != "eata":
        return None
    return f"eata_episodic_{str(bool(eata_episodic)).lower()}"


def cotta_episodic_file_suffix(test_adapt: str, cotta_episodic: bool) -> str | None:
    if test_adapt != "cotta":
        return None
    return f"cotta_episodic_{str(bool(cotta_episodic)).lower()}"


def slugify(value: object) -> str:
    text = str(value).strip().lower().replace(".", "p")
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def float_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def sweep_method_slug(args: EvaluationConfig) -> str:
    if args.sweep_name:
        return slugify(args.sweep_name)
    return slugify(args.test_adapt)


def output_path_for_args(args: EvaluationConfig) -> Path:
    if args.out_file:
        return Path(args.out_file)
    output_stem = f"{Path(args.checkpoint).stem}_shift_results"
    if args.test_adapt in {"rpt", "sarpt"}:
        episodic_suffix = tent_episodic_file_suffix(args.test_adapt, args.rpt_episodic)
    else:
        episodic_suffix = tent_episodic_file_suffix(args.test_adapt, args.tent_episodic)
    if episodic_suffix is None:
        episodic_suffix = eata_episodic_file_suffix(
            args.test_adapt,
            args.eata_episodic,
        )
    if episodic_suffix is None:
        episodic_suffix = cotta_episodic_file_suffix(
            args.test_adapt,
            args.cotta_episodic,
        )
    if episodic_suffix is not None:
        output_stem = f"{output_stem}_{episodic_suffix}"
    return Path(args.checkpoint).with_name(f"{output_stem}.csv")


def should_record_stream_accuracy(args: EvaluationConfig) -> bool:
    return (
        args.evaluation_mode == "mixed"
        and args.ood_dataset == "digitrobust_combined"
        and set(digitrobust_subsets(args)) == {"svhn", "mnist"}
    )


def stream_accuracy_paths(args: EvaluationConfig) -> tuple[Path, Path]:
    output_path = output_path_for_args(args)
    csv_path = output_path.with_name(f"{output_path.stem}_stream_accuracy.csv")
    plot_path = output_path.with_name(f"{output_path.stem}_stream_accuracy.png")
    return csv_path, plot_path


class StreamAccuracyRecorder:
    def __init__(self, row_label: str) -> None:
        self.row_label = row_label
        self.rows: list[dict[str, float | int | str]] = []

    def __call__(self, metrics: dict[str, float]) -> None:
        self.rows.append(
            {
                "step": int(metrics["step"]),
                "row_label": self.row_label,
                "batch_examples": int(metrics["batch_examples"]),
                "batch_loss": metrics["batch_loss"],
                "batch_accuracy": metrics["batch_accuracy"],
                "batch_correct": int(metrics["batch_correct"]),
                "cumulative_examples": int(metrics["cumulative_examples"]),
                "cumulative_loss": metrics["cumulative_loss"],
                "cumulative_accuracy": metrics["cumulative_accuracy"],
                "cumulative_correct": int(metrics["cumulative_correct"]),
            }
        )

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=STREAM_HISTORY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)

    def save_plot(self, path: Path, title: str) -> Path | None:
        if not self.rows:
            return None
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            print(f"Skipping stream accuracy plot because optional dependency is missing: {exc.name}")
            return None

        steps = [int(row["step"]) for row in self.rows]
        accuracies = [float(row["cumulative_accuracy"]) for row in self.rows]
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.plot(steps, accuracies, linewidth=1.8)
        ax.set_xlabel("step")
        ax.set_ylabel("cumulative accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path


def tensor_to_list(value: object) -> list:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Expected tensor/list payload, got {type(value).__name__}.")


def probability_log_path(args: EvaluationConfig) -> Path | None:
    if args.tent_prob_log_file is None:
        return None
    if args.tent_prob_log_file.strip().lower() in {"", "none", "null"}:
        return None
    if args.tent_prob_log_file.strip().lower() == "auto":
        output_path = output_path_for_args(args)
        return output_path.with_name(f"{output_path.stem}_probabilities.csv")
    return Path(args.tent_prob_log_file)


def should_record_probabilities(args: EvaluationConfig) -> bool:
    return probability_log_path(args) is not None and args.test_adapt in {"tent", "rpt", "sarpt"}


class ProbabilityLogRecorder:
    def __init__(self, path: Path, row_label: str) -> None:
        self.path = path
        self.row_label = row_label
        self.next_image_number = 1
        self.final_pred_counts: Counter[int] = Counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=PROBABILITY_LOG_FIELDNAMES,
        )
        self.writer.writeheader()

    @staticmethod
    def entropy(probs: list[float]) -> float:
        return -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)

    @staticmethod
    def probability_columns(prefix: str, probs: list[float]) -> dict[str, float]:
        return {
            f"{prefix}{digit}": float(probs[digit])
            for digit in range(min(DIGIT_CLASS_COUNT, len(probs)))
        }

    def __call__(self, metrics: dict[str, object]) -> None:
        if "probs" not in metrics or "targets" not in metrics:
            return

        probs_rows = tensor_to_list(metrics["probs"])
        targets = [int(value) for value in tensor_to_list(metrics["targets"])]
        preds = (
            [int(value) for value in tensor_to_list(metrics["preds"])]
            if "preds" in metrics
            else [int(max(range(len(probs)), key=lambda idx: probs[idx])) for probs in probs_rows]
        )
        aug1_probs_rows = (
            tensor_to_list(metrics["aug1_probs"])
            if "aug1_probs" in metrics
            else None
        )
        aug2_probs_rows = (
            tensor_to_list(metrics["aug2_probs"])
            if "aug2_probs" in metrics
            else None
        )
        aug1_preds = (
            [int(value) for value in tensor_to_list(metrics["aug1_preds"])]
            if "aug1_preds" in metrics
            else None
        )
        aug2_preds = (
            [int(value) for value in tensor_to_list(metrics["aug2_preds"])]
            if "aug2_preds" in metrics
            else None
        )
        jsd_values = tensor_to_list(metrics["jsd"]) if "jsd" in metrics else None
        step = int(float(metrics["step"]))

        for index, probs in enumerate(probs_rows):
            pred = preds[index]
            target = targets[index]
            row: dict[str, object] = {
                "image_number": self.next_image_number,
                "step": step,
                "row_label": self.row_label,
                "target": target,
                "pred": pred,
                "correct": int(pred == target),
                "confidence": max(float(value) for value in probs),
                "entropy": self.entropy([float(value) for value in probs]),
                "aug1_pred": "",
                "aug2_pred": "",
                "clean_aug1_agree": "",
                "clean_aug2_agree": "",
                "aug1_aug2_agree": "",
                "jsd": "",
            }
            row.update(self.probability_columns("prob_", [float(value) for value in probs]))
            if aug1_probs_rows is not None and aug2_probs_rows is not None:
                aug1_pred = int(aug1_preds[index]) if aug1_preds is not None else ""
                aug2_pred = int(aug2_preds[index]) if aug2_preds is not None else ""
                row.update(
                    {
                        "aug1_pred": aug1_pred,
                        "aug2_pred": aug2_pred,
                        "clean_aug1_agree": int(pred == aug1_pred),
                        "clean_aug2_agree": int(pred == aug2_pred),
                        "aug1_aug2_agree": int(aug1_pred == aug2_pred),
                        "jsd": (
                            float(jsd_values[index])
                            if jsd_values is not None
                            else ""
                        ),
                    }
                )
                row.update(
                    self.probability_columns(
                        "aug1_prob_",
                        [float(value) for value in aug1_probs_rows[index]],
                    )
                )
                row.update(
                    self.probability_columns(
                        "aug2_prob_",
                        [float(value) for value in aug2_probs_rows[index]],
                    )
                )
            self.writer.writerow(row)
            self.final_pred_counts[pred] += 1
            self.next_image_number += 1

    def close(self) -> None:
        self.handle.close()


class BatchRecorder:
    def __init__(self, *recorders: Callable[[dict[str, object]], None] | None) -> None:
        self.recorders = [recorder for recorder in recorders if recorder is not None]

    def __bool__(self) -> bool:
        return bool(self.recorders)

    def __call__(self, metrics: dict[str, object]) -> None:
        for recorder in self.recorders:
            recorder(metrics)

    def close(self) -> None:
        for recorder in self.recorders:
            close = getattr(recorder, "close", None)
            if close is not None:
                close()


def sweep_variant_slug(
    lr: float,
    jsd_weight: float,
    source_anchor_weight: float,
) -> str:
    parts = [
        f"lr{float_slug(lr)}",
        f"jsd{float_slug(jsd_weight)}",
        f"src{float_slug(source_anchor_weight)}",
    ]
    return "_".join(parts)


def iter_sweep_configs(args: EvaluationConfig) -> list[EvaluationConfig]:
    if args.test_adapt not in {"rpt", "sarpt"}:
        raise ValueError("Sweep currently supports test_adapt: rpt or sarpt.")

    lrs = args.rpt_lr_values or [args.rpt_lr]
    jsd_weights = args.rpt_jsd_weight_values or [args.rpt_jsd_weight]
    source_anchor_weights = (
        args.rpt_source_anchor_weight_values or [args.rpt_source_anchor_weight]
    )

    base_output_path = output_path_for_args(args)
    sweep_dir = base_output_path.parent / f"{sweep_method_slug(args)}_sweep"
    configs: list[EvaluationConfig] = []
    seen_output_paths: set[Path] = set()

    for lr, jsd_weight, source_anchor_weight in product(
        lrs,
        jsd_weights,
        source_anchor_weights,
    ):
        variant_slug = sweep_variant_slug(
            lr,
            jsd_weight,
            source_anchor_weight,
        )
        output_path = sweep_dir / f"{base_output_path.stem}__{variant_slug}.csv"
        if output_path in seen_output_paths:
            continue
        seen_output_paths.add(output_path)
        configs.append(
            replace(
                args,
                sweep_enabled=False,
                rpt_lr=lr,
                rpt_jsd_weight=jsd_weight,
                rpt_source_anchor_weight=source_anchor_weight,
                out_file=str(output_path),
            )
        )

    if not configs:
        raise ValueError("Sweep produced no configurations.")
    return configs


def evaluate_loader(
    args: EvaluationConfig,
    checkpoint_path: Path,
    device: torch.device,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
    row_label: str,
    batch_metrics_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[dict[str, float], dict]:
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    if args.test_adapt == "adabn":
        metrics = evaluate_model_per_batch_adabn(
            model=model,
            dataloader=loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_batches,
            progress_desc=f"AdaBN-batch {row_label}",
            show_progress=True,
            batch_metrics_callback=batch_metrics_callback,
        )
    elif args.test_adapt == "tent":
        metrics = evaluate_model_tent(
            model=model,
            dataloader=loader,
            criterion=criterion,
            device=device,
            steps=args.tent_steps,
            lr=args.tent_lr,
            optimizer_name=args.tent_optimizer,
            weight_decay=args.tent_weight_decay,
            episodic=args.tent_episodic,
            max_batches=args.max_batches,
            progress_desc=f"TENT {row_label}",
            show_progress=True,
            batch_metrics_callback=batch_metrics_callback,
        )
    elif args.test_adapt == "eata":
        metrics = evaluate_model_eata(
            model=model,
            dataloader=loader,
            criterion=criterion,
            device=device,
            steps=args.eata_steps,
            lr=args.eata_lr,
            optimizer_name=args.eata_optimizer,
            weight_decay=args.eata_weight_decay,
            episodic=args.eata_episodic,
            e_margin=args.eata_e_margin,
            d_margin=args.eata_d_margin,
            fisher_dataloader=build_eata_fisher_loader(args),
            fisher_size=args.eata_fisher_size,
            fisher_alpha=args.eata_fisher_alpha,
            fisher_clip_by_norm=args.eata_fisher_clip_by_norm,
            max_batches=args.max_batches,
            progress_desc=f"EATA {row_label}",
            show_progress=True,
            batch_metrics_callback=batch_metrics_callback,
        )
    elif args.test_adapt == "cotta":
        metrics = evaluate_model_cotta(
            model=model,
            dataloader=loader,
            criterion=criterion,
            device=device,
            steps=args.cotta_steps,
            lr=args.cotta_lr,
            optimizer_name=args.cotta_optimizer,
            weight_decay=args.cotta_weight_decay,
            episodic=args.cotta_episodic,
            mt_alpha=args.cotta_mt_alpha,
            rst_m=args.cotta_rst_m,
            ap=args.cotta_ap,
            augmentation_views=args.cotta_augmentation_views,
            normalization_dataset=args.source_dataset,
            gaussian_std=args.cotta_gaussian_std,
            soft_augmentations=args.cotta_soft_augmentations,
            beta=args.cotta_beta,
            max_batches=args.max_batches,
            progress_desc=f"CoTTA {row_label}",
            show_progress=True,
            batch_metrics_callback=batch_metrics_callback,
        )
    elif args.test_adapt in {"rpt", "sarpt"}:
        augmix_cache_metadata = build_augmix_cache_metadata(args, row_label)
        adapt_name = "SARPT" if args.test_adapt == "sarpt" else "RPT"
        metrics = evaluate_model_rpt(
            model=model,
            dataloader=loader,
            criterion=criterion,
            device=device,
            steps=args.rpt_steps,
            lr=args.rpt_lr,
            optimizer_name=args.rpt_optimizer,
            weight_decay=args.rpt_weight_decay,
            episodic=args.rpt_episodic,
            jsd_weight=args.rpt_jsd_weight,
            normalization_dataset=args.source_dataset,
            augmix_severity=args.rpt_augmix_severity,
            augmix_width=args.rpt_augmix_width,
            augmix_depth=args.rpt_augmix_depth,
            augmix_alpha=args.rpt_augmix_alpha,
            augmix_all_ops=args.rpt_augmix_all_ops,
            source_anchor_weight=args.rpt_source_anchor_weight,
            precompute_augmix=args.rpt_precompute_augmix,
            augmix_cache_dir=args.rpt_augmix_cache_dir,
            augmix_cache_key=build_augmix_cache_key(augmix_cache_metadata),
            augmix_cache_metadata=augmix_cache_metadata,
            rebuild_augmix_cache=args.rpt_rebuild_augmix_cache,
            max_batches=args.max_batches,
            progress_desc=f"{adapt_name} {row_label}",
            show_progress=True,
            batch_metrics_callback=batch_metrics_callback,
        )
    elif args.test_adapt == "none":
        metrics = evaluate_model(
            model=model,
            dataloader=loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_batches,
            progress_desc=f"Eval {row_label}",
            show_progress=True,
            batch_metrics_callback=batch_metrics_callback,
        )
    else:
        raise ValueError(f"Unsupported test adaptation method: {args.test_adapt}")

    return metrics, checkpoint


def evaluate_ood_condition(
    args: EvaluationConfig,
    checkpoint_path: Path,
    device: torch.device,
    criterion: nn.Module,
    corruption: str,
    severity: int,
) -> tuple[dict[str, float], dict]:
    loader = build_ood_loader(args, corruption, severity)
    return evaluate_loader(
        args=args,
        checkpoint_path=checkpoint_path,
        device=device,
        criterion=criterion,
        loader=loader,
        row_label=f"{corruption}@{severity}",
    )


def run_mixed_evaluation(
    args: EvaluationConfig,
    checkpoint_path: Path,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[list[dict[str, float | str]], dict, dict[str, object]]:
    stream_recorder = (
        StreamAccuracyRecorder(row_label="mixed")
        if should_record_stream_accuracy(args)
        else None
    )
    prob_log_csv_path = probability_log_path(args) if should_record_probabilities(args) else None
    probability_recorder = (
        ProbabilityLogRecorder(prob_log_csv_path, row_label="mixed")
        if prob_log_csv_path is not None
        else None
    )
    batch_recorder = BatchRecorder(stream_recorder, probability_recorder)
    try:
        metrics, checkpoint = evaluate_loader(
            args=args,
            checkpoint_path=checkpoint_path,
            device=device,
            criterion=criterion,
            loader=build_mixed_ood_loader(args),
            row_label="mixed",
            batch_metrics_callback=batch_recorder if batch_recorder else None,
        )
    finally:
        batch_recorder.close()
    metrics = dict(metrics)
    if prob_log_csv_path is not None:
        metrics["probability_log_csv"] = str(prob_log_csv_path)
        print(f"Saved probability CSV to: {prob_log_csv_path.resolve()}")
    if stream_recorder is not None:
        history_csv_path, history_plot_path = stream_accuracy_paths(args)
        stream_recorder.save_csv(history_csv_path)
        saved_plot = stream_recorder.save_plot(
            history_plot_path,
            title=f"{Path(args.out_file or args.checkpoint).stem} stream accuracy",
        )
        metrics["stream_accuracy_csv"] = str(history_csv_path)
        metrics["stream_accuracy_plot"] = str(saved_plot) if saved_plot is not None else None
        print(f"Saved stream accuracy CSV to: {history_csv_path.resolve()}")
        if saved_plot is not None:
            print(f"Saved stream accuracy plot to: {saved_plot.resolve()}")
    rows: list[dict[str, float | str]] = [
        {
            "model": checkpoint["model_name"],
            "test_adapt": args.test_adapt,
            "domain": "ood",
            "corruption": "mixed",
            "severity": 0,
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "error_rate": 1.0 - metrics["accuracy"],
            "ood_error": 1.0 - metrics["accuracy"],
            "examples": metrics["examples"],
        }
    ]
    subset_count = (
        len(digitrobust_subsets(args))
        if args.ood_dataset in DIGITROBUST_OOD_DATASETS
        else 1
    )
    condition_count = len(args.corruptions) * len(args.severity_levels) * subset_count
    if args.mixed_include_clean:
        condition_count += subset_count
    print(
        f"{'mixed':12s} | adapt={args.test_adapt} | "
        f"conditions={condition_count} | "
        f"acc={metrics['accuracy']:.4f}"
    )
    return rows, checkpoint, metrics


def run_clean_evaluation(
    args: EvaluationConfig,
    checkpoint_path: Path,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[list[dict[str, float | str]], dict, dict[str, float]]:
    clean_metrics, checkpoint = evaluate_loader(
        args=args,
        checkpoint_path=checkpoint_path,
        device=device,
        criterion=criterion,
        loader=build_clean_loader(args),
        row_label="clean",
    )
    rows: list[dict[str, float | str]] = [
        {
            "model": checkpoint["model_name"],
            "test_adapt": args.test_adapt,
            "domain": "clean",
            "corruption": "clean",
            "severity": 0,
            "loss": clean_metrics["loss"],
            "accuracy": clean_metrics["accuracy"],
            "error_rate": 1.0 - clean_metrics["accuracy"],
            "ood_error": "",
            "examples": clean_metrics["examples"],
        }
    ]
    print(
        f"clean        | adapt={args.test_adapt} | "
        f"severity=0 | acc={clean_metrics['accuracy']:.4f}"
    )
    return rows, checkpoint, clean_metrics


def run_standard_evaluation(
    args: EvaluationConfig,
    checkpoint_path: Path,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[list[dict[str, float | str]], dict, dict[str, float] | None]:
    rows: list[dict[str, float | str]] = []
    checkpoint = None
    clean_metrics = None

    if args.standard_include_clean:
        clean_metrics, checkpoint = evaluate_loader(
            args=args,
            checkpoint_path=checkpoint_path,
            device=device,
            criterion=criterion,
            loader=build_clean_loader(args),
            row_label="clean",
        )
        rows.append(
            {
                "model": checkpoint["model_name"],
                "test_adapt": args.test_adapt,
                "domain": "clean",
                "corruption": "clean",
                "severity": 0,
                "loss": clean_metrics["loss"],
                "accuracy": clean_metrics["accuracy"],
                "error_rate": 1.0 - clean_metrics["accuracy"],
                "ood_error": "",
                "examples": clean_metrics["examples"],
            }
        )
        print(
            f"clean        | adapt={args.test_adapt} | "
            f"severity=0 | acc={clean_metrics['accuracy']:.4f}"
        )

    for corruption in args.corruptions:
        for severity in args.severity_levels:
            metrics, checkpoint = evaluate_ood_condition(
                args=args,
                checkpoint_path=checkpoint_path,
                device=device,
                criterion=criterion,
                corruption=corruption,
                severity=int(severity),
            )
            rows.append(
                {
                    "model": checkpoint["model_name"],
                    "test_adapt": args.test_adapt,
                    "domain": "ood",
                    "corruption": corruption,
                    "severity": severity,
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "error_rate": 1.0 - metrics["accuracy"],
                    "ood_error": 1.0 - metrics["accuracy"],
                    "examples": metrics["examples"],
                }
            )
            print(
                f"{corruption:12s} | adapt={args.test_adapt} | "
                f"severity={severity} | acc={metrics['accuracy']:.4f}"
            )

    if checkpoint is None:
        raise RuntimeError(f"No {args.ood_dataset} conditions were evaluated.")
    return rows, checkpoint, clean_metrics


def annotate_adaptation_episodic(
    rows: list[dict[str, float | str]],
    args: EvaluationConfig,
) -> None:
    if args.test_adapt == "tent":
        value: bool | str = args.tent_episodic
    elif args.test_adapt in {"rpt", "sarpt"}:
        value = args.rpt_episodic
    else:
        value = ""
    for row in rows:
        row["tta_episodic"] = value
        row["tent_episodic"] = value
        row["rpt_episodic"] = value if args.test_adapt in {"rpt", "sarpt"} else ""


def run_evaluation(args: EvaluationConfig) -> tuple[Path, Path]:
    checkpoint_path = Path(args.checkpoint)
    if args.evaluation_mode == "clean":
        pass
    elif args.ood_dataset == "cifar10c":
        ensure_cifar10c_files(args.data_dir, args.corruptions)
    elif args.ood_dataset == "cifar10_1_c":
        ensure_cifar10_1_c_files(args.data_dir, args.corruptions)
    elif args.ood_dataset == "mnistc":
        ensure_mnistc_files(args.data_dir, args.corruptions)
    elif args.ood_dataset in DIGITROBUST_OOD_DATASETS:
        ensure_digitrobust_files(args.data_dir, digitrobust_subsets(args), args.corruptions)
    else:
        raise ValueError(f"Unsupported OOD dataset: {args.ood_dataset}")
    device = resolve_device(args.device)
    criterion = nn.CrossEntropyLoss()
    stream_accuracy_csv = None
    stream_accuracy_plot = None
    probability_log_csv = None

    if args.evaluation_mode == "clean":
        rows, checkpoint, clean_metrics = run_clean_evaluation(
            args=args,
            checkpoint_path=checkpoint_path,
            device=device,
            criterion=criterion,
        )
        clean_error = 1.0 - clean_metrics["accuracy"]
        clean_accuracy = clean_metrics["accuracy"]
    elif args.evaluation_mode == "standard":
        rows, checkpoint, clean_metrics = run_standard_evaluation(
            args=args,
            checkpoint_path=checkpoint_path,
            device=device,
            criterion=criterion,
        )
        clean_error = None if clean_metrics is None else 1.0 - clean_metrics["accuracy"]
        clean_accuracy = None if clean_metrics is None else clean_metrics["accuracy"]
    elif args.evaluation_mode == "mixed":
        rows, checkpoint, mixed_metrics = run_mixed_evaluation(
            args=args,
            checkpoint_path=checkpoint_path,
            device=device,
            criterion=criterion,
        )
        stream_accuracy_csv = mixed_metrics.get("stream_accuracy_csv")
        stream_accuracy_plot = mixed_metrics.get("stream_accuracy_plot")
        probability_log_csv = mixed_metrics.get("probability_log_csv")
        clean_error = None
        clean_accuracy = None
    else:
        raise ValueError(f"Unsupported evaluation mode: {args.evaluation_mode}")

    if not rows:
        raise RuntimeError(f"No {args.ood_dataset} conditions were evaluated.")
    annotate_adaptation_episodic(rows, args)

    ood_rows = [row for row in rows if row["domain"] == "ood"]
    if args.evaluation_mode == "clean":
        average_ood_error = None
        worst_domain = None
    elif not ood_rows:
        raise RuntimeError(f"No {args.ood_dataset} OOD conditions were evaluated.")
    else:
        ood_errors = [float(row["error_rate"]) for row in ood_rows]
        average_ood_error = statistics.mean(ood_errors)
        worst_domain = max(ood_rows, key=lambda row: float(row["error_rate"]))
    is_eata = args.test_adapt == "eata"
    is_cotta = args.test_adapt == "cotta"
    is_tent_like = args.test_adapt in {"tent", "rpt", "sarpt"}
    is_rpt = args.test_adapt in {"rpt", "sarpt"}
    rpt_variant = None
    if is_rpt:
        rpt_variant = args.test_adapt

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "model": checkpoint["model_name"],
        "test_adapt": args.test_adapt,
        "evaluation_mode": args.evaluation_mode,
        "standard_include_clean": (
            args.standard_include_clean if args.evaluation_mode == "standard" else None
        ),
        "clean_dataset": args.clean_dataset,
        "source_dataset": args.source_dataset,
        "ood_dataset": args.ood_dataset,
        "digitrobust_subset": args.digitrobust_subset,
        "digitrobust_subsets": args.digitrobust_subsets,
        "cifar10_1_version": args.cifar10_1_version,
        "cifar10_1_data_dir": args.cifar10_1_data_dir,
        "clean_shuffle": args.clean_shuffle,
        "clean_seed": args.clean_seed,
        "corruptions": args.corruptions,
        "severity_levels": args.severity_levels,
        "max_examples_per_condition": args.max_examples_per_condition,
        "condition_seed": args.condition_seed,
        "mixed_seed": args.mixed_seed if args.evaluation_mode == "mixed" else None,
        "mixed_max_examples": (
            args.mixed_max_examples if args.evaluation_mode == "mixed" else None
        ),
        "mixed_include_clean": (
            args.mixed_include_clean if args.evaluation_mode == "mixed" else None
        ),
        "stream_accuracy_csv": stream_accuracy_csv,
        "stream_accuracy_plot": stream_accuracy_plot,
        "probability_log_csv": probability_log_csv,
        "eata_lr": args.eata_lr if is_eata else None,
        "eata_steps": args.eata_steps if is_eata else None,
        "eata_optimizer": args.eata_optimizer if is_eata else None,
        "eata_weight_decay": args.eata_weight_decay if is_eata else None,
        "eata_episodic": args.eata_episodic if is_eata else None,
        "eata_e_margin": args.eata_e_margin if is_eata else None,
        "eata_d_margin": args.eata_d_margin if is_eata else None,
        "eata_fisher_size": args.eata_fisher_size if is_eata else None,
        "eata_fisher_alpha": args.eata_fisher_alpha if is_eata else None,
        "eata_fisher_clip_by_norm": (
            args.eata_fisher_clip_by_norm if is_eata else None
        ),
        "cotta_lr": args.cotta_lr if is_cotta else None,
        "cotta_steps": args.cotta_steps if is_cotta else None,
        "cotta_optimizer": args.cotta_optimizer if is_cotta else None,
        "cotta_weight_decay": args.cotta_weight_decay if is_cotta else None,
        "cotta_episodic": args.cotta_episodic if is_cotta else None,
        "cotta_mt_alpha": args.cotta_mt_alpha if is_cotta else None,
        "cotta_rst_m": args.cotta_rst_m if is_cotta else None,
        "cotta_ap": args.cotta_ap if is_cotta else None,
        "cotta_augmentation_views": args.cotta_augmentation_views if is_cotta else None,
        "cotta_gaussian_std": args.cotta_gaussian_std if is_cotta else None,
        "cotta_soft_augmentations": (
            args.cotta_soft_augmentations if is_cotta else None
        ),
        "cotta_beta": args.cotta_beta if is_cotta else None,
        "cotta_returns_teacher_logits": True if is_cotta else None,
        "tent_lr": args.tent_lr if args.test_adapt == "tent" else None,
        "tent_steps": args.tent_steps if args.test_adapt == "tent" else None,
        "tent_optimizer": args.tent_optimizer if args.test_adapt == "tent" else None,
        "tent_weight_decay": args.tent_weight_decay if args.test_adapt == "tent" else None,
        "tent_episodic": args.tent_episodic if args.test_adapt == "tent" else None,
        "rpt_lr": args.rpt_lr if is_rpt else None,
        "rpt_steps": args.rpt_steps if is_rpt else None,
        "rpt_optimizer": args.rpt_optimizer if is_rpt else None,
        "rpt_weight_decay": args.rpt_weight_decay if is_rpt else None,
        "rpt_episodic": args.rpt_episodic if is_rpt else None,
        "rpt_jsd_weight": args.rpt_jsd_weight if is_rpt else None,
        "rpt_augmix_severity": args.rpt_augmix_severity if is_rpt else None,
        "rpt_augmix_width": args.rpt_augmix_width if is_rpt else None,
        "rpt_augmix_depth": args.rpt_augmix_depth if is_rpt else None,
        "rpt_augmix_alpha": args.rpt_augmix_alpha if is_rpt else None,
        "rpt_augmix_all_ops": args.rpt_augmix_all_ops if is_rpt else None,
        "rpt_source_anchor_weight": args.rpt_source_anchor_weight if is_rpt else None,
        "rpt_precompute_augmix": args.rpt_precompute_augmix if is_rpt else None,
        "rpt_augmix_cache_dir": args.rpt_augmix_cache_dir if is_rpt else None,
        "rpt_rebuild_augmix_cache": args.rpt_rebuild_augmix_cache if is_rpt else None,
        "tent_jsd_weight": args.rpt_jsd_weight if is_rpt else None,
        "tent_source_anchor_weight": args.rpt_source_anchor_weight if is_rpt else None,
        "rpt_variant": rpt_variant,
        "rpt_returns_adapted_logits": True if is_rpt else None,
        "clean_accuracy": clean_accuracy,
        "clean_error": clean_error,
        "average_ood_error": average_ood_error,
        "worst_domain_error": (
            float(worst_domain["error_rate"]) if worst_domain is not None else None
        ),
        "worst_domain": (
            {
                "corruption": worst_domain["corruption"],
                "severity": worst_domain["severity"],
            }
            if worst_domain is not None
            else None
        ),
        "robustness_gap": (
            average_ood_error - clean_error
            if average_ood_error is not None and clean_error is not None
            else None
        ),
        "rows": rows,
    }

    output_path = output_path_for_args(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "test_adapt",
                "tta_episodic",
                "tent_episodic",
                "rpt_episodic",
                "domain",
                "corruption",
                "severity",
                "loss",
                "accuracy",
                "error_rate",
                "ood_error",
                "examples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_path.with_suffix(".json")
    save_json(summary, json_path)
    print(f"Saved CSV results to: {output_path.resolve()}")
    print(f"Saved JSON summary to: {json_path.resolve()}")
    return output_path, json_path


def run_evaluation_sweep(args: EvaluationConfig) -> list[tuple[Path, Path]]:
    sweep_configs = iter_sweep_configs(args)
    completed: list[tuple[Path, Path]] = []
    summary_rows: list[dict[str, object]] = []
    print(f"Running sweep with {len(sweep_configs)} configurations.")

    for index, sweep_args in enumerate(sweep_configs, start=1):
        print(
            f"Sweep {index}/{len(sweep_configs)} | "
            f"lr={sweep_args.rpt_lr:g} | "
            f"jsd={sweep_args.rpt_jsd_weight:g} | "
            f"src={sweep_args.rpt_source_anchor_weight:g}"
        )
        csv_path, json_path = run_evaluation(sweep_args)
        completed.append((csv_path, json_path))
        with json_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        summary_rows.append(
            {
                "rpt_lr": sweep_args.rpt_lr,
                "rpt_jsd_weight": sweep_args.rpt_jsd_weight,
                "rpt_source_anchor_weight": sweep_args.rpt_source_anchor_weight,
                "rpt_episodic": sweep_args.rpt_episodic,
                "tent_lr": sweep_args.rpt_lr,
                "tent_jsd_weight": sweep_args.rpt_jsd_weight,
                "tent_source_anchor_weight": sweep_args.rpt_source_anchor_weight,
                "tent_episodic": sweep_args.rpt_episodic,
                "clean_error": result.get("clean_error"),
                "average_ood_error": result.get("average_ood_error"),
                "robustness_gap": result.get("robustness_gap"),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
            }
        )

    sweep_dir = completed[0][0].parent
    summary_csv = sweep_dir / "sweep_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    save_json({"sweep": summary_rows}, sweep_dir / "sweep_summary.json")
    print(f"Saved sweep summary to: {summary_csv.resolve()}")
    return completed


def run_evaluation_or_sweep(args: EvaluationConfig) -> list[tuple[Path, Path]]:
    if args.sweep_enabled:
        return run_evaluation_sweep(args)
    return [run_evaluation(args)]


def main() -> None:
    run_evaluation_or_sweep(parse_args())


if __name__ == "__main__":
    main()

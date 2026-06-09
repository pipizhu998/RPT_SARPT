from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from data_utils.cifar10_1_c_data import CIFAR10_1_C_CORRUPTIONS
from entry_point.evaluate import (
    EvaluationConfig,
    cotta_episodic_file_suffix,
    eata_episodic_file_suffix,
    iter_sweep_configs,
    run_evaluation_or_sweep,
    tent_episodic_file_suffix,
)
from core.experiment_config import deep_merge, load_experiment_config, load_yaml
from core.training_method_interface import TrainingMethodConfig, TrainingMethodInterface
from core.training_method_train import run_training


TRAINING_KEYS = {
    "out_dir",
    "experiment_name",
    "epochs",
    "optimizer",
    "lr",
    "momentum",
    "weight_decay",
    "scheduler",
    "label_smoothing",
    "seed",
    "device",
    "max_train_batches",
    "max_val_batches",
    "mixed_precision",
    "channels_last",
    "tf32",
    "cudnn_benchmark",
    "deterministic",
    "progress_update_interval",
}

TRAIN_AUG_KEYS = {
    "augmix_severity",
    "augmix_width",
    "augmix_depth",
    "augmix_alpha",
    "augmix_jsd_weight",
    "augmix_all_ops",
}

TESTING_KEYS = {
    "device",
    "max_batches",
    "out_file",
    "evaluation_mode",
    "standard_include_clean",
    "mixed_seed",
    "mixed_max_examples",
}

TEST_ADAPT_KEYS = {
    "adabn_reset_stats",
    "adabn_momentum",
    "adapt_max_batches",
    "eata_lr",
    "eata_steps",
    "eata_optimizer",
    "eata_weight_decay",
    "eata_episodic",
    "eata_e_margin",
    "eata_d_margin",
    "eata_fisher_size",
    "eata_fisher_alpha",
    "eata_fisher_clip_by_norm",
    "cotta_lr",
    "cotta_steps",
    "cotta_optimizer",
    "cotta_weight_decay",
    "cotta_episodic",
    "cotta_mt_alpha",
    "cotta_rst_m",
    "cotta_ap",
    "cotta_augmentation_views",
    "cotta_gaussian_std",
    "cotta_soft_augmentations",
    "cotta_beta",
    "tent_lr",
    "tent_steps",
    "tent_optimizer",
    "tent_weight_decay",
    "tent_episodic",
    "rpt_lr",
    "rpt_steps",
    "rpt_optimizer",
    "rpt_weight_decay",
    "rpt_episodic",
    "rpt_jsd_weight",
    "rpt_augmix_severity",
    "rpt_augmix_width",
    "rpt_augmix_depth",
    "rpt_augmix_alpha",
    "rpt_augmix_all_ops",
    "rpt_source_anchor_weight",
    "rpt_precompute_augmix",
    "rpt_augmix_cache_dir",
    "rpt_rebuild_augmix_cache",
    "tent_prob_log_file",
    "sweep_enabled",
    "sweep_name",
    "rpt_lr_values",
    "rpt_jsd_weight_values",
    "rpt_source_anchor_weight_values",
}

RPT_TEST_ADAPT_ALIASES = {
    "tent_lr": "rpt_lr",
    "tent_steps": "rpt_steps",
    "tent_optimizer": "rpt_optimizer",
    "tent_weight_decay": "rpt_weight_decay",
    "tent_episodic": "rpt_episodic",
    "tent_jsd_weight": "rpt_jsd_weight",
    "tent_augmix_severity": "rpt_augmix_severity",
    "tent_augmix_width": "rpt_augmix_width",
    "tent_augmix_depth": "rpt_augmix_depth",
    "tent_augmix_alpha": "rpt_augmix_alpha",
    "tent_augmix_all_ops": "rpt_augmix_all_ops",
    "tent_source_anchor_weight": "rpt_source_anchor_weight",
    "tent_precompute_augmix": "rpt_precompute_augmix",
    "tent_augmix_cache_dir": "rpt_augmix_cache_dir",
    "tent_rebuild_augmix_cache": "rpt_rebuild_augmix_cache",
    "tent_lr_values": "rpt_lr_values",
    "tent_jsd_weight_values": "rpt_jsd_weight_values",
    "tent_source_anchor_weight_values": "rpt_source_anchor_weight_values",
}


def normalize_config_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists() or "\\" not in path:
        return candidate
    return Path(path.replace("\\", "/"))


def config_path_hint(path: str) -> str:
    if "\\" in path:
        return " Use forward slashes on Bash, or quote the path if you are using backslashes."
    if path.startswith("configs") and "/" not in path:
        return (
            " It looks like Bash removed backslashes from the path. Use "
            "'configs/experiment/training/baseline.yaml' instead."
        )
    return ""


def pick(source: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a mapping.")
    return value


def normalize_rpt_test_adapt_keys(test_adapt: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(test_adapt)
    if normalized.get("method") not in {"rpt", "sarpt"}:
        return normalized
    for old_key, new_key in RPT_TEST_ADAPT_ALIASES.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized[old_key]
    return normalized


def build_training_interface(config: dict[str, Any]) -> TrainingMethodInterface:
    dataset = require_mapping(config, "dataset")
    model = require_mapping(config, "model")
    train_aug = require_mapping(config, "train_aug")
    training = require_mapping(config, "training")

    dataset_name = dataset.get("name", "cifar10")
    if dataset_name not in {"cifar10", "mnist", "svhn"}:
        raise ValueError("Training supports dataset.name in {'cifar10', 'mnist', 'svhn'}.")

    payload = pick(training, TRAINING_KEYS)
    payload.update(pick(train_aug, TRAIN_AUG_KEYS))
    payload.update(
        {
            "method": train_aug.get("method", "clean"),
            "model": model.get("name", "resnet18"),
            "dataset_name": dataset_name,
            "data_dir": dataset.get("data_dir", "datasets"),
            "batch_size": dataset.get("batch_size", 128),
            "num_workers": dataset.get("num_workers", 2),
            "val_ratio": dataset.get("val_ratio", 0.1),
            "download": dataset.get("download", False),
        }
    )
    if payload["method"] == "none":
        payload["method"] = "clean"
    if "experiment_name" not in payload and config.get("name"):
        payload["experiment_name"] = config["name"]

    return TrainingMethodInterface(TrainingMethodConfig(**payload))


def stem_with_suffix(path: str | Path, suffix: str) -> str:
    output_path = Path(path)
    return str(output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}"))


def build_evaluation_config(config: dict[str, Any]) -> EvaluationConfig:
    dataset = require_mapping(config, "dataset")
    model = require_mapping(config, "model")
    test_adapt = normalize_rpt_test_adapt_keys(require_mapping(config, "test_adapt"))
    testing = require_mapping(config, "testing")

    payload = pick(testing, TESTING_KEYS)
    payload.update(pick(test_adapt, TEST_ADAPT_KEYS))
    test_adapt_method = test_adapt.get("method", "none")
    if test_adapt_method not in {"none", "adabn", "eata", "cotta", "tent", "rpt", "sarpt"}:
        raise ValueError(
            f"Unknown test adaptation method '{test_adapt_method}'. "
            "Available: none, adabn, eata, cotta, tent, rpt, sarpt."
        )

    checkpoint = testing.get("checkpoint") or model.get("checkpoint")
    if not checkpoint:
        raise ValueError("Testing config must define testing.checkpoint or model.checkpoint.")

    dataset_name = dataset.get("name", "digitrobust_mnistc")
    if dataset_name not in {
        "cifar10c",
        "cifar10_1",
        "cifar10_1_c",
        "mnistc",
        "digitrobust_mnistc",
        "digitrobust_svhnc",
        "digitrobust_combined",
    }:
        raise ValueError(
            "Testing expects dataset.name in "
            "{'cifar10c', 'cifar10_1', 'cifar10_1_c', 'mnistc', "
            "'digitrobust_mnistc', 'digitrobust_svhnc', 'digitrobust_combined'}."
        )
    if dataset_name == "cifar10c":
        default_clean_dataset = "cifar10"
        default_source_dataset = "cifar10"
        default_corruptions = [
            "gaussian_noise",
            "motion_blur",
            "fog",
            "jpeg_compression",
            "contrast",
        ]
    elif dataset_name == "cifar10_1":
        default_clean_dataset = "cifar10_1"
        default_source_dataset = "cifar10"
        default_corruptions = []
    elif dataset_name == "cifar10_1_c":
        default_clean_dataset = "cifar10_1"
        default_source_dataset = "cifar10"
        default_corruptions = list(CIFAR10_1_C_CORRUPTIONS)
    else:
        default_clean_dataset = "mnist"
        default_source_dataset = "svhn"
        default_corruptions = [
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
    corruptions = dataset.get("corruptions", default_corruptions)
    default_data_dir = (
        "datasets/DigitRobust"
        if dataset_name
        in {"digitrobust_mnistc", "digitrobust_svhnc", "digitrobust_combined"}
        else "datasets"
    )

    seed_override = config.get("_seed_override")
    seed_suffix = f"_seed{seed_override}" if seed_override is not None else ""

    if "out_file" not in payload:
        experiment_name = str(config.get("name") or Path(config["_config_path"]).stem)
        if test_adapt_method in {"rpt", "sarpt"}:
            episodic_value = bool(test_adapt.get("rpt_episodic", False))
        else:
            episodic_value = bool(test_adapt.get("tent_episodic", False))
        episodic_suffix = tent_episodic_file_suffix(test_adapt_method, episodic_value)
        if episodic_suffix is None:
            episodic_suffix = eata_episodic_file_suffix(
                test_adapt_method,
                bool(test_adapt.get("eata_episodic", False)),
            )
        if episodic_suffix is None:
            episodic_suffix = cotta_episodic_file_suffix(
                test_adapt_method,
                bool(test_adapt.get("cotta_episodic", False)),
            )
        if episodic_suffix is not None:
            experiment_name = f"{experiment_name}_{episodic_suffix}"
        if seed_suffix:
            experiment_name = f"{experiment_name}{seed_suffix}"
        out_dir = Path(testing.get("out_dir", "outputs/experiments/testing"))
        payload["out_file"] = str(out_dir / f"{experiment_name}_results.csv")
    elif seed_suffix:
        payload["out_file"] = stem_with_suffix(payload["out_file"], seed_suffix)

    payload.update(
        {
            "checkpoint": checkpoint,
            "clean_dataset": dataset.get("clean_dataset", default_clean_dataset),
            "source_dataset": dataset.get("source_dataset", default_source_dataset),
            "ood_dataset": dataset_name,
            "clean_data_dir": dataset.get("clean_data_dir", "datasets"),
            "clean_download": dataset.get("clean_download", False),
            "clean_shuffle": dataset.get("clean_shuffle", False),
            "clean_seed": dataset.get("clean_seed", 0),
            "data_dir": dataset.get("data_dir", default_data_dir),
            "cifar10_1_data_dir": dataset.get(
                "cifar10_1_data_dir",
                "datasets/CIFAR-10.1",
            ),
            "cifar10_1_version": dataset.get("cifar10_1_version", "v6"),
            "batch_size": dataset.get("batch_size", 128),
            "num_workers": dataset.get("num_workers", 2),
            "corruptions": corruptions,
            "severity_levels": dataset.get(
                "severity_levels",
                [1, 2, 3, 4, 5]
                if dataset_name in {"cifar10c", "cifar10_1_c"}
                else [1],
            ),
            "max_examples_per_condition": dataset.get("max_examples_per_condition"),
            "condition_seed": dataset.get("condition_seed", 0),
            "evaluation_mode": payload.get(
                "evaluation_mode",
                dataset.get("evaluation_mode", "standard"),
            ),
            "mixed_seed": payload.get("mixed_seed", dataset.get("mixed_seed", 0)),
            "mixed_max_examples": payload.get(
                "mixed_max_examples",
                dataset.get("mixed_max_examples"),
            ),
            "mixed_include_clean": payload.get(
                "mixed_include_clean",
                dataset.get("mixed_include_clean", False),
            ),
            "test_adapt": test_adapt_method,
            "digitrobust_subset": dataset.get("digitrobust_subset"),
            "digitrobust_subsets": dataset.get("digitrobust_subsets"),
        }
    )

    return EvaluationConfig(**payload)


def apply_device_override(config: dict[str, Any], device: str | None) -> None:
    if device is None:
        return

    stage = config.get("stage")
    if stage == "training":
        require_mapping(config, "training")["device"] = device
    elif stage == "testing":
        require_mapping(config, "testing")["device"] = device


def apply_protocol_override(config: dict[str, Any], protocol: str | None) -> dict[str, Any]:
    if protocol is None:
        return config

    protocol_path = Path("configs") / "protocol" / f"{protocol}.yaml"
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol config not found: {protocol_path}")
    protocol_config = load_yaml(protocol_path)
    method_overrides = protocol_config.pop("test_adapt_by_method", {})
    if not isinstance(method_overrides, dict):
        raise ValueError("'test_adapt_by_method' must be a mapping.")

    resolved = deep_merge(config, protocol_config)
    method = require_mapping(resolved, "test_adapt").get("method", "none")
    method_override = method_overrides.get(method, {})
    if not isinstance(method_override, dict):
        raise ValueError(f"Protocol override for '{method}' must be a mapping.")
    if method_override:
        resolved = deep_merge(resolved, {"test_adapt": method_override})
    return resolved


def apply_seed_override(config: dict[str, Any], seed: int | None) -> None:
    if seed is None:
        return

    config["_seed_override"] = seed
    stage = config.get("stage")
    if stage == "training":
        training = require_mapping(config, "training")
        training["seed"] = seed
        if "experiment_name" in training:
            training["experiment_name"] = f"{training['experiment_name']}_seed{seed}"
        elif config.get("name"):
            config["name"] = f"{config['name']}_seed{seed}"
        return

    if stage == "testing":
        dataset = require_mapping(config, "dataset")
        dataset["clean_seed"] = seed
        dataset["condition_seed"] = seed
        dataset["mixed_seed"] = seed
        require_mapping(config, "testing")["mixed_seed"] = seed


def run_config_file(
    path: Path,
    dry_run: bool,
    device_override: str | None = None,
    seed_override: int | None = None,
    protocol_override: str | None = None,
) -> None:
    config = load_experiment_config(path)
    config = apply_protocol_override(config, protocol_override)
    apply_device_override(config, device_override)
    apply_seed_override(config, seed_override)
    stage = config.get("stage")
    if stage == "training":
        interface = build_training_interface(config)
        if dry_run:
            print(json.dumps(interface.resolved_config(), indent=2))
            print(interface.train_command())
            return
        run_training(interface)
        return

    if stage == "testing":
        evaluation_config = build_evaluation_config(config)
        if dry_run:
            print(json.dumps(asdict(evaluation_config), indent=2))
            if evaluation_config.sweep_enabled:
                sweep_configs = iter_sweep_configs(evaluation_config)
                print(
                    json.dumps(
                        {
                            "sweep_count": len(sweep_configs),
                            "sweep_out_files": [
                                sweep_config.out_file for sweep_config in sweep_configs
                            ],
                        },
                        indent=2,
                    )
                )
            return
        run_evaluation_or_sweep(evaluation_config)
        return

    raise ValueError(f"Unknown stage '{stage}'. Expected 'training' or 'testing'.")


def extract_seed_args(argv: list[str]) -> tuple[list[str], list[int] | None]:
    if "--seeds" not in argv:
        seed_prefix = "--seeds="
        for index, value in enumerate(argv):
            if value.startswith(seed_prefix):
                raw_seeds = value[len(seed_prefix) :]
                if not raw_seeds:
                    raise SystemExit("--seeds requires at least one seed.")
                seeds = [int(seed) for seed in raw_seeds.replace(",", " ").split()]
                return [*argv[:index], *argv[index + 1 :]], seeds
        return argv, None

    index = argv.index("--seeds")
    seeds: list[int] = []
    end = index + 1
    while end < len(argv):
        try:
            seeds.append(int(argv[end]))
        except ValueError:
            break
        end += 1
    if not seeds:
        raise SystemExit("--seeds requires at least one seed.")
    return [*argv[:index], *argv[end:]], seeds


def parse_args() -> argparse.Namespace:
    argv, seeds = extract_seed_args(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Run composed YAML experiments.")
    parser.add_argument("configs", nargs="+", help="Experiment YAML files to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without running.")
    parser.add_argument(
        "--seeds",
        metavar="SEED",
        help="Run each config once per seed, e.g. --seeds 0 1 2.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated device pool for parallel config runs, e.g. "
            "'cuda:0,cuda:1'. A single config still runs on one device."
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=[
            "episodic",
            "continual_short",
            "continual_long",
            "continual_long_prob_log",
        ],
        default=None,
        help="Apply an immutable paper protocol override from configs/protocol.",
    )
    parser.add_argument("--device-override", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed-override", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.seeds = seeds
    return args


def parse_devices(raw: str | None) -> list[str]:
    if raw is None:
        return []
    devices = [device.strip() for device in raw.split(",") if device.strip()]
    if not devices:
        raise SystemExit("--devices must include at least one device, e.g. cuda:0,cuda:1.")
    return devices


def run_configs_parallel(
    jobs: list[tuple[Path, int | None]],
    devices: list[str],
    protocol: str | None,
) -> None:
    pending = deque(jobs)
    active: list[tuple[subprocess.Popen[bytes], Path, int | None, str]] = []
    script = Path(__file__).resolve()

    try:
        while pending or active:
            busy_devices = {device for _process, _path, _seed, device in active}
            for device in devices:
                if device in busy_devices or not pending:
                    continue
                path, seed = pending.popleft()
                command = [
                    sys.executable,
                    str(script),
                    "--device-override",
                    device,
                ]
                if seed is not None:
                    command.extend(["--seed-override", str(seed)])
                if protocol is not None:
                    command.extend(["--protocol", protocol])
                command.append(str(path))
                seed_text = f" seed={seed}" if seed is not None else ""
                print(f"[{device}] starting {path}{seed_text}", flush=True)
                active.append((subprocess.Popen(command), path, seed, device))

            time.sleep(1.0)

            for process, path, seed, device in list(active):
                return_code = process.poll()
                if return_code is None:
                    continue
                active.remove((process, path, seed, device))
                seed_text = f" seed={seed}" if seed is not None else ""
                print(f"[{device}] finished {path}{seed_text} exit={return_code}", flush=True)
                if return_code != 0:
                    pending.clear()
                    for running_process, running_path, running_seed, running_device in active:
                        running_seed_text = (
                            f" seed={running_seed}" if running_seed is not None else ""
                        )
                        print(
                            f"[{running_device}] terminating {running_path}{running_seed_text}",
                            flush=True,
                        )
                        running_process.terminate()
                    for running_process, _running_path, _running_seed, _running_device in active:
                        running_process.wait()
                    raise SystemExit(f"Config failed on {device}: {path}")
    except KeyboardInterrupt:
        for process, path, seed, device in active:
            seed_text = f" seed={seed}" if seed is not None else ""
            print(f"[{device}] terminating {path}{seed_text}", flush=True)
            process.terminate()
        for process, _path, _seed, _device in active:
            process.wait()
        raise


def main() -> None:
    args = parse_args()
    if args.devices is not None and args.device_override is not None:
        raise SystemExit("Use either --devices or --device-override, not both.")
    if args.seeds is not None and args.seed_override is not None:
        raise SystemExit("Use either --seeds or --seed-override, not both.")

    devices = parse_devices(args.devices)
    paths = [normalize_config_path(config_path) for config_path in args.configs]
    for config_path, path in zip(args.configs, paths, strict=True):
        if not path.exists():
            raise SystemExit(f"Config file not found: {config_path}.{config_path_hint(config_path)}")

    seeds = args.seeds if args.seeds is not None else [args.seed_override]
    jobs = [(path, seed) for path in paths for seed in seeds]

    if len(devices) > 1 and len(jobs) > 1 and not args.dry_run:
        run_configs_parallel(jobs, devices, args.protocol)
        return

    device_override = args.device_override or (devices[0] if devices else None)
    for config_path, path in zip(args.configs, paths, strict=True):
        for seed in seeds:
            try:
                run_config_file(
                    path,
                    dry_run=args.dry_run,
                    device_override=device_override,
                    seed_override=seed,
                    protocol_override=args.protocol,
                )
            except FileNotFoundError as exc:
                raise SystemExit(f"{exc}{config_path_hint(config_path)}") from None


if __name__ == "__main__":
    main()

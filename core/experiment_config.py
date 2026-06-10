from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_ROOT = Path("configs")
CONFIG_GROUPS = {"dataset", "model", "train_aug", "test_adapt"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file contains a non mapping root; expected a mapping: {path}")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_group_path(group: str, name: str) -> Path:
    return CONFIG_ROOT / group / f"{name}.yaml"


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    experiment_path = Path(path)
    raw = load_yaml(experiment_path)
    includes = raw.get("includes", {})
    if not isinstance(includes, dict):
        raise ValueError("'includes' must be a mapping of config group to config name.")

    resolved: dict[str, Any] = {}
    for group, name in includes.items():
        if group not in CONFIG_GROUPS:
            raise ValueError(f"Unknown config group '{group}'. Expected one of {sorted(CONFIG_GROUPS)}.")
        if not isinstance(name, str):
            raise ValueError(f"Config include '{group}' must be a string.")
        included_path = resolve_group_path(group, name)
        resolved = deep_merge(resolved, load_yaml(included_path))

    overrides = {key: value for key, value in raw.items() if key != "includes"}
    resolved = deep_merge(resolved, overrides)
    resolved["_config_path"] = str(experiment_path)
    return resolved

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MethodBars:
    label: str
    color: str
    result_suffix: str


CORRUPTIONS = [
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

METHODS = [
    MethodBars(
        label="AugMix+RPT",
        color="#4C78A8",
        result_suffix="augmix_rpt_resnet18_rpt_episodic_true",
    ),
    MethodBars(
        label="AugMix+TENT",
        color="#F58518",
        result_suffix="augmix_tent_resnet18_tent_episodic_true",
    ),
    MethodBars(
        label="AugMix",
        color="#54A24B",
        result_suffix="augmix_resnet18",
    ),
]

COLUMNS = [
    "svhn_to_mnist_single",
    "svhn_to_svhn_single",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot five-seed per-corruption accuracy bars for DigitRobust."
    )
    parser.add_argument("--results-dir", default="outputs/experiments/table")
    parser.add_argument("--out-dir", default="outputs/analysis_two/per_corruption")
    return parser.parse_args()


def import_plotting():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Cannot plot per-corruption accuracy because matplotlib is missing: {exc.name}"
        ) from exc
    return plt


def seed_from_path(path: Path) -> int:
    match = re.search(r"_seed(\d+)_results\.csv$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse seed from {path}")
    return int(match.group(1))


def result_paths(results_dir: Path, column: str, method: MethodBars) -> list[Path]:
    prefix = column.removesuffix("_single")
    paths = list(
        (results_dir / column).glob(
            f"{prefix}_{method.result_suffix}_seed*_results.csv"
        )
    )
    return sorted(paths, key=seed_from_path)


def load_accuracy_by_corruption(path: Path) -> dict[str, float]:
    accuracy_by_corruption: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["domain"] != "ood":
                continue
            corruption = row["corruption"]
            if corruption in CORRUPTIONS:
                accuracy_by_corruption[corruption] = 100.0 * float(row["accuracy"])

    missing = set(CORRUPTIONS) - set(accuracy_by_corruption)
    if missing:
        raise RuntimeError(f"Missing corruptions in {path}: {sorted(missing)}")
    return accuracy_by_corruption


def aggregate_method(
    results_dir: Path,
    column: str,
    method: MethodBars,
) -> tuple[list[float], list[float]]:
    paths = result_paths(results_dir, column, method)
    if not paths:
        raise FileNotFoundError(f"No result CSVs found for {column} {method.label}")

    by_seed = [load_accuracy_by_corruption(path) for path in paths]
    means = [
        statistics.mean(seed_values[corruption] for seed_values in by_seed)
        for corruption in CORRUPTIONS
    ]
    stds = [
        statistics.stdev(seed_values[corruption] for seed_values in by_seed)
        if len(by_seed) > 1
        else 0.0
        for corruption in CORRUPTIONS
    ]
    return means, stds


def plot_column(results_dir: Path, out_dir: Path, column: str) -> Path:
    plt = import_plotting()
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    positions = list(range(len(CORRUPTIONS)))
    width = 0.24

    for method_index, method in enumerate(METHODS):
        means, stds = aggregate_method(results_dir, column, method)
        offsets = [
            position + (method_index - (len(METHODS) - 1) / 2) * width
            for position in positions
        ]
        bars = ax.bar(
            offsets,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            label=method.label,
            color=method.color,
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=6)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels([name.replace("_", " ") for name in CORRUPTIONS])
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{column}_accuracy_by_noise.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    paths = [plot_column(results_dir, out_dir, column) for column in COLUMNS]
    print("Saved per-corruption accuracy plots:")
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()

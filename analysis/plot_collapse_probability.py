from __future__ import annotations

import argparse
import csv
import glob as globlib
import json
import math
from collections import Counter, deque
from pathlib import Path


DEFAULT_GLOB = "outputs/experiments/table/svhn_to_svhn_mnist_mixed/*_probabilities.csv"
DEFAULT_OUT_DIR = Path("outputs/analysis_two/collapse_probability")
METHOD_STYLES = {
    "TENT": "#e45756",
    "RPT": "#4c78a8",
    "SARPT": "#54a24b",
}


def digit_columns(row: dict[str, str], prefix: str) -> list[int]:
    digits: list[int] = []
    for key in row:
        if not key.startswith(prefix):
            continue
        suffix = key.removeprefix(prefix)
        if suffix.isdigit():
            digits.append(int(suffix))
    return sorted(digits)


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_entropy(counter: Counter[int], num_classes: int) -> float:
    total = sum(counter.values())
    if total == 0 or num_classes <= 1:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        prob = count / total
        entropy -= prob * math.log(prob)
    return entropy / math.log(num_classes)


def fractions(counter: Counter[int], num_classes: int) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {str(digit): 0.0 for digit in range(num_classes)}
    return {
        str(digit): counter.get(digit, 0) / total
        for digit in range(num_classes)
    }


def method_label(path: Path) -> str:
    name = path.name.lower()
    if "sarpt" in name:
        return "SARPT"
    if "rpt" in name:
        return "RPT"
    if "tent" in name:
        return "TENT"
    return path.stem


def mean_optional(rows: list[dict[str, str]], key: str) -> float | None:
    values = [parse_float(row.get(key, "")) for row in rows]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("Rolling window must be positive.")
    recent: deque[float] = deque()
    total = 0.0
    means: list[float] = []
    for value in values:
        recent.append(value)
        total += value
        if len(recent) > window:
            total -= recent.popleft()
        means.append(total / window if len(recent) == window else float("nan"))
    return means


def save_probability_plot(
    rows: list[dict[str, str]],
    path: Path,
    out_dir: Path,
    collapse_digit: int,
    rolling_window: int,
) -> Path:
    plotting = load_plotting()
    if plotting is None:
        raise RuntimeError("matplotlib is required to save collapse probability plots.")
    plt = plotting

    xs = [int(row["image_number"]) for row in rows]
    prob_key = f"prob_{collapse_digit}"
    ys = [float(row[prob_key]) for row in rows]
    plot_path = out_dir / f"{path.stem}_collapse_digit{collapse_digit}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(
        xs,
        rolling_mean(ys, rolling_window),
        color="#4c78a8",
        linewidth=1.2,
        label="Adapted prediction",
    )

    aug1_key = f"aug1_prob_{collapse_digit}"
    aug2_key = f"aug2_prob_{collapse_digit}"
    has_aug = any(row.get(aug1_key, "") != "" for row in rows)
    if has_aug:
        aug1 = [float(row[aug1_key]) for row in rows]
        aug2 = [float(row[aug2_key]) for row in rows]
        ax.plot(
            xs,
            rolling_mean(aug1, rolling_window),
            color="#f58518",
            linewidth=1.0,
            label="Augmented view 1",
        )
        ax.plot(
            xs,
            rolling_mean(aug2, rolling_window),
            color="#54a24b",
            linewidth=1.0,
            label="Augmented view 2",
        )

    ax.set_xlabel("Image index")
    ax.set_ylabel(f"Probability assigned to digit {collapse_digit}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=400)
    plt.close(fig)
    return plot_path


def save_combined_probability_plot(
    paths: list[Path],
    summaries: list[dict[str, object]],
    out_dir: Path,
    rolling_window: int,
    collapse_digit: int | None = None,
) -> Path | None:
    if len(paths) < 2:
        return None
    plotting = load_plotting()
    if plotting is None:
        raise RuntimeError("matplotlib is required to save collapse probability plots.")
    plt = plotting

    if collapse_digit is None:
        collapse_digits = [int(row["collapse_digit"]) for row in summaries]
        collapse_digit = Counter(collapse_digits).most_common(1)[0][0]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for path in sorted(paths, key=lambda item: method_label(item)):
        rows = load_rows(path)
        prob_key = f"prob_{collapse_digit}"
        if prob_key not in rows[0]:
            continue
        xs = [int(row["image_number"]) for row in rows]
        ys = [float(row[prob_key]) for row in rows]
        label = method_label(path)
        color = METHOD_STYLES.get(label, None)
        ax.plot(
            xs,
            rolling_mean(ys, rolling_window),
            linewidth=1.2,
            color=color,
            label=label,
        )

    ax.set_xlabel("Image index")
    ax.set_ylabel(f"Adapted probability assigned to digit {collapse_digit}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / f"combined_clean_probability_digit{collapse_digit}.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    return path


def load_plotting():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def summarize_and_plot(
    path: Path,
    out_dir: Path,
    final_window: int,
    rolling_window: int,
) -> dict[str, object]:
    rows = load_rows(path)
    if not rows:
        raise RuntimeError(f"No probability rows found in {path}.")

    digits = digit_columns(rows[0], "prob_")
    if not digits:
        raise RuntimeError(f"No prob_* columns found in {path}.")
    num_classes = max(digits) + 1
    first_rows = rows[: min(final_window, len(rows))]
    final_rows = rows[-min(final_window, len(rows)) :]
    first_counts = Counter(int(row["pred"]) for row in first_rows)
    final_counts = Counter(int(row["pred"]) for row in final_rows)
    collapse_digit, collapse_count = final_counts.most_common(1)[0]
    final_top_fraction = collapse_count / len(final_rows)
    plot_path = save_probability_plot(
        rows=rows,
        path=path,
        out_dir=out_dir,
        collapse_digit=collapse_digit,
        rolling_window=rolling_window,
    )
    first_top_digit, first_top_count = first_counts.most_common(1)[0]
    final_prob_key = f"prob_{collapse_digit}"

    return {
        "csv_path": str(path),
        "plot_path": str(plot_path),
        "num_images": len(rows),
        "window": len(final_rows),
        "collapse_digit": collapse_digit,
        "final_top_fraction": final_top_fraction,
        "final_mean_collapse_probability": mean_optional(final_rows, final_prob_key),
        "final_accuracy": mean_optional(final_rows, "correct"),
        "final_mean_confidence": mean_optional(final_rows, "confidence"),
        "final_mean_jsd": mean_optional(final_rows, "jsd"),
        "final_clean_aug1_agree": mean_optional(final_rows, "clean_aug1_agree"),
        "final_clean_aug2_agree": mean_optional(final_rows, "clean_aug2_agree"),
        "final_aug1_aug2_agree": mean_optional(final_rows, "aug1_aug2_agree"),
        "first_top_digit": first_top_digit,
        "first_top_fraction": first_top_count / len(first_rows),
        "first_pred_entropy_norm": normalized_entropy(first_counts, num_classes),
        "final_pred_entropy_norm": normalized_entropy(final_counts, num_classes),
        "first_pred_fraction_by_digit": json.dumps(
            fractions(first_counts, num_classes),
            sort_keys=True,
        ),
        "final_pred_fraction_by_digit": json.dumps(
            fractions(final_counts, num_classes),
            sort_keys=True,
        ),
    }


def resolve_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.csv_paths]
    for pattern in args.glob:
        paths.extend(Path(value) for value in sorted(globlib.glob(pattern)))
    unique_paths = sorted(dict.fromkeys(path for path in paths if path.exists()))
    if not unique_paths:
        raise FileNotFoundError("No probability CSVs found.")
    return unique_paths


def write_summary(rows: list[dict[str, object]], out_dir: Path) -> Path:
    path = out_dir / "collapse_probability_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot collapse-class probability curves from per-image probability logs."
    )
    parser.add_argument("csv_paths", nargs="*", help="Probability CSV files to plot.")
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob for probability CSV files. Can be provided more than once.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--final-window", type=int, default=10000)
    parser.add_argument("--rolling-window", type=int, default=500)
    parser.add_argument(
        "--combined-digit",
        type=int,
        default=None,
        help="Digit to use in the combined direct-prediction probability plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv_paths and not args.glob:
        args.glob = [DEFAULT_GLOB]
    out_dir = Path(args.out_dir)
    paths = resolve_paths(args)
    summaries = [
        summarize_and_plot(
            path=path,
            out_dir=out_dir,
            final_window=args.final_window,
            rolling_window=args.rolling_window,
        )
        for path in paths
    ]
    summary_path = write_summary(summaries, out_dir)
    combined_path = save_combined_probability_plot(
        paths=paths,
        summaries=summaries,
        out_dir=out_dir,
        rolling_window=args.rolling_window,
        collapse_digit=args.combined_digit,
    )
    print(f"Saved collapse summary: {summary_path.resolve()}")
    if combined_path is not None:
        print(combined_path)
    for row in summaries:
        print(row["plot_path"])


if __name__ == "__main__":
    main()

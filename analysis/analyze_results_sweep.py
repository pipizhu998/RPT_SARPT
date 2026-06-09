from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

try:
    from .analyze_results import format_metric, load_plotting
except ImportError:
    from analyze_results import format_metric, load_plotting


METRICS = ("average_ood_error", "clean_error", "robustness_gap")


def as_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def as_optional_bool(value: Any) -> bool | None:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def slugify(value: object) -> str:
    text = str(value).strip().lower().replace(".", "p")
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def load_sweep_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Sweep summary is empty: {path}")

    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                "tent_lr": float(row["tent_lr"]),
                "tent_jsd_weight": float(row["tent_jsd_weight"]),
                "tent_source_anchor_weight": as_optional_float(
                    row.get("tent_source_anchor_weight")
                ),
                "tent_episodic": as_optional_bool(row.get("tent_episodic")),
                "clean_error": as_optional_float(row.get("clean_error")),
                "average_ood_error": as_optional_float(row.get("average_ood_error")),
                "robustness_gap": as_optional_float(row.get("robustness_gap")),
                "csv_path": row.get("csv_path", ""),
                "json_path": row.get("json_path", ""),
            }
        )
    return parsed


def choose_metric(rows: list[dict[str, Any]], requested: str) -> str:
    if requested != "auto":
        return requested
    for metric in ("average_ood_error", "clean_error", "robustness_gap"):
        if any(row.get(metric) is not None for row in rows):
            return metric
    raise RuntimeError("No plottable metric found in sweep summary.")


def sorted_unique(rows: list[dict[str, Any]], key: str) -> list[float | str]:
    return sorted({row[key] for row in rows})


def sorted_numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return sorted({float(row[key]) for row in rows if row.get(key) is not None})


def format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):g}"


def matrix_for_rows(
    rows: list[dict[str, Any]],
    metric: str,
    lrs: list[float],
    jsd_weights: list[float],
) -> list[list[float]]:
    lookup = {
        (row["tent_lr"], row["tent_jsd_weight"]): row.get(metric)
        for row in rows
    }
    matrix: list[list[float]] = []
    for lr in lrs:
        matrix_row = []
        for jsd_weight in jsd_weights:
            value = lookup.get((lr, jsd_weight))
            matrix_row.append(float("nan") if value is None else float(value))
        matrix.append(matrix_row)
    return matrix


def write_matrix_csv(
    summary_path: Path,
    out_dir: Path,
    metric: str,
    rows: list[dict[str, Any]],
    lrs: list[float],
    jsd_weights: list[float],
    facet_key: str | None,
    facet_values: list[float | None],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{summary_path.parent.name}_{slugify(metric)}_matrix.csv"
    jsd_columns = [f"jsd_{value:g}" for value in jsd_weights]
    fieldnames = [
        "sweep_dir",
        "metric",
        "tent_source_anchor_weight",
        "tent_episodic",
        "tent_lr",
        *jsd_columns,
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for facet_value in facet_values:
            filtered_rows = [
                row
                for row in rows
                if facet_key is None or row.get(facet_key) == facet_value
            ]
            matrix = matrix_for_rows(filtered_rows, metric, lrs, jsd_weights)
            for lr, matrix_row in zip(lrs, matrix, strict=True):
                csv_row: dict[str, Any] = {
                    "sweep_dir": str(summary_path.parent),
                    "metric": metric,
                    "tent_source_anchor_weight": (
                        f"{facet_value:g}"
                        if facet_key == "tent_source_anchor_weight"
                        else rows[0].get("tent_source_anchor_weight")
                    ),
                    "tent_episodic": rows[0].get("tent_episodic"),
                    "tent_lr": f"{lr:g}",
                }
                for column_name, value in zip(jsd_columns, matrix_row, strict=True):
                    csv_row[column_name] = "" if math.isnan(value) else f"{value:.8g}"
                writer.writerow(csv_row)
    return output_path


def best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get(metric) is not None]
    if not valid_rows:
        raise RuntimeError(f"No values available for metric '{metric}'.")
    return min(valid_rows, key=lambda row: float(row[metric]))


def draw_heatmap(
    ax: Any,
    np: Any,
    matrix: list[list[float]],
    lrs: list[float],
    jsd_weights: list[float],
    title: str,
    metric: str,
) -> Any:
    data = np.array(matrix, dtype=float)
    image = ax.imshow(data, cmap="viridis_r", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("tent_jsd_weight")
    ax.set_ylabel("tent_lr")
    ax.set_xticks(range(len(jsd_weights)))
    ax.set_yticks(range(len(lrs)))
    ax.set_xticklabels([f"{value:g}" for value in jsd_weights])
    ax.set_yticklabels([f"{value:g}" for value in lrs])

    for row_idx, _lr in enumerate(lrs):
        for col_idx, _jsd_weight in enumerate(jsd_weights):
            value = data[row_idx, col_idx]
            if math.isnan(float(value)):
                label = ""
            else:
                label = f"{value:.4f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="black", fontsize=8)
    ax.grid(False)
    ax.text(
        0.0,
        -0.18,
        f"metric: {metric}; lower is better",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    return image


def plot_sweep_summary(
    summary_path: Path,
    out_dir: Path,
    requested_metric: str,
) -> tuple[Path, Path, dict[str, Any]]:
    plotting = load_plotting()
    if plotting is None:
        raise RuntimeError("matplotlib and numpy are required to plot sweep summaries.")
    plt, np = plotting

    rows = load_sweep_summary(summary_path)
    metric = choose_metric(rows, requested_metric)

    lrs = [float(value) for value in sorted_unique(rows, "tent_lr")]
    jsd_weights = [float(value) for value in sorted_unique(rows, "tent_jsd_weight")]
    facet_key = None
    facet_values: list[float | None] = [None]
    source_anchor_weights = sorted_numeric_values(rows, "tent_source_anchor_weight")
    if len(source_anchor_weights) > 1:
        facet_key = "tent_source_anchor_weight"
        facet_values = source_anchor_weights

    matrix_csv_path = write_matrix_csv(
        summary_path=summary_path,
        out_dir=out_dir,
        metric=metric,
        rows=rows,
        lrs=lrs,
        jsd_weights=jsd_weights,
        facet_key=facet_key,
        facet_values=facet_values,
    )

    fig_width = max(5.5, 4.2 * len(facet_values))
    fig_height = 4.8
    fig, axes = plt.subplots(1, len(facet_values), figsize=(fig_width, fig_height), squeeze=False)
    images = []
    for col_idx, facet_value in enumerate(facet_values):
        ax = axes[0][col_idx]
        filtered_rows = [
            row
            for row in rows
            if facet_key is None or row.get(facet_key) == facet_value
        ]
        matrix = matrix_for_rows(filtered_rows, metric, lrs, jsd_weights)
        title = f"{summary_path.parent.name}\nSARPT"
        if facet_key == "tent_source_anchor_weight" and facet_value is not None:
            title += f", src={facet_value:g}"
        images.append(draw_heatmap(ax, np, matrix, lrs, jsd_weights, title, metric))

    fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.8)
    best = best_row(rows, metric)
    fig.suptitle(
        "Best "
        f"{metric}={best[metric]:.4f} | "
        f"lr={best['tent_lr']:g}, jsd={best['tent_jsd_weight']:g}, "
        f"src={format_optional_float(best.get('tent_source_anchor_weight'))}",
        fontsize=10,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{summary_path.parent.name}_{slugify(metric)}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path, matrix_csv_path, {
        "sweep_dir": str(summary_path.parent),
        "metric": metric,
        "best_value": best[metric],
        "tent_lr": best["tent_lr"],
        "tent_jsd_weight": best["tent_jsd_weight"],
        "tent_source_anchor_weight": best.get("tent_source_anchor_weight"),
        "tent_episodic": best.get("tent_episodic"),
        "plot_path": str(output_path),
        "matrix_csv_path": str(matrix_csv_path),
    }


def find_sweep_summaries(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("**/*_sweep/sweep_summary.csv"))


def write_best_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "sweep_best.csv"
    fieldnames = [
        "sweep_dir",
        "metric",
        "best_value",
        "tent_lr",
        "tent_jsd_weight",
        "tent_source_anchor_weight",
        "tent_episodic",
        "plot_path",
        "matrix_csv_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def analyze_sweeps(results_dir: Path, out_dir: Path, metric: str) -> None:
    summaries = find_sweep_summaries(results_dir)
    if not summaries:
        print(f"No sweep_summary.csv files found under {results_dir}")
        return

    best_rows = []
    for summary_path in summaries:
        plot_path, matrix_csv_path, best = plot_sweep_summary(summary_path, out_dir, metric)
        best_rows.append(best)
        print(f"Saved sweep plot: {plot_path}")
        print(f"Saved sweep matrix CSV: {matrix_csv_path}")
    best_csv = write_best_csv(best_rows, out_dir)
    print(f"Saved sweep best summary: {best_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SARPT sweep heatmaps.")
    parser.add_argument("--results-dir", default="outputs/experiments/table")
    parser.add_argument("--out-dir", default="outputs/analysis_sweep")
    parser.add_argument(
        "--metric",
        choices=("auto", *METRICS),
        default="auto",
        help="Metric to plot. auto uses average_ood_error when available, otherwise clean_error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_sweeps(
        results_dir=Path(args.results_dir),
        out_dir=Path(args.out_dir),
        metric=args.metric,
    )


if __name__ == "__main__":
    main()

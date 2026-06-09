from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MethodCurve:
    slug: str
    label: str
    color: str
    stream_suffix: str


@dataclass(frozen=True)
class ColumnCurve:
    slug: str
    label: str
    y_min: float
    y_max: float


METHODS = [
    MethodCurve(
        slug="baseline",
        label="Baseline",
        color="#4C78A8",
        stream_suffix="baseline_resnet18_results_stream_accuracy.csv",
    ),
    MethodCurve(
        slug="augmix_tent",
        label="AugMix+c-TENT",
        color="#F58518",
        stream_suffix="augmix_tent_resnet18_tent_episodic_false_results_stream_accuracy.csv",
    ),
    MethodCurve(
        slug="augmix_eata",
        label="AugMix+c-EATA",
        color="#54A24B",
        stream_suffix="augmix_eata_resnet18_eata_episodic_false_results_stream_accuracy.csv",
    ),
    MethodCurve(
        slug="augmix_cotta",
        label="AugMix+CoTTA",
        color="#E45756",
        stream_suffix="augmix_cotta_resnet18_cotta_episodic_false_results_stream_accuracy.csv",
    ),
    MethodCurve(
        slug="augmix_rpt",
        label="AugMix+c-RPT",
        color="#72B7B2",
        stream_suffix=(
            "augmix_rpt_resnet18_rpt_episodic_false_"
            "results_stream_accuracy.csv"
        ),
    ),
    MethodCurve(
        slug="augmix_sarpt",
        label="AugMix+c-SARPT",
        color="#B279A2",
        stream_suffix=(
            "augmix_sarpt_resnet18_sarpt_episodic_false_"
            "results_stream_accuracy.csv"
        ),
    ),
]

COLUMNS = [
    ColumnCurve(
        slug="svhn_to_svhn_mnist_mixed",
        label="SVHN->SVHN&MNIST Mixed long stream",
        y_min=0.60,
        y_max=0.82,
    ),
    ColumnCurve(
        slug="mnist_to_svhn_mnist_mixed",
        label="MNIST->SVHN&MNIST Mixed long stream",
        y_min=0.20,
        y_max=0.50,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot long-stream accuracy curves for the combined DigitRobust streams."
    )
    parser.add_argument("--results-dir", default="outputs/experiments/table")
    parser.add_argument("--out-dir", default="outputs/analysis_two/long_stream_accuracy")
    parser.add_argument(
        "--metric",
        choices=["cumulative_accuracy", "batch_accuracy"],
        default="cumulative_accuracy",
    )
    parser.add_argument(
        "--x-axis",
        choices=["step", "cumulative_examples"],
        default="cumulative_examples",
    )
    return parser.parse_args()


def stream_csv_path(results_dir: Path, column: ColumnCurve, method: MethodCurve) -> Path:
    return results_dir / column.slug / f"{column.slug}_{method.stream_suffix}"


def stream_csv_paths(
    results_dir: Path,
    column: ColumnCurve,
    method: MethodCurve,
) -> list[Path]:
    unseeded_path = stream_csv_path(results_dir, column, method)
    base_name = unseeded_path.name.removesuffix("_results_stream_accuracy.csv")
    seeded_paths = sorted(
        path
        for path in unseeded_path.parent.glob(
            f"{base_name}_seed*_results_stream_accuracy.csv"
        )
        if path.name.removeprefix(f"{base_name}_seed")
        .removesuffix("_results_stream_accuracy.csv")
        .isdigit()
    )
    return seeded_paths or ([unseeded_path] if unseeded_path.exists() else [])


def load_curve(path: Path, x_axis: str, metric: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            xs.append(float(row[x_axis]))
            ys.append(float(row[metric]))
    return xs, ys


def load_average_curve(
    paths: list[Path],
    x_axis: str,
    metric: str,
) -> tuple[list[float], list[float]]:
    curves = [load_curve(path, x_axis=x_axis, metric=metric) for path in paths]
    xs = curves[0][0]
    if any(curve_xs != xs for curve_xs, _curve_ys in curves[1:]):
        raise RuntimeError(
            f"Cannot average long-stream curves with different {x_axis}: {paths}"
        )
    ys = [
        statistics.mean(values)
        for values in zip(*(curve_ys for _curve_xs, curve_ys in curves), strict=True)
    ]
    return xs, ys


def collect_curves(
    results_dir: Path,
    x_axis: str,
    metric: str,
) -> tuple[dict[str, dict[str, tuple[list[float], list[float], list[Path]]]], list[str]]:
    curves: dict[str, dict[str, tuple[list[float], list[float], list[Path]]]] = {}
    missing: list[str] = []
    for column in COLUMNS:
        column_curves: dict[str, tuple[list[float], list[float], list[Path]]] = {}
        for method in METHODS:
            paths = stream_csv_paths(results_dir, column, method)
            if not paths:
                missing.append(str(stream_csv_path(results_dir, column, method)))
                continue
            xs, ys = load_average_curve(paths, x_axis=x_axis, metric=metric)
            if xs and ys:
                column_curves[method.slug] = (xs, ys, paths)
        curves[column.slug] = column_curves
    return curves, missing


def write_merged_csv(
    curves: dict[str, dict[str, tuple[list[float], list[float], list[Path]]]],
    out_dir: Path,
    x_axis: str,
    metric: str,
) -> Path:
    path = out_dir / "long_stream_accuracy_curves.csv"
    fieldnames = [
        "column_slug",
        "column",
        "method_slug",
        "method",
        x_axis,
        metric,
        "num_runs",
        "source_csv",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for column in COLUMNS:
            for method in METHODS:
                curve = curves.get(column.slug, {}).get(method.slug)
                if curve is None:
                    continue
                xs, ys, source_paths = curve
                for x_value, y_value in zip(xs, ys, strict=True):
                    writer.writerow(
                        {
                            "column_slug": column.slug,
                            "column": column.label,
                            "method_slug": method.slug,
                            "method": method.label,
                            x_axis: x_value,
                            metric: y_value,
                            "num_runs": len(source_paths),
                            "source_csv": ";".join(str(path) for path in source_paths),
                        }
                    )
    return path


def import_plotting():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Cannot plot long-stream accuracy because matplotlib is missing: {exc.name}"
        ) from exc
    return plt


def plot_column(ax, column: ColumnCurve, curves, metric: str, x_axis: str) -> None:
    any_curve = False
    for method in METHODS:
        curve = curves.get(column.slug, {}).get(method.slug)
        if curve is None:
            continue
        xs, ys, _source_path = curve
        ax.plot(xs, ys, label=method.label, color=method.color, linewidth=2.0)
        any_curve = True

    ax.set_xlabel("Step" if x_axis == "step" else "Number of images")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(column.y_min, column.y_max)
    ax.grid(True, alpha=0.25)
    if not any_curve:
        ax.text(
            0.5,
            0.5,
            "No stream accuracy CSVs found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


def save_individual_plots(curves, out_dir: Path, metric: str, x_axis: str) -> list[Path]:
    plt = import_plotting()
    paths: list[Path] = []
    for column in COLUMNS:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        plot_column(ax, column, curves=curves, metric=metric, x_axis=x_axis)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{column.slug}_{metric}_curves.png"
        fig.savefig(path, dpi=400)
        plt.close(fig)
        paths.append(path)
    return paths


def save_combined_plot(curves, out_dir: Path, metric: str, x_axis: str) -> Path:
    plt = import_plotting()
    fig, axes = plt.subplots(1, len(COLUMNS), figsize=(15.5, 4.8), sharey=False)
    if len(COLUMNS) == 1:
        axes = [axes]
    for ax, column in zip(axes, COLUMNS, strict=True):
        plot_column(ax, column, curves=curves, metric=metric, x_axis=x_axis)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(handles), 6),
            frameon=False,
            fontsize=9,
        )
        fig.subplots_adjust(bottom=0.22)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = out_dir / f"long_stream_{metric}_curves.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves, missing = collect_curves(
        results_dir=results_dir,
        x_axis=args.x_axis,
        metric=args.metric,
    )
    merged_csv = write_merged_csv(
        curves=curves,
        out_dir=out_dir,
        x_axis=args.x_axis,
        metric=args.metric,
    )
    individual_plots = save_individual_plots(
        curves=curves,
        out_dir=out_dir,
        metric=args.metric,
        x_axis=args.x_axis,
    )
    combined_plot = save_combined_plot(
        curves=curves,
        out_dir=out_dir,
        metric=args.metric,
        x_axis=args.x_axis,
    )

    print("Saved long-stream accuracy outputs:")
    print(merged_csv.resolve())
    for path in individual_plots:
        print(path.resolve())
    print(combined_plot.resolve())
    if missing:
        print("\nMissing stream CSVs (rerun those experiments to include them):")
        for path in missing:
            print(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


METRIC_COLUMNS = [
    "clean_error",
    "average_ood_error",
    "worst_domain_error",
    "robustness_gap",
]

_PLOTTING_UNSET = object()
_PLOTTING_CACHE: object | tuple[Any, Any] | None = _PLOTTING_UNSET


def ood_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in result["rows"] if row.get("domain") == "ood"]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def as_float(value: Any) -> float:
    return float(value)


def as_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def load_plotting() -> tuple[Any, Any] | None:
    global _PLOTTING_CACHE
    if _PLOTTING_CACHE is _PLOTTING_UNSET:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ModuleNotFoundError as exc:
            print(f"Skipping plot output because optional dependency is missing: {exc.name}")
            _PLOTTING_CACHE = None
        else:
            _PLOTTING_CACHE = (plt, np)
    if _PLOTTING_CACHE is None:
        return None
    return _PLOTTING_CACHE


def experiment_name(path: Path) -> str:
    name = path.stem
    return name[:-8] if name.endswith("_results") else name


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    rows = result.get("rows", [])
    clean_rows = [row for row in rows if row.get("domain") == "clean"]
    ood_rows = [row for row in rows if row.get("domain") == "ood"]

    clean_error = result.get("clean_error")
    if clean_error is None and clean_rows:
        clean_error = clean_rows[0].get("error_rate")

    average_ood_error = result.get("average_ood_error")
    if average_ood_error is None and ood_rows:
        average_ood_error = sum(as_float(row["error_rate"]) for row in ood_rows) / len(ood_rows)

    worst_domain_error = result.get("worst_domain_error")
    if worst_domain_error is None and ood_rows:
        worst_domain_error = max(as_float(row["error_rate"]) for row in ood_rows)

    robustness_gap = result.get("robustness_gap")
    if robustness_gap is None and clean_error is not None and average_ood_error is not None:
        robustness_gap = as_float(average_ood_error) - as_float(clean_error)

    return {
        "experiment": experiment_name(path),
        "model": result.get("model", ""),
        "test_adapt": result.get("test_adapt", ""),
        "evaluation_mode": result.get("evaluation_mode", "standard"),
        "clean_error": as_optional_float(clean_error),
        "average_ood_error": as_optional_float(average_ood_error),
        "worst_domain_error": as_optional_float(worst_domain_error),
        "robustness_gap": as_optional_float(robustness_gap),
        "worst_domain": result.get("worst_domain", {}),
        "rows": rows,
    }


def load_all_results(results_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(results_dir.glob("*_results.json"))
    if not paths:
        raise FileNotFoundError(f"No *_results.json files found in {results_dir}")
    return [load_result(path) for path in paths]


def write_summary_csv(results: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "summary_metrics.csv"
    fieldnames = [
        "experiment",
        "model",
        "test_adapt",
        "evaluation_mode",
        *METRIC_COLUMNS,
        "worst_corruption",
        "worst_severity",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            worst = result.get("worst_domain") or {}
            writer.writerow(
                {
                    "experiment": result["experiment"],
                    "model": result["model"],
                    "test_adapt": result["test_adapt"],
                    "evaluation_mode": result["evaluation_mode"],
                    "clean_error": result["clean_error"],
                    "average_ood_error": result["average_ood_error"],
                    "worst_domain_error": result["worst_domain_error"],
                    "robustness_gap": result["robustness_gap"],
                    "worst_corruption": worst.get("corruption", ""),
                    "worst_severity": worst.get("severity", ""),
                }
            )
    return path


def all_ood_severities(results: list[dict[str, Any]]) -> list[int]:
    severities: set[int] = set()
    for result in results:
        severities.update(int(row["severity"]) for row in ood_rows(result))
    return sorted(severities)


def corruption_error_summaries(result: dict[str, Any]) -> list[dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in ood_rows(result):
        grouped_rows.setdefault(str(row["corruption"]), []).append(row)

    summaries = []
    for corruption, rows in sorted(grouped_rows.items()):
        severity_errors: dict[int, list[float]] = {}
        for row in rows:
            severity_errors.setdefault(int(row["severity"]), []).append(as_float(row["error_rate"]))

        mean_severity_errors = {
            severity: sum(errors) / len(errors)
            for severity, errors in severity_errors.items()
        }
        all_errors = [error for errors in severity_errors.values() for error in errors]
        worst_severity, worst_error = max(mean_severity_errors.items(), key=lambda item: item[1])
        summaries.append(
            {
                "corruption": corruption,
                "average_error": sum(all_errors) / len(all_errors),
                "worst_error": worst_error,
                "worst_severity": worst_severity,
                "severity_errors": mean_severity_errors,
            }
        )
    return summaries


def write_corruption_csv(results: list[dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "corruption_metrics.csv"
    severities = all_ood_severities(results)
    severity_columns = [f"severity_{severity}_error" for severity in severities]
    fieldnames = [
        "experiment",
        "model",
        "test_adapt",
        "evaluation_mode",
        "corruption",
        "average_error",
        "worst_error",
        "worst_severity",
        *severity_columns,
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for summary in corruption_error_summaries(result):
                row = {
                    "experiment": result["experiment"],
                    "model": result["model"],
                    "test_adapt": result["test_adapt"],
                    "evaluation_mode": result["evaluation_mode"],
                    "corruption": summary["corruption"],
                    "average_error": summary["average_error"],
                    "worst_error": summary["worst_error"],
                    "worst_severity": summary["worst_severity"],
                }
                severity_errors = summary["severity_errors"]
                row.update(
                    {
                        f"severity_{severity}_error": severity_errors.get(severity, "")
                        for severity in severities
                    }
                )
                writer.writerow(row)
    return path


def save_summary_table(results: list[dict[str, Any]], out_dir: Path) -> Path | None:
    plotting = load_plotting()
    if plotting is None:
        return None
    plt, _np = plotting

    headers = [
        "Experiment",
        "Clean Err",
        "Avg OOD Err",
        "Worst Err",
        "Gap",
        "Worst Domain",
    ]
    table_rows = []
    for result in results:
        worst = result.get("worst_domain") or {}
        worst_text = f"{worst.get('corruption', '')}@{worst.get('severity', '')}"
        table_rows.append(
            [
                result["experiment"],
                format_metric(result["clean_error"]),
                format_metric(result["average_ood_error"]),
                format_metric(result["worst_domain_error"]),
                format_metric(result["robustness_gap"]),
                worst_text,
            ]
        )

    fig_width = max(13, len(headers) * 2.2)
    fig_height = max(3.0, 0.5 * len(table_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.34, 0.13, 0.13, 0.13, 0.09, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef7")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f7f7f7")
    path = out_dir / "summary_table.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_summary_bars(results: list[dict[str, Any]], out_dir: Path) -> Path | None:
    plotting = load_plotting()
    if plotting is None:
        return None
    plt, np = plotting

    labels = [result["experiment"] for result in results]
    x = np.arange(len(labels))
    width = 0.2

    fig_width = max(10, len(labels) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    for index, metric in enumerate(METRIC_COLUMNS):
        values = [
            result[metric] if result[metric] is not None else np.nan
            for result in results
        ]
        ax.bar(x + (index - 1.5) * width, values, width, label=metric)

    ax.set_ylabel("Error rate")
    ax.set_title("Clean / OOD Robustness Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    path = out_dir / "summary_bars.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_ood_heatmap(result: dict[str, Any], out_dir: Path) -> Path | None:
    plotting = load_plotting()
    if plotting is None:
        return None
    plt, np = plotting

    rows = ood_rows(result)
    if not rows:
        return None

    corruptions = sorted({str(row["corruption"]) for row in rows})
    severities = sorted({int(row["severity"]) for row in rows})
    matrix = np.full((len(corruptions), len(severities)), np.nan)

    corruption_index = {name: idx for idx, name in enumerate(corruptions)}
    severity_index = {level: idx for idx, level in enumerate(severities)}
    for row in rows:
        matrix[
            corruption_index[str(row["corruption"])],
            severity_index[int(row["severity"])],
        ] = as_float(row["error_rate"])

    fig_width = max(6, len(severities) * 1.4)
    fig_height = max(4, len(corruptions) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)

    ax.set_title(f"OOD Error Heatmap: {result['experiment']}")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Corruption")
    ax.set_xticks(np.arange(len(severities)))
    ax.set_xticklabels(severities)
    ax.set_yticks(np.arange(len(corruptions)))
    ax.set_yticklabels(corruptions)

    for row_idx in range(len(corruptions)):
        for col_idx in range(len(severities)):
            value = matrix[row_idx, col_idx]
            if not np.isnan(value):
                color = "white" if value > 0.5 else "black"
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, label="Error rate")
    path = out_dir / f"heatmap_{safe_name(result['experiment'])}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CIFAR-10-C experiment metrics.")
    parser.add_argument("--results-dir", default="outputs/experiments/testing")
    parser.add_argument("--out-dir", default="outputs/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results(results_dir)
    outputs: list[Path] = [
        write_summary_csv(results, out_dir),
        write_corruption_csv(results, out_dir),
    ]
    for path in [save_summary_table(results, out_dir), save_summary_bars(results, out_dir)]:
        if path is not None:
            outputs.append(path)
    for result in results:
        heatmap_path = save_ood_heatmap(result, out_dir)
        if heatmap_path is not None:
            outputs.append(heatmap_path)

    print("Saved analysis outputs:")
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()

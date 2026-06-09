from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .analyze_results import format_metric, load_plotting
except ImportError:
    from analyze_results import format_metric, load_plotting
from data_utils.cifar10_1_c_data import CIFAR10_1_C_CORRUPTIONS
from data_utils.cifar10c_data import CIFAR10C_CORRUPTIONS


ROW_ORDER = [
    ("baseline", "Baseline"),
    ("adabn", "AdaBN"),
    ("tent", "TENT"),
    ("eata", "EATA"),
    ("cotta", "CoTTA"),
    ("rpt", "RPT"),
    ("sarpt", "SARPT"),
    ("augmix", "AUGMIX"),
    ("augmix_adabn", "AUGMIX+AdaBN"),
    ("augmix_tent", "AUGMIX+TENT"),
    ("augmix_eata", "AUGMIX+EATA"),
    ("augmix_cotta", "AUGMIX+CoTTA"),
    ("augmix_rpt", "AUGMIX+RPT"),
    ("augmix_sarpt", "AUGMIX+SARPT"),
    ("continual_tent", "continual_tent"),
    ("continual_eata", "continual_eata"),
    ("continual_cotta", "continual_cotta"),
    ("continual_rpt", "continual_rpt"),
    ("continual_sarpt", "continual_sarpt"),
    ("augmix_continual_tent", "AUGMIX+continual_tent"),
    ("augmix_continual_eata", "AUGMIX+continual_eata"),
    ("augmix_continual_cotta", "AUGMIX+continual_cotta"),
    ("augmix_continual_rpt", "AUGMIX+continual_rpt"),
    (
        "augmix_continual_sarpt",
        "AUGMIX+continual_sarpt",
    ),
]

COLUMN_ORDER = [
    ("cifar10_to_cifar10_clean", "CIFAR10->CIFAR10 clean"),
    ("cifar10_to_cifar10_1_clean", "CIFAR10->CIFAR10.1 clean"),
    ("cifar10_to_cifar10_1_c_single", "CIFAR10->CIFAR10.1-C single corruption"),
    ("cifar10_to_cifar10_1_c_mixed", "CIFAR10->CIFAR10.1-C mixed corruption"),
    ("cifar10_to_cifar10_single", "CIFAR10->CIFAR10-C single corruption"),
    ("cifar10_to_cifar10_mixed", "CIFAR10->CIFAR10-C mixed corruption"),
    ("svhn_to_svhn_clean", "SVHN->SVHN clean"),
    ("svhn_to_svhn_single", "SVHN->SVHN single corruption"),
    ("svhn_to_svhn_mixed", "SVHN->SVHN mixed corruption"),
    ("svhn_to_svhn_mnist_mixed", "SVHN->SVHN&MNIST mixed corruption"),
    ("svhn_to_mnist_clean", "SVHN->MNIST clean"),
    ("svhn_to_mnist_single", "SVHN->MNIST single corruption"),
    ("svhn_to_mnist_mixed", "SVHN->MNIST mixed corruption"),
    ("mnist_to_mnist_clean", "MNIST->MNIST clean"),
    ("mnist_to_mnist_single", "MNIST->MNIST single corruption"),
    ("mnist_to_mnist_mixed", "MNIST->MNIST mixed corruption"),
    ("mnist_to_svhn_clean", "MNIST->SVHN clean"),
    ("mnist_to_svhn_single", "MNIST->SVHN single corruption"),
    ("mnist_to_svhn_mixed", "MNIST->SVHN mixed corruption"),
    ("mnist_to_svhn_mnist_mixed", "MNIST->SVHN&MNIST mixed corruption"),
]

ROW_SLUGS = {slug for slug, _label in ROW_ORDER}
MIXED_COLUMNS = {slug for slug, _label in COLUMN_ORDER if slug.endswith("_mixed")}
MIXED_PREFIXES = tuple(
    f"{slug}_" for slug, _label in COLUMN_ORDER if slug.endswith("_mixed")
)
SOURCE_TARGET_PREFIXES = (
    "cifar10_to_cifar10_1_c_",
    "cifar10_to_cifar10_1_",
    "cifar10_to_cifar10_",
    "svhn_to_svhn_mnist_",
    "svhn_to_svhn_",
    "svhn_to_mnist_",
    "mnist_to_mnist_",
    "mnist_to_svhn_mnist_",
    "mnist_to_svhn_",
)
ROW_ALIASES = {
    "rpt": "rpt",
    "augmix_rpt": "augmix_rpt",
}
CONTINUAL_ROW_ALIASES = {
    "tent": "continual_tent",
    "eata": "continual_eata",
    "cotta": "continual_cotta",
    "rpt": "continual_rpt",
    "sarpt": "continual_sarpt",
    "augmix_tent": "augmix_continual_tent",
    "augmix_eata": "augmix_continual_eata",
    "augmix_cotta": "augmix_continual_cotta",
    "augmix_rpt": "augmix_continual_rpt",
    "augmix_sarpt": (
        "augmix_continual_sarpt"
    ),
}
EPISODIC_NAME_SUFFIXES = (
    "_tent_episodic_true",
    "_tent_episodic_false",
    "_rpt_episodic_true",
    "_rpt_episodic_false",
    "_sarpt_episodic_true",
    "_sarpt_episodic_false",
    "_eata_episodic_true",
    "_eata_episodic_false",
    "_cotta_episodic_true",
    "_cotta_episodic_false",
)
CIFAR10C_FULL_SEVERITIES = [1, 2, 3, 4, 5]
CIFAR10C_MAX_EXAMPLES_PER_CONDITION = 1000
CIFAR10C_COLUMNS = {
    "cifar10_to_cifar10_clean",
    "cifar10_to_cifar10_single",
    "cifar10_to_cifar10_mixed",
}
CIFAR10_1_C_COLUMNS = {
    "cifar10_to_cifar10_1_c_single",
    "cifar10_to_cifar10_1_c_mixed",
}


def experiment_name(path: Path) -> str:
    name = path.stem
    return name[:-8] if name.endswith("_results") else name


def strip_seed_suffix(name: str) -> tuple[str, int | None]:
    if "_seed" in name:
        prefix, maybe_seed = name.rsplit("_seed", 1)
        if maybe_seed.isdigit():
            return prefix, int(maybe_seed)
    return name, None


def row_slug_from_experiment(name: str) -> str | None:
    name, _seed = strip_seed_suffix(name)
    for prefix in MIXED_PREFIXES:
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    for prefix in SOURCE_TARGET_PREFIXES:
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    for suffix in EPISODIC_NAME_SUFFIXES:
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            break
    if name.endswith("_resnet18"):
        name = name.removesuffix("_resnet18")
    name = ROW_ALIASES.get(name, name)
    return name if name in ROW_SLUGS else None


def is_continual_result(row_slug: str, result: dict[str, Any]) -> bool:
    if "eata" in row_slug:
        return result.get("eata_episodic") is False
    if "cotta" in row_slug:
        return result.get("cotta_episodic") is False
    if "tent" in row_slug or "rpt" in row_slug:
        return result.get("tent_episodic") is False
    return False


def row_slug_for_result(row_slug: str, result: dict[str, Any]) -> str:
    if is_continual_result(row_slug, result):
        return CONTINUAL_ROW_ALIASES.get(row_slug, row_slug)
    return row_slug


def adaptation_episodic_value(result: dict[str, Any]) -> Any:
    if result.get("test_adapt") == "eata":
        return result.get("eata_episodic")
    if result.get("test_adapt") == "cotta":
        return result.get("cotta_episodic")
    return result.get("tent_episodic")


def as_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_from_result(result: dict[str, Any], column_slug: str) -> float | None:
    if column_slug.endswith("_clean"):
        value = result.get("clean_error")
        if value is None:
            for row in result.get("rows", []):
                if row.get("domain") == "clean":
                    value = row.get("error_rate")
                    break
        return as_optional_float(value)

    value = result.get("average_ood_error")
    if value is not None:
        return as_optional_float(value)

    ood_errors = [
        float(row["error_rate"])
        for row in result.get("rows", [])
        if row.get("domain") == "ood"
    ]
    if not ood_errors:
        return None
    return sum(ood_errors) / len(ood_errors)


def is_stale_mixed_result(result: dict[str, Any], column_slug: str) -> bool:
    return False


def is_stale_cifar10c_result(result: dict[str, Any], column_slug: str) -> bool:
    if column_slug not in CIFAR10C_COLUMNS:
        return False
    if column_slug.endswith("_mixed"):
        return False
    return (
        result.get("ood_dataset") != "cifar10c"
        or list(result.get("corruptions", [])) != list(CIFAR10C_CORRUPTIONS)
        or list(result.get("severity_levels", [])) != CIFAR10C_FULL_SEVERITIES
        or result.get("max_examples_per_condition")
        != CIFAR10C_MAX_EXAMPLES_PER_CONDITION
    )


def is_stale_cifar10_1_result(result: dict[str, Any], column_slug: str) -> bool:
    return (
        column_slug == "cifar10_to_cifar10_1_clean"
        and result.get("clean_shuffle") is not True
    )


def is_stale_cifar10_1_c_result(result: dict[str, Any], column_slug: str) -> bool:
    if column_slug not in CIFAR10_1_C_COLUMNS:
        return False
    if column_slug.endswith("_mixed"):
        return False
    return (
        result.get("ood_dataset") != "cifar10_1_c"
        or list(result.get("corruptions", [])) != list(CIFAR10_1_C_CORRUPTIONS)
        or list(result.get("severity_levels", [])) != CIFAR10C_FULL_SEVERITIES
    )


def is_stale_single_result(result: dict[str, Any], column_slug: str) -> bool:
    return (
        column_slug.endswith("_single")
        and result.get("evaluation_mode") == "standard"
        and result.get("standard_include_clean") is not False
    )


def is_rpt_row(row_slug: str) -> bool:
    return "rpt" in row_slug


def allowed_rpt_variants(row_slug: str) -> set[str]:
    if "sarpt" in row_slug:
        return {"sarpt"}
    return {"rpt"}


def is_stale_rpt_result(result: dict[str, Any], row_slug: str) -> bool:
    return (
        is_rpt_row(row_slug)
        and result.get("test_adapt") in {"rpt", "sarpt"}
        and (
            result.get("rpt_returns_adapted_logits") is not True
            or result.get("rpt_variant")
            not in allowed_rpt_variants(row_slug)
        )
    )


def load_table(results_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    table: dict[str, dict[str, dict[str, Any]]] = {
        row_slug: {} for row_slug, _row_label in ROW_ORDER
    }
    json_paths = sorted(results_dir.glob("*/*_results.json"))
    if not json_paths:
        raise FileNotFoundError(f"No table result JSON files found under {results_dir}")

    for path in json_paths:
        column_slug = path.parent.name
        if column_slug not in {slug for slug, _label in COLUMN_ORDER}:
            continue
        result_name = experiment_name(path)
        _, output_seed = strip_seed_suffix(result_name)
        row_slug = row_slug_from_experiment(result_name)
        if row_slug is None:
            print(f"Skipping unrecognized result name: {path}")
            continue

        result = load_json(path)
        row_slug = row_slug_for_result(row_slug, result)
        if is_stale_mixed_result(result, column_slug):
            print(
                "Skipping stale mixed result without clean samples "
                f"(rerun this config): {path}"
            )
            continue
        if is_stale_cifar10c_result(result, column_slug):
            print(
                "Skipping stale CIFAR10-C result with old corruption/severity sampling "
                f"(rerun this config): {path}"
            )
            continue
        if is_stale_cifar10_1_result(result, column_slug):
            print(
                "Skipping stale CIFAR10.1 result without deterministic clean shuffle "
                f"(rerun this config): {path}"
            )
            continue
        if is_stale_cifar10_1_c_result(result, column_slug):
            print(
                "Skipping stale CIFAR10.1-C result with old synthetic setup "
                f"(rerun this config): {path}"
            )
            continue
        if is_stale_single_result(result, column_slug):
            print(
                "Skipping stale single-corruption result that still includes clean "
                f"(rerun this config): {path}"
            )
            continue
        if is_stale_rpt_result(result, row_slug):
            print(
                "Skipping stale pre-RPT Revised TENT result "
                f"(rerun this config): {path}"
            )
            continue
        value = metric_from_result(result, column_slug)
        if value is None:
            continue
        cell = table[row_slug].setdefault(column_slug, {"runs": []})
        cell["runs"].append(
            {
                "error": value,
                "accuracy": 1.0 - value,
                "tent_episodic": adaptation_episodic_value(result),
                "source_json": str(path),
                "seed": output_seed
                if output_seed is not None
                else (
                    result.get("mixed_seed")
                    if result.get("evaluation_mode") == "mixed"
                    else result.get("condition_seed", result.get("clean_seed"))
                ),
                "seeded_output": output_seed is not None,
            }
        )

    for row in table.values():
        for cell in row.values():
            runs = cell.get("runs", [])
            if not runs:
                continue
            seeded_runs = [run for run in runs if run.get("seeded_output")]
            if seeded_runs:
                runs = seeded_runs
            errors = [run["error"] for run in runs]
            accuracies = [run["accuracy"] for run in runs]
            cell["error"] = statistics.mean(errors)
            cell["accuracy"] = statistics.mean(accuracies)
            cell["error_std"] = statistics.stdev(errors) if len(errors) > 1 else 0.0
            cell["accuracy_std"] = (
                statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
            )
            cell["num_seeds"] = len(runs)
            cell["seeds"] = ";".join(str(run.get("seed", "")) for run in runs)
            cell["tent_episodic"] = runs[0].get("tent_episodic", "")
            cell["source_json"] = ";".join(run["source_json"] for run in runs)
    return table


def write_matrix_csv(
    table: dict[str, dict[str, dict[str, Any]]],
    out_dir: Path,
    metric: str,
) -> Path:
    path = out_dir / f"big_table_{metric}.csv"
    fieldnames = ["method", *[label for _slug, label in COLUMN_ORDER]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_slug, row_label in ROW_ORDER:
            row = {"method": row_label}
            for column_slug, column_label in COLUMN_ORDER:
                cell = table[row_slug].get(column_slug, {})
                row[column_label] = cell.get(metric, "")
            writer.writerow(row)
    return path


def format_mean_std(cell: dict[str, Any], metric: str) -> str:
    value = cell.get(metric)
    if value in {None, ""}:
        return ""
    std = cell.get(f"{metric}_std") or 0.0
    return f"{float(value) * 100:.2f} +/- {float(std) * 100:.2f}"


def write_mean_std_csv(
    table: dict[str, dict[str, dict[str, Any]]],
    out_dir: Path,
    metric: str,
) -> Path:
    path = out_dir / f"big_table_{metric}_mean_std.csv"
    fieldnames = ["method", *[label for _slug, label in COLUMN_ORDER]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_slug, row_label in ROW_ORDER:
            row = {"method": row_label}
            for column_slug, column_label in COLUMN_ORDER:
                cell = table[row_slug].get(column_slug, {})
                row[column_label] = format_mean_std(cell, metric)
            writer.writerow(row)
    return path


def write_long_csv(table: dict[str, dict[str, dict[str, Any]]], out_dir: Path) -> Path:
    path = out_dir / "big_table_long.csv"
    fieldnames = [
        "method_slug",
        "method",
        "column_slug",
        "column",
        "error",
        "accuracy",
        "error_std",
        "accuracy_std",
        "num_seeds",
        "seeds",
        "tent_episodic",
        "source_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_slug, row_label in ROW_ORDER:
            for column_slug, column_label in COLUMN_ORDER:
                cell = table[row_slug].get(column_slug, {})
                writer.writerow(
                    {
                        "method_slug": row_slug,
                        "method": row_label,
                        "column_slug": column_slug,
                        "column": column_label,
                        "error": cell.get("error", ""),
                        "accuracy": cell.get("accuracy", ""),
                        "error_std": cell.get("error_std", ""),
                        "accuracy_std": cell.get("accuracy_std", ""),
                        "num_seeds": cell.get("num_seeds", ""),
                        "seeds": cell.get("seeds", ""),
                        "tent_episodic": cell.get("tent_episodic", ""),
                        "source_json": cell.get("source_json", ""),
                    }
                )
    return path


def write_markdown_table(
    table: dict[str, dict[str, dict[str, Any]]],
    out_dir: Path,
    metric: str,
) -> Path:
    path = out_dir / f"big_table_{metric}.md"
    headers = ["Method", *[label for _slug, label in COLUMN_ORDER]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row_slug, row_label in ROW_ORDER:
        values = [row_label]
        for column_slug, _column_label in COLUMN_ORDER:
            cell = table[row_slug].get(column_slug, {})
            values.append(format_metric(cell.get(metric)))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_mean_std_markdown_table(
    table: dict[str, dict[str, dict[str, Any]]],
    out_dir: Path,
    metric: str,
) -> Path:
    path = out_dir / f"big_table_{metric}_mean_std.md"
    headers = ["Method", *[label for _slug, label in COLUMN_ORDER]]
    lines = [
        f"Values are {metric} mean +/- sample standard deviation across seeds, in percent.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row_slug, row_label in ROW_ORDER:
        values = [row_label]
        for column_slug, _column_label in COLUMN_ORDER:
            cell = table[row_slug].get(column_slug, {})
            values.append(format_mean_std(cell, metric))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_table_png(
    table: dict[str, dict[str, dict[str, Any]]],
    out_dir: Path,
    metric: str,
    include_std: bool = False,
) -> Path | None:
    plotting = load_plotting()
    if plotting is None:
        return None
    plt, _np = plotting

    headers = ["Method", *[label.replace(" ", "\n", 1) for _slug, label in COLUMN_ORDER]]
    rows = []
    for row_slug, row_label in ROW_ORDER:
        row = [row_label]
        for column_slug, _column_label in COLUMN_ORDER:
            cell = table[row_slug].get(column_slug, {})
            if include_std:
                row.append(format_mean_std(cell, metric))
            else:
                row.append(format_metric(cell.get(metric)))
        rows.append(row)

    fig_width = max(28 if include_std else 24, (3.1 if include_std else 2.4) * len(headers))
    fig_height = max(5.5, 0.5 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    method_col_width = 0.22
    metric_col_width = (1.0 - method_col_width) / (len(headers) - 1)
    mpl_table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[method_col_width, *([metric_col_width] * (len(headers) - 1))],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(7 if include_std else 8)
    mpl_table.scale(1, 1.7)
    for (row_idx, _col_idx), cell in mpl_table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef7")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 else "#f7f7f7")

    suffix = "_mean_std" if include_std else ""
    path = out_dir / f"big_table_{metric}{suffix}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def print_table(table: dict[str, dict[str, dict[str, Any]]], metric: str) -> None:
    headers = ["Method", *[label for _slug, label in COLUMN_ORDER]]
    rows = []
    for row_slug, row_label in ROW_ORDER:
        row = [row_label]
        for column_slug, _column_label in COLUMN_ORDER:
            cell = table[row_slug].get(column_slug, {})
            row.append(format_metric(cell.get(metric)))
        rows.append(row)

    widths = [
        max(len(str(row[col_idx])) for row in [headers, *rows])
        for col_idx in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the table experiment summary.")
    parser.add_argument("--results-dir", default="outputs/experiments/table")
    parser.add_argument("--out-dir", default="outputs/analysis_two")
    parser.add_argument(
        "--print-metric",
        choices=["error", "accuracy"],
        default="error",
        help="Which table to print to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_table(results_dir)
    outputs: list[Path] = [
        write_matrix_csv(table, out_dir, "error"),
        write_matrix_csv(table, out_dir, "accuracy"),
        write_mean_std_csv(table, out_dir, "error"),
        write_mean_std_csv(table, out_dir, "accuracy"),
        write_long_csv(table, out_dir),
        write_markdown_table(table, out_dir, "error"),
        write_markdown_table(table, out_dir, "accuracy"),
        write_mean_std_markdown_table(table, out_dir, "error"),
        write_mean_std_markdown_table(table, out_dir, "accuracy"),
    ]
    for maybe_path in [
        save_table_png(table, out_dir, "error"),
        save_table_png(table, out_dir, "accuracy"),
        save_table_png(table, out_dir, "error", include_std=True),
        save_table_png(table, out_dir, "accuracy", include_std=True),
    ]:
        if maybe_path is not None:
            outputs.append(maybe_path)

    print_table(table, args.print_metric)
    print("\nSaved analysis outputs:")
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()

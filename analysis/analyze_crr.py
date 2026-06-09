from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .analyze_results import format_metric, load_plotting
except ImportError:
    from analyze_results import format_metric, load_plotting


@dataclass(frozen=True)
class CRRPair:
    slug: str
    label: str
    no_domain_clean_column: str
    no_domain_corrupt_column: str
    domain_shift_clean_column: str
    domain_shift_corrupt_column: str


METRIC_OUTPUT_STEMS = {
    "crr": ["crr"],
    "rar_domain_shift": ["robust_accuracy_ratio_domain_shift"],
    "rar_no_domain": ["robust_accuracy_ratio_no_domain"],
    "rar_retention": ["robust_accuracy_ratio_retention"],
}

METRIC_DESCRIPTIONS = {
    "crr": (
        "CRR = Acc_corrupt(domain-shift) / Acc_corrupt(no-domain). "
        "Greater than 1 means corrupted accuracy is higher after domain shift; "
        "near 1 means retained; less than 1 means lower."
    ),
    "rar_domain_shift": (
        "RAR under domain shift. RAR = Acc_corrupt / Acc_clean. Higher is better."
    ),
    "rar_no_domain": (
        "RAR under no domain shift. RAR = Acc_corrupt / Acc_clean. Higher is better."
    ),
    "rar_retention": (
        "RAR retention = RAR(domain-shift) / RAR(no-domain). This is an auxiliary "
        "clean-normalized retention metric, not the paper CRR."
    ),
}


CRR_PAIRS = [
    CRRPair(
        slug="cifar10_1_c_single",
        label="CIFAR10-C -> CIFAR10.1-C single",
        no_domain_clean_column="CIFAR10->CIFAR10 clean",
        no_domain_corrupt_column="CIFAR10->CIFAR10-C single corruption",
        domain_shift_clean_column="CIFAR10->CIFAR10.1 clean",
        domain_shift_corrupt_column="CIFAR10->CIFAR10.1-C single corruption",
    ),
    CRRPair(
        slug="cifar10_1_c_mixed",
        label="CIFAR10-C -> CIFAR10.1-C mixed",
        no_domain_clean_column="CIFAR10->CIFAR10 clean",
        no_domain_corrupt_column="CIFAR10->CIFAR10-C mixed corruption",
        domain_shift_clean_column="CIFAR10->CIFAR10.1 clean",
        domain_shift_corrupt_column="CIFAR10->CIFAR10.1-C mixed corruption",
    ),
    CRRPair(
        slug="svhn_to_mnist_single",
        label="SVHN->SVHN-C -> SVHN->MNIST-C single",
        no_domain_clean_column="SVHN->SVHN clean",
        no_domain_corrupt_column="SVHN->SVHN single corruption",
        domain_shift_clean_column="SVHN->MNIST clean",
        domain_shift_corrupt_column="SVHN->MNIST single corruption",
    ),
    CRRPair(
        slug="svhn_to_mnist_mixed",
        label="SVHN->SVHN-C -> SVHN->MNIST-C mixed",
        no_domain_clean_column="SVHN->SVHN clean",
        no_domain_corrupt_column="SVHN->SVHN mixed corruption",
        domain_shift_clean_column="SVHN->MNIST clean",
        domain_shift_corrupt_column="SVHN->MNIST mixed corruption",
    ),
    CRRPair(
        slug="mnist_to_svhn_single",
        label="MNIST->MNIST-C -> MNIST->SVHN-C single",
        no_domain_clean_column="MNIST->MNIST clean",
        no_domain_corrupt_column="MNIST->MNIST single corruption",
        domain_shift_clean_column="MNIST->SVHN clean",
        domain_shift_corrupt_column="MNIST->SVHN single corruption",
    ),
    CRRPair(
        slug="mnist_to_svhn_mixed",
        label="MNIST->MNIST-C -> MNIST->SVHN-C mixed",
        no_domain_clean_column="MNIST->MNIST clean",
        no_domain_corrupt_column="MNIST->MNIST mixed corruption",
        domain_shift_clean_column="MNIST->SVHN clean",
        domain_shift_corrupt_column="MNIST->SVHN mixed corruption",
    ),
]


def optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def load_accuracy_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def available_pairs(fieldnames: list[str]) -> list[CRRPair]:
    fields = set(fieldnames)
    return [
        pair
        for pair in CRR_PAIRS
        if pair.no_domain_corrupt_column in fields
        and pair.domain_shift_corrupt_column in fields
    ]


def build_long_rows(
    rows: list[dict[str, str]],
    pairs: list[CRRPair],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []

    for pair in pairs:
        for row in rows:
            method = row["method"]
            acc_no_domain_clean = optional_float(row.get(pair.no_domain_clean_column))
            acc_no_domain_corrupt = optional_float(row.get(pair.no_domain_corrupt_column))
            acc_domain_clean = optional_float(row.get(pair.domain_shift_clean_column))
            acc_domain_corrupt = optional_float(row.get(pair.domain_shift_corrupt_column))

            rar_no_domain = None
            rar_domain_shift = None
            rar_retention = None
            crr = None
            crr_status = "ok"
            crr_values = [acc_no_domain_corrupt, acc_domain_corrupt]
            if any(value is None for value in crr_values):
                crr_status = "missing"
            elif acc_no_domain_corrupt == 0:
                crr_status = "zero_no_domain_corrupt_accuracy"
            else:
                crr = acc_domain_corrupt / acc_no_domain_corrupt

            if (
                acc_no_domain_clean is not None
                and acc_domain_clean is not None
                and acc_no_domain_corrupt is not None
                and acc_domain_corrupt is not None
                and acc_no_domain_clean != 0
                and acc_domain_clean != 0
            ):
                rar_no_domain = acc_no_domain_corrupt / acc_no_domain_clean
                rar_domain_shift = acc_domain_corrupt / acc_domain_clean
                if rar_no_domain != 0:
                    rar_retention = rar_domain_shift / rar_no_domain

            output_rows.append(
                {
                    "method": method,
                    "setting": pair.label,
                    "setting_slug": pair.slug,
                    "no_domain_clean_column": pair.no_domain_clean_column,
                    "no_domain_corrupt_column": pair.no_domain_corrupt_column,
                    "domain_shift_clean_column": pair.domain_shift_clean_column,
                    "domain_shift_corrupt_column": pair.domain_shift_corrupt_column,
                    "acc_no_domain_clean": acc_no_domain_clean,
                    "acc_no_domain_corrupt": acc_no_domain_corrupt,
                    "rar_no_domain": rar_no_domain,
                    "acc_domain_shift_clean": acc_domain_clean,
                    "acc_domain_shift_corrupt": acc_domain_corrupt,
                    "rar_domain_shift": rar_domain_shift,
                    "rar_retention": rar_retention,
                    "crr": crr,
                    "crr_status": crr_status,
                }
            )
    return output_rows


def cleanup_old_outputs(out_dir: Path) -> None:
    for pattern in [
        "crr*",
        "corruption_gain_*",
        "corruption_drop_reduction_*",
        "robust_accuracy_ratio_*",
        "corruption_robustness_ratio_*",
    ]:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def write_long_csv(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    paths = [
        out_dir / "crr_long.csv",
    ]
    fieldnames = [
        "method",
        "setting",
        "setting_slug",
        "no_domain_clean_column",
        "no_domain_corrupt_column",
        "domain_shift_clean_column",
        "domain_shift_corrupt_column",
        "acc_no_domain_clean",
        "acc_no_domain_corrupt",
        "rar_no_domain",
        "acc_domain_shift_clean",
        "acc_domain_shift_corrupt",
        "rar_domain_shift",
        "rar_retention",
        "crr",
        "crr_status",
    ]
    for path in paths:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return paths


def write_matrix_csv(
    long_rows: list[dict[str, Any]],
    method_order: list[str],
    pairs: list[CRRPair],
    metric: str,
    out_dir: Path,
) -> list[Path]:
    labels = [pair.label for pair in pairs]
    by_key = {
        (row["method"], row["setting"]): row
        for row in long_rows
    }
    paths = []
    for stem in METRIC_OUTPUT_STEMS[metric]:
        path = out_dir / f"{stem}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method", *labels])
            writer.writeheader()
            for method in method_order:
                row_out = {"method": method}
                for pair in pairs:
                    row_out[pair.label] = by_key.get((method, pair.label), {}).get(metric, "")
                writer.writerow(row_out)
        paths.append(path)
    return paths


def write_markdown_table(
    long_rows: list[dict[str, Any]],
    method_order: list[str],
    pairs: list[CRRPair],
    metric: str,
    out_dir: Path,
) -> list[Path]:
    headers = ["Method", *[pair.label for pair in pairs]]
    by_key = {
        (row["method"], row["setting"]): row
        for row in long_rows
    }
    lines = [
        METRIC_DESCRIPTIONS[metric],
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for method in method_order:
        values = [method]
        for pair in pairs:
            values.append(format_metric(by_key.get((method, pair.label), {}).get(metric)))
        lines.append("| " + " | ".join(values) + " |")
    paths = []
    for stem in METRIC_OUTPUT_STEMS[metric]:
        path = out_dir / f"{stem}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def save_table_png(
    long_rows: list[dict[str, Any]],
    method_order: list[str],
    pairs: list[CRRPair],
    metric: str,
    out_dir: Path,
) -> list[Path]:
    plotting = load_plotting()
    if plotting is None:
        return []
    plt, _np = plotting
    headers = ["Method", *[pair.label.replace(" -> ", "\n-> ") for pair in pairs]]
    by_key = {
        (row["method"], row["setting"]): row
        for row in long_rows
    }
    rows = []
    for method in method_order:
        row = [method]
        for pair in pairs:
            row.append(format_metric(by_key.get((method, pair.label), {}).get(metric)))
        rows.append(row)

    fig_width = max(18, 2.7 * len(headers))
    fig_height = max(5.5, 0.5 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(METRIC_DESCRIPTIONS[metric], fontsize=9, weight="bold", pad=12)
    method_col_width = 0.24
    metric_col_width = (1.0 - method_col_width) / (len(headers) - 1)
    mpl_table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[method_col_width, *([metric_col_width] * (len(headers) - 1))],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8)
    mpl_table.scale(1, 1.7)
    for (row_idx, _col_idx), cell in mpl_table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef7")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 else "#f7f7f7")

    paths = []
    fig.tight_layout()
    for stem in METRIC_OUTPUT_STEMS[metric]:
        path = out_dir / f"{stem}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def print_table(
    long_rows: list[dict[str, Any]],
    method_order: list[str],
    pairs: list[CRRPair],
    metric: str,
) -> None:
    print(METRIC_DESCRIPTIONS[metric])
    print()
    headers = ["Method", *[pair.label for pair in pairs]]
    by_key = {
        (row["method"], row["setting"]): row
        for row in long_rows
    }
    matrix_rows = []
    for method in method_order:
        values = [method]
        for pair in pairs:
            values.append(format_metric(by_key.get((method, pair.label), {}).get(metric)))
        matrix_rows.append(values)

    widths = [
        max(len(str(row[index])) for row in [headers, *matrix_rows])
        for index in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in matrix_rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paper CRR under domain shift."
    )
    parser.add_argument("--accuracy-csv", default="outputs/analysis_two/big_table_accuracy.csv")
    parser.add_argument("--out-dir", default="outputs/analysis_crr")
    parser.add_argument(
        "--print-metric",
        choices=sorted(METRIC_OUTPUT_STEMS),
        default="crr",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accuracy_csv = Path(args.accuracy_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_old_outputs(out_dir)

    rows = load_accuracy_table(accuracy_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {accuracy_csv}")
    pairs = available_pairs(list(rows[0].keys()))
    if not pairs:
        raise RuntimeError("No configured CRR pairs were present in the accuracy table.")

    method_order = [row["method"] for row in rows]
    long_rows = build_long_rows(
        rows=rows,
        pairs=pairs,
    )
    outputs = []
    outputs.extend(write_long_csv(long_rows, out_dir))
    for metric in ["crr", "rar_domain_shift", "rar_no_domain", "rar_retention"]:
        outputs.extend(write_matrix_csv(long_rows, method_order, pairs, metric, out_dir))
        outputs.extend(write_markdown_table(long_rows, method_order, pairs, metric, out_dir))
        outputs.extend(save_table_png(long_rows, method_order, pairs, metric, out_dir))

    print_table(long_rows, method_order, pairs, args.print_metric)
    print("\nSaved CRR analysis outputs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

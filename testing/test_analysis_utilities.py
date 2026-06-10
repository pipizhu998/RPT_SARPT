from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing._bootstrap import use_workspace_tempdir

use_workspace_tempdir()

from analysis.analyze_crr import CRRPair, build_long_rows
from analysis.analyze_results_two import (
    format_mean_std,
    load_table,
    metric_from_result,
    row_slug_from_experiment,
    strip_seed_suffix,
)
from analysis.plot_collapse_probability import fractions, normalized_entropy, rolling_mean
from analysis.plot_long_stream_accuracy import load_average_curve


class AnalysisUtilityTests(unittest.TestCase):
    def test_result_name_parsing(self) -> None:
        name, seed = strip_seed_suffix("svhn_to_mnist_mixed_augmix_rpt_seed4")
        self.assertEqual(name, "svhn_to_mnist_mixed_augmix_rpt")
        self.assertEqual(seed, 4)
        self.assertEqual(
            row_slug_from_experiment(
                "svhn_to_mnist_mixed_augmix_rpt_resnet18_rpt_episodic_true_seed4"
            ),
            "augmix_rpt",
        )

    def test_metric_from_result(self) -> None:
        self.assertEqual(
            metric_from_result({"clean_error": 0.1}, "svhn_to_svhn_clean"),
            0.1,
        )
        self.assertEqual(
            round(metric_from_result(
                {
                    "rows": [
                        {"domain": "ood", "error_rate": 0.2},
                        {"domain": "ood", "error_rate": 0.4},
                    ]
                },
                "svhn_to_svhn_single",
            ), 10),
            0.3,
        )

    def test_load_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            column = root / "svhn_to_mnist_mixed"
            column.mkdir()
            for seed, error in [(0, 0.2), (1, 0.4)]:
                payload = {
                    "average_ood_error": error,
                    "test_adapt": "rpt",
                    "rpt_returns_adapted_logits": True,
                    "rpt_variant": "rpt",
                    "tent_episodic": True,
                }
                path = column / (
                    f"svhn_to_mnist_mixed_augmix_rpt_resnet18_"
                    f"rpt_episodic_true_seed{seed}_results.json"
                )
                path.write_text(json.dumps(payload), encoding="utf-8")

            table = load_table(root)

        cell = table["augmix_rpt"]["svhn_to_mnist_mixed"]
        self.assertAlmostEqual(cell["error"], 0.3)
        self.assertAlmostEqual(cell["accuracy"], 0.7)
        self.assertAlmostEqual(cell["error_std"], 0.1414213562)
        self.assertEqual(cell["num_seeds"], 2)

    def test_format_mean_std_formats_percent_values(self) -> None:
        self.assertEqual(
            format_mean_std({"accuracy": 0.75, "accuracy_std": 0.02}, "accuracy"),
            "75.00 +/- 2.00",
        )


    def test_collapse_helpers(self) -> None:
        from collections import Counter

        counts = Counter({0: 3, 1: 1})
        self.assertEqual(fractions(counts, 2), {"0": 0.75, "1": 0.25})
        self.assertGreater(normalized_entropy(counts, 2), 0.0)
        means = rolling_mean([1.0, 3.0, 5.0], window=2)
        self.assertTrue(means[0] != means[0])
        self.assertEqual(means[1:], [2.0, 4.0])
        with self.assertRaises(ValueError):
            rolling_mean([1.0], window=0)

    def test_long_stream_curve_average(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for index, values in enumerate(([0.4, 0.6], [0.6, 0.8])):
                path = root / f"curve{index}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["step", "cumulative_accuracy"],
                    )
                    writer.writeheader()
                    writer.writerows(
                        [
                            {"step": 1, "cumulative_accuracy": values[0]},
                            {"step": 2, "cumulative_accuracy": values[1]},
                        ]
                    )
                paths.append(path)

            xs, ys = load_average_curve(paths, "step", "cumulative_accuracy")

        self.assertEqual(xs, [1.0, 2.0])
        self.assertEqual(ys, [0.5, 0.7])


if __name__ == "__main__":
    unittest.main()

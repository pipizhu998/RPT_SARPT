from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing._bootstrap import use_workspace_tempdir

use_workspace_tempdir()

from entry_point.evaluate import (
    EvaluationConfig,
    ProbabilityLogRecorder,
    StreamAccuracyRecorder,
    build_augmix_cache_key,
    output_path_for_args,
    probability_log_path,
    should_record_probabilities,
    slugify,
    sweep_variant_slug,
    tensor_to_list,
)


class EvaluationUtilityTests(unittest.TestCase):
    def test_cache_key_is_stable_and_sensitive_to_metadata(self) -> None:
        metadata = {
            "row_label": "noise@1",
            "ood_dataset": "digitrobust_mnistc",
            "evaluation_mode": "mixed",
            "seed": 1,
        }
        self.assertEqual(build_augmix_cache_key(metadata), build_augmix_cache_key(dict(metadata)))
        changed = dict(metadata, seed=2)
        self.assertNotEqual(build_augmix_cache_key(metadata), build_augmix_cache_key(changed))

    def test_slug_helpers_generate_filename(self) -> None:
        self.assertEqual(slugify(" RPT Sweep 1.0 "), "rpt_sweep_1p0")
        self.assertEqual(sweep_variant_slug(0.001, 12.0, 0.05), "lr0p001_jsd12_src0p05")

    def test_output_and_probability_paths(self) -> None:
        args = EvaluationConfig(
            checkpoint="models/best.pt",
            test_adapt="rpt",
            rpt_episodic=True,
            tent_prob_log_file="auto",
        )
        output = output_path_for_args(args)
        self.assertEqual(output.name, "best_shift_results_rpt_episodic_true.csv")
        self.assertEqual(
            probability_log_path(args).name,
            "best_shift_results_rpt_episodic_true_probabilities.csv",
        )
        self.assertTrue(should_record_probabilities(args))

    def test_tensor_to_list_accepts_supported_inputs_only(self) -> None:
        self.assertEqual(tensor_to_list(torch.tensor([1, 2])), [1, 2])
        self.assertEqual(tensor_to_list((1, 2)), [1, 2])
        with self.assertRaises(TypeError):
            tensor_to_list("invalid")

    def test_stream_accuracy_recorder_writes_expected_csv(self) -> None:
        recorder = StreamAccuracyRecorder("mixed")
        recorder(
            {
                "step": 1.0,
                "batch_examples": 2.0,
                "batch_loss": 0.4,
                "batch_accuracy": 0.5,
                "batch_correct": 1.0,
                "cumulative_examples": 2.0,
                "cumulative_loss": 0.4,
                "cumulative_accuracy": 0.5,
                "cumulative_correct": 1.0,
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stream.csv"
            recorder.save_csv(path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["row_label"], "mixed")
        self.assertEqual(rows[0]["cumulative_accuracy"], "0.5")
        self.assertEqual(rows[0]["batch_loss"], "0.4")
        self.assertEqual(rows[0]["step"], "1")
        self.assertEqual(rows[0]["cumulative_correct"], "1")

    def test_probability_log_recorder_writes_predictions_and_augmented_views(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "probabilities.csv"
            recorder = ProbabilityLogRecorder(path, "mixed")
            recorder(
                {
                    "step": 1,
                    "targets": torch.tensor([1]),
                    "preds": torch.tensor([1]),
                    "probs": torch.tensor([[0.1, 0.9]]),
                    "aug1_preds": torch.tensor([1]),
                    "aug2_preds": torch.tensor([0]),
                    "aug1_probs": torch.tensor([[0.2, 0.8]]),
                    "aug2_probs": torch.tensor([[0.7, 0.3]]),
                    "jsd": torch.tensor([0.12]),
                }
            )
            recorder.close()
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["correct"], "1")
        self.assertEqual(row["clean_aug1_agree"], "1")
        self.assertEqual(row["clean_aug2_agree"], "0")
        self.assertAlmostEqual(float(row["jsd"]), 0.12, places=5)
        self.assertAlmostEqual(
            float(row["entropy"]),
            -(0.1 * math.log(0.1) + 0.9 * math.log(0.9)),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()

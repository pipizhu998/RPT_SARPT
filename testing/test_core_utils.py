from __future__ import annotations

import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.utils import (
    evaluate_model,
    progress_total,
    resolve_device,
    save_csv,
    save_json,
    set_seed,
)
from testing._bootstrap import use_workspace_tempdir

use_workspace_tempdir()

class FixedClassifier(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class CoreUtilsTests(unittest.TestCase):
    def test_set_seed(self) -> None:
        set_seed(17)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        set_seed(17)
        second = (random.random(), np.random.rand(), torch.rand(1).item())

        self.assertEqual(first, second)

    def test_resolve_device(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))

    def test_save_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "nested" / "payload.json"
            csv_path = root / "nested" / "rows.csv"

            save_json({"value": 3}, json_path)
            save_csv([{"name": "rpt", "accuracy": 0.75}, {"name": "tent", "accuracy": 0.8}], csv_path)

            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), {"value": 3})
            with csv_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(
                    list(csv.DictReader(handle)),
                    [{"name": "rpt", "accuracy": "0.75"}, {"name": "tent", "accuracy": "0.8"}],
                )

    def test_progress_total(self) -> None:
        loader = DataLoader(TensorDataset(torch.arange(10)), batch_size=2)
        self.assertEqual(progress_total(loader), 5)
        self.assertEqual(progress_total(loader, max_batches=3), 3)
        self.assertEqual(progress_total(loader, max_batches=8), 5)

    def test_evaluate_model_computes_accuracy(self) -> None:
        logits = torch.tensor(
            [[5.0, 0.0], [0.0, 5.0], [4.0, 1.0], [0.0, 3.0]]
        )
        targets = torch.tensor([0, 1, 1, 1])
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        callback_rows: list[dict[str, float]] = []

        metrics = evaluate_model(
            model=FixedClassifier(),
            dataloader=loader,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            batch_metrics_callback=callback_rows.append,
        )

        self.assertEqual(metrics["examples"], 4.0)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(len(callback_rows), 2)
        self.assertAlmostEqual(callback_rows[-1]["cumulative_accuracy"], 0.75)

    def test_evaluate_model_rejects_empty_loader(self) -> None:
        loader = DataLoader(
            TensorDataset(torch.empty((0, 2)), torch.empty((0,), dtype=torch.long)),
            batch_size=2,
        )
        with self.assertRaisesRegex(RuntimeError, "No evaluation examples"):
            evaluate_model(
                FixedClassifier(),
                loader,
                nn.CrossEntropyLoss(),
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()

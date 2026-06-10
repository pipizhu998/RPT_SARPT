from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import experiment_config
from testing._bootstrap import use_workspace_tempdir

use_workspace_tempdir()


class ExperimentConfigTests(unittest.TestCase):
    def test_load_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mapping_path = root / "mapping.yaml"
            mapping_path.write_text("dataset:\n  name: svhn\n", encoding="utf-8")
            empty_path = root / "empty.yaml"
            empty_path.write_text("", encoding="utf-8")

            self.assertEqual(
                experiment_config.load_yaml(mapping_path),
                {"dataset": {"name": "svhn"}},
            )
            self.assertEqual(experiment_config.load_yaml(empty_path), {})

    def test_load_yaml_rejects_non_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "list.yaml"
            path.write_text("- one\n- two\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non mapping"):
                experiment_config.load_yaml(path)

    def test_deep_merge_merges(self) -> None:
        base = {"dataset": {"name": "svhn", "batch_size": 128}, "seed": 0}
        override = {"dataset": {"batch_size": 64}, "seed": 1}

        result = experiment_config.deep_merge(base, override)

        self.assertEqual(
            result,
            {"dataset": {"name": "svhn", "batch_size": 64}, "seed": 1},
        )
        self.assertEqual(base["dataset"]["batch_size"], 128)
        self.assertEqual(override["dataset"]["batch_size"], 64)

    def test_load_experiment_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "dataset").mkdir()
            (root / "model").mkdir()
            (root / "dataset" / "digits.yaml").write_text(
                "dataset:\n  name: svhn\n  batch_size: 128\n",
                encoding="utf-8",
            )
            (root / "model" / "small.yaml").write_text(
                "model:\n  name: resnet18\n",
                encoding="utf-8",
            )
            experiment = root / "experiment.yaml"
            experiment.write_text(
                "includes:\n"
                "  dataset: digits\n"
                "  model: small\n"
                "dataset:\n"
                "  batch_size: 32\n"
                "stage: testing\n",
                encoding="utf-8",
            )

            with patch.object(experiment_config, "CONFIG_ROOT", root):
                result = experiment_config.load_experiment_config(experiment)

            self.assertEqual(result["dataset"], {"name": "svhn", "batch_size": 32})
            self.assertEqual(result["model"]["name"], "resnet18")
            self.assertEqual(result["stage"], "testing")
            self.assertEqual(result["_config_path"], str(experiment))

    def test_load_experiment_config_rejects_unknown_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiment.yaml"
            path.write_text("includes:\n  unknown: value\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unknown config group"):
                experiment_config.load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()

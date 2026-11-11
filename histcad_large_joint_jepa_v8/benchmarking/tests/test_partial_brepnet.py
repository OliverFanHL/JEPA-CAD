from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from benchmarking.baselines.external.prepare_brepnet import collect  # noqa: E402
from benchmarking.src.cad_benchmark.io import Representation  # noqa: E402
from benchmarking.src.cad_benchmark.registry import Registry  # noqa: E402


class PartialBRepNetTest(unittest.TestCase):
    def test_collect_and_shared_coverage_keep_every_method_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = root / "work" / "embeddings"
            failures = root / "work" / "preprocessing_failures"
            embeddings.mkdir(parents=True)
            failures.mkdir()
            uids = ["a", "b", "c", "d"]
            splits = ["train", "val", "test", "train"]
            labels = {"class": [0, 1, 1, 0]}
            targets = {"dimensions": torch.arange(8).reshape(4, 2).float()}
            reference = Representation("base", "jepa", torch.arange(8).reshape(4, 2).float(),
                                       uids, splits, labels, targets)
            rows = [{"uid": uid} for uid in uids]
            for index in (0, 2, 3):
                (embeddings / f"{index:08d}.embeddings").write_text("1 2\n3 4\n", encoding="utf-8")
            (failures / "00000001.json").write_text(json.dumps({"error": "Open body"}), encoding="utf-8")
            output = root / "brepnet.pt"
            with patch("benchmarking.baselines.external.prepare_brepnet.repository_revision", return_value="test"), \
                    patch("benchmarking.baselines.external.prepare_brepnet.file_sha256", return_value="test"), \
                    patch("benchmarking.baselines.external.prepare_brepnet.checkpoint_tensor_count", return_value=1):
                with self.assertRaisesRegex(RuntimeError, "--allow-partial"):
                    collect("histcad", rows, reference, embeddings, output, root, root / "checkpoint")
                self.assertFalse(output.exists())
                collect("histcad", rows, reference, embeddings, output, root,
                        root / "checkpoint", allow_partial=True)

            base = root / "base.pt"
            torch.save({"uids": uids, "splits": splits, "labels": labels, "targets": targets,
                        "embeddings": {"base": reference.values}}, base)
            specs = [
                {"name": "base", "family": "jepa", "dataset": "histcad", "format": "generic_pt",
                 "artifact": base, "storage": "embeddings", "key": "base", "tasks": ["frozen"]},
                {"name": "brepnet", "family": "external_brepnet", "dataset": "histcad",
                 "format": "generic_pt", "artifact": output, "storage": "embeddings",
                 "key": "brepnet_official", "tasks": ["frozen"]},
            ]
            config = {"representations": specs,
                      "shared_coverage": {"histcad": {"file": output.with_suffix(".coverage.json")}}}
            registry = Registry(config, apply_shared_coverage=True)
            registry.validate_alignment("frozen")
            for name in ("base", "brepnet"):
                loaded = registry.load(name)
                self.assertEqual(loaded.uids, ["a", "c", "d"])
                self.assertEqual(loaded.splits, ["train", "test", "train"])
                self.assertEqual(loaded.labels["class"], [0, 1, 0])
                self.assertEqual(loaded.targets["dimensions"].shape[0], 3)
            self.assertEqual(registry.coverage_summary()["histcad"]["omitted_rows"], 1)


if __name__ == "__main__":
    unittest.main()

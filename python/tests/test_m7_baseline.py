from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nre_baseline.artifacts import load_model_artifact, save_model_artifact
from nre_baseline.checksums import fnv1a64_file
from nre_baseline.dataset import DatasetError, LabelRow, REQUIRED_COLUMNS, load_dataset
from nre_baseline.metrics import error_metrics
from nre_baseline.model import fit_model
from nre_baseline.preprocessing import Preprocessor, raw_features


def raw_row(parameter_id: str, split: str, included: str = "true") -> dict[str, str]:
    row = {field: "" for field in REQUIRED_COLUMNS}
    row.update(
        {
            "schema_version": "nre.dataset.v1",
            "parameter_id": parameter_id,
            "split": split,
            "included_for_training": included,
            "quality_status": "accepted" if included == "true" else "rejected",
            "quality_flags": "none" if included == "true" else "test_rejection",
            "option_style": "european",
            "option_type": "call",
            "spot": "100",
            "strike": "100",
            "maturity_years": "1",
            "volatility": "0.2",
            "risk_free_rate": "0.03",
            "dividend_yield": "0.01",
            "observations": "1",
            "backend": "monte_carlo",
            "estimator": "plain",
            "label_tier": "bulk_training" if split == "train" else "heldout_reference",
            "price": "10",
            "price_standard_error": "0.1",
            "price_ci_95_lower": "9.804",
            "price_ci_95_upper": "10.196",
            "delta": "0.55",
            "delta_standard_error": "0.01",
            "delta_ci_95_lower": "0.5304",
            "delta_ci_95_upper": "0.5696",
            "effective_paths": "1000",
            "raw_paths": "1000",
            "pricing_seed": "123",
            "requested_threads": "2",
            "active_threads": "2",
            "analytical_price": "10.01",
            "analytical_delta": "0.551",
            "analytical_price_absolute_error": "0.01",
            "analytical_delta_absolute_error": "0.001",
        }
    )
    return row


def write_fixture(root: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    dataset = root / "dataset"
    dataset.mkdir()
    config = root / "config.cfg"
    config.write_text("schema_version=nre.dataset.v1\n", encoding="utf-8")
    labels = dataset / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}
    manifest = {
        "schema_version": "nre.dataset.v1",
        "config": {"fnv1a64": fnv1a64_file(config)},
        "counts": {
            "total_rows": len(rows),
            "train_rows": counts["train"],
            "validation_rows": counts["validation"],
            "test_rows": counts["test"],
        },
        "artifacts": {"labels_fnv1a64": fnv1a64_file(labels)},
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset, config


def label(identifier: str, split: str, spot: float, price: float, delta: float) -> LabelRow:
    return LabelRow(
        identifier,
        split,
        "european",
        "call",
        spot,
        100.0,
        1.0,
        0.2,
        0.03,
        0.01,
        1,
        price,
        delta,
        {},
    )


class DatasetLoaderTests(unittest.TestCase):
    def test_validates_checksums_and_excludes_rejected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [raw_row("train", "train"), raw_row("rejected", "train", "false"), raw_row("val", "validation"), raw_row("test", "test")]
            dataset_path, config = write_fixture(root, rows)
            dataset = load_dataset(dataset_path, config)
            self.assertEqual({split: len(dataset.by_split[split]) for split in dataset.by_split}, {"train": 1, "validation": 1, "test": 1})
            self.assertNotIn("rejected", {row.parameter_id for row in dataset.rows})

    def test_rejects_label_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path, config = write_fixture(root, [raw_row("train", "train"), raw_row("val", "validation"), raw_row("test", "test")])
            with (dataset_path / "labels.csv").open("a", encoding="utf-8") as output:
                output.write("tamper\n")
            with self.assertRaisesRegex(DatasetError, "checksum"):
                load_dataset(dataset_path, config)

    def test_rejects_duplicate_ids_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [raw_row("same", "train"), raw_row("same", "validation"), raw_row("test", "test")]
            dataset_path, config = write_fixture(root, rows)
            with self.assertRaisesRegex(DatasetError, "duplicate"):
                load_dataset(dataset_path, config)

    def test_rejects_non_finite_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [raw_row("train", "train"), raw_row("val", "validation"), raw_row("test", "test")]
            rows[0]["price"] = "nan"
            dataset_path, config = write_fixture(root, rows)
            with self.assertRaisesRegex(DatasetError, "not finite"):
                load_dataset(dataset_path, config)


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = tuple(
            label(f"p{index}", "train", 80.0 + 5.0 * index, 5.0 + index, 0.3 + 0.02 * index)
            for index in range(8)
        )

    def test_preprocessing_is_fit_on_training_only(self) -> None:
        validation = (label("v", "validation", 10000.0, 1.0, 0.1),)
        first = Preprocessor.fit(self.training, degree=2)
        second = Preprocessor.fit(self.training, degree=2)
        second.transform(validation)
        np.testing.assert_array_equal(first.means, second.means)
        np.testing.assert_array_equal(first.means, raw_features(self.training).mean(axis=0))

    def test_model_fit_is_reproducible(self) -> None:
        first = fit_model(self.training, degree=2, alpha=1.0e-2)
        second = fit_model(self.training, degree=2, alpha=1.0e-2)
        np.testing.assert_array_equal(first.coefficients, second.coefficients)

    def test_metric_values(self) -> None:
        result = error_metrics(
            np.asarray([2.0, 12.0]),
            np.asarray([1.0, 10.0]),
            np.asarray([0.5, -0.4]),
            np.asarray([0.4, -0.6]),
            price_floor=1.0,
        )
        self.assertAlmostEqual(result["normalized_price_error"]["median"], 0.6)
        self.assertAlmostEqual(result["delta_rmse"], np.sqrt(0.025))

    def test_artifact_round_trip_preserves_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path, config = write_fixture(root, [raw_row("train", "train"), raw_row("val", "validation"), raw_row("test", "test")])
            dataset = load_dataset(dataset_path, config)
            model = fit_model(self.training, degree=2, alpha=1.0e-2)
            artifact = root / "model.json"
            save_model_artifact(artifact, model, dataset, [])
            restored, _ = load_model_artifact(artifact)
            expected = model.predict(self.training)
            actual = restored.predict(self.training)
            np.testing.assert_array_equal(expected[0], actual[0])
            np.testing.assert_array_equal(expected[1], actual[1])


if __name__ == "__main__":
    unittest.main()

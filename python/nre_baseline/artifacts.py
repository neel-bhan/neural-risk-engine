"""Canonical model artifact serialization and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checksums import fnv1a64_file
from .dataset import Dataset
from .model import PolynomialRidgeModel


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def save_model_artifact(
    path: Path,
    model: PolynomialRidgeModel,
    dataset: Dataset,
    validation_candidates: list[dict[str, float]],
) -> str:
    value = {
        "artifact_version": "nre.baseline.artifact.v1",
        "source_implementation_commit": "PENDING",
        "dataset": {
            "schema_version": dataset.schema_version,
            "manifest_fnv1a64": dataset.manifest_checksum,
            "config_fnv1a64": dataset.config_checksum,
            "labels_fnv1a64": dataset.labels_checksum,
        },
        "training_policy": {
            "fit_split": "train",
            "selection_split": "validation",
            "test_access_during_training": False,
            "selection_metric": "median normalized price error; p99 then lower degree break ties",
            "price_floor": 1.0,
            "declared_degrees": [1, 2, 3],
            "declared_ridge_alphas": [1.0e-8, 1.0e-5, 1.0e-2, 1.0],
        },
        "model": model.to_dict(),
        "validation_candidates": validation_candidates,
    }
    write_json(path, value)
    return fnv1a64_file(path)


def load_model_artifact(path: Path) -> tuple[PolynomialRidgeModel, dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("model artifact is not valid JSON") from error
    if not isinstance(value, dict) or value.get("artifact_version") != "nre.baseline.artifact.v1":
        raise ValueError("unsupported model artifact version")
    model = PolynomialRidgeModel.from_dict(value["model"])
    return model, value


"""Versioned M8 checkpoint state and export-ready metadata."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nre_baseline.artifacts import write_json
from nre_baseline.checksums import fnv1a64_file
from nre_baseline.dataset import Dataset
from nre_baseline.preprocessing import BASE_FEATURE_NAMES

from .model import ScalarPriceMLP
from .preprocessing import NeuralPreprocessor
from .training import ExperimentConfig, TrainingResult


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def dataset_provenance(dataset: Dataset) -> dict[str, str]:
    return {
        "schema_version": dataset.schema_version,
        "manifest_fnv1a64": dataset.manifest_checksum,
        "config_fnv1a64": dataset.config_checksum,
        "labels_fnv1a64": dataset.labels_checksum,
    }


def save_neural_artifact(
    metadata_path: Path,
    state_path: Path,
    result: TrainingResult,
    dataset: Dataset,
    experiment: ExperimentConfig,
    experiment_path: Path,
) -> dict[str, Any]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = result.model.state_dict()
    torch.save(state, state_path)
    state_sha256 = state_dict_sha256(state)
    value: dict[str, Any] = {
        "artifact_version": "nre.neural.artifact.v1",
        "source_implementation_commit": "PENDING",
        "variant": result.variant,
        "dataset": dataset_provenance(dataset),
        "experiment": {
            "config_path": "python/config/m8-neural-v1.json",
            "config_fnv1a64": fnv1a64_file(experiment_path),
            "seed": result.seed,
            "fit_split": "train",
            "selection_split": "validation",
            "test_access_during_training": False,
            "primary_selection_metric": "validation median normalized price error",
            "optimizer": experiment.raw["optimizer"],
            "learning_rate": experiment.learning_rate,
            "schedule": experiment.raw["schedule"],
            "batch_size": experiment.batch_size,
            "maximum_epochs": experiment.maximum_epochs,
            "early_stopping_patience": experiment.early_stopping_patience,
            "minimum_epochs": experiment.minimum_epochs,
            "price_loss": experiment.raw["price_loss"],
            "delta_loss": experiment.raw["delta_loss"],
            "derivative_supervision_weight": (
                experiment.derivative_supervision_weight
                if result.variant == "derivative_supervised"
                else 0.0
            ),
            "elapsed_training_seconds": result.elapsed_seconds,
        },
        "model": {
            "model_version": "nre.model.scalar_price_mlp.v1",
            "input_feature_names": list(BASE_FEATURE_NAMES),
            "hidden_layers": list(result.model.hidden_layers),
            "activation": "tanh",
            "output_count": 1,
            "output_representation": "price_divided_by_strike",
            "delta_representation": "autograd derivative of physical scalar price with respect to unscaled spot; no Delta head",
            "parameter_count": result.model.parameter_count,
            "dtype": "float64",
        },
        "preprocessing": result.preprocessor.to_dict(),
        "selected_candidate": result.selected_candidate,
        "validation_candidates": result.validation_candidates,
        "state": {
            "file": state_path.name,
            "file_fnv1a64": fnv1a64_file(state_path),
            "canonical_tensor_sha256": state_sha256,
            "format": "PyTorch state_dict; weights_only load",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "device": "cpu",
            "platform": platform.platform(),
            "processor": platform.processor() or "arm64 reported by platform",
            "deterministic_algorithms": True,
            "torch_threads": 1,
        },
        "m9_export_contract": {
            "input": "nine scaled float64 base features in recorded order",
            "output": "one normalized scalar price per row",
            "physical_price": "strike * model_output",
            "onnx_exported": False,
        },
    }
    write_json(metadata_path, value)
    return value


def load_neural_artifact(
    metadata_path: Path | str, state_path: Path | str | None = None
) -> tuple[ScalarPriceMLP, NeuralPreprocessor, dict[str, Any]]:
    metadata_file = Path(metadata_path)
    try:
        value = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("neural metadata is not valid JSON") from error
    if not isinstance(value, dict) or value.get("artifact_version") != "nre.neural.artifact.v1":
        raise ValueError("unsupported neural artifact version")
    model_value = value["model"]
    if (
        model_value.get("model_version") != "nre.model.scalar_price_mlp.v1"
        or model_value.get("output_count") != 1
        or model_value.get("input_feature_names") != list(BASE_FEATURE_NAMES)
    ):
        raise ValueError("invalid scalar-price model metadata")
    preprocessor = NeuralPreprocessor.from_dict(value["preprocessing"])
    model = ScalarPriceMLP(len(BASE_FEATURE_NAMES), model_value["hidden_layers"]).to(dtype=torch.float64)
    state_file = Path(state_path) if state_path is not None else metadata_file.with_name(value["state"]["file"])
    if fnv1a64_file(state_file) != value["state"]["file_fnv1a64"]:
        raise ValueError("neural state file checksum does not match metadata")
    loaded = torch.load(state_file, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or state_dict_sha256(loaded) != value["state"]["canonical_tensor_sha256"]:
        raise ValueError("neural tensor checksum does not match metadata")
    model.load_state_dict(loaded, strict=True)
    if model.parameter_count != int(model_value["parameter_count"]):
        raise ValueError("neural parameter count does not match metadata")
    model.eval()
    return model, preprocessor, value

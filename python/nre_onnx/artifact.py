"""Frozen scalar-price ONNX artifact and strict deployment metadata."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from nre_baseline.artifacts import write_json
from nre_baseline.checksums import fnv1a64_file
from nre_baseline.preprocessing import BASE_FEATURE_NAMES
from nre_neural.artifacts import load_neural_artifact
from nre_neural.preprocessing import NeuralPreprocessor


ARTIFACT_VERSION = "nre.onnx.scalar_price.v1"
INPUT_NAME = "scaled_features"
OUTPUT_NAME = "normalized_price"
OPSET_VERSION = 18
DEPLOYMENT_DOMAIN: dict[str, object] = {
    "spot": [60.0, 140.0],
    "strike": [60.0, 140.0],
    "maturity_years": [0.25, 2.0],
    "volatility": [0.05, 0.60],
    "risk_free_rate": [-0.02, 0.10],
    "dividend_yield": [-0.01, 0.08],
    "asian_observations": [2, 52],
    "european_observations": 1,
}
DELTA_POLICY: dict[str, object] = {
    "name": "centered_spot_bump",
    "relative_bump": 1.0e-4,
    "absolute_floor": 1.0e-6,
}
GUARDRAILS: dict[str, float] = {
    "bound_tolerance": 1.0e-8,
    "monotonicity_tolerance": 1.0e-8,
    "spot_probe_relative": 1.0e-3,
    "volatility_probe_absolute": 1.0e-3,
}
PARITY_TOLERANCES: dict[str, float] = {
    "normalized_price_absolute": 1.0e-10,
    "normalized_price_relative": 1.0e-10,
    "physical_price_absolute": 1.0e-8,
    "physical_price_relative": 1.0e-10,
    "delta_autograd_absolute": 1.0e-7,
    "delta_autograd_relative": 1.0e-6,
}


@dataclass(frozen=True)
class OnnxArtifact:
    metadata_path: Path
    model_path: Path
    metadata: dict[str, Any]
    preprocessor: NeuralPreprocessor


def _require_exact_list(value: object, expected: list[str], name: str) -> None:
    if value != expected:
        raise ValueError(f"ONNX metadata {name} does not match the frozen contract")


def load_onnx_artifact(metadata_path: Path | str) -> OnnxArtifact:
    path = Path(metadata_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("ONNX metadata is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or value.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported ONNX artifact version")
    required = {
        "model_file",
        "model_fnv1a64",
        "input_name",
        "output_name",
        "input_dtype",
        "output_representation",
        "dynamic_batch",
        "opset",
        "feature_names",
        "preprocessing",
        "deployment_domain",
        "delta_policy",
        "guardrails",
        "parity_tolerances",
        "source_neural_artifact",
    }
    if not required.issubset(value):
        raise ValueError("ONNX metadata is missing a required contract field")
    if (
        value["input_name"] != INPUT_NAME
        or value["output_name"] != OUTPUT_NAME
        or value["input_dtype"] != "float64"
        or value["output_representation"] != "price_divided_by_strike"
        or value["dynamic_batch"] is not True
        or int(value["opset"]) != OPSET_VERSION
    ):
        raise ValueError("ONNX tensor contract does not match the supported version")
    _require_exact_list(value["feature_names"], list(BASE_FEATURE_NAMES), "feature order")
    if value["deployment_domain"] != DEPLOYMENT_DOMAIN:
        raise ValueError("ONNX deployment domain does not match the supported version")
    if value["delta_policy"] != DELTA_POLICY:
        raise ValueError("ONNX Delta policy does not match the supported version")
    if value["guardrails"] != GUARDRAILS:
        raise ValueError("ONNX guardrail policy does not match the supported version")
    if value["parity_tolerances"] != PARITY_TOLERANCES:
        raise ValueError("ONNX parity tolerances do not match the supported version")
    source = value["source_neural_artifact"]
    if not isinstance(source, dict) or not {
        "artifact_version",
        "canonical_tensor_sha256",
        "dataset",
        "experiment_config_fnv1a64",
        "metadata_fnv1a64",
        "model_version",
        "state_file_fnv1a64",
    }.issubset(source):
        raise ValueError("ONNX source provenance is incomplete")
    preprocessor_value = value["preprocessing"]
    if not isinstance(preprocessor_value, dict):
        raise ValueError("ONNX preprocessing metadata must be an object")
    preprocessor = NeuralPreprocessor.from_dict(preprocessor_value)
    model_path = path.with_name(str(value["model_file"]))
    if not model_path.is_file() or fnv1a64_file(model_path) != value["model_fnv1a64"]:
        raise ValueError("ONNX model checksum does not match metadata")
    try:
        exported = onnx.load(model_path)
        onnx.checker.check_model(exported)
    except (OSError, onnx.checker.ValidationError) as error:
        raise ValueError("ONNX model is not a valid graph") from error
    if (
        len(exported.graph.input) != 1
        or len(exported.graph.output) != 1
        or exported.graph.input[0].name != INPUT_NAME
        or exported.graph.output[0].name != OUTPUT_NAME
        or exported.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.DOUBLE
    ):
        raise ValueError("ONNX graph does not match the scalar-price tensor contract")
    return OnnxArtifact(path, model_path, value, preprocessor)


def export_frozen_model(
    neural_metadata_path: Path | str,
    onnx_path: Path | str,
    metadata_path: Path | str,
) -> dict[str, Any]:
    neural_path = Path(neural_metadata_path)
    model_path = Path(onnx_path)
    output_metadata_path = Path(metadata_path)
    model, preprocessor, neural = load_neural_artifact(neural_path)
    if neural.get("variant") != "derivative_supervised":
        raise ValueError("M9 export accepts only the frozen derivative-supervised model")
    if neural["model"].get("output_count") != 1:
        raise ValueError("M9 export requires exactly one scalar price output")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros((2, len(BASE_FEATURE_NAMES)), dtype=torch.float64)
    torch.onnx.export(
        model,
        example,
        model_path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=OPSET_VERSION,
        dynamo=False,
    )
    exported = onnx.load(model_path)
    onnx.checker.check_model(exported)
    if len(exported.graph.output) != 1:
        raise ValueError("exported graph must have exactly one output")
    value: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "source_implementation_commit": "PENDING",
        "model_file": model_path.name,
        "model_fnv1a64": fnv1a64_file(model_path),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "input_dtype": "float64",
        "output_representation": "price_divided_by_strike",
        "physical_price": "strike * normalized_price",
        "dynamic_batch": True,
        "opset": OPSET_VERSION,
        "feature_names": list(BASE_FEATURE_NAMES),
        "preprocessing": preprocessor.to_dict(),
        "deployment_domain": DEPLOYMENT_DOMAIN,
        "delta_policy": DELTA_POLICY,
        "guardrails": GUARDRAILS,
        "parity_tolerances": PARITY_TOLERANCES,
        "source_neural_artifact": {
            "metadata_file": neural_path.name,
            "metadata_fnv1a64": fnv1a64_file(neural_path),
            "artifact_version": neural["artifact_version"],
            "model_version": neural["model"]["model_version"],
            "state_file_fnv1a64": neural["state"]["file_fnv1a64"],
            "canonical_tensor_sha256": neural["state"]["canonical_tensor_sha256"],
            "experiment_config_fnv1a64": neural["experiment"]["config_fnv1a64"],
            "dataset": neural["dataset"],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "platform": platform.platform(),
            "exporter": "torch.onnx legacy TorchScript exporter with dynamo=false",
        },
        "limitations": [
            "The ONNX graph exports only scalar normalized price; it has no Delta head.",
            "Deployment Delta is a centered finite difference of the same price graph.",
            "The deployment domain is an engineering guardrail, not an accuracy guarantee.",
        ],
    }
    write_json(output_metadata_path, value)
    load_onnx_artifact(output_metadata_path)
    return value

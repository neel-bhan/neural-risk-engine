"""Predeclared PyTorch-to-ONNX parity and held-out evaluation protocol."""

from __future__ import annotations

import platform
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch

from nre_baseline.dataset import Dataset, LabelRow
from nre_baseline.metrics import PRICE_FLOOR, error_metrics
from nre_neural.model import price_and_delta

from .artifact import OnnxArtifact
from .inference import create_session, normalized_price, onnx_price_and_delta


def boundary_points() -> tuple[LabelRow, ...]:
    """A fixed non-label probe set spanning domain endpoints and all contract categories."""
    result: list[LabelRow] = []
    styles = ("european", "geometric_asian", "arithmetic_asian")
    for combination, (style, option_type) in enumerate(
        (pair for style in styles for pair in ((style, "call"), (style, "put")))
    ):
        for endpoint, low in enumerate((True, False)):
            result.append(
                LabelRow(
                    parameter_id=f"m9-boundary-{combination}-{endpoint}",
                    split="probe",
                    option_style=style,
                    option_type=option_type,
                    spot=60.0 if low else 140.0,
                    strike=140.0 if low else 60.0,
                    maturity_years=0.25 if low else 2.0,
                    volatility=0.05 if low else 0.60,
                    risk_free_rate=-0.02 if low else 0.10,
                    dividend_yield=0.08 if low else -0.01,
                    observations=1 if style == "european" else (2 if low else 52),
                    price=0.0,
                    delta=0.0,
                    raw={},
                )
            )
    return tuple(result)


def _errors(actual: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]) -> dict[str, float]:
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), 1.0e-12)
    return {
        "maximum_absolute": float(np.max(absolute, initial=0.0)),
        "maximum_relative": float(np.max(relative, initial=0.0)),
    }


def _passes(measured: dict[str, float], absolute: float, relative: float) -> bool:
    return measured["maximum_absolute"] <= absolute or measured["maximum_relative"] <= relative


def evaluate_parity(
    artifact: OnnxArtifact,
    neural_model: torch.nn.Module,
    dataset: Dataset,
) -> dict[str, Any]:
    if artifact.metadata["source_neural_artifact"]["dataset"] != {
        "schema_version": dataset.schema_version,
        "manifest_fnv1a64": dataset.manifest_checksum,
        "config_fnv1a64": dataset.config_checksum,
        "labels_fnv1a64": dataset.labels_checksum,
    }:
        raise ValueError("ONNX source provenance does not match evaluation dataset")
    session = create_session(artifact)
    test = dataset.by_split["test"]
    probes = boundary_points()
    all_rows = test + probes
    pytorch_price, pytorch_delta = price_and_delta(neural_model, artifact.preprocessor, all_rows)
    onnx_price, onnx_delta = onnx_price_and_delta(artifact, session, all_rows)
    strikes = np.asarray([row.strike for row in all_rows], dtype=np.float64)
    pytorch_normalized = pytorch_price / strikes
    onnx_normalized = normalized_price(artifact, session, all_rows)
    normalized_errors = _errors(onnx_normalized, pytorch_normalized)
    physical_errors = _errors(onnx_price, pytorch_price)
    delta_errors = _errors(onnx_delta, pytorch_delta)
    near_zero_reference_price_count = sum(row.price < PRICE_FLOOR for row in test)
    tolerance = artifact.metadata["parity_tolerances"]
    checks = {
        "normalized_price": _passes(
            normalized_errors,
            float(tolerance["normalized_price_absolute"]),
            float(tolerance["normalized_price_relative"]),
        ),
        "physical_price": _passes(
            physical_errors,
            float(tolerance["physical_price_absolute"]),
            float(tolerance["physical_price_relative"]),
        ),
        "deployment_delta_vs_pytorch_autograd": _passes(
            delta_errors,
            float(tolerance["delta_autograd_absolute"]),
            float(tolerance["delta_autograd_relative"]),
        ),
    }
    batch_checks: list[dict[str, object]] = []
    for size in (1, 2, 7, 32, len(all_rows)):
        selected = all_rows[:size]
        torch_batch, _ = price_and_delta(neural_model, artifact.preprocessor, selected)
        onnx_batch, _ = onnx_price_and_delta(artifact, session, selected)
        errors = _errors(onnx_batch, torch_batch)
        batch_checks.append(
            {
                "batch_size": size,
                "physical_price": errors,
                "within_tolerance": _passes(
                    errors,
                    float(tolerance["physical_price_absolute"]),
                    float(tolerance["physical_price_relative"]),
                ),
            }
        )
    checks["dynamic_batch_sizes"] = all(bool(item["within_tolerance"]) for item in batch_checks)
    if not all(checks.values()):
        raise ValueError(f"PyTorch/ONNX parity failed: {checks}")
    heldout_price, heldout_delta = onnx_price_and_delta(artifact, session, test)
    heldout_reference_price = np.asarray([row.price for row in test], dtype=np.float64)
    heldout_reference_delta = np.asarray([row.delta for row in test], dtype=np.float64)
    return {
        "result_version": "nre.onnx.python_parity.v1",
        "source_implementation_commit": "PENDING",
        "evaluation_policy": "frozen M8 held-out rows and fixed boundary probes; tolerances declared in artifact metadata before evaluation",
        "row_counts": {"heldout": len(test), "boundary_probes": len(probes), "total_parity": len(all_rows)},
        "coverage": {
            "styles": sorted({row.option_style for row in all_rows}),
            "types": sorted({row.option_type for row in all_rows}),
            "spot_minimum": min(row.spot for row in all_rows),
            "spot_maximum": max(row.spot for row in all_rows),
            "near_zero_reference_price_count": near_zero_reference_price_count,
            "boundary_low_spot_count": sum(row.spot == 60.0 for row in probes),
            "boundary_high_spot_count": sum(row.spot == 140.0 for row in probes),
        },
        "tolerances": tolerance,
        "parity": {
            "normalized_price": normalized_errors,
            "physical_price": physical_errors,
            "deployment_delta_vs_pytorch_autograd": delta_errors,
            "batch_checks": batch_checks,
            "checks": checks,
        },
        "heldout_onnx_metrics": error_metrics(
            heldout_price,
            heldout_reference_price,
            heldout_delta,
            heldout_reference_delta,
            PRICE_FLOOR,
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "provider": session.get_providers()[0],
            "platform": platform.platform(),
        },
        "limitations": [
            "Centered-bump Delta is derivative-consistent with the scalar price graph but is not runtime autograd.",
            "Held-out metrics inherit documented Monte Carlo label noise.",
            "In-domain parity and boundary probes do not establish out-of-distribution accuracy or formal no-arbitrage guarantees.",
        ],
    }

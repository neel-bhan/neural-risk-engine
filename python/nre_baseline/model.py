"""Deterministic multi-output polynomial ridge baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .dataset import LabelRow
from .metrics import PRICE_FLOOR, error_metrics
from .preprocessing import Preprocessor

DECLARED_DEGREES = (1, 2, 3)
DECLARED_RIDGE_ALPHAS = (1.0e-8, 1.0e-5, 1.0e-2, 1.0)


def targets(rows: tuple[LabelRow, ...]) -> npt.NDArray[np.float64]:
    return np.asarray([(row.price / row.strike, row.delta) for row in rows], dtype=np.float64)


@dataclass(frozen=True)
class PolynomialRidgeModel:
    preprocessor: Preprocessor
    coefficients: npt.NDArray[np.float64]
    alpha: float
    version: str = "nre.model.polynomial_ridge.v1"

    def predict(self, rows: tuple[LabelRow, ...] | list[LabelRow]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        design = self.preprocessor.transform(rows)
        normalized_price_and_delta = design @ self.coefficients
        strikes = np.asarray([row.strike for row in rows], dtype=np.float64)
        return normalized_price_and_delta[:, 0] * strikes, normalized_price_and_delta[:, 1]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "alpha": self.alpha,
            "target_representation": ["price_divided_by_strike", "separately_fitted_delta"],
            "delta_note": "Delta is a separate ridge target; it is not derived from the price output.",
            "preprocessor": self.preprocessor.to_dict(),
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PolynomialRidgeModel":
        if value.get("version") != "nre.model.polynomial_ridge.v1":
            raise ValueError("unsupported model version")
        preprocessor = Preprocessor.from_dict(value["preprocessor"])
        coefficients = np.asarray(value["coefficients"], dtype=np.float64)
        if coefficients.shape != (len(preprocessor.powers), 2) or not np.all(np.isfinite(coefficients)):
            raise ValueError("invalid coefficient matrix")
        alpha = float(value["alpha"])
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("invalid ridge alpha")
        return cls(preprocessor, coefficients, alpha)


def fit_model(training_rows: tuple[LabelRow, ...], degree: int, alpha: float) -> PolynomialRidgeModel:
    preprocessor = Preprocessor.fit(training_rows, degree)
    design = preprocessor.transform(training_rows)
    target = targets(training_rows)
    penalty = np.sqrt(alpha) * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    augmented_design = np.vstack((design, penalty))
    augmented_target = np.vstack((target, np.zeros((design.shape[1], 2), dtype=np.float64)))
    coefficients, _, _, _ = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)
    return PolynomialRidgeModel(preprocessor, coefficients, alpha)


def select_model(
    training_rows: tuple[LabelRow, ...], validation_rows: tuple[LabelRow, ...]
) -> tuple[PolynomialRidgeModel, list[dict[str, float]]]:
    reference_price = np.asarray([row.price for row in validation_rows], dtype=np.float64)
    reference_delta = np.asarray([row.delta for row in validation_rows], dtype=np.float64)
    candidates: list[tuple[tuple[float, float, int], PolynomialRidgeModel, dict[str, float]]] = []
    for degree in DECLARED_DEGREES:
        for alpha in DECLARED_RIDGE_ALPHAS:
            model = fit_model(training_rows, degree, alpha)
            price, delta = model.predict(validation_rows)
            metrics = error_metrics(price, reference_price, delta, reference_delta, PRICE_FLOOR)
            normalized = metrics["normalized_price_error"]
            record = {
                "degree": float(degree),
                "alpha": alpha,
                "median_normalized_price_error": float(normalized["median"]),
                "p99_normalized_price_error": float(normalized["p99"]),
                "delta_rmse": float(metrics["delta_rmse"]),
            }
            key = (record["median_normalized_price_error"], record["p99_normalized_price_error"], degree)
            candidates.append((key, model, record))
    candidates.sort(key=lambda candidate: candidate[0])
    return candidates[0][1], [candidate[2] for candidate in candidates]


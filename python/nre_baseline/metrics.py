"""Predeclared M7 price and Delta metrics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

PRICE_FLOOR = 1.0


def percentile(values: npt.NDArray[np.float64], quantile: float) -> float:
    if values.size == 0:
        raise ValueError("cannot summarize an empty metric")
    return float(np.percentile(values, quantile, method="linear"))


def error_metrics(
    predicted_price: npt.NDArray[np.float64],
    reference_price: npt.NDArray[np.float64],
    predicted_delta: npt.NDArray[np.float64],
    reference_delta: npt.NDArray[np.float64],
    price_floor: float = PRICE_FLOOR,
) -> dict[str, object]:
    if price_floor <= 0.0:
        raise ValueError("price floor must be positive")
    arrays = (predicted_price, reference_price, predicted_delta, reference_delta)
    if any(array.ndim != 1 for array in arrays) or len({array.size for array in arrays}) != 1:
        raise ValueError("metric arrays must be equally sized vectors")
    if predicted_price.size == 0 or any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("metric arrays must be non-empty and finite")
    absolute = np.abs(predicted_price - reference_price)
    normalized = absolute / np.maximum(reference_price, price_floor)
    delta_error = predicted_delta - reference_delta
    return {
        "count": int(predicted_price.size),
        "price_floor": price_floor,
        "normalized_price_error": {
            "median": percentile(normalized, 50.0),
            "p95": percentile(normalized, 95.0),
            "p99": percentile(normalized, 99.0),
            "maximum": float(normalized.max()),
        },
        "absolute_price_error": {
            "median": percentile(absolute, 50.0),
            "p95": percentile(absolute, 95.0),
            "p99": percentile(absolute, 99.0),
            "maximum": float(absolute.max()),
        },
        "delta_rmse": float(np.sqrt(np.mean(delta_error * delta_error))),
        "near_zero_reference_count": int(np.count_nonzero(reference_price < price_floor)),
    }


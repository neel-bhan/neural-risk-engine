"""Held-out metrics, slice analysis, and warmed batch timing."""

from __future__ import annotations

import platform
import time
from typing import Any

import numpy as np

from .dataset import LabelRow
from .metrics import PRICE_FLOOR, error_metrics
from .model import PolynomialRidgeModel


def evaluate_rows(model: PolynomialRidgeModel, rows: tuple[LabelRow, ...]) -> dict[str, object]:
    price, delta = model.predict(rows)
    reference_price = np.asarray([row.price for row in rows], dtype=np.float64)
    reference_delta = np.asarray([row.delta for row in rows], dtype=np.float64)
    return error_metrics(price, reference_price, delta, reference_delta, PRICE_FLOOR)


def slice_metrics(model: PolynomialRidgeModel, rows: tuple[LabelRow, ...]) -> dict[str, dict[str, object]]:
    slices: dict[str, tuple[LabelRow, ...]] = {}
    for style in ("european", "geometric_asian", "arithmetic_asian"):
        slices[f"style:{style}"] = tuple(row for row in rows if row.option_style == style)
    for option_type in ("call", "put"):
        slices[f"type:{option_type}"] = tuple(row for row in rows if row.option_type == option_type)
    slices["near_zero_price"] = tuple(row for row in rows if row.price < PRICE_FLOOR)
    # Held-out points do not own the exact deterministic endpoints, so a boundary slice is the
    # outer 10% of each predeclared M6 continuous range. These cutoffs are fixed from the manifest
    # domain, not selected after seeing prediction errors.
    slices["spot_or_strike_outer_10_percent"] = tuple(
        row
        for row in rows
        if row.spot <= 68.0 or row.spot >= 132.0 or row.strike <= 68.0 or row.strike >= 132.0
    )
    slices["maturity_outer_10_percent"] = tuple(
        row for row in rows if row.maturity_years <= 0.425 or row.maturity_years >= 1.825
    )
    slices["volatility_outer_10_percent"] = tuple(
        row for row in rows if row.volatility <= 0.105 or row.volatility >= 0.545
    )
    slices["rate_outer_10_percent"] = tuple(
        row for row in rows if row.risk_free_rate <= -0.008 or row.risk_free_rate >= 0.088
    )
    slices["dividend_outer_10_percent"] = tuple(
        row for row in rows if row.dividend_yield <= -0.001 or row.dividend_yield >= 0.071
    )
    result: dict[str, dict[str, object]] = {}
    for name, selected in slices.items():
        if selected:
            result[name] = evaluate_rows(model, selected)
        else:
            result[name] = {
                "count": 0,
                "status": "no accepted held-out points in predeclared slice",
            }
    return result


def time_inference(
    model: PolynomialRidgeModel,
    rows: tuple[LabelRow, ...],
    repetitions: int = 300,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot time an empty batch source")
    records: list[dict[str, object]] = []
    for requested_batch in (1, 32, 128, len(rows)):
        batch_size = min(requested_batch, len(rows))
        batch = tuple(rows[index % len(rows)] for index in range(batch_size))
        for _ in range(20):
            model.predict(batch)
        elapsed_us = np.empty(repetitions, dtype=np.float64)
        for repetition in range(repetitions):
            begin = time.perf_counter_ns()
            model.predict(batch)
            elapsed_us[repetition] = (time.perf_counter_ns() - begin) / 1000.0
        records.append(
            {
                "batch_size": batch_size,
                "repetitions": repetitions,
                "warmup_repetitions": 20,
                "median_microseconds": float(np.median(elapsed_us)),
                "empirical_p99_microseconds": float(np.percentile(elapsed_us, 99.0, method="linear")),
            }
        )
    return {
        "scope": "Python feature transformation plus NumPy matrix multiplication",
        "timer": "time.perf_counter_ns",
        "runtime": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "arm64 reported by platform",
        "batches": records,
    }

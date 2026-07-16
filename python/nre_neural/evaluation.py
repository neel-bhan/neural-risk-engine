"""Fair held-out comparison, failure slices, and warmed PyTorch timing."""

from __future__ import annotations

import platform
import time
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import torch

from nre_baseline.dataset import LabelRow
from nre_baseline.metrics import PRICE_FLOOR, error_metrics


PredictionFunction = Callable[
    [tuple[LabelRow, ...]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
]


def evaluate_predictions(
    rows: tuple[LabelRow, ...],
    price: npt.NDArray[np.float64],
    delta: npt.NDArray[np.float64],
) -> dict[str, object]:
    reference_price = np.asarray([row.price for row in rows], dtype=np.float64)
    reference_delta = np.asarray([row.delta for row in rows], dtype=np.float64)
    return error_metrics(price, reference_price, delta, reference_delta, PRICE_FLOOR)


def declared_slices(rows: tuple[LabelRow, ...]) -> dict[str, tuple[LabelRow, ...]]:
    slices: dict[str, tuple[LabelRow, ...]] = {}
    for style in ("european", "geometric_asian", "arithmetic_asian"):
        slices[f"style:{style}"] = tuple(row for row in rows if row.option_style == style)
    for option_type in ("call", "put"):
        slices[f"type:{option_type}"] = tuple(row for row in rows if row.option_type == option_type)
    slices["near_zero_price"] = tuple(row for row in rows if row.price < PRICE_FLOOR)
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
    return slices


def evaluate_model(predict: PredictionFunction, rows: tuple[LabelRow, ...]) -> dict[str, object]:
    price, delta = predict(rows)
    return evaluate_predictions(rows, price, delta)


def slice_metrics(predict: PredictionFunction, rows: tuple[LabelRow, ...]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, selected in declared_slices(rows).items():
        if selected:
            result[name] = evaluate_model(predict, selected)
        else:
            result[name] = {"count": 0, "status": "no accepted held-out points in predeclared slice"}
    return result


def failure_examples(
    predict: PredictionFunction, rows: tuple[LabelRow, ...], count: int = 5
) -> list[dict[str, object]]:
    price, delta = predict(rows)
    reference = np.asarray([row.price for row in rows], dtype=np.float64)
    normalized = np.abs(price - reference) / np.maximum(reference, PRICE_FLOOR)
    order = np.argsort(-normalized, kind="stable")[:count]
    return [
        {
            "parameter_id": rows[index].parameter_id,
            "option_style": rows[index].option_style,
            "option_type": rows[index].option_type,
            "spot": rows[index].spot,
            "strike": rows[index].strike,
            "reference_price": rows[index].price,
            "predicted_price": float(price[index]),
            "normalized_price_error": float(normalized[index]),
            "reference_delta": rows[index].delta,
            "predicted_delta": float(delta[index]),
        }
        for index in order
    ]


def time_inference(
    predict: PredictionFunction,
    rows: tuple[LabelRow, ...],
    repetitions: int = 300,
) -> dict[str, Any]:
    if not rows or repetitions <= 0:
        raise ValueError("timing requires rows and positive repetitions")
    records: list[dict[str, object]] = []
    for requested_batch in (1, 32, 128, len(rows)):
        batch_size = min(requested_batch, len(rows))
        batch = tuple(rows[index % len(rows)] for index in range(batch_size))
        for _ in range(20):
            predict(batch)
        elapsed = np.empty(repetitions, dtype=np.float64)
        for repetition in range(repetitions):
            begin = time.perf_counter_ns()
            predict(batch)
            elapsed[repetition] = (time.perf_counter_ns() - begin) / 1000.0
        records.append(
            {
                "batch_size": batch_size,
                "repetitions": repetitions,
                "warmup_repetitions": 20,
                "median_microseconds": float(np.median(elapsed)),
                "empirical_p99_microseconds": float(np.percentile(elapsed, 99.0, method="linear")),
            }
        )
    return {
        "scope": "Python row tensorization, PyTorch scalar-price forward, and autograd spot Delta",
        "timer": "time.perf_counter_ns",
        "device": "cpu",
        "dtype": "float64",
        "torch_threads": torch.get_num_threads(),
        "runtime": platform.python_version(),
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "arm64 reported by platform",
        "batches": records,
        "tail_note": "empirical p99 is a repeated local-run summary, not a production latency guarantee",
    }

"""Training-only feature scaling and deterministic polynomial expansion."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .dataset import LabelRow

BASE_FEATURE_NAMES = (
    "log_spot_over_strike",
    "maturity_years",
    "volatility",
    "risk_free_rate",
    "dividend_yield",
    "log_observations",
    "style_geometric_asian",
    "style_arithmetic_asian",
    "type_put",
)


def raw_features(rows: tuple[LabelRow, ...] | list[LabelRow]) -> npt.NDArray[np.float64]:
    result = np.empty((len(rows), len(BASE_FEATURE_NAMES)), dtype=np.float64)
    for index, row in enumerate(rows):
        result[index] = (
            math.log(row.spot / row.strike),
            row.maturity_years,
            row.volatility,
            row.risk_free_rate,
            row.dividend_yield,
            math.log(float(row.observations)),
            float(row.option_style == "geometric_asian"),
            float(row.option_style == "arithmetic_asian"),
            float(row.option_type == "put"),
        )
    return result


def polynomial_powers(feature_count: int, degree: int) -> tuple[tuple[int, ...], ...]:
    powers: list[tuple[int, ...]] = [tuple(0 for _ in range(feature_count))]
    for current_degree in range(1, degree + 1):
        for combination in itertools.combinations_with_replacement(range(feature_count), current_degree):
            power = [0] * feature_count
            for feature in combination:
                power[feature] += 1
            powers.append(tuple(power))
    return tuple(powers)


@dataclass(frozen=True)
class Preprocessor:
    means: npt.NDArray[np.float64]
    scales: npt.NDArray[np.float64]
    degree: int
    powers: tuple[tuple[int, ...], ...]
    version: str = "nre.preprocessing.polynomial.v1"

    @classmethod
    def fit(cls, training_rows: tuple[LabelRow, ...], degree: int) -> "Preprocessor":
        if not training_rows:
            raise ValueError("cannot fit preprocessing without training rows")
        if degree < 1 or degree > 3:
            raise ValueError("supported polynomial degrees are 1 through 3")
        features = raw_features(training_rows)
        means = features.mean(axis=0)
        scales = features.std(axis=0)
        scales = np.where(scales > 1.0e-12, scales, 1.0)
        return cls(means, scales, degree, polynomial_powers(features.shape[1], degree))

    def transform(self, rows: tuple[LabelRow, ...] | list[LabelRow]) -> npt.NDArray[np.float64]:
        scaled = (raw_features(rows) - self.means) / self.scales
        design = np.ones((len(rows), len(self.powers)), dtype=np.float64)
        for column, powers in enumerate(self.powers[1:], start=1):
            value = np.ones(len(rows), dtype=np.float64)
            for feature, exponent in enumerate(powers):
                if exponent:
                    value *= scaled[:, feature] ** exponent
            design[:, column] = value
        return design

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "base_feature_names": list(BASE_FEATURE_NAMES),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "degree": self.degree,
            "powers": [list(power) for power in self.powers],
            "fit_split": "train",
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Preprocessor":
        if value.get("version") != "nre.preprocessing.polynomial.v1":
            raise ValueError("unsupported preprocessing version")
        if value.get("base_feature_names") != list(BASE_FEATURE_NAMES):
            raise ValueError("preprocessing feature names do not match")
        means = np.asarray(value["means"], dtype=np.float64)
        scales = np.asarray(value["scales"], dtype=np.float64)
        powers = tuple(tuple(int(item) for item in power) for power in value["powers"])
        degree = int(value["degree"])
        if means.shape != (len(BASE_FEATURE_NAMES),) or scales.shape != means.shape:
            raise ValueError("invalid preprocessing dimensions")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("invalid preprocessing parameters")
        if powers != polynomial_powers(len(BASE_FEATURE_NAMES), degree):
            raise ValueError("invalid polynomial power ordering")
        return cls(means, scales, degree, powers)


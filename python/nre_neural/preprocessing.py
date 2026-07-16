"""Training-only preprocessing shared by training and future export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from nre_baseline.dataset import LabelRow
from nre_baseline.preprocessing import BASE_FEATURE_NAMES, raw_features


@dataclass(frozen=True)
class NeuralPreprocessor:
    means: npt.NDArray[np.float64]
    scales: npt.NDArray[np.float64]
    normalized_price_scale: float
    delta_scale: float
    version: str = "nre.preprocessing.neural.v1"

    @classmethod
    def fit(cls, training_rows: tuple[LabelRow, ...]) -> "NeuralPreprocessor":
        if not training_rows:
            raise ValueError("cannot fit neural preprocessing without training rows")
        features = raw_features(training_rows)
        means = features.mean(axis=0)
        scales = features.std(axis=0)
        scales = np.where(scales > 1.0e-12, scales, 1.0)
        normalized_prices = np.asarray(
            [row.price / row.strike for row in training_rows], dtype=np.float64
        )
        normalized_price_scale = float(normalized_prices.std())
        if not np.isfinite(normalized_price_scale) or normalized_price_scale <= 1.0e-12:
            normalized_price_scale = 1.0
        deltas = np.asarray([row.delta for row in training_rows], dtype=np.float64)
        delta_scale = float(deltas.std())
        if not np.isfinite(delta_scale) or delta_scale <= 1.0e-12:
            delta_scale = 1.0
        return cls(means, scales, normalized_price_scale, delta_scale)

    def tensors(self, rows: tuple[LabelRow, ...] | list[LabelRow]) -> dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("cannot tensorize an empty row collection")
        floating = torch.float64
        return {
            "spot": torch.tensor([row.spot for row in rows], dtype=floating),
            "strike": torch.tensor([row.strike for row in rows], dtype=floating),
            "maturity": torch.tensor([row.maturity_years for row in rows], dtype=floating),
            "volatility": torch.tensor([row.volatility for row in rows], dtype=floating),
            "rate": torch.tensor([row.risk_free_rate for row in rows], dtype=floating),
            "dividend": torch.tensor([row.dividend_yield for row in rows], dtype=floating),
            "observations": torch.tensor([row.observations for row in rows], dtype=floating),
            "style_geometric": torch.tensor(
                [row.option_style == "geometric_asian" for row in rows], dtype=floating
            ),
            "style_arithmetic": torch.tensor(
                [row.option_style == "arithmetic_asian" for row in rows], dtype=floating
            ),
            "type_put": torch.tensor([row.option_type == "put" for row in rows], dtype=floating),
            "price": torch.tensor([row.price for row in rows], dtype=floating),
            "delta": torch.tensor([row.delta for row in rows], dtype=floating),
        }

    def scaled_features(
        self, tensors: dict[str, torch.Tensor], spot: torch.Tensor
    ) -> torch.Tensor:
        raw = torch.stack(
            (
                torch.log(spot / tensors["strike"]),
                tensors["maturity"],
                tensors["volatility"],
                tensors["rate"],
                tensors["dividend"],
                torch.log(tensors["observations"]),
                tensors["style_geometric"],
                tensors["style_arithmetic"],
                tensors["type_put"],
            ),
            dim=1,
        )
        means = torch.as_tensor(self.means, dtype=raw.dtype, device=raw.device)
        scales = torch.as_tensor(self.scales, dtype=raw.dtype, device=raw.device)
        return (raw - means) / scales

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "base_feature_names": list(BASE_FEATURE_NAMES),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "normalized_price_scale": self.normalized_price_scale,
            "delta_scale": self.delta_scale,
            "fit_split": "train",
            "spot_chain_rule": "z0=(log(spot/strike)-mean0)/scale0; price=strike*network(z); Delta=dprice/dspot via autograd",
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "NeuralPreprocessor":
        if value.get("version") != "nre.preprocessing.neural.v1":
            raise ValueError("unsupported neural preprocessing version")
        if value.get("base_feature_names") != list(BASE_FEATURE_NAMES):
            raise ValueError("neural preprocessing feature names do not match")
        means = np.asarray(value["means"], dtype=np.float64)
        scales = np.asarray(value["scales"], dtype=np.float64)
        normalized_price_scale = float(value["normalized_price_scale"])
        delta_scale = float(value["delta_scale"])
        expected = (len(BASE_FEATURE_NAMES),)
        if means.shape != expected or scales.shape != expected:
            raise ValueError("invalid neural preprocessing dimensions")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("invalid neural preprocessing parameters")
        if not np.isfinite(normalized_price_scale) or normalized_price_scale <= 0.0:
            raise ValueError("invalid normalized-price scale")
        if not np.isfinite(delta_scale) or delta_scale <= 0.0:
            raise ValueError("invalid Delta scale")
        return cls(means, scales, normalized_price_scale, delta_scale)

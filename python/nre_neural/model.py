"""Scalar-price MLP and derivative-consistent inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from nre_baseline.dataset import LabelRow

from .preprocessing import NeuralPreprocessor


class ScalarPriceMLP(nn.Module):
    """A smooth MLP with exactly one normalized-price output and no Delta head."""

    def __init__(self, input_count: int, hidden_layers: Sequence[int]) -> None:
        super().__init__()
        if input_count <= 0 or not hidden_layers or any(width <= 0 for width in hidden_layers):
            raise ValueError("model dimensions must be positive")
        layers: list[nn.Module] = []
        previous = input_count
        for width in hidden_layers:
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.input_count = input_count
        self.hidden_layers = tuple(int(width) for width in hidden_layers)

    def forward(self, scaled_features: torch.Tensor) -> torch.Tensor:
        return self.network(scaled_features).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def torch_price_and_delta(
    model: nn.Module,
    preprocessor: NeuralPreprocessor,
    tensors: dict[str, torch.Tensor],
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    spot = tensors["spot"].detach().clone().requires_grad_(True)
    scaled = preprocessor.scaled_features(tensors, spot)
    normalized_price = model(scaled)
    if normalized_price.ndim != 1 or normalized_price.shape[0] != spot.shape[0]:
        raise ValueError("neural model must return one scalar price per row")
    price = tensors["strike"] * normalized_price
    delta = torch.autograd.grad(price.sum(), spot, create_graph=create_graph)[0]
    return price, delta


def price_and_delta(
    model: nn.Module,
    preprocessor: NeuralPreprocessor,
    rows: tuple[LabelRow, ...] | list[LabelRow],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    model.eval()
    tensors = preprocessor.tensors(rows)
    price, delta = torch_price_and_delta(model, preprocessor, tensors, create_graph=False)
    predicted_price = price.detach().cpu().numpy().astype(np.float64, copy=False)
    predicted_delta = delta.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.all(np.isfinite(predicted_price)) or not np.all(np.isfinite(predicted_delta)):
        raise ValueError("neural inference produced a non-finite value")
    return predicted_price, predicted_delta

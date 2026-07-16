"""M8 derivative-consistent PyTorch surrogate."""

from .artifacts import load_neural_artifact, save_neural_artifact
from .model import ScalarPriceMLP, price_and_delta
from .preprocessing import NeuralPreprocessor
from .training import ExperimentConfig, train_variant

__all__ = [
    "ExperimentConfig",
    "NeuralPreprocessor",
    "ScalarPriceMLP",
    "load_neural_artifact",
    "price_and_delta",
    "save_neural_artifact",
    "train_variant",
]

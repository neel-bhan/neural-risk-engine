from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nre_baseline.dataset import Dataset, LabelRow
from nre_baseline.preprocessing import BASE_FEATURE_NAMES, raw_features
from nre_neural.artifacts import (
    load_neural_artifact,
    save_neural_artifact,
    state_dict_sha256,
)
from nre_neural.model import ScalarPriceMLP, price_and_delta
from nre_neural.preprocessing import NeuralPreprocessor
from nre_neural.training import ExperimentConfig, TrainingResult, train_variant


def label(
    identifier: str,
    split: str,
    spot: float,
    strike: float = 100.0,
    option_type: str = "call",
) -> LabelRow:
    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    price = intrinsic + 3.0
    delta = 0.7 if option_type == "call" else -0.3
    return LabelRow(
        identifier,
        split,
        "european",
        option_type,
        spot,
        strike,
        0.5 + spot / 1000.0,
        0.15 + spot / 2000.0,
        0.03,
        0.01,
        1,
        price,
        delta,
        {},
    )


def tiny_config() -> ExperimentConfig:
    raw = {
        "optimizer": "Adam",
        "schedule": {"name": "StepLR", "step_size_epochs": 2, "gamma": 0.5},
        "price_loss": "test normalized price loss",
        "delta_loss": "test normalized Delta loss",
    }
    return ExperimentConfig(
        version="nre.neural.experiment.v1",
        seed=2026071608,
        hidden_layer_candidates=((4,),),
        weight_decay_candidates=(0.0,),
        learning_rate=0.01,
        schedule_step=2,
        schedule_gamma=0.5,
        batch_size=4,
        maximum_epochs=4,
        early_stopping_patience=2,
        minimum_epochs=2,
        price_floor=1.0,
        derivative_supervision_weight=1.0,
        raw=raw,
    )


class ScaledLogPrice(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0]


class NeuralModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = tuple(
            label(f"train-{index}", "train", 70.0 + 5.0 * index, option_type="call" if index % 2 == 0 else "put")
            for index in range(12)
        )
        self.validation = tuple(
            label(f"validation-{index}", "validation", 75.0 + 10.0 * index, option_type="call" if index % 2 == 0 else "put")
            for index in range(4)
        )

    def test_delta_is_autograd_of_price_with_full_spot_chain_rule(self) -> None:
        means = np.zeros(len(BASE_FEATURE_NAMES), dtype=np.float64)
        scales = np.ones(len(BASE_FEATURE_NAMES), dtype=np.float64)
        means[0] = 0.2
        scales[0] = 2.5
        preprocessor = NeuralPreprocessor(means, scales, 1.0, 1.0)
        row = label("chain", "test", 100.0, strike=80.0)
        price, delta = price_and_delta(ScaledLogPrice(), preprocessor, (row,))
        self.assertAlmostEqual(price[0], 80.0 * (np.log(1.25) - 0.2) / 2.5, places=12)
        self.assertAlmostEqual(delta[0], 0.8 / 2.5, places=12)

    def test_model_has_exactly_one_output_and_no_delta_head(self) -> None:
        model = ScalarPriceMLP(len(BASE_FEATURE_NAMES), (8, 4)).to(dtype=torch.float64)
        output = model(torch.zeros((3, len(BASE_FEATURE_NAMES)), dtype=torch.float64))
        self.assertEqual(output.shape, (3,))
        self.assertEqual(model.network[-1].out_features, 1)
        self.assertFalse(any("delta" in name.lower() for name, _ in model.named_parameters()))

    def test_preprocessing_is_fit_on_train_only(self) -> None:
        preprocessing = NeuralPreprocessor.fit(self.training)
        np.testing.assert_array_equal(preprocessing.means, raw_features(self.training).mean(axis=0))
        extreme_test = (label("unseen", "test", 1_000_000.0),)
        preprocessing.tensors(extreme_test)
        np.testing.assert_array_equal(preprocessing.means, raw_features(self.training).mean(axis=0))

    def test_training_seed_reproduces_exact_tensor_state(self) -> None:
        first = train_variant(self.training, self.validation, tiny_config(), "derivative_supervised")
        second = train_variant(self.training, self.validation, tiny_config(), "derivative_supervised")
        self.assertEqual(state_dict_sha256(first.model.state_dict()), state_dict_sha256(second.model.state_dict()))
        self.assertEqual(first.selected_candidate, second.selected_candidate)

    def test_training_rejects_test_rows_in_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "train and validation"):
            train_variant(self.training, (label("test", "test", 100.0),), tiny_config(), "price_only")

    def test_artifact_reload_preserves_price_and_autograd_delta(self) -> None:
        preprocessor = NeuralPreprocessor.fit(self.training)
        torch.manual_seed(9)
        model = ScalarPriceMLP(len(BASE_FEATURE_NAMES), (4,)).to(dtype=torch.float64)
        result = TrainingResult(
            variant="price_only",
            model=model,
            preprocessor=preprocessor,
            selected_candidate={"hidden_layers": [4]},
            validation_candidates=[],
            elapsed_seconds=0.0,
            seed=9,
        )
        dataset = Dataset(
            rows=self.training + self.validation,
            by_split={"train": self.training, "validation": self.validation, "test": (label("test", "test", 95.0),)},
            schema_version="nre.dataset.v1",
            manifest={},
            manifest_checksum="manifest",
            config_checksum="config",
            labels_checksum="labels",
            dataset_directory=Path("dataset"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment_path = root / "experiment.json"
            experiment_path.write_text("{}\n", encoding="utf-8")
            metadata = root / "model.json"
            state = root / "model.pt"
            save_neural_artifact(metadata, state, result, dataset, tiny_config(), experiment_path)
            restored, restored_preprocessing, _ = load_neural_artifact(metadata)
            expected = price_and_delta(model, preprocessor, self.validation)
            actual = price_and_delta(restored, restored_preprocessing, self.validation)
            np.testing.assert_array_equal(expected[0], actual[0])
            np.testing.assert_array_equal(expected[1], actual[1])
            self.assertTrue(np.all(np.isfinite(actual[0])))
            self.assertTrue(np.all(np.isfinite(actual[1])))


if __name__ == "__main__":
    unittest.main()

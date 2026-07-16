"""Deterministic validation-only model selection for both M8 ablations."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nre_baseline.dataset import LabelRow
from nre_baseline.metrics import PRICE_FLOOR, error_metrics
from nre_baseline.preprocessing import BASE_FEATURE_NAMES

from .model import ScalarPriceMLP, price_and_delta, torch_price_and_delta
from .preprocessing import NeuralPreprocessor


@dataclass(frozen=True)
class ExperimentConfig:
    version: str
    seed: int
    hidden_layer_candidates: tuple[tuple[int, ...], ...]
    weight_decay_candidates: tuple[float, ...]
    learning_rate: float
    schedule_step: int
    schedule_gamma: float
    batch_size: int
    maximum_epochs: int
    early_stopping_patience: int
    minimum_epochs: int
    price_floor: float
    derivative_supervision_weight: float
    raw: dict[str, Any]

    @classmethod
    def from_path(cls, path: Path | str) -> "ExperimentConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("config_version") != "nre.neural.experiment.v1":
            raise ValueError("unsupported neural experiment configuration")
        if value.get("device") != "cpu" or value.get("dtype") != "float64":
            raise ValueError("M8 reproducibility protocol requires CPU float64")
        if value.get("activation") != "tanh" or value.get("optimizer") != "Adam":
            raise ValueError("unsupported neural architecture or optimizer")
        schedule = value.get("schedule")
        if not isinstance(schedule, dict) or schedule.get("name") != "StepLR":
            raise ValueError("unsupported neural schedule")
        candidates = tuple(tuple(int(width) for width in item) for item in value["hidden_layer_candidates"])
        weight_decays = tuple(float(item) for item in value["weight_decay_candidates"])
        result = cls(
            version=str(value["config_version"]),
            seed=int(value["seed"]),
            hidden_layer_candidates=candidates,
            weight_decay_candidates=weight_decays,
            learning_rate=float(value["learning_rate"]),
            schedule_step=int(schedule["step_size_epochs"]),
            schedule_gamma=float(schedule["gamma"]),
            batch_size=int(value["batch_size"]),
            maximum_epochs=int(value["maximum_epochs"]),
            early_stopping_patience=int(value["early_stopping_patience"]),
            minimum_epochs=int(value["minimum_epochs"]),
            price_floor=float(value["price_floor"]),
            derivative_supervision_weight=float(value["derivative_supervision_weight"]),
            raw=value,
        )
        if (
            not result.hidden_layer_candidates
            or any(not item or any(width <= 0 for width in item) for item in result.hidden_layer_candidates)
            or not result.weight_decay_candidates
            or any(item < 0.0 for item in result.weight_decay_candidates)
            or result.learning_rate <= 0.0
            or result.schedule_step <= 0
            or not 0.0 < result.schedule_gamma <= 1.0
            or result.batch_size <= 0
            or result.maximum_epochs <= 0
            or result.minimum_epochs > result.maximum_epochs
            or result.early_stopping_patience <= 0
            or result.price_floor != PRICE_FLOOR
            or result.derivative_supervision_weight < 0.0
        ):
            raise ValueError("invalid neural experiment configuration value")
        return result


@dataclass
class TrainingResult:
    variant: str
    model: ScalarPriceMLP
    preprocessor: NeuralPreprocessor
    selected_candidate: dict[str, object]
    validation_candidates: list[dict[str, object]]
    elapsed_seconds: float
    seed: int


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _clone_state(model: ScalarPriceMLP) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _batch_loss(
    model: ScalarPriceMLP,
    preprocessor: NeuralPreprocessor,
    rows: tuple[LabelRow, ...],
    price_floor: float,
    delta_weight: float,
) -> torch.Tensor:
    tensors = preprocessor.tensors(rows)
    if delta_weight > 0.0:
        price, delta = torch_price_and_delta(model, preprocessor, tensors, create_graph=True)
    else:
        scaled = preprocessor.scaled_features(tensors, tensors["spot"])
        price = tensors["strike"] * model(scaled)
        delta = None
    del price_floor
    normalized_error = (price - tensors["price"]) / tensors["strike"]
    price_loss = torch.mean((normalized_error / preprocessor.normalized_price_scale) ** 2)
    if delta is None:
        return price_loss
    delta_loss = torch.mean(((delta - tensors["delta"]) / preprocessor.delta_scale) ** 2)
    return price_loss + delta_weight * delta_loss


def _validation_record(
    model: ScalarPriceMLP,
    preprocessor: NeuralPreprocessor,
    rows: tuple[LabelRow, ...],
    price_floor: float,
) -> dict[str, float]:
    price, delta = price_and_delta(model, preprocessor, rows)
    reference_price = np.asarray([row.price for row in rows], dtype=np.float64)
    reference_delta = np.asarray([row.delta for row in rows], dtype=np.float64)
    metrics = error_metrics(price, reference_price, delta, reference_delta, price_floor)
    normalized = metrics["normalized_price_error"]
    return {
        "median_normalized_price_error": float(normalized["median"]),
        "p99_normalized_price_error": float(normalized["p99"]),
        "delta_rmse": float(metrics["delta_rmse"]),
    }


def _selection_key(metrics: dict[str, float], parameter_count: int) -> tuple[float, float, float, int]:
    return (
        metrics["median_normalized_price_error"],
        metrics["p99_normalized_price_error"],
        metrics["delta_rmse"],
        parameter_count,
    )


def train_variant(
    training_rows: tuple[LabelRow, ...],
    validation_rows: tuple[LabelRow, ...],
    config: ExperimentConfig,
    variant: str,
) -> TrainingResult:
    if variant not in ("price_only", "derivative_supervised"):
        raise ValueError("unknown neural training variant")
    if not training_rows or not validation_rows:
        raise ValueError("training and validation rows must be non-empty")
    if any(row.split != "train" for row in training_rows) or any(
        row.split != "validation" for row in validation_rows
    ):
        raise ValueError("training selection accepts only train and validation rows")
    preprocessor = NeuralPreprocessor.fit(training_rows)
    delta_weight = config.derivative_supervision_weight if variant == "derivative_supervised" else 0.0
    started = time.perf_counter()
    candidate_results: list[tuple[tuple[float, float, float, int], dict[str, torch.Tensor], dict[str, object]]] = []
    candidate_index = 0
    for hidden_layers in config.hidden_layer_candidates:
        for weight_decay in config.weight_decay_candidates:
            candidate_seed = config.seed + candidate_index
            configure_determinism(candidate_seed)
            model = ScalarPriceMLP(len(BASE_FEATURE_NAMES), hidden_layers).to(dtype=torch.float64)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate, weight_decay=weight_decay
            )
            schedule = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.schedule_step, gamma=config.schedule_gamma
            )
            permutation_generator = torch.Generator().manual_seed(candidate_seed + 100_000)
            best_key: tuple[float, float, float, int] | None = None
            best_state: dict[str, torch.Tensor] | None = None
            best_metrics: dict[str, float] | None = None
            best_epoch = 0
            epochs_without_improvement = 0
            completed_epoch = 0
            for epoch in range(1, config.maximum_epochs + 1):
                model.train()
                order = torch.randperm(len(training_rows), generator=permutation_generator).tolist()
                for offset in range(0, len(order), config.batch_size):
                    selected = order[offset : offset + config.batch_size]
                    batch = tuple(training_rows[index] for index in selected)
                    optimizer.zero_grad(set_to_none=True)
                    loss = _batch_loss(model, preprocessor, batch, config.price_floor, delta_weight)
                    if not torch.isfinite(loss):
                        raise ValueError("training produced a non-finite loss")
                    loss.backward()
                    optimizer.step()
                schedule.step()
                completed_epoch = epoch
                metrics = _validation_record(model, preprocessor, validation_rows, config.price_floor)
                key = _selection_key(metrics, model.parameter_count)
                if best_key is None or key < best_key:
                    best_key = key
                    best_state = _clone_state(model)
                    best_metrics = metrics
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epoch >= config.minimum_epochs and epochs_without_improvement >= config.early_stopping_patience:
                    break
            if best_state is None or best_key is None or best_metrics is None:
                raise RuntimeError("candidate training did not produce a checkpoint")
            record: dict[str, object] = {
                "candidate_index": candidate_index,
                "seed": candidate_seed,
                "hidden_layers": list(hidden_layers),
                "weight_decay": weight_decay,
                "parameter_count": model.parameter_count,
                "best_epoch": best_epoch,
                "epochs_completed": completed_epoch,
                "validation_metrics": best_metrics,
            }
            candidate_results.append((best_key, best_state, record))
            candidate_index += 1
    candidate_results.sort(key=lambda item: item[0])
    _, selected_state, selected_record = candidate_results[0]
    selected_layers = tuple(int(item) for item in selected_record["hidden_layers"])
    selected_model = ScalarPriceMLP(len(BASE_FEATURE_NAMES), selected_layers).to(dtype=torch.float64)
    selected_model.load_state_dict(selected_state)
    selected_model.eval()
    return TrainingResult(
        variant=variant,
        model=selected_model,
        preprocessor=preprocessor,
        selected_candidate=copy.deepcopy(selected_record),
        validation_candidates=[copy.deepcopy(item[2]) for item in candidate_results],
        elapsed_seconds=time.perf_counter() - started,
        seed=config.seed,
    )

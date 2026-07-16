"""M8 training, reproducibility, and sealed held-out evaluation commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nre_baseline.artifacts import load_model_artifact, write_json
from nre_baseline.checksums import fnv1a64_file
from nre_baseline.dataset import load_dataset

from .artifacts import dataset_provenance, load_neural_artifact, save_neural_artifact
from .evaluation import evaluate_model, failure_examples, slice_metrics, time_inference
from .model import price_and_delta
from .training import ExperimentConfig, train_variant


ARTIFACT_NAMES = {
    "price_only": "price-only-v1",
    "derivative_supervised": "derivative-supervised-v1",
}


def train(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset, args.config)
    experiment_path = Path(args.experiment)
    experiment = ExperimentConfig.from_path(experiment_path)
    output = Path(args.output_dir)
    summaries: dict[str, object] = {}
    for variant, name in ARTIFACT_NAMES.items():
        result = train_variant(
            dataset.by_split["train"], dataset.by_split["validation"], experiment, variant
        )
        metadata_path = output / f"{name}.json"
        state_path = output / f"{name}.pt"
        metadata = save_neural_artifact(
            metadata_path, state_path, result, dataset, experiment, experiment_path
        )
        summaries[variant] = {
            "metadata": str(metadata_path),
            "state": str(state_path),
            "canonical_tensor_sha256": metadata["state"]["canonical_tensor_sha256"],
            "selected_candidate": metadata["selected_candidate"],
            "elapsed_training_seconds": metadata["experiment"]["elapsed_training_seconds"],
        }
    print(json.dumps({"status": "trained", "variants": summaries}, sort_keys=True))


def _verify_dataset(metadata: dict[str, object], expected: dict[str, str]) -> None:
    if metadata.get("dataset") != expected:
        raise ValueError("neural artifact provenance does not match evaluation dataset")


def evaluate(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset, args.config)
    expected = dataset_provenance(dataset)
    test_rows = dataset.by_split["test"]
    models: dict[str, tuple[object, object, dict[str, object]]] = {}
    for variant, metadata_path in (
        ("price_only", args.price_only),
        ("derivative_supervised", args.derivative_supervised),
    ):
        model, preprocessor, metadata = load_neural_artifact(metadata_path)
        _verify_dataset(metadata, expected)
        models[variant] = (model, preprocessor, metadata)

    baseline_model, baseline_artifact = load_model_artifact(Path(args.baseline))
    if baseline_artifact.get("dataset") != expected:
        raise ValueError("M7 baseline provenance does not match evaluation dataset")
    predictors = {
        variant: (lambda rows, model=model, preprocessor=preprocessor: price_and_delta(model, preprocessor, rows))
        for variant, (model, preprocessor, _) in models.items()
    }
    predictors["m7_polynomial_ridge"] = baseline_model.predict
    comparisons: dict[str, object] = {}
    for name, predict in predictors.items():
        comparison: dict[str, object] = {
            "test_metrics": evaluate_model(predict, test_rows),
            "slices": slice_metrics(predict, test_rows),
            "worst_failure_examples": failure_examples(predict, test_rows),
        }
        if name != "m7_polynomial_ridge":
            comparison["timing"] = time_inference(predict, test_rows)
        comparisons[name] = comparison
    result = {
        "result_version": "nre.neural.evaluation.v1",
        "source_implementation_commit": "PENDING",
        "dataset": expected,
        "accepted_split_counts": {split: len(dataset.by_split[split]) for split in dataset.by_split},
        "test_access_policy": "one final held-out evaluation after validation-only selection; no post-test tuning",
        "primary_comparison_metric": "held-out median normalized price error with fixed one-currency-unit floor",
        "artifacts": {
            variant: {
                "metadata_fnv1a64": fnv1a64_file(Path(path)),
                "canonical_tensor_sha256": models[variant][2]["state"]["canonical_tensor_sha256"],
            }
            for variant, path in (
                ("price_only", args.price_only),
                ("derivative_supervised", args.derivative_supervised),
            )
        }
        | {"m7_polynomial_ridge_fnv1a64": fnv1a64_file(Path(args.baseline))},
        "comparisons": comparisons,
        "limitations": [
            "The held-out labels retain documented Monte Carlo sampling noise.",
            "The test set contains 180 accepted points and some predeclared boundary slices are empty.",
            "This is in-domain held-out evidence only; it supports no out-of-distribution claim.",
            "No formal no-arbitrage guarantee, calibrated confidence, C++ inference, or fallback exists in M8.",
            "Python CPU timings include tensorization and autograd and are not low-latency claims.",
        ],
    }
    write_json(Path(args.output), result)
    print(json.dumps({"status": "evaluated", "output": args.output, "test_rows": len(test_rows)}, sort_keys=True))


def compare_reproduction(args: argparse.Namespace) -> None:
    differences: list[str] = []
    for name in ARTIFACT_NAMES.values():
        reference = json.loads((Path(args.reference_dir) / f"{name}.json").read_text(encoding="utf-8"))
        reproduced = json.loads((Path(args.reproduction_dir) / f"{name}.json").read_text(encoding="utf-8"))
        for field in ("variant", "dataset", "model", "preprocessing", "selected_candidate", "validation_candidates"):
            if reference[field] != reproduced[field]:
                differences.append(f"{name}:{field}")
        if reference["state"]["canonical_tensor_sha256"] != reproduced["state"]["canonical_tensor_sha256"]:
            differences.append(f"{name}:canonical_tensor_sha256")
    if differences:
        raise ValueError(f"neural reproduction mismatch: {', '.join(differences)}")
    print(json.dumps({"status": "reproduced", "exact_fields": True, "variants": list(ARTIFACT_NAMES)}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    train_command = commands.add_parser("train")
    train_command.add_argument("--dataset", required=True)
    train_command.add_argument("--config", required=True)
    train_command.add_argument("--experiment", required=True)
    train_command.add_argument("--output-dir", required=True)
    train_command.set_defaults(function=train)

    evaluate_command = commands.add_parser("evaluate")
    evaluate_command.add_argument("--dataset", required=True)
    evaluate_command.add_argument("--config", required=True)
    evaluate_command.add_argument("--price-only", required=True)
    evaluate_command.add_argument("--derivative-supervised", required=True)
    evaluate_command.add_argument("--baseline", required=True)
    evaluate_command.add_argument("--output", required=True)
    evaluate_command.set_defaults(function=evaluate)

    compare_command = commands.add_parser("compare-reproduction")
    compare_command.add_argument("--reference-dir", required=True)
    compare_command.add_argument("--reproduction-dir", required=True)
    compare_command.set_defaults(function=compare_reproduction)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

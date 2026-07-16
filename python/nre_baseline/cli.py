"""Command-line entry points for M7 training and one-time test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import load_model_artifact, save_model_artifact, write_json
from .checksums import fnv1a64_file
from .dataset import load_dataset
from .evaluation import evaluate_rows, slice_metrics, time_inference
from .model import select_model


def train(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset, args.config)
    model, candidates = select_model(dataset.by_split["train"], dataset.by_split["validation"])
    checksum = save_model_artifact(Path(args.artifact), model, dataset, candidates)
    result = {
        "status": "trained",
        "artifact_fnv1a64": checksum,
        "degree": model.preprocessor.degree,
        "alpha": model.alpha,
        "accepted_split_counts": {split: len(dataset.by_split[split]) for split in dataset.by_split},
        "dataset_manifest_fnv1a64": dataset.manifest_checksum,
        "dataset_labels_fnv1a64": dataset.labels_checksum,
    }
    print(json.dumps(result, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset, args.config)
    artifact_path = Path(args.artifact)
    model, artifact = load_model_artifact(artifact_path)
    artifact_dataset = artifact["dataset"]
    expected = {
        "schema_version": dataset.schema_version,
        "manifest_fnv1a64": dataset.manifest_checksum,
        "config_fnv1a64": dataset.config_checksum,
        "labels_fnv1a64": dataset.labels_checksum,
    }
    if artifact_dataset != expected:
        raise ValueError("model artifact provenance does not match evaluation dataset")
    test_rows = dataset.by_split["test"]
    slices = slice_metrics(model, test_rows)
    result = {
        "result_version": "nre.baseline.evaluation.v1",
        "source_implementation_commit": "efeafd90af7814912567c59688e1e8dc624fdd1f",
        "artifact_fnv1a64": fnv1a64_file(artifact_path),
        "dataset": expected,
        "test_access_policy": "held-out evaluation only after validation selection; no post-test model tuning",
        "test_metrics": evaluate_rows(model, test_rows),
        "slices": slices,
        "timing": time_inference(model, test_rows),
        "limitations": [
            "Delta is fitted separately and is not the derivative of the price prediction.",
            "Normalized price error uses a fixed one-currency-unit floor and remains large for some near-zero options.",
            "Python timings include preprocessing and NumPy execution and are not low-latency claims.",
        ],
    }
    write_json(Path(args.output), result)
    print(json.dumps({"status": "evaluated", "output": args.output, "test_rows": len(test_rows)}, sort_keys=True))


def validate(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset, args.config)
    print(
        json.dumps(
            {
                "status": "valid",
                "accepted_split_counts": {split: len(dataset.by_split[split]) for split in dataset.by_split},
                "manifest_fnv1a64": dataset.manifest_checksum,
                "labels_fnv1a64": dataset.labels_checksum,
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for name, function in (("validate", validate), ("train", train), ("evaluate", evaluate)):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset", required=True)
        command.add_argument("--config", required=True)
        if name in ("train", "evaluate"):
            command.add_argument("--artifact", required=True)
        if name == "evaluate":
            command.add_argument("--output", required=True)
        command.set_defaults(function=function)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

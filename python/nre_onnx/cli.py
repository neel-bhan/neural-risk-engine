"""M9 frozen ONNX export and Python parity commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nre_baseline.artifacts import write_json
from nre_baseline.dataset import load_dataset
from nre_neural.artifacts import load_neural_artifact

from .artifact import export_frozen_model, load_onnx_artifact
from .evaluation import evaluate_parity


def export_command(args: argparse.Namespace) -> None:
    metadata = export_frozen_model(args.neural, args.model, args.metadata)
    print(
        json.dumps(
            {
                "status": "exported",
                "artifact_version": metadata["artifact_version"],
                "model_fnv1a64": metadata["model_fnv1a64"],
            },
            sort_keys=True,
        )
    )


def evaluate_command(args: argparse.Namespace) -> None:
    artifact = load_onnx_artifact(args.metadata)
    model, _, neural = load_neural_artifact(args.neural)
    if artifact.metadata["source_neural_artifact"]["canonical_tensor_sha256"] != neural["state"][
        "canonical_tensor_sha256"
    ]:
        raise ValueError("ONNX and PyTorch artifacts do not share the frozen state checksum")
    dataset = load_dataset(args.dataset, args.config)
    report = evaluate_parity(artifact, model, dataset)
    write_json(Path(args.output), report)
    print(json.dumps({"status": "evaluated", "output": args.output}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--neural", required=True)
    export_parser.add_argument("--model", required=True)
    export_parser.add_argument("--metadata", required=True)
    export_parser.set_defaults(function=export_command)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--metadata", required=True)
    evaluate_parser.add_argument("--neural", required=True)
    evaluate_parser.add_argument("--dataset", required=True)
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.set_defaults(function=evaluate_command)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

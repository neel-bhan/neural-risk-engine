"""Strict loader for the versioned M6 label schema."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checksums import fnv1a64_file

SCHEMA_VERSION = "nre.dataset.v1"
SPLITS = ("train", "validation", "test")
OPTION_STYLES = ("european", "geometric_asian", "arithmetic_asian")
OPTION_TYPES = ("call", "put")

REQUIRED_COLUMNS = (
    "schema_version",
    "parameter_id",
    "split",
    "included_for_training",
    "quality_status",
    "quality_flags",
    "option_style",
    "option_type",
    "spot",
    "strike",
    "maturity_years",
    "volatility",
    "risk_free_rate",
    "dividend_yield",
    "observations",
    "backend",
    "estimator",
    "label_tier",
    "price",
    "price_standard_error",
    "price_ci_95_lower",
    "price_ci_95_upper",
    "delta",
    "delta_standard_error",
    "delta_ci_95_lower",
    "delta_ci_95_upper",
    "effective_paths",
    "raw_paths",
    "pricing_seed",
    "requested_threads",
    "active_threads",
    "pilot_paths",
    "pilot_seed",
    "pilot_active_threads",
    "price_control_coefficient",
    "price_control_expectation",
    "price_control_applied",
    "delta_control_coefficient",
    "delta_control_expectation",
    "delta_control_applied",
    "analytical_price",
    "analytical_delta",
    "analytical_price_absolute_error",
    "analytical_delta_absolute_error",
)

REQUIRED_FLOATS = (
    "spot",
    "strike",
    "maturity_years",
    "volatility",
    "risk_free_rate",
    "dividend_yield",
    "price",
    "price_standard_error",
    "price_ci_95_lower",
    "price_ci_95_upper",
    "delta",
    "delta_standard_error",
    "delta_ci_95_lower",
    "delta_ci_95_upper",
)
OPTIONAL_FLOATS = (
    "price_control_coefficient",
    "price_control_expectation",
    "delta_control_coefficient",
    "delta_control_expectation",
    "analytical_price",
    "analytical_delta",
    "analytical_price_absolute_error",
    "analytical_delta_absolute_error",
)
REQUIRED_INTS = (
    "observations",
    "effective_paths",
    "raw_paths",
    "pricing_seed",
    "requested_threads",
    "active_threads",
)
OPTIONAL_INTS = ("pilot_paths", "pilot_seed", "pilot_active_threads")


class DatasetError(ValueError):
    """Raised when a dataset fails an M7 integrity or schema check."""


@dataclass(frozen=True)
class LabelRow:
    parameter_id: str
    split: str
    option_style: str
    option_type: str
    spot: float
    strike: float
    maturity_years: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float
    observations: int
    price: float
    delta: float
    raw: dict[str, str]


@dataclass(frozen=True)
class Dataset:
    rows: tuple[LabelRow, ...]
    by_split: dict[str, tuple[LabelRow, ...]]
    schema_version: str
    manifest: dict[str, Any]
    manifest_checksum: str
    config_checksum: str
    labels_checksum: str
    dataset_directory: Path


def _finite_float(value: str, field: str, parameter_id: str, *, optional: bool) -> None:
    if not value and optional:
        return
    try:
        parsed = float(value)
    except ValueError as error:
        raise DatasetError(f"{parameter_id}: {field} is not numeric") from error
    if not math.isfinite(parsed):
        raise DatasetError(f"{parameter_id}: {field} is not finite")


def _integer(value: str, field: str, parameter_id: str, *, optional: bool) -> None:
    if not value and optional:
        return
    try:
        parsed = int(value)
    except ValueError as error:
        raise DatasetError(f"{parameter_id}: {field} is not an integer") from error
    if parsed < 0:
        raise DatasetError(f"{parameter_id}: {field} is negative")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError("unsupported manifest schema version")
    for path in (
        ("config", "fnv1a64"),
        ("artifacts", "labels_fnv1a64"),
        ("counts", "total_rows"),
        ("counts", "train_rows"),
        ("counts", "validation_rows"),
        ("counts", "test_rows"),
    ):
        current: Any = manifest
        for key in path:
            if not isinstance(current, dict) or key not in current:
                raise DatasetError(f"manifest is missing {'.'.join(path)}")
            current = current[key]


def load_dataset(dataset_directory: Path | str, config_path: Path | str) -> Dataset:
    directory = Path(dataset_directory)
    labels_path = directory / "labels.csv"
    manifest_path = directory / "manifest.json"
    config = Path(config_path)
    for path in (labels_path, manifest_path, config):
        if not path.is_file():
            raise DatasetError(f"required file does not exist: {path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise DatasetError("manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest, dict):
        raise DatasetError("manifest root must be an object")
    _validate_manifest(manifest)

    config_checksum = fnv1a64_file(config)
    expected_config_checksum = str(manifest["config"]["fnv1a64"])
    if config_checksum != expected_config_checksum:
        raise DatasetError("configuration checksum does not match manifest")
    labels_checksum = fnv1a64_file(labels_path)
    expected_labels_checksum = str(manifest["artifacts"]["labels_fnv1a64"])
    if labels_checksum != expected_labels_checksum:
        raise DatasetError("label checksum does not match manifest")

    rows: list[LabelRow] = []
    all_ids: set[str] = set()
    split_ids: dict[str, set[str]] = {split: set() for split in SPLITS}
    with labels_path.open(newline="", encoding="utf-8") as labels_file:
        reader = csv.DictReader(labels_file)
        if reader.fieldnames != list(REQUIRED_COLUMNS):
            raise DatasetError("labels.csv columns do not exactly match nre.dataset.v1")
        for number, raw in enumerate(reader, start=2):
            parameter_id = raw["parameter_id"]
            if not parameter_id:
                raise DatasetError(f"row {number}: empty parameter_id")
            if parameter_id in all_ids:
                raise DatasetError(f"duplicate parameter_id: {parameter_id}")
            all_ids.add(parameter_id)
            if raw["schema_version"] != SCHEMA_VERSION:
                raise DatasetError(f"{parameter_id}: unsupported row schema version")
            split = raw["split"]
            if split not in SPLITS:
                raise DatasetError(f"{parameter_id}: unknown split")
            split_ids[split].add(parameter_id)
            if raw["included_for_training"] not in ("true", "false"):
                raise DatasetError(f"{parameter_id}: invalid included_for_training flag")
            if raw["option_style"] not in OPTION_STYLES:
                raise DatasetError(f"{parameter_id}: unknown option style")
            if raw["option_type"] not in OPTION_TYPES:
                raise DatasetError(f"{parameter_id}: unknown option type")
            for field in REQUIRED_FLOATS:
                _finite_float(raw[field], field, parameter_id, optional=False)
            for field in OPTIONAL_FLOATS:
                _finite_float(raw[field], field, parameter_id, optional=True)
            for field in REQUIRED_INTS:
                _integer(raw[field], field, parameter_id, optional=False)
            for field in OPTIONAL_INTS:
                _integer(raw[field], field, parameter_id, optional=True)
            if raw["price_control_applied"] not in ("", "true", "false") or raw[
                "delta_control_applied"
            ] not in ("", "true", "false"):
                raise DatasetError(f"{parameter_id}: invalid control application flag")
            if raw["included_for_training"] == "true":
                rows.append(
                    LabelRow(
                        parameter_id=parameter_id,
                        split=split,
                        option_style=raw["option_style"],
                        option_type=raw["option_type"],
                        spot=float(raw["spot"]),
                        strike=float(raw["strike"]),
                        maturity_years=float(raw["maturity_years"]),
                        volatility=float(raw["volatility"]),
                        risk_free_rate=float(raw["risk_free_rate"]),
                        dividend_yield=float(raw["dividend_yield"]),
                        observations=int(raw["observations"]),
                        price=float(raw["price"]),
                        delta=float(raw["delta"]),
                        raw=dict(raw),
                    )
                )

    if any(split_ids[left] & split_ids[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise DatasetError("parameter ids overlap across splits")
    counts = manifest["counts"]
    observed_counts = {split: len(split_ids[split]) for split in SPLITS}
    if int(counts["total_rows"]) != len(all_ids) or any(
        int(counts[f"{split}_rows"]) != observed_counts[split] for split in SPLITS
    ):
        raise DatasetError("manifest row counts do not match labels.csv")

    by_split = {split: tuple(row for row in rows if row.split == split) for split in SPLITS}
    if any(not by_split[split] for split in SPLITS):
        raise DatasetError("each split must contain at least one accepted row")
    return Dataset(
        rows=tuple(rows),
        by_split=by_split,
        schema_version=SCHEMA_VERSION,
        manifest=manifest,
        manifest_checksum=fnv1a64_file(manifest_path),
        config_checksum=config_checksum,
        labels_checksum=labels_checksum,
        dataset_directory=directory,
    )


"""ONNX Runtime price inference and derivative-consistent deployment Delta."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from nre_baseline.dataset import LabelRow
from nre_baseline.preprocessing import raw_features

from .artifact import OnnxArtifact


def scaled_features(
    artifact: OnnxArtifact, rows: tuple[LabelRow, ...] | list[LabelRow]
) -> npt.NDArray[np.float64]:
    if not rows:
        raise ValueError("cannot infer an empty batch")
    raw = raw_features(rows)
    result = (raw - artifact.preprocessor.means) / artifact.preprocessor.scales
    if not np.all(np.isfinite(result)):
        raise ValueError("preprocessing produced a non-finite feature")
    return np.ascontiguousarray(result, dtype=np.float64)


def create_session(artifact: OnnxArtifact) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(artifact.model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def normalized_price(
    artifact: OnnxArtifact,
    session: ort.InferenceSession,
    rows: tuple[LabelRow, ...] | list[LabelRow],
) -> npt.NDArray[np.float64]:
    features = scaled_features(artifact, rows)
    output = session.run(
        [str(artifact.metadata["output_name"])],
        {str(artifact.metadata["input_name"]): features},
    )[0]
    result = np.asarray(output, dtype=np.float64)
    if result.shape != (len(rows),):
        raise ValueError("ONNX model did not return one scalar price per row")
    if not np.all(np.isfinite(result)):
        raise ValueError("ONNX model returned a non-finite price")
    return result


def onnx_price_and_delta(
    artifact: OnnxArtifact,
    session: ort.InferenceSession,
    rows: tuple[LabelRow, ...] | list[LabelRow],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if not rows:
        raise ValueError("cannot infer an empty batch")
    base = tuple(rows)
    relative = float(artifact.metadata["delta_policy"]["relative_bump"])
    floor = float(artifact.metadata["delta_policy"]["absolute_floor"])
    bumps = np.asarray([max(relative * row.spot, floor) for row in base], dtype=np.float64)
    if np.any(~np.isfinite(bumps)) or any(bump <= 0.0 or bump >= row.spot for row, bump in zip(base, bumps)):
        raise ValueError("invalid centered spot bump")
    up = tuple(replace(row, spot=row.spot + float(bump)) for row, bump in zip(base, bumps))
    down = tuple(replace(row, spot=row.spot - float(bump)) for row, bump in zip(base, bumps))
    base_normalized = normalized_price(artifact, session, base)
    up_normalized = normalized_price(artifact, session, up)
    down_normalized = normalized_price(artifact, session, down)
    strikes = np.asarray([row.strike for row in base], dtype=np.float64)
    price = strikes * base_normalized
    delta = strikes * (up_normalized - down_normalized) / (2.0 * bumps)
    if not np.all(np.isfinite(price)) or not np.all(np.isfinite(delta)):
        raise ValueError("ONNX inference produced a non-finite result")
    return price, delta

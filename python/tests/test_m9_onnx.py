from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

from nre_baseline.dataset import LabelRow
from nre_neural.artifacts import load_neural_artifact
from nre_neural.model import price_and_delta
from nre_onnx.artifact import export_frozen_model, load_onnx_artifact
from nre_onnx.evaluation import boundary_points
from nre_onnx.inference import create_session, normalized_price, onnx_price_and_delta


ROOT = Path(__file__).resolve().parents[2]
NEURAL = ROOT / "models/m8/derivative-supervised-v1.json"


class OnnxExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.model_path = root / "scalar-price-v1.onnx"
        cls.metadata_path = root / "scalar-price-v1.json"
        export_frozen_model(NEURAL, cls.model_path, cls.metadata_path)
        cls.artifact = load_onnx_artifact(cls.metadata_path)
        cls.session = create_session(cls.artifact)
        cls.torch_model, _, _ = load_neural_artifact(NEURAL)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_graph_has_dynamic_float64_input_and_one_price_output(self) -> None:
        graph = onnx.load(self.model_path).graph
        self.assertEqual(len(graph.input), 1)
        self.assertEqual(len(graph.output), 1)
        self.assertEqual(graph.input[0].name, "scaled_features")
        self.assertEqual(graph.output[0].name, "normalized_price")
        self.assertEqual(graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.DOUBLE)
        self.assertEqual(graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")
        self.assertNotIn("delta", graph.output[0].name.lower())

    def test_dynamic_batches_match_pytorch_for_all_styles_and_types(self) -> None:
        rows = boundary_points()
        for size in (1, 2, 5, len(rows)):
            selected = rows[:size]
            torch_price, _ = price_and_delta(self.torch_model, self.artifact.preprocessor, selected)
            onnx_price, _ = onnx_price_and_delta(self.artifact, self.session, selected)
            np.testing.assert_allclose(onnx_price, torch_price, rtol=1.0e-10, atol=1.0e-8)

    def test_centered_spot_bump_matches_pytorch_autograd_at_boundaries(self) -> None:
        rows = boundary_points()
        _, torch_delta = price_and_delta(self.torch_model, self.artifact.preprocessor, rows)
        _, onnx_delta = onnx_price_and_delta(self.artifact, self.session, rows)
        np.testing.assert_allclose(onnx_delta, torch_delta, rtol=1.0e-6, atol=1.0e-7)

    def test_metadata_rejects_version_feature_order_and_checksum_mismatch(self) -> None:
        original = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        for name, mutation, message in (
            ("version", lambda value: value.update(artifact_version="wrong"), "version"),
            ("features", lambda value: value["feature_names"].reverse(), "feature order"),
            ("checksum", lambda value: value.update(model_fnv1a64="0000000000000000"), "checksum"),
            ("schema", lambda value: value.pop("delta_policy"), "required contract"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / "metadata.json"
                value = json.loads(json.dumps(original))
                mutation(value)
                shutil.copyfile(self.model_path, root / value["model_file"])
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_onnx_artifact(path)

    def test_non_finite_runtime_output_is_rejected(self) -> None:
        class NonFiniteSession:
            def run(self, *_: object) -> list[np.ndarray]:
                return [np.asarray([np.nan], dtype=np.float64)]

        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalized_price(self.artifact, NonFiniteSession(), boundary_points()[:1])  # type: ignore[arg-type]

    def test_export_rejects_price_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "derivative-supervised"):
                export_frozen_model(
                    ROOT / "models/m8/price-only-v1.json",
                    root / "wrong.onnx",
                    root / "wrong.json",
                )


if __name__ == "__main__":
    unittest.main()

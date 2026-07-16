# Versioned model artifacts

`m7/polynomial-ridge-v1.json` is the deterministic simple-surrogate baseline.

`m8/` contains two PyTorch state dictionaries and their JSON metadata:

- `price-only-v1` is trained without Delta loss but still derives reported Delta from price via
  automatic differentiation.
- `derivative-supervised-v1` adds normalized trusted Delta labels to the loss and still has only
  one scalar price output.

The JSON files record dataset/config checksums, training-only preprocessing, feature order,
architecture, selection candidates, environment, and canonical tensor SHA-256. The `.pt` files are
state dictionaries loaded with `weights_only=True`; they are not ONNX models. Generated
reproduction artifacts belong under ignored `models/generated/`.

`m9/scalar-price-v1.onnx` is the frozen derivative-supervised scalar-price graph.
`m9/scalar-price-v1.json` is its runtime contract: source checksums, exact preprocessing, tensor
names, feature order, domain, Delta bump, guardrail tolerances, and ONNX checksum. The graph has no
learned Delta output. Do not replace either file independently; loaders reject checksum or schema
mismatch.

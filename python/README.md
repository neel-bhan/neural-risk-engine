# Python workspace

Dataset generation remains in the dependency-free trusted C++ engine. M7 adds a strict loader and
a deterministic polynomial-ridge baseline using only pinned NumPy 2.3.5. PyTorch and ONNX remain
out of scope until M8 and M9.

Create the ignored environment and run the tests:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r python/requirements-m7.txt
make python-test
```

Python 3.11 through 3.14 is supported by the project metadata; the measured M7 run used Python
3.14.6 on Apple Silicon. Commands use `PYTHONPATH=python`, so an editable install is unnecessary.

## Baseline protocol

Generate the trusted labels, prove deterministic fitting, and perform the final test evaluation:

```bash
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' dataset-m7
make baseline-reproduce
make baseline-train
make baseline-evaluate
```

The loader accepts only `nre.dataset.v1`. It checks the manifest and configuration checksum, label
checksum, exact columns, schema, finite numeric fields, unique ids, split integrity, manifest row
counts, and quality inclusion flag. Rejected rows remain auditable but are never returned to model
fitting or evaluation.

Preprocessing is fit on accepted training rows only. Validation chooses from predeclared degrees
1/2/3 and ridge penalties `1e-8`, `1e-5`, `1e-2`, and `1`; the test split is used only by the final
evaluation command. The model predicts price divided by strike and Delta as two separate targets.
Its Delta is explicitly not a derivative of its price output.

The small JSON model artifact is `models/m7/polynomial-ridge-v1.json`. Generated labels and
reproduction copies remain ignored. The comparison-ready machine-readable result is
`benchmarks/m7-polynomial-ridge-v1.json`.

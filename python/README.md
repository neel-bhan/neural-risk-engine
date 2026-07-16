# Python workspace

Dataset generation remains in the dependency-free trusted C++ engine. M7 provides the strict
loader and deterministic polynomial-ridge baseline. M8 layers pinned PyTorch 2.13.0 onto the same
environment for the scalar-price neural ablations. ONNX remains out of scope until M9.

Create the ignored environment and run the tests:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r python/requirements-m8.txt
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

## Neural protocol

The frozen experiment is `python/config/m8-neural-v1.json`. Run both fair ablations, reproduce
their exact selected tensor state, and perform the sealed held-out comparison with:

```bash
make neural-train
make neural-reproduce
make neural-evaluate
```

The MLP has one `price/strike` output. Delta is `d(strike * output)/d(spot)` from PyTorch autograd,
including the log-moneyness preprocessing chain. Price-only and derivative-supervised models use
the same four capacity/weight-decay candidates and validation-only selection. State dictionaries
and export-ready JSON metadata live under `models/m8/`; machine-readable evidence is
`benchmarks/m8-neural-v1.json`. The training device is deterministic float64 CPU; Apple MPS was not
used or tested.

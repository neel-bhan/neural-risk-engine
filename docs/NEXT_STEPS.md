# M8 work queue — Derivative-supervised PyTorch model

M7 is complete with a strict versioned-data loader, training-only preprocessing, deterministic
polynomial-ridge artifact, separately fitted Delta baseline, and held-out/slice/timing evidence.

## Task 1 — Freeze the neural experiment

- Add a minimal pinned PyTorch environment compatible with the existing Python version and Apple
  Silicon; retain the M7 baseline commands and artifact unchanged.
- Reuse the M7 validated loader, exact dataset checksums, split policy, price floor, slice
  definitions, and final comparison protocol.
- Predeclare model capacity candidates, seeds, optimizer, training budget, early-stopping rule,
  price/Delta loss scaling, and validation selection metric before inspecting neural test results.

## Task 2 — Implement derivative-consistent models

- Build an MLP with one scalar price output and obtain Delta by automatic differentiation with
  respect to unscaled spot, accounting explicitly for preprocessing and price normalization.
- Train a price-only ablation and a derivative-supervised model using the same accepted training
  parameter points. Validation alone selects capacity/checkpoint and the test split remains sealed.
- Add deterministic tests for gradient units, batching, loss scaling, seed control, checkpoint round
  trips, preprocessing reuse, and finite outputs.

## Task 3 — Evaluate once and compare fairly

- Evaluate the frozen price-only and derivative-supervised checkpoints once on the same M7 test
  rows and slices; report median/p95/p99/maximum normalized and absolute price errors plus Delta RMSE.
- Compare both neural variants to the exact M7 artifact on the predeclared primary metric and expose
  near-zero, boundary, style, and type failures rather than retuning on them.
- Time warmed Python batch inference using the M7 timer/batch protocol, version checkpoints and
  machine-readable results, and document training cost, hardware, runtime, seeds, and limitations.

## Exit gate

M8 is complete only when price-only and derivative-supervised models share a fair protocol, neural
Delta is verified as the derivative of price output, held-out/slice evidence compares both with M7,
and all C++ and Python tests pass. If the neural model does not beat the baseline on the predeclared
metric, report that result and justify any distinct deployment tradeoff without changing the test
metric.

## Scope boundary

M8 does not add ONNX export/runtime, C++ neural inference, production guardrails/fallback, portfolio
scenarios, or matched-error speedup claims. Those remain M9 and M10.

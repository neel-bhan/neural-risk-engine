# M7 work queue — Simple surrogate baseline

M6 is complete: C++ generation produces versioned price/Delta labels, point-level splits, quality
flags, deterministic manifests, and byte-identical small-run reproduction. The implementation
commit field in `docs/M6_DATASET_GENERATION.md` and M6 configs remains `PENDING` for finalization by
the committing agent.

## Task 1 — Freeze the M7 environment and loader

- Create an isolated project Python environment with the smallest pinned dependency set needed for
  the chosen simple baseline; do not add PyTorch or ONNX yet.
- Validate schema version, manifest checksum, unique parameter ids, split disjointness, finite
  fields, and `included_for_training=true` before fitting.
- Fit preprocessing on training only and version its parameters.

## Task 2 — Train a declared simple baseline

- Predeclare polynomial regression or gradient boosting and its feature representation.
- Use only accepted training rows for fitting; validation may select declared hyperparameters and
  test remains untouched until final evaluation.
- Predict price and, if the model supports it honestly, Delta. Do not infer a Delta capability from
  a price-only baseline without implementing and validating the derivative.

## Task 3 — Evaluate and report

- Report median, p95, p99, and maximum normalized price error using a price floor declared before
  evaluation; include option-style/type and boundary slices.
- Report Delta RMSE only if the baseline emits Delta, and time warmed batched inference with batch
  size, repetitions, hardware, and timer stated.
- Version the command, seeds, dataset manifest checksum, preprocessing, model artifact, and
  machine-readable metrics. Keep generated models/results ignored.

## Scope boundary

M7 does not add derivative-supervised PyTorch training (M8), ONNX export/runtime (M9), neural
guardrails/fallback, portfolio scenarios, or fabricated comparisons. The simple baseline must be a
real benchmark for the later neural model, not an intentionally weak straw man.

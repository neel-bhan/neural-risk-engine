# M6 work queue — Reproducible dataset generation

M5 multithreading, profiling, strict builds, and the machine-specific performance exit gate are
complete. The source field in `docs/M5_PERFORMANCE.md` must be replaced with the final implementation
commit after this change is committed. The active milestone is **M6: reproducible dataset generation**.

## Task 1 — Freeze the dataset schema and parameter domain

- Version the feature, label, units, option-style, monitoring, engine, and estimator fields.
- Declare parameter ranges before sampling and include boundary and analytical subsets.
- Keep train/validation/test splits disjoint by parameter point.

## Task 2 — Add deterministic C++ generation

- Generate price and pathwise-Delta labels through the trusted backend-neutral interface.
- Record master seeds, thread policy, estimator, path counts, CI diagnostics, engine commit, and
  build configuration in a manifest.
- Write generated artifacts only under ignored `data/generated/` paths.

## Task 3 — Validate label quality and regeneration

- Cross-check European and geometric-Asian rows against analytical prices and Delta.
- Reject or flag rows whose reference confidence interval misses the declared label tolerance.
- Provide one small, fast regeneration command and a larger explicitly optional generation command.

## Scope boundary

M6 does not train a surrogate, add PyTorch or ONNX Runtime, claim dataset-scale numbers before a
measured run, or weaken the dependency-free C++ pricing path.

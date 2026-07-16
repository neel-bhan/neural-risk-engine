# M10 work queue — portfolio risk benchmark and final evidence

M9 is complete with a checksum-bound ONNX scalar-price graph, optional C++ ONNX Runtime backend,
cross-language price/Delta parity, engineering guardrails, reasoned Monte Carlo fallback, and
accepted/full-set held-out evidence. M10 measures the actual many-contract market-shock workload.

## Prerequisite — CI stabilization gate

- Run the complete default and optional builds with Apple Clang and real GNU GCC warnings-as-errors.
- Keep dependency-free `make check`, CMake/CTest, Python tests, ONNX parity, and artifact checksum
  regeneration green before adding portfolio code.
- Fix portability issues without weakening strict warning flags or hiding third-party diagnostics.

## Task 1 — Freeze the portfolio and scenario protocol

- Version a deterministic portfolio spanning all supported styles/types, strikes, maturities,
  observation counts, and declared in/out-of-domain cases.
- Version spot and volatility shock grids, request ordering, fallback Monte Carlo budgets/seeds,
  repetitions, warm-up, and timing scope before measuring.
- Define the matched-error comparison rule using held-out/portfolio reference error and confidence
  intervals; do not silently change the numerical problem between backends.

## Task 2 — Add portfolio repricing orchestration

- Build contiguous batched requests across contracts and shocks while preserving contract/scenario
  identity and output ordering.
- Run optimized Monte Carlo and guarded neural routing through the existing public boundaries.
- Aggregate price, Delta, neural acceptance, total fallback, and reason counters by portfolio,
  style/type, and scenario slice.

## Task 3 — Benchmark and profile

- Measure warm release builds with steady-clock median and empirical p99 latency, contracts/scenarios
  per second, and total workload size.
- Report neural versus Monte Carlo only at an explicitly matched measured error tolerance. Include
  fallback work in guarded-neural end-to-end timing.
- Profile the dominant paths before any final optimization; retain before/after evidence and exact
  compiler, flags, hardware, threads, seeds, paths, warm-ups, and repetitions.

## Task 4 — Validate numerical and routing evidence

- Report median/p99 normalized price error, Delta RMSE, confidence-interval context, neural
  acceptance/fallback rates, and fallback distribution for the full portfolio/scenario grid.
- Include near-zero prices, domain boundaries, out-of-domain requests, and worst examples. Keep
  accepted-neural metrics beside full routed metrics.
- Test deterministic scenario generation, mixed-batch ordering, reason totals, fallback equality,
  and machine-readable report schema.

## Task 5 — Final project polish

- Produce one concise final report linking every measured claim to a reproducible command and
  versioned result.
- Update README architecture/setup examples and fill resume placeholders only with measurements
  directly supported by tracked evidence.
- Run the full Apple Clang/GNU GCC, default/optional CMake, Python, export/parity, and benchmark
  verification matrix from a clean checkout.

## Exit gate

M10 is complete only when the versioned portfolio benchmark reports end-to-end guarded-neural and
optimized-Monte-Carlo results at a stated matched error tolerance, includes fallback cost and reason
rates, passes the full strict verification matrix, and supports every final README/resume number.

## Scope boundary

Do not claim production low latency, formal no-arbitrage, calibrated confidence, or general OOD
accuracy. Do not add new products, stochastic models, calibration, or trading-system features.

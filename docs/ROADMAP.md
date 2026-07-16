# Roadmap and milestone gates

Milestones are intentionally sequential. A later milestone may be explored in a branch, but it
must not be used to claim a result before its prerequisite gates pass.

## M0 — Foundation scaffold (complete)

**Learn:** how contract inputs and financial conventions determine the problem.

**Deliver:** C++20 build, domain types, validation, documentation, and a deterministic smoke test.

**Exit gate:** `make check` passes with warnings enabled; no pricing or performance claim is made.

## M1 — Analytical reference pricing (complete)

**Learn:** discounting, calls versus puts, normal CDF, Black-Scholes, Delta, and put-call parity.

**Deliver:** European Black-Scholes price and Delta, followed by the discrete geometric-Asian
formula using the exact monitoring schedule in `docs/CONVENTIONS.md`.

**Exit gate:** tests cover published/high-precision fixtures, call-put parity, sensible boundaries,
and finite-difference Delta. Edge-case behavior is documented.

## M2 — Correct scalar Monte Carlo (complete)

**Learn:** exact GBM time steps, pseudo-random sampling, online mean/variance, standard error, and
confidence intervals.

**Deliver:** one-thread scalar European and geometric-Asian simulation with a configurable seed and
path count. Separate random draws, path evolution, payoff, and aggregation.

**Exit gate:** convergence experiments across many seeds show expected `1/sqrt(N)` behavior and
approximately correct 95% interval coverage against both analytical prices. Test runtime stays
short; experiments live outside unit tests.

## M3 — Arithmetic Asian and variance reduction (complete)

**Learn:** arithmetic versus geometric averaging, covariance, antithetic pairs, and control
variates.

**Deliver:** arithmetic-Asian pricing, antithetic sampling, and a geometric-Asian control variate
whose coefficient is estimated without biasing the reported comparison.

**Exit gate:** matched-draw experiments report variance and time-to-target-CI for plain,
antithetic, and control-variate estimators. Degenerate covariance is handled safely.

## M4 — Delta and pricing interface (complete)

**Learn:** bump-and-revalue, common random numbers, and pathwise sensitivity estimators.

**Deliver:** a backend-neutral request/result API, Delta with a documented estimator, and result
diagnostics including confidence intervals and effective sample count.

**Exit gate:** Monte Carlo Delta agrees with analytical European/geometric Delta within statistical
tolerance and with finite differences using common random numbers.

## M5 — Multithreading and performance engineering (complete)

**Learn:** work partitioning, deterministic reduction, false sharing, profiling, compiler flags,
and throughput-versus-latency measurement.

**Deliver:** multithreaded Monte Carlo, reusable per-thread memory/state, a scalar baseline target,
profiling notes, and a reproducible benchmark harness. Add SIMD only if profiling supports it.

**Exit gate:** correctness is invariant across configured thread counts within documented floating-
point tolerance. Reports include paths/s, scaling, time to target CI, compiler/config, and hardware.

The implementation, profile evidence, release protocol, and machine-specific measurements are in
[`M5_PERFORMANCE.md`](M5_PERFORMANCE.md). Source commit finalization remains a mechanical metadata
update by the committing agent.

At this point the project is already a credible C++/numerical-computing portfolio piece. Do not wait
for ML before explaining or applying with it.

## M6 — Reproducible dataset generation (complete)

**Learn:** parameter-space design, data leakage, label noise, and schema/version management.

**Deliver:** C++-generated price and Delta labels, deterministic manifests, train/validation/test
splits by parameter point, and label-quality checks. Use tighter reference tolerances for held-out
evaluation than for bulk training if cost requires it, and record both.

**Exit gate:** a fresh command regenerates a small dataset; manifests capture engine commit and
configuration; analytical subsets pass cross-checks.

The schema, deterministic generator, quality policy, small-run measurements, and regeneration
evidence are in [`M6_DATASET_GENERATION.md`](M6_DATASET_GENERATION.md).

## M7 — Simple surrogate baseline (complete)

**Learn:** feature scaling, polynomial regression or gradient boosting, error distributions, and
slice-based evaluation.

**Deliver:** at least one simple model, timed batched inference, and a held-out report with median,
p95/p99, worst-slice price error, and Delta RMSE if the baseline supports Delta.

**Exit gate:** metrics and artifacts are reproducible. The neural model must beat this baseline on a
predeclared metric or justify a different deployment tradeoff.

The deterministic environment, strict loader, versioned artifact, one-time held-out evaluation,
slice failures, and warmed Python inference measurements are recorded in
[`M7_BASELINE.md`](M7_BASELINE.md). The implementation source commit remains a mechanical metadata
finalization by the committing agent.

## M8 — Derivative-supervised PyTorch model (complete)

**Learn:** automatic differentiation, multi-objective loss scaling, early stopping, and model
capacity selection.

**Deliver:** an MLP whose scalar price output supplies Delta through automatic differentiation;
training combines price and Delta supervision. Include price-only and derivative-supervised
ablations.

**Exit gate:** held-out and boundary-slice reports compare both neural variants and the simple
baseline. Seeds, model selection, and failure cases are recorded.

The frozen protocol, exact-reproduction evidence, held-out comparison, failures, and warmed Python
timing are recorded in [`M8_NEURAL_MODEL.md`](M8_NEURAL_MODEL.md). Source commit finalization remains
a mechanical metadata update by the committing agent.

## M9 — ONNX C++ acceleration with fallback (complete)

**Learn:** model export, preprocessing parity, batched inference, deployment domains, and runtime
policy.

**Deliver:** ONNX export, C++ inference backend, cross-language parity tests, domain/finite/bound/
monotonicity checks, reasoned counters, and Monte Carlo fallback.

**Exit gate:** Python and C++ outputs match within declared tolerance; all rejection reasons are
tested; accepted-set error and total fallback rate are reported together.

The frozen export contract, cross-language parity, guarded routing evidence, runtime setup, and
measured limitations are recorded in [`M9_ONNX_DEPLOYMENT.md`](M9_ONNX_DEPLOYMENT.md). Source
commit finalization remains a mechanical metadata update by the committing agent.

## M10 — Portfolio risk benchmark and final report (complete)

**Learn:** scenario grids, warm-up, median versus p99 latency, matched-error comparisons, and honest
performance reporting.

**Deliver:** many-contract spot/volatility shock repricing; Monte Carlo and guarded-neural runs;
machine-readable results and a concise project report.

**Exit gate:** report includes portfolio latency, throughput, price-error distribution, Delta RMSE,
neural acceptance/fallback rates, and neural speedup versus Monte Carlo at an explicitly matched
error tolerance.

The frozen workload, matched-error method, routing/error evidence, measured timing, reproducible
commands, and limitations are recorded in [`M10_FINAL_REPORT.md`](M10_FINAL_REPORT.md).

## Resume-ready finish line

Use these measured bullets with the experiment scope intact:

- Engineered a multithreaded C++20 Monte Carlo engine for arithmetic Asian options with antithetic
  and geometric control-variate estimators, sustaining 15.7M control-variate raw path evolutions/s
  at 10 threads (5.72x scalar latency) on Apple M4 and validating European/geometric convergence
  against analytical prices.
- Trained a derivative-supervised scalar-price PyTorch MLP on 840 accepted Monte Carlo-labeled
  parameter points, deriving Delta by autograd and reducing held-out p99 normalized price error
  from 8.68 to 1.74 and Delta RMSE from 0.121 to 0.0495 versus polynomial ridge.
- Embedded float64 ONNX Runtime inference in C++ with domain, finite-output, price-bound, sampled
  monotonicity checks, and reasoned Monte Carlo fallback; repriced 18 synthetic contracts across
  9 shocks in 15.2 ms median versus 31.6 ms for all Monte Carlo at the documented matched-error
  rule (58.6% neural acceptance, Apple M4).

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

## M4 — Delta and pricing interface

**Learn:** bump-and-revalue, common random numbers, and pathwise sensitivity estimators.

**Deliver:** a backend-neutral request/result API, Delta with a documented estimator, and result
diagnostics including confidence intervals and effective sample count.

**Exit gate:** Monte Carlo Delta agrees with analytical European/geometric Delta within statistical
tolerance and with finite differences using common random numbers.

## M5 — Multithreading and performance engineering

**Learn:** work partitioning, deterministic reduction, false sharing, profiling, compiler flags,
and throughput-versus-latency measurement.

**Deliver:** multithreaded Monte Carlo, reusable per-thread memory/state, a scalar baseline target,
profiling notes, and a reproducible benchmark harness. Add SIMD only if profiling supports it.

**Exit gate:** correctness is invariant across configured thread counts within documented floating-
point tolerance. Reports include paths/s, scaling, time to target CI, compiler/config, and hardware.

At this point the project is already a credible C++/numerical-computing portfolio piece. Do not wait
for ML before explaining or applying with it.

## M6 — Reproducible dataset generation

**Learn:** parameter-space design, data leakage, label noise, and schema/version management.

**Deliver:** C++-generated price and Delta labels, deterministic manifests, train/validation/test
splits by parameter point, and label-quality checks. Use tighter reference tolerances for held-out
evaluation than for bulk training if cost requires it, and record both.

**Exit gate:** a fresh command regenerates a small dataset; manifests capture engine commit and
configuration; analytical subsets pass cross-checks.

## M7 — Simple surrogate baseline

**Learn:** feature scaling, polynomial regression or gradient boosting, error distributions, and
slice-based evaluation.

**Deliver:** at least one simple model, timed batched inference, and a held-out report with median,
p95/p99, worst-slice price error, and Delta RMSE if the baseline supports Delta.

**Exit gate:** metrics and artifacts are reproducible. The neural model must beat this baseline on a
predeclared metric or justify a different deployment tradeoff.

## M8 — Derivative-supervised PyTorch model

**Learn:** automatic differentiation, multi-objective loss scaling, early stopping, and model
capacity selection.

**Deliver:** an MLP whose scalar price output supplies Delta through automatic differentiation;
training combines price and Delta supervision. Include price-only and derivative-supervised
ablations.

**Exit gate:** held-out and boundary-slice reports compare both neural variants and the simple
baseline. Seeds, model selection, and failure cases are recorded.

## M9 — ONNX C++ acceleration with fallback

**Learn:** model export, preprocessing parity, batched inference, deployment domains, and runtime
policy.

**Deliver:** ONNX export, C++ inference backend, cross-language parity tests, domain/finite/bound/
monotonicity checks, reasoned counters, and Monte Carlo fallback.

**Exit gate:** Python and C++ outputs match within declared tolerance; all rejection reasons are
tested; accepted-set error and total fallback rate are reported together.

## M10 — Portfolio risk benchmark and final report

**Learn:** scenario grids, warm-up, median versus p99 latency, matched-error comparisons, and honest
performance reporting.

**Deliver:** many-contract spot/volatility shock repricing; Monte Carlo and guarded-neural runs;
machine-readable results and a concise project report.

**Exit gate:** report includes portfolio latency, throughput, price-error distribution, Delta RMSE,
neural acceptance/fallback rates, and neural speedup versus Monte Carlo at an explicitly matched
error tolerance.

## Resume-ready finish line

Fill resume placeholders only from versioned benchmark summaries. Each bullet must link internally
to a command/report that establishes the number. A truthful intermediate resume can describe the
completed C++ engine after M5 and mark the neural backend as in progress.

Target template (not a current claim):

- Engineered a multithreaded C++20 Monte Carlo engine for arithmetic Asian options, sustaining
  `[N]M` paths/s (`[X]x` a single-thread scalar baseline) and validating convergence against
  analytical pricing benchmarks.
- Trained a derivative-supervised PyTorch neural surrogate on `[N]` Monte Carlo-labeled parameter
  sets, outperforming `[baseline]` while holding p99 pricing error below `[Y]` defined bps and Delta
  RMSE below `[Z]`.
- Embedded batched ONNX Runtime inference in the C++ risk engine with domain, price-bound, and
  monotonicity checks plus Monte Carlo fallback, repricing `[N]` contracts across `[M]` shocks in
  `[A]` ms versus `[B]` ms for optimized Monte Carlo.

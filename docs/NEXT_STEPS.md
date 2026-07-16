# M4 work queue — Delta and pricing interface

M3 is complete at implementation commit `1364d217a187a1a4d8c030bdaba4de0ff42fc2e6`.
Arithmetic-Asian calls and puts now have plain, antithetic, and independent-pilot geometric control
variate estimators; matched-draw measurements are recorded in `docs/M3_VARIANCE_REDUCTION.md`. The
active milestone is **M4: Delta and pricing interface**.

## Task 1 — Define the backend-neutral pricing request and result

- Introduce a request that owns a validated contract, market state, backend/estimator selection, and
  numerical configuration without making domain types depend on Monte Carlo.
- Define result diagnostics for price and Delta estimates, standard errors, confidence intervals,
  effective/raw samples, seeds, and fallback or estimator metadata where applicable.
- Preserve the existing style-specific analytical and Monte Carlo entry points while the router is
  introduced.
- Test unsupported backend/estimator combinations and stable metadata propagation.

**Done when:** callers can request supported analytical or Monte Carlo pricing through one C++
interface without weakening style validation.

## Task 2 — Add a scalar reference Delta estimator

- Select and document the reference estimator for each supported payoff, including behavior at
  payoff kinks.
- Reuse the same normal draws between price and Delta calculations and avoid new path-loop
  allocations.
- Use `double` throughout and retain the exact monitoring schedule for both Asian styles.
- Cover deterministic zero-volatility cases, supplied draws, call/put signs, and fixed-seed
  reproducibility.

**Done when:** every scalar Monte Carlo contract returns a price and Delta with separately defined
sampling diagnostics.

## Task 3 — Validate common-random-number bump-and-revalue

- Add centered spot bump-and-revalue as an independent validation estimator using common random
  numbers for up/down paths.
- Document bump size and a scale-aware rule for avoiding invalid down-spots.
- Compare the reference Delta estimator with common-random-number finite differences across
  representative European, geometric-Asian, and arithmetic-Asian cases.
- Keep stochastic convergence checks outside the short unit-test target.

**Done when:** deterministic fixtures catch draw mismatches and the two Monte Carlo Delta methods
agree within declared statistical tolerance.

## Task 4 — Close M4 with analytical and convergence checks

- Compare European and geometric-Asian Monte Carlo Delta against M1 analytical Delta over many
  seeds and increasing path counts.
- Report error, estimated uncertainty, and confidence-interval behavior with reproducible compiler,
  hardware, seed, path-count, and timing metadata.
- Add arithmetic-Asian finite-difference comparisons because it has no matching analytical formula.
- Update README, architecture, conventions, roadmap status, and a versioned M4 report only after the
  exit gate is supported.

**Done when:** the M4 exit criteria in `docs/ROADMAP.md` pass strict Make and CMake/CTest builds from
a clean checkout.

## Scope boundary

M4 remains single-threaded and scalar. Multithreading, SIMD, profiling-driven optimization,
PyTorch, ONNX, and external dependencies remain in later milestones.

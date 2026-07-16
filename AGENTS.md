# AGENTS.md

This file is the operating guide for coding agents working in this repository.

## Mission

Build a **C++ pricing and risk engine with an ML acceleration backend**. The trusted numerical
backend is C++20 Monte Carlo. PyTorch and ONNX may accelerate eligible batched requests, but they
do not replace the reference engine.

Keep the project credible for quant software engineering interviews. Prefer measured results,
reproducible experiments, clear assumptions, and understandable code over breadth or impressive
but unsupported claims.

## Current phase

The repository is in the foundation phase. Read these files before changing code:

1. `README.md` for the project entry point and commands.
2. `docs/CONVENTIONS.md` for the financial model and units.
3. `docs/ROADMAP.md` for milestone order and exit criteria.
4. `docs/NEXT_STEPS.md` for the immediate work queue.
5. `docs/ARCHITECTURE.md` before adding a new component or dependency.

Work on the earliest incomplete milestone unless the user explicitly changes priorities.

## Non-negotiable truthfulness rules

- Never invent benchmark results, error statistics, training-set sizes, or resume numbers.
- Label unmeasured targets as targets; label placeholders as `[N]`, `[X]`, and so on.
- Do not claim low latency, formal no-arbitrage guarantees, calibrated uncertainty, or reliable
  out-of-distribution behavior without evidence specifically supporting the claim.
- Do not describe a feature as complete until automated tests cover its important invariants.
- Store enough benchmark metadata to reproduce results: commit, compiler, flags, hardware, thread
  count, seed policy, path count, repetitions, and timing method.

## Engineering workflow

For each change:

1. State the assumption or convention the change relies on.
2. Make the smallest coherent implementation for the current milestone.
3. Add or update tests in the same change.
4. Run `make check`. If CMake is installed, also keep the CMake build working.
5. Update documentation when behavior, commands, scope, or milestone status changes.
6. Report what was measured separately from what is expected.

Do not add PyTorch, ONNX Runtime, a benchmarking framework, or another large dependency before its
roadmap milestone. Keep a dependency-free reference path available.

## Numerical rules

- Use `double` for reference pricing and risk calculations.
- Centralize tolerances and document why each tolerance is appropriate.
- Use fixed seeds in correctness tests. A fixed seed is for reproducibility, not proof of accuracy.
- Report Monte Carlo estimates with standard error and a 95% confidence interval.
- Prefer streaming statistics (for example, Welford's algorithm) over storing every payoff.
- Ensure variance-reduction comparisons use the same underlying random draws.
- Validate stochastic pricers against analytical cases and convergence behavior, not a single
  lucky seed.
- Keep random-number generation, path evolution, payoff calculation, and statistical aggregation
  separable enough to test independently.

## Performance rules

- Establish a correct single-thread scalar baseline before optimizing.
- Benchmark release builds outside unit tests; use a steady clock and repeated trials.
- Profile before optimizing. Record the profile and the before/after measurement.
- Never silently change the numerical problem to report a speedup.
- Avoid allocations in path loops. Prefer per-thread state and deterministic reductions.
- Treat SIMD as optional until profiling shows that the path loop is the right target.

## ML and deployment rules

- Generate labels with the trusted pricing engine and version the label-generation configuration.
- Split train/validation/test data by parameter points, not duplicate stochastic observations.
- Compare against at least one simple surrogate using the same held-out set.
- Derive neural Delta from the price output using automatic differentiation during training and
  evaluation; document loss scaling between price and Delta.
- Measure median, tail, and slice-based errors. An aggregate score alone is insufficient.
- The C++ caller owns domain, finite-value, price-bound, and monotonicity checks.
- Rejected neural requests must fall back to Monte Carlo and be counted.

## Repository boundaries

- `include/nre/`: public C++ interfaces.
- `src/`: C++ implementations and command-line programs.
- `tests/`: fast deterministic C++ tests.
- `python/`: later dataset, baseline, training, and export code.
- `docs/`: assumptions, architecture, plans, and experiment protocols.
- `benchmarks/`: benchmark programs and tracked summaries, not unreviewed raw output.
- `data/generated/`, `models/generated/`, `profiles/`, and `benchmarks/results/` are generated and
  ignored by Git.

Avoid circular dependencies. Domain types must not depend on Monte Carlo, ML, or CLI code. The
ONNX backend must implement the same pricing interface as Monte Carlo without making the core
engine depend on Python.

## Definition of done

A milestone is complete only when:

- its exit criteria in `docs/ROADMAP.md` are satisfied;
- `make check` passes from a clean checkout;
- public behavior and assumptions are documented;
- no generated binaries, datasets, models, or local machine paths are committed; and
- claimed metrics are produced by a reproducible command and clearly identified as measurements.


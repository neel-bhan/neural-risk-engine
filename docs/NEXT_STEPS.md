# Completed M2 work queue — correct scalar Monte Carlo

M1 analytical reference pricing and the M2 single-thread scalar Monte Carlo reference are complete.
The next active milestone is **M3: arithmetic Asian and variance reduction** in
`docs/ROADMAP.md`. The completed M2 sequence is retained below as an implementation record.

## Task 1 — Add streaming statistics (complete)

- Accumulate scalar `double` samples with Welford's online mean and squared-deviation update.
- Report the mean estimate, sample standard error, and the normal-approximation 95% confidence
  interval specified in `docs/CONVENTIONS.md`.
- Reject summary requests with fewer than two samples because sample variance is undefined.
- Test hand-calculated sequences, the two-sample boundary, constant samples, and empty/one-sample
  behavior.
- Register the component and its tests in both Make and CMake.

**Done when:** strict Make and CMake/CTest builds pass without warnings and no samples need to be
stored.

## Task 2 — Define deterministic random draws (complete)

- Select and document the standard-library pseudo-random engine, seed type, and normal transform.
- Keep uniform-engine state and normal draws behind a small interface that does not depend on
  contracts, payoffs, or statistics.
- Add deterministic tests appropriate to the chosen toolchain; do not treat a fixed sequence as a
  portability guarantee across different standard-library implementations.

**Done when:** a caller can reproduce a normal-draw stream from a documented seed and tests detect
accidental seed or draw-order changes.

## Task 3 — Implement exact scalar GBM evolution (complete)

- Implement the exact risk-neutral GBM step using `double` and the model in
  `docs/CONVENTIONS.md`.
- Keep path evolution independent of random-number generation by accepting supplied normal draws.
- Cover zero volatility, nonzero dividend yield, negative rates, and deterministic supplied draws.
- Evolve the exact observation schedule `t_i = iT/m`, excluding the initial spot.

**Done when:** deterministic path fixtures and the zero-volatility path agree with hand-calculated
values without allocating inside an observation loop.

## Task 4 — Add European and geometric-Asian payoffs (complete)

- Implement call and put payoffs separately from path evolution and aggregation.
- Compute the discrete geometric average on the documented observation schedule without storing
  every path payoff.
- Test in-, at-, and out-of-the-money values and geometric averages from small supplied paths.

**Done when:** payoff tests are deterministic and would catch call/put sign errors or accidental
inclusion of the initial spot.

## Task 5 — Compose the scalar Monte Carlo pricer (complete)

- Add a configuration containing a seed and path count, requiring at least two paths.
- Compose random draws, exact GBM evolution, payoff evaluation, discounting, and streaming
  statistics without merging their responsibilities.
- Price European and discrete geometric-Asian calls and puts on one thread.
- Return the estimate, sample standard error, 95% confidence interval, effective path count, and
  seed/configuration metadata required by `docs/CONVENTIONS.md`.

**Done when:** fixed-seed integration tests are reproducible and analytical edge cases such as zero
volatility agree to deterministic numerical tolerance.

## Task 6 — Validate convergence and interval coverage (complete)

- Add a reproducible experiment outside unit tests that compares both pricers with their M1
  analytical references across many seeds and increasing path counts.
- Record errors, interval inclusion, and enough configuration metadata to reproduce the run.
- Check whether error and interval width exhibit the expected `1/sqrt(N)` trend and report measured
  coverage; do not encode a single lucky seed or a long stochastic experiment as a unit test.
- Document the command and measured results separately from the expected 95% target.
- Keep the measured output and interpretation in `docs/M2_CONVERGENCE.md`.

**Done when:** M2's exit gate in `docs/ROADMAP.md` is supported by a short-test suite plus a
reproducible external convergence report.

## Scope boundary

M2 remains single-threaded and scalar. Random sampling and path simulation begin only in Tasks 2
and 3. Antithetic sampling, control variates, arithmetic-Asian pricing, threading, SIMD, ML, ONNX,
and external dependencies belong to later roadmap milestones.

# Completed M1 work queue

The analytical reference-pricing tasks for **M1** are complete. The next active milestone is
**M2: correct scalar Monte Carlo** in `docs/ROADMAP.md`.

## Task 1 — Define pricing result and analytical API (complete)

- Add `include/nre/analytics.hpp`.
- Define a small result type containing `price` and `delta`.
- Expose a European Black-Scholes function accepting validated `OptionContract` and `MarketState`.
- Decide and document whether the function rejects non-European input with an error result or an
  exception. Do not silently price the wrong style.
- Add the source to both Make and CMake builds.

**Done when:** the API compiles and invalid-style behavior has a test.

## Task 2 — Implement European Black-Scholes (complete)

- Implement the normal CDF with the C++ standard library.
- Implement call and put price and spot Delta with continuous dividend yield.
- Handle zero volatility explicitly using the discounted deterministic terminal spot.
- Keep near-zero maturity behavior consistent with the public validation contract.

**Done when:** independent high-precision fixtures pass, including a nonzero dividend yield and a
negative-rate case.

## Task 3 — Test invariants, not just examples (complete)

- Test put-call parity.
- Test call price bounds and put price bounds.
- Test monotonicity on a small deterministic grid: call price increases with spot; put price
  decreases with spot.
- Compare analytical Delta with a centered finite difference of analytical price.

**Done when:** tests would catch sign mistakes in discounting, dividend yield, and put Delta.

## Task 4 — Derive the geometric-Asian formula on paper first (complete)

- Derive the mean and variance of the average log spot for `t_i = iT/m`, excluding `t=0`.
- Put the derivation and final formula in `docs/GEOMETRIC_ASIAN_DERIVATION.md`.
- Check the formula numerically against a slow experimental simulation before making it a test
  oracle.

**Done when:** another reader can reproduce every discrete-monitoring coefficient from the doc.

## Task 5 — Implement and validate geometric Asian analytics (complete)

- Add geometric-Asian call/put price and Delta.
- Add high-precision fixtures produced independently of the C++ implementation.
- Check geometric put-call parity for the discounted expected geometric average.
- Check finite-difference Delta.

**Done when:** M1's exit gate in `docs/ROADMAP.md` passes and the roadmap status is updated.

## Suggested rhythm for a beginner

Use one task per session. Before asking an agent to implement it, ask for a two-minute explanation of
the formula and the tests that will detect common bugs. After implementation, read the relevant test
names and run `make check` yourself. Keep a short note of what you can now explain without the code;
that becomes interview preparation.

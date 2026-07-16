# Architecture

## Design goal

Maintain one trusted pricing contract while allowing different backends. Analytical formulas
provide validation cases, Monte Carlo is the general reference backend, and ONNX is an optional
batched accelerator guarded by the C++ engine.

## Intended component flow

```text
CLI / portfolio runner
        |
request validation and backend policy
        |-----------------------------|
        v                             v
Monte Carlo backend              ONNX backend
        |                             |
price + Delta + diagnostics       candidate price + autodiff-trained Delta
        ^                             |
        |                    domain / finite-value / bounds /
        |                    monotonicity checks and counters
        |                             |
        +--------- fallback <---------+
```

Analytical pricers are test oracles and control-variate inputs, not a universal backend. Python is
an offline training and evaluation environment. Production-style C++ inference must not embed a
Python interpreter.

## Planned C++ modules

- `domain`: contracts, market inputs, validation, stable units.
- `analytics`: Black-Scholes and discrete geometric-Asian reference formulas.
- `statistics`: streaming mean, variance, covariance, confidence intervals.
- `random`: seeded streams and normal draws.
- `monte_carlo`: path generation, payoffs, variance reduction, threading.
- `pricing`: backend-neutral request/result interface and routing policy.
- `onnx`: batched inference adapter, added only when ONNX Runtime is introduced.
- `risk`: portfolio scenarios, Delta results, acceptance/fallback counters.
- `benchmark`: isolated performance entry points; never part of correctness tests.

Dependencies should point downward from orchestration to small numerical components. In particular,
`domain` and `analytics` must not depend on Monte Carlo or ML.

## Planned Python modules

- Dataset generation orchestration and schema checks.
- Polynomial regression or gradient-boosting baseline.
- PyTorch model, derivative-supervised loss, evaluation, and export.
- Cross-language parity tests for preprocessing and ONNX output.

Feature scaling parameters and domain bounds are versioned model artifacts. Training and C++
inference must consume the same exported metadata instead of duplicating constants by hand.

## Guardrail policy

The final router may accept a neural result only when all configured checks pass:

1. Every input is inside the declared training/deployment domain.
2. Outputs are finite.
3. Price lies inside contract-specific lower and upper bounds, allowing a documented numerical
   tolerance.
4. Batched finite-difference probes satisfy configured spot/volatility monotonicity checks where
   applicable.

These are engineering checks, not a proof of no arbitrage. Every rejection increments a reasoned
counter and routes the request to Monte Carlo. The evaluation suite reports both accepted-set error
and overall fallback rate to prevent selective reporting.

## Dependency policy

The foundation and trusted scalar engine build with the C++ standard library. Add dependencies only
when a milestone needs them and document installation and version constraints then. Candidate later
dependencies include a test/benchmark framework and ONNX Runtime, but none is required yet.


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

## Analytical API error policy

Analytical pricing functions accept contracts and market states that callers have already checked
with the domain `validate` functions. A style-specific analytical function throws
`std::invalid_argument` when given a different, otherwise valid option style rather than silently
applying the wrong formula. Successful calls return a small `PricingResult` containing `price` and
spot `delta`; pricing formulas are added only in their corresponding roadmap tasks.

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

The scalar implementation instantiates `statistics`, `random`, and `monte_carlo` as separate
standard-library-only modules. Random generation owns only seeded normal draws; deterministic GBM
evolution accepts supplied draws; payoff calculation is independent; and discounted payoff samples
flow into streaming statistics. The geometric- and arithmetic-Asian pricers share the same path
simulation loop and allocate one reusable draw buffer outside it. Their path-average helpers use the
same equally spaced post-initial observation schedule.

M3 adds two arithmetic-Asian variance-reduction paths without changing the domain layer.
Antithetic sampling treats the mean payoff from `z` and `-z` as one effective sample. The geometric
control variate uses paired arithmetic and geometric payoffs from one path evolution, fits its
coefficient on a separate explicitly seeded pilot sample, and uses the M1 analytical geometric
price as the known expectation. A numerically degenerate control variance falls back to the plain
estimator and is exposed in result diagnostics.

M4 adds `pricing` as the validated orchestration boundary. A request owns its contract, market,
backend/estimator selection, and exactly one compatible numerical configuration. The router rejects
invalid domain inputs, unsupported style/estimator pairs, missing configuration, and extraneous
configuration rather than silently substituting an estimator. Analytical and stochastic outputs
share typed price/Delta estimates; sampling errors, confidence intervals, path counts, seeds, pilot
metadata, and control metadata are optional only where they are genuinely unavailable.

The scalar Monte Carlo layer now produces price and pathwise Delta from each path's same draws.
Price and Delta statistics remain separate. The arithmetic control variate fits independent price
and Delta coefficients on one separate pilot stream, while the CRN bump-and-revalue functions are
validation APIs rather than router backends.

M5 extends `MonteCarloConfig` with a requested thread count while preserving `thread_count = 1` as
the scalar reference. Effective samples are split by quotient and remainder; requested workers are
capped at the number of samples. Every active worker owns its RNG, reusable draw buffer, price
statistics, and Delta statistics. Cache-line-aligned worker records keep independently written
accumulators apart, and the caller merges them in ascending worker-index order with Chan/Welford
formulas after joining all threads. Pilot and pricing phases of the control variate use the same
policy independently. No allocation occurs inside an effective-sample or observation loop.

The multithreaded finite estimate is deterministic for a fixed thread count, but it is not required
to equal scalar output bit for bit because it uses distinct worker streams. Analytical agreement,
statistical scalar/threaded agreement, deterministic zero-volatility cases, and exact metadata/count
invariants provide the correctness checks. Requested and active worker counts are returned through
the backend-neutral pricing result.

M6 adds a standard-library-only offline `dataset` module above the pricing router. Versioned config
files declare the feature domain, sample budgets, thread policy, quality tolerances, master seed,
and source-finalization field before generation. A deterministic endpoint-plus-radical-inverse
design covers all six style/type combinations. The unique parameter point owns its 70/15/15 split;
stochastic observations are never split independently.

Every label is generated through the validated `nre::price` interface with the M5 execution
controls. European and geometric-Asian labels use plain Monte Carlo and analytical results as
cross-checks; arithmetic Asians use the independent-pilot geometric control variate. CSV rows carry
price/Delta uncertainty and complete execution/control metadata. A deterministic manifest records
the schema, domain, compiler/build/hardware summary, config and label checksums, seed rule, split
counts, and quality results. Failed quality gates retain the row for audit but set
`included_for_training=false`. Generated artifacts never become source dependencies.

M7 adds an offline Python layer without changing the trusted C++ engine. The loader binds a dataset
to its versioned schema, manifest/config checksum, and label checksum before exposing accepted rows.
It rejects non-finite fields, duplicate ids, inconsistent split counts, and schema drift. A
training-only preprocessor maps dimensionless log moneyness and the remaining market/contract
features into a polynomial basis. A deterministic ridge solve fits price divided by strike and a
plainly separate Delta target; validation alone selects the declared degree and regularization.
The JSON artifact carries the fitted preprocessing, basis order, coefficients, dataset provenance,
and selection table. This baseline is offline comparison evidence, not a C++ pricing backend.

M8 adds a second offline Python package above the same validated loader. A smooth float64 MLP emits
one `price/strike` scalar and has no Delta head. Training constructs physical price, differentiates
it with respect to unscaled spot through log-moneyness preprocessing, and optionally supervises
that derivative against the trusted pathwise labels. Both ablations share the same fixed
capacity/regularization search and validation-only checkpoint policy. Versioned `.pt` state plus
JSON feature order, preprocessing, architecture, checksums, and output semantics form the M9 export
input; no ONNX or C++ neural dependency is introduced in M8.

M9 freezes the selected derivative-supervised state as a dynamic-batch float64 ONNX graph with one
`price/strike` output. Exact feature order, train-only means/scales, deployment-domain limits,
tensor names, bump policy, tolerances, and checksums live in one versioned JSON artifact consumed
by Python and C++. Because ONNX Runtime does not expose training autograd to this C++ inference
path, deployed Delta is the centered finite difference of that same scalar price graph using
`max(1e-4 * spot, 1e-6)`; it is not a learned second output.

The optional `OnnxPricingBackend` owns one reusable session and buffers batched base/up/down graph
rows. `price_guarded_neural_batch` remains in the dependency-free core and accepts only requests
that already declare their exact trusted Monte Carlo fallback. It evaluates eligible base/spot/
volatility probes in batches, preserves request order, returns neural results only after all checks
pass, and otherwise calls the ordinary `price` route while recording one reason. Invalid financial
inputs are rejected before inference because no valid Monte Carlo price exists for them.

M10 adds the dependency-free `risk` workload layer. It owns deterministic synthetic portfolio and
scenario construction, identity/order preservation, fallback-request seed policy, portfolio error
aggregation, and matched-tolerance selection. The optional benchmark executable composes that layer
with `OnnxPricingBackend`; ONNX remains absent from `nre_core`, so trusted pricing and its tests do
not acquire an inference dependency. Portfolio results remain position-major then scenario-minor,
and item-level rejection reasons must reconcile exactly with aggregate counters.

Dependencies should point downward from orchestration to small numerical components. In particular,
`domain` and `analytics` must not depend on Monte Carlo or ML.

## Planned Python modules (M7 and later)

- Dataset loading and schema/manifest checks.
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

The trusted engine and default tests remain C++-standard-library-only. M9 optionally links ONNX
Runtime 1.27.x when `NRE_ENABLE_ONNX=ON` or an ONNX Make target is selected. Python export uses
ONNX 1.22.0 and ONNX Runtime 1.27.0; the measured Homebrew C++ runtime is 1.27.1. No Python
interpreter or generated training dataset is embedded in the C++ process.

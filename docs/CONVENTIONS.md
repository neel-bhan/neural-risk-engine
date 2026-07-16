# Financial and numerical conventions

These conventions define the problem. Changing one can change the correct answer, so update this
file and the relevant tests whenever a convention changes.

## Market model

The first version uses a risk-neutral geometric Brownian motion model with constant parameters:

```text
dS(t) = (r - q) S(t) dt + sigma S(t) dW(t)
```

- `S` is spot in currency units.
- `r` is the continuously compounded annual risk-free rate as a decimal.
- `q` is the continuous annual dividend yield as a decimal.
- `sigma` is annualized volatility as a decimal.
- `T` is time to maturity in years.
- One year is a model unit; no calendar or day-count library is used in version one.

Rates and dividend yields may be negative. Spot and strike must be strictly positive. Volatility
must be non-negative. Maturity must be strictly positive at the public interface; zero-maturity
limits belong in analytical tests, not ordinary contracts.

This is a deliberately simple model for numerical engineering. It is not market calibration and
does not claim to reproduce an options surface.

## Analytical edge cases

Analytical pricing functions require inputs that have already passed the public domain validation.
They accept any strictly positive maturity, including positive values close to zero, but a contract
with zero maturity remains invalid. A zero-volatility market is priced from its discounted
deterministic payoff rather than by dividing by volatility. When that deterministic value is
exactly at the strike, the reported Delta is half of the payoff's left/right Delta jump; this is
the convention approached as positive volatility tends to zero.

## Contracts

Version one supports fixed-strike European, geometric Asian, and arithmetic Asian calls and puts.
Payoffs are:

```text
European call:        max(S(T) - K, 0)
European put:         max(K - S(T), 0)
Geometric Asian call: max(G - K, 0)
Arithmetic Asian call:max(A - K, 0)
```

Puts reverse the difference. `A` and `G` use `m` equally spaced observations at
`t_i = i T / m` for `i = 1, ..., m`. The initial spot at `t=0` is excluded. The European contract
has one observation at maturity.

All analytical formulas, simulations, datasets, and tests must use this same monitoring schedule.

## Prices, Delta, and error units

- A price is the time-zero discounted expected payoff in the same currency units as spot.
- Delta is the derivative of price with respect to one unit of spot.
- The reference engine uses `double`.
- A Monte Carlo result must contain the estimate, sample standard error, 95% confidence interval,
  number of effective paths, and seed/configuration metadata.
- Unless an experiment defines another metric, normalized pricing error is
  `abs(predicted - reference) / max(reference, price_floor)`. The experiment must state its
  `price_floor`; do not hide near-zero-option behavior by choosing it after seeing results.
- Report basis points only after defining the denominator. “Price bps” is otherwise ambiguous.

## Random sampling

- The scalar reference uses `std::mt19937_64` with a 64-bit seed. It converts the top 53 output
  bits explicitly to an open-interval uniform and applies the Box-Muller transform, caching the
  paired draw. This avoids the implementation-defined sequence of `std::normal_distribution`.
- The engine output and draw order are stable by construction. Last-place differences in `log`,
  `sqrt`, `sin`, and `cos` may still occur across math-library implementations, so reproducibility
  claims apply to a fixed toolchain and platform.
- Correctness tests use fixed seeds and statistical tolerances.
- For plain pricing, one effective path means one independently generated discounted payoff sample,
  and raw paths equal effective paths.
- For antithetic pricing, one effective sample is the mean of discounted payoffs from `z` and `-z`;
  it uses two raw paths.
- For control-variate pricing, effective paths count adjusted pricing samples. Raw paths include both
  the independent coefficient-estimation pilot and the pricing sample.
- Performance results must state whether a “path” means a raw evolution, one plain payoff sample, or
  an antithetic pair.
- Antithetic and control-variate comparisons must reuse the same draws.
- A scalar run (`thread_count = 1`) retains the original master-seed draw order. A multiworker run
  derives one distinct seed per logical worker by applying the SplitMix64 finalizer to the master
  seed and worker index, then gives each worker a private `std::mt19937_64`/Box-Muller generator.
  SplitMix64 is a permutation, so configured workers receive distinct seed values and never share
  generator state or owned draws. This is deterministic stream separation, not a proof that two
  finite pseudorandom sequences can never contain the same value.
- Thread-count reproducibility applies to a fixed master seed, requested thread count, worker
  partition, toolchain, and platform. Changing thread count intentionally changes worker streams
  and therefore the finite Monte Carlo estimate.

## Confidence intervals

The initial engine reports the normal-approximation interval:

```text
estimate +/- 1.96 * standard_error
```

This is a numerical diagnostic, not calibrated uncertainty about model error. Coverage tests should
use many independent seeds against analytical values; a single interval containing the answer is
not sufficient validation.

## Monte Carlo Delta

The scalar reference uses a pathwise spot derivative. Under GBM, a simulated terminal spot,
geometric average, or arithmetic average `U` is proportional to initial spot, so `dU/dS(0) =
U/S(0)`. The discounted pathwise sample is `D * U/S(0)` for an in-the-money call, its negative for
an in-the-money put, and zero out of the money. At an exact payoff kink, the implementation uses
half of the left/right derivative jump, matching the analytical zero-volatility convention.

Price and Delta reuse the identical normal draws within each effective sample but accumulate
separate sampling diagnostics. Antithetic Delta averages the derivatives from `z` and `-z`.
Arithmetic-Asian control-variate pricing fits separate price and Delta coefficients on the same
independent pilot stream; the Delta control expectation is the analytical geometric-Asian Delta.

Centered spot bump-and-revalue is an independent validation estimator. Up/down valuations use
common random numbers and the default scale-aware bump `max(1e-4 * spot, 1e-6)`. A bump is invalid
when it is non-finite, non-positive, or at least the current spot. Its standard error comes from
paired finite-difference samples, not from treating up/down price estimates as independent.

## M6 dataset domain and labels

Schema `nre.dataset.v1` fixes a deployment-candidate domain before model training: spot and strike
60–140 currency units, maturity 0.25–2 years, volatility 0.05–0.60, risk-free rate -0.02–0.10,
dividend yield -0.01–0.08, and 2–52 observations for Asian options. These are engineering dataset
bounds, not a statement that future models are accurate throughout the domain. European contracts
retain one maturity observation.

The deterministic design includes each endpoint and radical-inverse interior points for European,
geometric-Asian, and arithmetic-Asian calls and puts. A unique parameter point owns exactly one of
train, validation, or test using a fixed 70/15/15 index mapping. The label generator does not create
duplicate stochastic observations of one point.

All labels use the C++ backend-neutral pricing interface and include price and pathwise Delta,
their separate standard errors and 95% intervals, effective/raw paths, seeds, thread counts, and
control-pilot diagnostics where applicable. Validation/test use greater path and pilot budgets and
tighter declared SE limits than training. Rejected rows remain auditable but are not eligible for
training. Analytical price/Delta values for European and geometric Asians are validation fields,
not replacements for the Monte Carlo label.

## M7 baseline error convention

The predeclared M7 normalized price error is
`abs(predicted-reference) / max(reference, 1 currency unit)`. The one-unit floor was fixed before
test evaluation. Results also report absolute errors and the count and dedicated slice of reference
prices below the floor, so the denominator does not conceal poor near-zero-option behavior.

The polynomial-ridge baseline fits price divided by strike and Delta as separate regression
targets. Its reported Delta RMSE is a useful empirical comparison, but the Delta is not derived
from the price output and has no derivative-consistency claim. M8 neural Delta must instead be
derived from the scalar neural price through automatic differentiation.

## M8 neural representation and derivative supervision

The M8 MLP consumes the same nine M7 base features after means and scales fitted on accepted train
rows only. Its single output is price divided by strike; physical price is strike times that
output. Spot Delta is never a separately predicted output. PyTorch differentiates physical price
with respect to an unscaled spot tensor through log moneyness and the stored feature scaling.

The price loss normalizes squared `price/strike` error by the training-only standard deviation of
`price/strike`. The Delta loss normalizes squared Delta error by the training-only Delta standard
deviation. Derivative supervision gives the two normalized terms equal weight. Model and
checkpoint selection use validation median normalized price error under the unchanged one-unit
M7 price floor. Held-out test rows do not influence selection.

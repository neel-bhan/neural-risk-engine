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

- The initial scalar reference will use a documented standard-library generator and normal
  transform for portability and reproducibility within a fixed toolchain.
- Correctness tests use fixed seeds and statistical tolerances.
- Performance results must state whether a “path” means one payoff sample or an antithetic pair.
- Antithetic and control-variate comparisons must reuse the same draws.
- Threaded runs derive non-overlapping deterministic streams from a master seed; the precise stream
  construction will be chosen and documented in its milestone.

## Confidence intervals

The initial engine reports the normal-approximation interval:

```text
estimate +/- 1.96 * standard_error
```

This is a numerical diagnostic, not calibrated uncertainty about model error. Coverage tests should
use many independent seeds against analytical values; a single interval containing the answer is
not sufficient validation.


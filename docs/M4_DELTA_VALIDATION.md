# M4 Delta validation

This report records the M4 numerical experiment for the scalar pathwise Delta estimator and its
independent common-random-number (CRN) finite-difference check. It is a validation experiment, not
a throughput or latency benchmark.

Source implementation commit: `5c2adcb007d26a2e3935ca79f2072e5c4b653588`.

## Estimators

Under GBM, every simulated European terminal spot and Asian average `U` is homogeneous in initial
spot: `dU/dS(0) = U/S(0)`. For each path, the reference estimator differentiates the discounted
payoff:

```text
call: D * 1{U > K} * U/S(0)
put: -D * 1{U < K} * U/S(0)
```

where `D = exp(-rT)`. At exactly `U = K`, the indicator is one half, matching the analytical
zero-volatility convention. Price and Delta are accumulated from the same draws and have separate
sample standard errors and normal-approximation 95% confidence intervals.

For antithetic arithmetic Asians, one Delta sample is the mean of the `z` and `-z` pathwise
derivatives. For the geometric control variate, an independent pilot fits separate coefficients for
price and Delta. The Delta control uses the pathwise geometric-Asian Delta and its M1 analytical
expectation. Degenerate Delta-control variance safely gives a zero coefficient.

The independent validation estimator uses centered spot bump-and-revalue with common random
numbers between the up/down path evolutions:

```text
h = max(1e-4 * S(0), 1e-6)
Delta_CRN = [payoff(S(0) + h, z) - payoff(S(0) - h, z)] / (2h)
```

The public helper rejects non-finite or non-positive bump parameters and `h >= S(0)`, so the down
spot remains valid. CRN uncertainty is calculated from paired per-sample finite differences; it is
not reconstructed from two independent price errors.

## Method and reproducibility

- Date: 2026-07-16.
- Compiler: Apple Clang 17.0.0 (`clang-1700.0.13.5`).
- Flags: C++20, `-O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`.
- Hardware: Apple M4, arm64 MacBook Air.
- OS: macOS 26.5.1, Darwin 25.5.0.
- Threads: one.
- Effective path counts: 1,000, 4,000, and 16,000.
- Repetitions: 100 independently seeded runs per row.
- Pricing seeds: 5,000,000 through 5,000,099, restarted for each contract, estimator, and path
  count.
- Control pilot: 1,000 paths with seeds 6,000,000 through 6,000,099; pilot and pricing streams are
  distinct.
- Timing: `std::chrono::steady_clock` around each complete call. Timings document experiment cost
  only and are not M5 performance claims.

The analytical cases use spot 100, volatility 0.20, rate 0.05, dividend yield 0.02, and one-year
maturity. The European call has strike 100. The geometric-Asian put has strike 100 and 12
post-initial observations. The arithmetic-Asian call uses spot 105, strike 100, volatility 0.30,
rate 0.02, dividend yield 0.01, 1.5-year maturity, and 18 post-initial observations.

Reproduce from a clean tree with:

```bash
make clean
make CXXFLAGS='-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror' delta-validation
```

## Analytical Delta results

`Error SE` is the standard error of the mean signed error across the 100 runs. `Mean reported SE`
is the average within-run Delta standard error.

| Contract | Paths | Analytical Delta | RMSE | Mean absolute error | Mean error | Error SE | Mean reported SE | 95% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| European call | 1,000 | 0.5868511461 | 0.0168763926 | 0.0131412450 | 0.0018649993 | 0.0016857526 | 0.0181597638 | 95% |
| European call | 4,000 | 0.5868511461 | 0.0091199648 | 0.0075099338 | 0.0007951474 | 0.0009131005 | 0.0090721885 | 96% |
| European call | 16,000 | 0.5868511461 | 0.0039589612 | 0.0031304977 | 0.0000660953 | 0.0003978351 | 0.0045357157 | 97% |
| Geometric-Asian put | 1,000 | -0.4179741806 | 0.0134299016 | 0.0108327988 | -0.0013087457 | 0.0013433316 | 0.0137612103 | 98% |
| Geometric-Asian put | 4,000 | -0.4179741806 | 0.0064576403 | 0.0050143962 | -0.0005530472 | 0.0006466327 | 0.0068794415 | 96% |
| Geometric-Asian put | 16,000 | -0.4179741806 | 0.0039452872 | 0.0030879614 | -0.0000763786 | 0.0003964420 | 0.0034392602 | 92% |

## European and geometric-Asian CRN comparison

The generic CRN estimator is also checked directly for both analytically tractable styles. The
difference is again `pathwise Delta - CRN Delta` with its uncertainty estimated across paired runs.

| Contract | Paths | Mean difference | Difference SE | Difference RMSE | Mean absolute difference | Mean pathwise SE | Mean CRN SE |
|---|---:|---:|---:|---:|---:|---:|---:|
| European call | 1,000 | -0.0000074023 | 0.000015520 | 0.000154602 | 0.000068236 | 0.018159764 | 0.018158327 |
| European call | 4,000 | 0.0000009276 | 0.000008634 | 0.000085917 | 0.000057548 | 0.009072189 | 0.009071385 |
| European call | 16,000 | -0.0000033706 | 0.000004096 | 0.000040897 | 0.000033531 | 0.004535716 | 0.004535311 |
| Geometric-Asian put | 1,000 | -0.0000110060 | 0.000020449 | 0.000203765 | 0.000107178 | 0.013761210 | 0.013758176 |
| Geometric-Asian put | 4,000 | -0.0000059281 | 0.000010149 | 0.000101155 | 0.000076654 | 0.006879441 | 0.006877858 |
| Geometric-Asian put | 16,000 | -0.0000042942 | 0.000004933 | 0.000049273 | 0.000038170 | 0.003439260 | 0.003438330 |

## Arithmetic-Asian CRN comparison

The difference is `pathwise Delta - CRN Delta`. Because the estimators deliberately share draws,
their individual standard errors must not be combined as though independent. `Difference SE` is
instead calculated from the 100 independently seeded paired run differences.

| Estimator | Paths | Mean difference | Difference SE | Difference RMSE | Mean absolute difference | Mean pathwise SE | Mean CRN SE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain | 1,000 | -0.0000259641 | 0.000016000 | 0.000161300 | 0.000081178 | 0.018213555 | 0.018211715 |
| Plain | 4,000 | -0.0000074411 | 0.000007285 | 0.000072861 | 0.000052316 | 0.009118252 | 0.009117535 |
| Plain | 16,000 | -0.0000019996 | 0.000004399 | 0.000043812 | 0.000034550 | 0.004556299 | 0.004555922 |
| Antithetic | 1,000 | -0.0000014706 | 0.000010836 | 0.000107830 | 0.000066568 | 0.004318753 | 0.004315719 |
| Antithetic | 4,000 | -0.0000029138 | 0.000005185 | 0.000051673 | 0.000040855 | 0.002166841 | 0.002165466 |
| Antithetic | 16,000 | 0.0000005211 | 0.000002965 | 0.000029506 | 0.000023364 | 0.001084408 | 0.001083652 |
| Control variate | 1,000 | 0.0000019519 | 0.000021923 | 0.000218143 | 0.000137462 | 0.003275445 | 0.003258636 |
| Control variate | 4,000 | 0.0000004946 | 0.000010058 | 0.000100074 | 0.000082488 | 0.001668155 | 0.001660371 |
| Control variate | 16,000 | -0.0000045252 | 0.000006148 | 0.000061338 | 0.000049787 | 0.000837194 | 0.000833460 |

## Interpretation

European and geometric-Asian Delta RMSE falls as path count increases, and their mean signed errors
are within roughly 1.2 error standard errors of zero in every row. Observed analytical-Delta
coverage ranges from 92% to 98% across 100 runs per row. This supports the M4 analytical agreement
gate for these stated cases; it is not calibrated uncertainty about model error.

For every European, geometric-Asian, and arithmetic row, the mean pathwise-minus-CRN difference is
within two paired difference standard errors of zero. Difference RMSE is also small relative to
either estimator's within-run standard error. Small nonzero sample differences are expected when a
finite bump causes a path to cross the payoff kink. These results validate the implementation and
chosen bump rule for the stated cases, not every parameter point.

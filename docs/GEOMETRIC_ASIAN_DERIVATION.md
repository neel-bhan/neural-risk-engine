# Discrete geometric-Asian analytical formula

This derivation uses the version-one monitoring schedule from `docs/CONVENTIONS.md`: `m` equally
spaced observations at `t_i = iT/m`, for `i = 1, ..., m`. Spot at `t=0` is excluded.

## Distribution of the log geometric average

Under the risk-neutral GBM model,

```text
log S(t_i) = log S(0) + (r - q - sigma^2 / 2) t_i + sigma W(t_i).
```

Let `Y = log G = (1/m) sum_i log S(t_i)`. A linear combination of jointly normal Brownian
values is normal. Its mean is

```text
E[Y] = log S(0) + (r - q - sigma^2 / 2) (1/m) sum_i t_i
     = log S(0) + (r - q - sigma^2 / 2) T(m + 1)/(2m),
```

because `sum_i i = m(m+1)/2`. Define the average monitoring time

```text
t_bar = T(m + 1)/(2m).
```

For the variance, `Cov(W(t_i), W(t_j)) = min(t_i, t_j)`, so

```text
Var[Y] = sigma^2/m^2 sum_i sum_j min(t_i, t_j)
       = sigma^2 T/m^3 sum_i sum_j min(i, j).
```

The double sum can be counted by noting that `min(i,j) >= k` for exactly `(m-k+1)^2` pairs:

```text
sum_i sum_j min(i,j)
    = sum_(k=1)^m (m-k+1)^2
    = sum_(k=1)^m k^2
    = m(m+1)(2m+1)/6.
```

Therefore

```text
mu = E[Y] = log S(0) + (r - q - sigma^2/2) t_bar,
v  = Var[Y] = sigma^2 T (m+1)(2m+1)/(6m^2).
```

These coefficients depend on excluding `t=0`; including it would produce a different formula.

## Price and Delta

Since `Y` is normal, `G = exp(Y)` is lognormal and

```text
M  = E[G] = exp(mu + v/2),
d1 = (mu - log K + v)/sqrt(v),
d2 = d1 - sqrt(v),
D  = exp(-rT).
```

The time-zero prices are

```text
call = D [M Phi(d1) - K Phi(d2)],
put  = D [K Phi(-d2) - M Phi(-d1)].
```

Because `mu` changes one-for-one with `log S(0)`, `M/S(0)` is independent of spot. The spot
Deltas are

```text
call Delta = D M/S(0) Phi(d1),
put Delta  = D M/S(0) [Phi(d1) - 1].
```

The corresponding parity relation is `call - put = D(M-K)`.

## Zero-volatility limit

When `sigma = 0`, division by `sqrt(v)` is avoided. The geometric average is deterministic:

```text
G = S(0) exp((r-q)t_bar).
```

The implementation discounts its intrinsic payoff directly. At the payoff kink it reports half
of the left/right Delta jump, matching the `sigma` approaching zero convention used by the
European analytical function.

## Independent numerical checks

The test fixtures were generated at 80 decimal digits with Python's `decimal` arithmetic, using
its independent `ln`, `exp`, and `sqrt` operations and a directly summed power series for `erf`.
For `S=100`, `K=105`, `T=1.5`, `m=12`, `sigma=0.25`, `r=0.03`, and `q=0.01`, this calculation gives:

```text
call price = 5.5467739788053796957793650603696...
call Delta = 0.4357617080392449116129826854484...
put price  = 9.5114952740204389946501468907484...
put Delta  = -0.5283884349333593977629335072207...
```

Before the formula was used as a C++ test oracle, a separate scalar path simulation checked the
same parameter point. It used exact GBM steps, all 12 monitored log spots, Welford streaming
statistics, 1,000,000 paths, `std::mt19937_64`, `std::normal_distribution<double>`, and seed
`20260716`, compiled with AppleClang 17 and libc++ on 2026-07-16. The measured results were:

```text
call = 5.53254440874225, SE = 0.0105003774475083,
       95% CI [5.51196366894513, 5.55312514853937]
put  = 9.50054878961116, SE = 0.0108293462610501,
       95% CI [9.47932327093950, 9.52177430828282]
```

Both independent analytical values lie inside the corresponding simulation intervals. This is a
consistency check, not evidence about general Monte Carlo coverage or performance.

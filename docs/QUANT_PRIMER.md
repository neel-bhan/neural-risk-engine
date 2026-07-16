# Quant primer for building this project

You do not need a finance degree to build the first version. You do need a precise mental model for
what is being computed.

## The product in one paragraph

An option is a contract with a payoff determined by a future asset price. A pricer estimates what
that future payoff is worth today under an explicit model. A risk engine repeats pricing under
changed market inputs and computes sensitivities such as Delta. This project implements the same
calculation several ways, proves agreement on cases with known answers, and measures their accuracy
and speed.

## Five ideas to learn first

1. **Discounted expectation:** in the chosen model, a price is the discounted average of future
   payoffs under the risk-neutral distribution. “Risk-neutral” is a pricing construction, not a
   forecast of real market returns.
2. **Geometric Brownian motion:** this model makes log returns normal and keeps simulated spot
   positive. It assumes constant rate, dividend yield, and volatility.
3. **Monte Carlo:** sample many possible paths, evaluate each payoff, and average. Its standard
   error usually shrinks like `1 / sqrt(number of samples)`, so twice the accuracy can cost roughly
   four times as many paths.
4. **Variance reduction:** antithetic sampling pairs opposite random shocks. A control variate uses
   a correlated quantity with a known expected value. Both aim to reduce error without simply adding
   paths.
5. **Delta:** the local change in option price for a one-unit spot change. The analytical formula,
   finite differences, pathwise estimators, and neural automatic differentiation give independent
   ways to cross-check it.

## Why the three option types matter

- A European option depends only on the maturity spot and has a Black-Scholes formula. It is the
  simplest correctness oracle.
- A geometric Asian depends on a geometric average of monitored spots. In this model it also has an
  analytical formula, but it exercises path simulation and monitoring conventions.
- An arithmetic Asian uses the ordinary average. It generally lacks the same simple closed form, so
  Monte Carlo is useful. Its close relationship with the geometric Asian makes the latter a strong
  control variate.

## What “correct” means here

There are three different errors to keep separate:

- **Model error:** the simple model may not describe markets. This project does not solve that.
- **Numerical error:** an implementation may approximate the model price. Analytical validation,
  confidence intervals, and convergence tests target this.
- **Surrogate error:** the neural or baseline model approximates the reference engine. Held-out
  evaluation and runtime guardrails target this.

Never use a small surrogate error to claim the market model itself is accurate.

## A practical learning loop

For each roadmap milestone: read the formula, derive or explain its units, implement the smallest
version, test a known case, then measure. Keep a short experiment note about surprises. In an
interview, being able to explain one numerical choice and one discovered bug is more credible than
listing many libraries.

Good prerequisite topics are basic probability (expectation, variance, normal distribution),
single-variable calculus (derivatives), vectors/loops in modern C++, threads, and basic supervised
learning. Learn them just in time as the roadmap introduces them.


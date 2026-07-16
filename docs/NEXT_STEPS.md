# Optional post-project backlog

M0–M10 are complete. The current repository is a coherent recruiting project; none of the items
below is required to support its measured claims.

## Highest-value follow-ups

- Rework Monte Carlo portfolio execution to retain a worker pool across requests. M10 component
  timing shows fallback work and per-request thread creation dominate guarded-route latency.
- Expand training coverage around the observed low-price failure slice, retrain under a newly
  versioned protocol, and compare without altering the frozen M8–M10 evidence.
- Add Linux ONNX Runtime CI when a stable package/install cache is available; keep the default
  dependency-free GCC job regardless.
- Add property-based tests for put-call relationships and sampled monotonicity across broader valid
  inputs, while continuing to describe them as tests rather than formal proofs.
- Package benchmark artifacts and plots for a project write-up without changing the underlying
  measured JSON.

## Explicitly out of current scope

Market calibration, stochastic volatility, American exercise, GPUs, distributed pricing, live
market feeds, and trading-system integration would each require a new design and validation phase.
They should not be appended casually to the completed benchmark.

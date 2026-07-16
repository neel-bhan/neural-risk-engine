# M9 work queue — ONNX C++ acceleration with guarded fallback

M8 is complete with pinned deterministic PyTorch training, one scalar price output, autograd spot
Delta, fair price-only/derivative-supervised ablations, exact checkpoint reproduction, and frozen
held-out/slice/timing evidence. M9 deploys the selected derivative-supervised state without changing
the trusted Monte Carlo backend.

## Task 1 — Freeze export and runtime contracts

- Pin compatible ONNX, ONNX Runtime Python, and C++ runtime versions; document supported platforms
  and preserve the dependency-free C++ reference build when neural support is disabled.
- Define a versioned batched tensor contract from raw domain inputs through the exact M8 feature
  order, means/scales, `price/strike` output, and physical-price rescaling. Do not copy constants
  independently from the M8 metadata.
- Export only the frozen derivative-supervised checkpoint. Record source model/state/config/data
  checksums, opset, exporter/runtime versions, tensor names, dtypes, and dynamic batch axes.
- Choose and document a derivative-consistent deployment Delta policy. If C++ uses centered spot
  bumps of the same scalar ONNX price because ONNX Runtime does not provide autograd, use a
  predeclared scale-aware bump and validate it against M8 PyTorch autograd; never add a learned
  Delta head.

## Task 2 — Prove Python/export parity

- Add deterministic tests comparing reloaded PyTorch and ONNX outputs over training-domain points,
  boundary points, all style/type combinations, and multiple batch sizes.
- Compare deployed Delta policy against PyTorch autograd with explicit absolute/relative tolerances,
  including near-zero and high/low spot cases.
- Reject metadata/checksum/schema/feature-order mismatches and non-finite exporter outputs.
- Store a compact machine-readable parity report; do not tune export tolerances on the final test
  split.

## Task 3 — Add an optional C++ ONNX backend

- Add a build option that finds/links pinned ONNX Runtime only when enabled; ordinary `make check`
  and the standard-library Monte Carlo path must remain usable without it.
- Implement a reusable batched C++ session with preallocated/reused input and output buffers,
  versioned metadata loading, exact Python preprocessing parity, physical price, and the frozen
  Delta policy.
- Integrate it behind the backend-neutral pricing boundary without embedding Python or allowing
  direct neural calls to bypass request validation.
- Add cross-language golden tests for raw inputs, scaled features, normalized output, physical
  price, Delta, batch ordering, and artifact-version failures.

## Task 4 — Implement guardrails and fallback

- Enforce the declared M6/M8 deployment domain before inference: finite inputs, spot/strike,
  maturity, volatility, rates/yields, observation counts, style, and type.
- Check finite outputs and contract-specific price lower/upper bounds with one predeclared numerical
  tolerance. Add batched spot and volatility monotonicity probes where applicable.
- Give every rejection one explicit reason counter (domain, non-finite, price bound, spot
  monotonicity, volatility monotonicity, artifact/runtime failure) and route the original request to
  the trusted Monte Carlo estimator without silently changing its configuration.
- Test every rejection reason, accepted routing, fallback result/diagnostics, counter totals, and
  mixed accepted/rejected batch ordering. These checks are engineering guardrails, not formal
  no-arbitrage or OOD guarantees.

## Task 5 — Evaluate the guarded backend

- Run the exact frozen M8 held-out rows plus a separately declared boundary/probe set through the
  C++ router. Report overall acceptance, fallback rate by reason, accepted-set errors, and full
  routed errors together so selective acceptance cannot hide failures.
- Record Python-to-ONNX and Python-to-C++ parity, price/Delta metrics, batch sizes, warm-up,
  repetitions, compiler flags, hardware, runtime versions, artifact checksums, and source commit.
- Report C++ ONNX batch timing descriptively. Do not claim matched-error neural speedup or portfolio
  latency until the M10 scenario benchmark.
- Update architecture, conventions, build instructions, artifact documentation, and the M9 report
  only from measured evidence.

## Exit gate

M9 is complete only when Python and optional C++ outputs match within declared tolerances; the
default C++ reference build remains dependency-free; all guardrail rejection reasons and Monte
Carlo fallback paths are tested; and accepted-set error is reported alongside total/reasoned
fallback rates.

## Scope boundary

M9 does not tune the M8 model on held-out data, add a separate Delta head, claim formal
no-arbitrage/OOD reliability, or produce the final many-contract market-shock benchmark. Portfolio
repricing and matched-error neural-versus-Monte-Carlo speedup remain M10.

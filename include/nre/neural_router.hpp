#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "nre/domain.hpp"
#include "nre/pricing.hpp"

namespace nre {

enum class NeuralRejectionReason : std::size_t {
  none = 0,
  input_domain,
  non_finite_output,
  price_bound,
  spot_monotonicity,
  volatility_monotonicity,
  artifact_runtime_failure,
  count,
};

struct NeuralCandidate {
  double price;
  double delta;
};

struct NeuralInput {
  OptionContract contract;
  MarketState market;
};

class NeuralBatchBackend {
 public:
  virtual ~NeuralBatchBackend() = default;
  [[nodiscard]] virtual std::vector<NeuralCandidate> predict(
      const std::vector<NeuralInput>& inputs) = 0;
};

struct NeuralGuardrailConfig {
  double spot_min{60.0};
  double spot_max{140.0};
  double strike_min{60.0};
  double strike_max{140.0};
  double maturity_min{0.25};
  double maturity_max{2.0};
  double volatility_min{0.05};
  double volatility_max{0.60};
  double rate_min{-0.02};
  double rate_max{0.10};
  double dividend_min{-0.01};
  double dividend_max{0.08};
  std::size_t asian_observations_min{2U};
  std::size_t asian_observations_max{52U};
  double bound_tolerance{1.0e-8};
  double monotonicity_tolerance{1.0e-8};
  double spot_probe_relative{1.0e-3};
  double volatility_probe_absolute{1.0e-3};
};

struct GuardedPricingItem {
  UnifiedPricingResult result;
  bool neural_accepted;
  NeuralRejectionReason rejection_reason;
};

struct GuardedBatchResult {
  std::vector<GuardedPricingItem> items;
  std::array<std::size_t, static_cast<std::size_t>(NeuralRejectionReason::count)> counters{};
};

// Each request must describe the exact Monte Carlo fallback configuration. The router validates
// every request, evaluates eligible candidates in one batch, applies engineering checks, and calls
// the trusted pricing route for every rejection without changing that configuration.
[[nodiscard]] GuardedBatchResult price_guarded_neural_batch(
    const std::vector<PricingRequest>& fallback_requests, NeuralBatchBackend& backend,
    const NeuralGuardrailConfig& config = {});

[[nodiscard]] std::string to_string(NeuralRejectionReason reason);

}  // namespace nre

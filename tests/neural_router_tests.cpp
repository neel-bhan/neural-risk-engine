#include "nre/neural_router.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "nre/analytics.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

nre::PricingRequest fallback_request(double spot = 100.0) {
  return {
      .contract = {.type = nre::OptionType::call,
                   .style = nre::OptionStyle::european,
                   .strike = 100.0,
                   .maturity_years = 1.0,
                   .observations = 1U},
      .market = {.spot = spot,
                 .volatility = 0.20,
                 .risk_free_rate = 0.03,
                 .dividend_yield = 0.01},
      .backend = nre::PricingBackend::monte_carlo,
      .estimator = nre::PricingEstimator::plain,
      .monte_carlo_config = nre::MonteCarloConfig{.seed = 42U,
                                                  .path_count = 128U,
                                                  .thread_count = 1U},
      .control_variate_config = std::nullopt,
  };
}

enum class FakeMode { analytical, non_finite, price_bound, spot_failure, volatility_failure, fail };

class FakeBackend final : public nre::NeuralBatchBackend {
 public:
  explicit FakeBackend(FakeMode mode) : mode_(mode) {}

  std::vector<nre::NeuralCandidate> predict(
      const std::vector<nre::NeuralInput>& inputs) override {
    calls += 1U;
    last_size = inputs.size();
    if (mode_ == FakeMode::fail) {
      throw std::runtime_error("synthetic runtime failure");
    }
    std::vector<nre::NeuralCandidate> result;
    result.reserve(inputs.size());
    for (const auto& input : inputs) {
      const auto value = nre::black_scholes_european(input.contract, input.market);
      result.push_back({.price = value.price, .delta = value.delta});
    }
    for (std::size_t base = 0U; base + 4U < result.size(); base += 5U) {
      switch (mode_) {
        case FakeMode::analytical:
        case FakeMode::fail:
          break;
        case FakeMode::non_finite:
          result[base].price = std::numeric_limits<double>::quiet_NaN();
          break;
        case FakeMode::price_bound:
          result[base].price = -1.0;
          break;
        case FakeMode::spot_failure:
          result[base + 1U].price = 10.0;
          result[base + 2U].price = 9.0;
          break;
        case FakeMode::volatility_failure:
          result[base + 3U].price = 10.0;
          result[base + 4U].price = 9.0;
          break;
      }
    }
    return result;
  }

  std::size_t calls{0U};
  std::size_t last_size{0U};

 private:
  FakeMode mode_;
};

void require_reason(FakeMode mode, nre::NeuralRejectionReason expected) {
  FakeBackend backend(mode);
  const auto routed = nre::price_guarded_neural_batch({fallback_request()}, backend);
  require(routed.items.size() == 1U, "router must retain item count");
  require(!routed.items[0].neural_accepted, "synthetic bad candidate must fall back");
  require(routed.items[0].rejection_reason == expected, "fallback reason must be explicit");
  require(routed.items[0].result.backend == nre::PricingBackend::monte_carlo,
          "fallback must use trusted Monte Carlo");
  require(routed.counters[static_cast<std::size_t>(expected)] == 1U,
          "fallback reason counter must increment");
}

}  // namespace

int main() {
  {
    FakeBackend backend(FakeMode::analytical);
    const auto routed = nre::price_guarded_neural_batch({fallback_request()}, backend);
    require(routed.items[0].neural_accepted, "valid analytical-like candidate must be accepted");
    require(routed.items[0].result.backend == nre::PricingBackend::neural,
            "accepted result must identify neural backend");
    require(backend.last_size == 5U, "one item must use one base and four guardrail probes");
    require(routed.counters[static_cast<std::size_t>(nre::NeuralRejectionReason::none)] == 1U,
            "accepted result must be counted");
  }

  require_reason(FakeMode::non_finite, nre::NeuralRejectionReason::non_finite_output);
  require_reason(FakeMode::price_bound, nre::NeuralRejectionReason::price_bound);
  require_reason(FakeMode::spot_failure, nre::NeuralRejectionReason::spot_monotonicity);
  require_reason(FakeMode::volatility_failure,
                 nre::NeuralRejectionReason::volatility_monotonicity);
  require_reason(FakeMode::fail, nre::NeuralRejectionReason::artifact_runtime_failure);

  {
    FakeBackend backend(FakeMode::analytical);
    const auto routed = nre::price_guarded_neural_batch({fallback_request(150.0)}, backend);
    require(routed.items[0].rejection_reason == nre::NeuralRejectionReason::input_domain,
            "out-of-deployment-domain input must have an explicit reason");
    require(routed.items[0].result.backend == nre::PricingBackend::monte_carlo,
            "out-of-domain input must fall back");
    require(backend.last_size == 0U, "out-of-domain input must not reach the model");
  }

  {
    FakeBackend backend(FakeMode::analytical);
    const auto routed = nre::price_guarded_neural_batch(
        {fallback_request(100.0), fallback_request(150.0), fallback_request(110.0)}, backend);
    require(routed.items.size() == 3U, "mixed batch size must be preserved");
    require(routed.items[0].neural_accepted && !routed.items[1].neural_accepted &&
                routed.items[2].neural_accepted,
            "mixed batch ordering must be preserved");
    require(backend.last_size == 10U, "only eligible mixed-batch items must be inferred");
  }

  {
    FakeBackend backend(FakeMode::analytical);
    auto invalid = fallback_request();
    invalid.market.spot = std::numeric_limits<double>::quiet_NaN();
    bool threw = false;
    try {
      static_cast<void>(nre::price_guarded_neural_batch({invalid}, backend));
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "non-finite financial input must be rejected before inference and fallback");
  }

  std::cout << "neural router tests passed\n";
  return 0;
}

#include "nre/risk.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

nre::UnifiedPricingResult result(double price, double delta,
                                 nre::PricingBackend backend = nre::PricingBackend::monte_carlo) {
  return {.price = {.estimate = price,
                    .sample_standard_error = std::nullopt,
                    .confidence_interval_95 = std::nullopt},
          .delta = {.estimate = delta,
                    .sample_standard_error = std::nullopt,
                    .confidence_interval_95 = std::nullopt},
          .backend = backend,
          .estimator = backend == nre::PricingBackend::neural ? nre::PricingEstimator::neural
                                                              : nre::PricingEstimator::plain,
          .metadata = {}};
}

}  // namespace

int main() {
  const auto portfolio = nre::m10_synthetic_portfolio();
  const auto scenarios = nre::m10_market_scenarios();
  const auto items = nre::expand_portfolio_scenarios(portfolio, scenarios);
  require(portfolio.size() == 18U, "M10 portfolio size must stay frozen");
  require(scenarios.size() == 9U, "M10 scenario count must stay frozen");
  require(items.size() == 162U, "portfolio/scenario cross-product must be complete");
  for (std::size_t index = 0U; index < items.size(); ++index) {
    require(items[index].ordinal == index, "work-item ordinals must preserve result ordering");
    require(items[index].position_id == portfolio[index / scenarios.size()].id,
            "ordering must be position-major");
    require(items[index].scenario_id == scenarios[index % scenarios.size()].id,
            "ordering must be scenario-minor");
  }

  const auto requests =
      nre::make_portfolio_fallback_requests(items, 512U, 128U, 4U, 202607160010ULL);
  const auto repeated =
      nre::make_portfolio_fallback_requests(items, 512U, 128U, 4U, 202607160010ULL);
  require(requests.size() == items.size(), "each work item must have one fallback request");
  for (std::size_t index = 0U; index < requests.size(); ++index) {
    require(requests[index].contract.strike == repeated[index].contract.strike,
            "request construction must be deterministic");
    if (items[index].contract.style == nre::OptionStyle::arithmetic_asian) {
      require(requests[index].control_variate_config.has_value(),
              "arithmetic Asian must use control-variate fallback");
      require(requests[index].control_variate_config->pricing.seed ==
                  repeated[index].control_variate_config->pricing.seed,
              "control pricing seeds must be reproducible");
      require(requests[index].control_variate_config->pilot_seed !=
                  requests[index].control_variate_config->pricing.seed,
              "control pilot and pricing streams must differ");
    } else {
      require(requests[index].monte_carlo_config.has_value(),
              "analytical styles must use plain fallback");
      require(requests[index].monte_carlo_config->seed ==
                  repeated[index].monte_carlo_config->seed,
              "plain pricing seeds must be reproducible");
    }
  }

  const std::vector<nre::UnifiedPricingResult> references{result(0.5, 0.1), result(10.0, -0.4)};
  const std::vector<nre::UnifiedPricingResult> predictions{result(0.6, 0.2), result(9.0, -0.2)};
  const auto metrics = nre::calculate_error_metrics(predictions, references);
  require(metrics.count == 2U, "metrics must retain count");
  require(metrics.p99_normalized_price_error >= metrics.median_normalized_price_error,
          "p99 must not be below median");

  const nre::ErrorMetrics tolerance{.count = 2U,
                                    .median_normalized_price_error = 0.10,
                                    .p99_normalized_price_error = 0.20,
                                    .delta_rmse = 0.15};
  const std::vector<nre::MatchedToleranceCandidate> candidates{
      {.effective_paths = 128U,
       .metrics = {.count = 2U,
                   .median_normalized_price_error = 0.11,
                   .p99_normalized_price_error = 0.19,
                   .delta_rmse = 0.10}},
      {.effective_paths = 512U,
       .metrics = {.count = 2U,
                   .median_normalized_price_error = 0.09,
                   .p99_normalized_price_error = 0.18,
                   .delta_rmse = 0.14}},
      {.effective_paths = 2048U,
       .metrics = {.count = 2U,
                   .median_normalized_price_error = 0.05,
                   .p99_normalized_price_error = 0.10,
                   .delta_rmse = 0.08}},
  };
  require(nre::select_matched_tolerance(candidates, tolerance).effective_paths == 512U,
          "selector must choose the first candidate meeting every criterion");

  nre::GuardedBatchResult routed{};
  routed.items = {
      {.result = result(0.6, 0.2, nre::PricingBackend::neural),
       .neural_accepted = true,
       .rejection_reason = nre::NeuralRejectionReason::none},
      {.result = result(10.0, -0.4),
       .neural_accepted = false,
       .rejection_reason = nre::NeuralRejectionReason::input_domain},
  };
  routed.counters[static_cast<std::size_t>(nre::NeuralRejectionReason::none)] = 1U;
  routed.counters[static_cast<std::size_t>(nre::NeuralRejectionReason::input_domain)] = 1U;
  const auto counts = nre::aggregate_fallback_reason_counts(routed);
  require(counts[static_cast<std::size_t>(nre::NeuralRejectionReason::none)] == 1U &&
              counts[static_cast<std::size_t>(nre::NeuralRejectionReason::input_domain)] == 1U,
          "fallback aggregation must match item-level accounting");
  require(nre::calculate_accepted_neural_error_metrics(routed, references).count == 1U,
          "accepted metrics must exclude fallback results");

  std::cout << "risk tests passed\n";
  return 0;
}

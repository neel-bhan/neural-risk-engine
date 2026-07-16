#include "nre/neural_router.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace nre {
namespace {

constexpr std::size_t kProbeCount = 5U;

bool finite_input(const PricingRequest& request) noexcept {
  return std::isfinite(request.contract.strike) &&
         std::isfinite(request.contract.maturity_years) && std::isfinite(request.market.spot) &&
         std::isfinite(request.market.volatility) &&
         std::isfinite(request.market.risk_free_rate) &&
         std::isfinite(request.market.dividend_yield);
}

void validate_fallback_request(const PricingRequest& request) {
  if (!finite_input(request)) {
    throw std::invalid_argument("guarded pricing requires finite contract and market inputs");
  }
  if (!validate(request.contract).empty() || !validate(request.market).empty()) {
    throw std::invalid_argument("guarded pricing requires a valid contract and market state");
  }
  if (request.backend != PricingBackend::monte_carlo) {
    throw std::invalid_argument("guarded neural requests must declare a Monte Carlo fallback");
  }
  if (request.estimator == PricingEstimator::plain ||
      request.estimator == PricingEstimator::antithetic) {
    if (!request.monte_carlo_config.has_value() || request.control_variate_config.has_value()) {
      throw std::invalid_argument("plain/antithetic fallback requires one MonteCarloConfig");
    }
    if (request.monte_carlo_config->path_count < 2U ||
        request.monte_carlo_config->thread_count == 0U) {
      throw std::invalid_argument("invalid Monte Carlo fallback configuration");
    }
  } else if (request.estimator == PricingEstimator::geometric_control_variate) {
    if (request.monte_carlo_config.has_value() || !request.control_variate_config.has_value()) {
      throw std::invalid_argument("control fallback requires one ControlVariateConfig");
    }
    const auto& control = *request.control_variate_config;
    if (control.pricing.path_count < 2U || control.pricing.thread_count == 0U ||
        control.pilot_path_count < 2U || control.pilot_seed == control.pricing.seed) {
      throw std::invalid_argument("invalid control-variate fallback configuration");
    }
  } else {
    throw std::invalid_argument("unsupported guarded neural fallback estimator");
  }
  if (request.estimator == PricingEstimator::antithetic &&
      request.contract.style != OptionStyle::arithmetic_asian) {
    throw std::invalid_argument("antithetic fallback requires an arithmetic Asian contract");
  }
  if (request.estimator == PricingEstimator::geometric_control_variate &&
      request.contract.style != OptionStyle::arithmetic_asian) {
    throw std::invalid_argument("control fallback requires an arithmetic Asian contract");
  }
}

bool in_deployment_domain(const PricingRequest& request,
                          const NeuralGuardrailConfig& config) noexcept {
  const auto& contract = request.contract;
  const auto& market = request.market;
  if (market.spot < config.spot_min || market.spot > config.spot_max ||
      contract.strike < config.strike_min || contract.strike > config.strike_max ||
      contract.maturity_years < config.maturity_min ||
      contract.maturity_years > config.maturity_max ||
      market.volatility < config.volatility_min ||
      market.volatility > config.volatility_max || market.risk_free_rate < config.rate_min ||
      market.risk_free_rate > config.rate_max || market.dividend_yield < config.dividend_min ||
      market.dividend_yield > config.dividend_max) {
    return false;
  }
  if (contract.style == OptionStyle::european) {
    return contract.observations == 1U;
  }
  return contract.observations >= config.asian_observations_min &&
         contract.observations <= config.asian_observations_max;
}

double discounted_expected_underlying(const OptionContract& contract,
                                      const MarketState& market) {
  const double maturity = contract.maturity_years;
  const double discount = std::exp(-market.risk_free_rate * maturity);
  if (contract.style == OptionStyle::european) {
    return market.spot * std::exp(-market.dividend_yield * maturity);
  }

  const double count = static_cast<double>(contract.observations);
  const double time_step = maturity / count;
  if (contract.style == OptionStyle::arithmetic_asian) {
    double expected_average = 0.0;
    for (std::size_t index = 1U; index <= contract.observations; ++index) {
      const double time = static_cast<double>(index) * time_step;
      expected_average +=
          market.spot * std::exp((market.risk_free_rate - market.dividend_yield) * time);
    }
    return discount * expected_average / count;
  }

  const double average_time = maturity * (count + 1.0) / (2.0 * count);
  const double brownian_average_variance =
      maturity * (count + 1.0) * (2.0 * count + 1.0) / (6.0 * count * count);
  const double variance = market.volatility * market.volatility;
  const double expected_geometric = std::exp(
      std::log(market.spot) +
      (market.risk_free_rate - market.dividend_yield - 0.5 * variance) * average_time +
      0.5 * variance * brownian_average_variance);
  return discount * expected_geometric;
}

std::pair<double, double> price_bounds(const OptionContract& contract,
                                       const MarketState& market) {
  const double discounted_strike =
      contract.strike * std::exp(-market.risk_free_rate * contract.maturity_years);
  const double discounted_underlying = discounted_expected_underlying(contract, market);
  if (contract.type == OptionType::call) {
    return {std::max(discounted_underlying - discounted_strike, 0.0), discounted_underlying};
  }
  return {std::max(discounted_strike - discounted_underlying, 0.0), discounted_strike};
}

UnifiedPricingResult accepted_result(const NeuralCandidate& candidate) {
  const PricingEstimate price_estimate{
      .estimate = candidate.price,
      .sample_standard_error = std::nullopt,
      .confidence_interval_95 = std::nullopt,
  };
  const PricingEstimate delta_estimate{
      .estimate = candidate.delta,
      .sample_standard_error = std::nullopt,
      .confidence_interval_95 = std::nullopt,
  };
  return {
      .price = price_estimate,
      .delta = delta_estimate,
      .backend = PricingBackend::neural,
      .estimator = PricingEstimator::neural,
      .metadata = {},
  };
}

struct ProbeIndices {
  std::size_t base;
  std::size_t spot_low;
  std::size_t spot_high;
  std::size_t volatility_low;
  std::size_t volatility_high;
};

void count(GuardedBatchResult& result, NeuralRejectionReason reason) {
  ++result.counters[static_cast<std::size_t>(reason)];
}

}  // namespace

GuardedBatchResult price_guarded_neural_batch(
    const std::vector<PricingRequest>& fallback_requests, NeuralBatchBackend& backend,
    const NeuralGuardrailConfig& config) {
  for (const auto& request : fallback_requests) {
    validate_fallback_request(request);
  }

  GuardedBatchResult routed{};
  routed.items.reserve(fallback_requests.size());
  std::vector<std::optional<ProbeIndices>> mappings(fallback_requests.size());
  std::vector<NeuralInput> inference_inputs;
  inference_inputs.reserve(fallback_requests.size() * kProbeCount);

  for (std::size_t index = 0U; index < fallback_requests.size(); ++index) {
    const auto& request = fallback_requests[index];
    if (!in_deployment_domain(request, config)) {
      continue;
    }
    const double spot_step = std::max(config.spot_probe_relative * request.market.spot, 1.0e-6);
    const double spot_low = std::max(config.spot_min, request.market.spot - spot_step);
    const double spot_high = std::min(config.spot_max, request.market.spot + spot_step);
    const double volatility_low =
        std::max(config.volatility_min,
                 request.market.volatility - config.volatility_probe_absolute);
    const double volatility_high =
        std::min(config.volatility_max,
                 request.market.volatility + config.volatility_probe_absolute);

    const std::size_t base = inference_inputs.size();
    inference_inputs.push_back({request.contract, request.market});
    auto low_spot_market = request.market;
    low_spot_market.spot = spot_low;
    inference_inputs.push_back({request.contract, low_spot_market});
    auto high_spot_market = request.market;
    high_spot_market.spot = spot_high;
    inference_inputs.push_back({request.contract, high_spot_market});
    auto low_volatility_market = request.market;
    low_volatility_market.volatility = volatility_low;
    inference_inputs.push_back({request.contract, low_volatility_market});
    auto high_volatility_market = request.market;
    high_volatility_market.volatility = volatility_high;
    inference_inputs.push_back({request.contract, high_volatility_market});
    mappings[index] = ProbeIndices{base, base + 1U, base + 2U, base + 3U, base + 4U};
  }

  std::vector<NeuralCandidate> candidates;
  bool runtime_failure = false;
  try {
    candidates = backend.predict(inference_inputs);
    runtime_failure = candidates.size() != inference_inputs.size();
  } catch (const std::exception&) {
    runtime_failure = true;
  }

  for (std::size_t index = 0U; index < fallback_requests.size(); ++index) {
    const auto& request = fallback_requests[index];
    NeuralRejectionReason reason = NeuralRejectionReason::none;
    if (!mappings[index].has_value()) {
      reason = NeuralRejectionReason::input_domain;
    } else if (runtime_failure) {
      reason = NeuralRejectionReason::artifact_runtime_failure;
    } else {
      const auto mapping = *mappings[index];
      const auto& base = candidates[mapping.base];
      const auto& spot_low = candidates[mapping.spot_low];
      const auto& spot_high = candidates[mapping.spot_high];
      const auto& volatility_low = candidates[mapping.volatility_low];
      const auto& volatility_high = candidates[mapping.volatility_high];
      const bool all_finite =
          std::isfinite(base.price) && std::isfinite(base.delta) &&
          std::isfinite(spot_low.price) && std::isfinite(spot_high.price) &&
          std::isfinite(volatility_low.price) && std::isfinite(volatility_high.price);
      if (!all_finite) {
        reason = NeuralRejectionReason::non_finite_output;
      } else {
        const auto bounds = price_bounds(request.contract, request.market);
        if (base.price < bounds.first - config.bound_tolerance ||
            base.price > bounds.second + config.bound_tolerance) {
          reason = NeuralRejectionReason::price_bound;
        } else {
          const bool spot_ok =
              request.contract.type == OptionType::call
                  ? spot_high.price + config.monotonicity_tolerance >= spot_low.price
                  : spot_high.price <= spot_low.price + config.monotonicity_tolerance;
          if (!spot_ok) {
            reason = NeuralRejectionReason::spot_monotonicity;
          } else if (volatility_high.price + config.monotonicity_tolerance <
                     volatility_low.price) {
            reason = NeuralRejectionReason::volatility_monotonicity;
          }
        }
      }
    }

    if (reason == NeuralRejectionReason::none) {
      routed.items.push_back({
          .result = accepted_result(candidates[mappings[index]->base]),
          .neural_accepted = true,
          .rejection_reason = reason,
      });
    } else {
      routed.items.push_back({
          .result = price(request),
          .neural_accepted = false,
          .rejection_reason = reason,
      });
    }
    count(routed, reason);
  }
  return routed;
}

std::string to_string(NeuralRejectionReason reason) {
  switch (reason) {
    case NeuralRejectionReason::none:
      return "none";
    case NeuralRejectionReason::input_domain:
      return "input_domain";
    case NeuralRejectionReason::non_finite_output:
      return "non_finite_output";
    case NeuralRejectionReason::price_bound:
      return "price_bound";
    case NeuralRejectionReason::spot_monotonicity:
      return "spot_monotonicity";
    case NeuralRejectionReason::volatility_monotonicity:
      return "volatility_monotonicity";
    case NeuralRejectionReason::artifact_runtime_failure:
      return "artifact_runtime_failure";
    case NeuralRejectionReason::count:
      return "count";
  }
  return "unknown";
}

}  // namespace nre

#include "nre/pricing.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nre/analytics.hpp"

namespace nre {
namespace {

std::string validation_message(const std::vector<std::string>& errors, const char* input_name) {
  std::string message = std::string("invalid ") + input_name + ": ";
  for (std::size_t index = 0; index < errors.size(); ++index) {
    if (index != 0U) {
      message += "; ";
    }
    message += errors[index];
  }
  return message;
}

PricingEstimate deterministic_estimate(double estimate) {
  return {
      .estimate = estimate,
      .sample_standard_error = std::nullopt,
      .confidence_interval_95 = std::nullopt,
  };
}

PricingEstimate stochastic_price_estimate(const MonteCarloResult& result) {
  return {
      .estimate = result.estimate,
      .sample_standard_error = result.sample_standard_error,
      .confidence_interval_95 = result.confidence_interval_95,
  };
}

PricingEstimate stochastic_delta_estimate(const MonteCarloResult& result) {
  return {
      .estimate = result.delta.estimate,
      .sample_standard_error = result.delta.sample_standard_error,
      .confidence_interval_95 = result.delta.confidence_interval_95,
  };
}

PricingMetadata monte_carlo_metadata(const MonteCarloResult& result) {
  PricingMetadata metadata{};
  metadata.effective_paths = result.effective_paths;
  metadata.raw_paths = result.raw_paths;
  metadata.seed = result.seed;
  metadata.requested_threads = result.requested_threads;
  metadata.active_threads = result.active_threads;
  return metadata;
}

UnifiedPricingResult unified_monte_carlo_result(const MonteCarloResult& result,
                                                PricingEstimator estimator,
                                                PricingMetadata metadata) {
  return {
      .price = stochastic_price_estimate(result),
      .delta = stochastic_delta_estimate(result),
      .backend = PricingBackend::monte_carlo,
      .estimator = estimator,
      .metadata = std::move(metadata),
  };
}

void require_no_numerical_config(const PricingRequest& request) {
  if (request.monte_carlo_config.has_value() || request.control_variate_config.has_value()) {
    throw std::invalid_argument("analytical pricing does not accept Monte Carlo configuration");
  }
}

const MonteCarloConfig& require_plain_config(const PricingRequest& request) {
  if (!request.monte_carlo_config.has_value() || request.control_variate_config.has_value()) {
    throw std::invalid_argument(
        "plain or antithetic Monte Carlo requires exactly one MonteCarloConfig");
  }
  return *request.monte_carlo_config;
}

const ControlVariateConfig& require_control_config(const PricingRequest& request) {
  if (request.monte_carlo_config.has_value() || !request.control_variate_config.has_value()) {
    throw std::invalid_argument(
        "geometric control-variate pricing requires exactly one ControlVariateConfig");
  }
  return *request.control_variate_config;
}

}  // namespace

UnifiedPricingResult price(const PricingRequest& request) {
  const auto contract_errors = validate(request.contract);
  if (!contract_errors.empty()) {
    throw std::invalid_argument(validation_message(contract_errors, "contract"));
  }
  const auto market_errors = validate(request.market);
  if (!market_errors.empty()) {
    throw std::invalid_argument(validation_message(market_errors, "market"));
  }

  switch (request.backend) {
    case PricingBackend::analytical: {
      if (request.estimator != PricingEstimator::analytical) {
        throw std::invalid_argument("analytical backend requires the analytical estimator");
      }
      require_no_numerical_config(request);
      PricingResult analytical{};
      if (request.contract.style == OptionStyle::european) {
        analytical = black_scholes_european(request.contract, request.market);
      } else if (request.contract.style == OptionStyle::geometric_asian) {
        analytical = geometric_asian_analytical(request.contract, request.market);
      } else {
        throw std::invalid_argument("no analytical arithmetic-Asian pricer is available");
      }
      return {
          .price = deterministic_estimate(analytical.price),
          .delta = deterministic_estimate(analytical.delta),
          .backend = request.backend,
          .estimator = request.estimator,
          .metadata = {},
      };
    }
    case PricingBackend::monte_carlo:
      break;
    case PricingBackend::neural:
      throw std::invalid_argument(
          "direct neural pricing is not allowed; use the guarded batch router");
    default:
      throw std::invalid_argument("unsupported pricing backend");
  }

  switch (request.estimator) {
    case PricingEstimator::analytical:
      throw std::invalid_argument("Monte Carlo backend cannot use the analytical estimator");
    case PricingEstimator::plain: {
      const auto& config = require_plain_config(request);
      MonteCarloResult result{};
      if (request.contract.style == OptionStyle::european) {
        result = price_european_monte_carlo(request.contract, request.market, config);
      } else if (request.contract.style == OptionStyle::geometric_asian) {
        result = price_geometric_asian_monte_carlo(request.contract, request.market, config);
      } else if (request.contract.style == OptionStyle::arithmetic_asian) {
        result = price_arithmetic_asian_monte_carlo(request.contract, request.market, config);
      } else {
        throw std::invalid_argument("unsupported option style for plain Monte Carlo");
      }
      return unified_monte_carlo_result(result, request.estimator, monte_carlo_metadata(result));
    }
    case PricingEstimator::antithetic: {
      const auto& config = require_plain_config(request);
      if (request.contract.style != OptionStyle::arithmetic_asian) {
        throw std::invalid_argument(
            "the current antithetic estimator supports arithmetic-Asian options only");
      }
      const auto result =
          price_arithmetic_asian_antithetic_monte_carlo(request.contract, request.market, config);
      return unified_monte_carlo_result(result, request.estimator, monte_carlo_metadata(result));
    }
    case PricingEstimator::geometric_control_variate: {
      const auto& config = require_control_config(request);
      if (request.contract.style != OptionStyle::arithmetic_asian) {
        throw std::invalid_argument(
            "the geometric control variate supports arithmetic-Asian options only");
      }
      const auto result = price_arithmetic_asian_control_variate_monte_carlo(
          request.contract, request.market, config);
      auto metadata = monte_carlo_metadata(result.monte_carlo);
      metadata.pilot_paths = result.pilot_paths;
      metadata.pilot_seed = result.pilot_seed;
      metadata.pilot_active_threads = result.pilot_active_threads;
      metadata.price_control_coefficient = result.coefficient;
      metadata.price_control_expectation = result.control_expectation;
      metadata.price_control_applied = result.control_applied;
      metadata.delta_control_coefficient = result.delta_coefficient;
      metadata.delta_control_expectation = result.delta_control_expectation;
      metadata.delta_control_applied = result.delta_control_applied;
      return unified_monte_carlo_result(result.monte_carlo, request.estimator, std::move(metadata));
    }
    case PricingEstimator::neural:
      throw std::invalid_argument("Monte Carlo backend cannot use the neural estimator");
    default:
      throw std::invalid_argument("unsupported pricing estimator");
  }
}

std::string to_string(PricingBackend backend) {
  switch (backend) {
    case PricingBackend::analytical:
      return "analytical";
    case PricingBackend::monte_carlo:
      return "Monte Carlo";
    case PricingBackend::neural:
      return "neural";
  }
  return "unknown";
}

std::string to_string(PricingEstimator estimator) {
  switch (estimator) {
    case PricingEstimator::analytical:
      return "analytical";
    case PricingEstimator::plain:
      return "plain";
    case PricingEstimator::antithetic:
      return "antithetic";
    case PricingEstimator::geometric_control_variate:
      return "geometric control variate";
    case PricingEstimator::neural:
      return "neural";
  }
  return "unknown";
}

}  // namespace nre

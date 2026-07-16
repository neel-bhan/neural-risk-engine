#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "nre/domain.hpp"
#include "nre/monte_carlo.hpp"

namespace nre {

enum class PricingBackend { analytical, monte_carlo };

enum class PricingEstimator { analytical, plain, antithetic, geometric_control_variate };

struct PricingRequest {
  OptionContract contract;
  MarketState market;
  PricingBackend backend{PricingBackend::analytical};
  PricingEstimator estimator{PricingEstimator::analytical};
  std::optional<MonteCarloConfig> monte_carlo_config;
  std::optional<ControlVariateConfig> control_variate_config;
};

struct PricingEstimate {
  double estimate;
  std::optional<double> sample_standard_error;
  std::optional<ConfidenceInterval> confidence_interval_95;
};

struct PricingMetadata {
  std::optional<std::size_t> effective_paths;
  std::optional<std::size_t> raw_paths;
  std::optional<std::uint64_t> seed;
  std::optional<std::size_t> requested_threads;
  std::optional<std::size_t> active_threads;
  std::optional<std::size_t> pilot_paths;
  std::optional<std::uint64_t> pilot_seed;
  std::optional<std::size_t> pilot_active_threads;
  std::optional<double> price_control_coefficient;
  std::optional<double> price_control_expectation;
  std::optional<bool> price_control_applied;
  std::optional<double> delta_control_coefficient;
  std::optional<double> delta_control_expectation;
  std::optional<bool> delta_control_applied;
};

struct UnifiedPricingResult {
  PricingEstimate price;
  PricingEstimate delta;
  PricingBackend backend;
  PricingEstimator estimator;
  PricingMetadata metadata;
};

// Validates contract and market inputs at the backend-neutral boundary. Unsupported combinations
// and missing/extraneous numerical configurations throw std::invalid_argument; no estimator is
// silently substituted.
[[nodiscard]] UnifiedPricingResult price(const PricingRequest& request);

[[nodiscard]] std::string to_string(PricingBackend backend);
[[nodiscard]] std::string to_string(PricingEstimator estimator);

}  // namespace nre

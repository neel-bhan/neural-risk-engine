#include "nre/risk.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nre {
namespace {

std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

double quantile(std::vector<double> values, double probability) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double position = probability * static_cast<double>(values.size() - 1U);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] + fraction * (values[upper] - values[lower]);
}

ErrorMetrics metrics_from_indices(const std::vector<UnifiedPricingResult>& results,
                                  const std::vector<UnifiedPricingResult>& references,
                                  const std::vector<std::size_t>& indices, double price_floor) {
  if (!std::isfinite(price_floor) || price_floor <= 0.0) {
    throw std::invalid_argument("price floor must be positive and finite");
  }
  std::vector<double> normalized_errors;
  normalized_errors.reserve(indices.size());
  double squared_delta_error = 0.0;
  for (const std::size_t index : indices) {
    if (index >= results.size() || index >= references.size()) {
      throw std::invalid_argument("error metric index is out of range");
    }
    const double predicted_price = results[index].price.estimate;
    const double predicted_delta = results[index].delta.estimate;
    const double reference_price = references[index].price.estimate;
    const double reference_delta = references[index].delta.estimate;
    if (!std::isfinite(predicted_price) || !std::isfinite(predicted_delta) ||
        !std::isfinite(reference_price) || !std::isfinite(reference_delta)) {
      throw std::invalid_argument("error metrics require finite estimates");
    }
    normalized_errors.push_back(
        std::abs(predicted_price - reference_price) / std::max(reference_price, price_floor));
    const double delta_error = predicted_delta - reference_delta;
    squared_delta_error += delta_error * delta_error;
  }
  return {
      .count = indices.size(),
      .median_normalized_price_error = quantile(normalized_errors, 0.50),
      .p99_normalized_price_error = quantile(normalized_errors, 0.99),
      .delta_rmse = indices.empty()
                        ? 0.0
                        : std::sqrt(squared_delta_error / static_cast<double>(indices.size())),
  };
}

}  // namespace

std::vector<PortfolioPosition> m10_synthetic_portfolio() {
  std::vector<PortfolioPosition> result;
  result.reserve(18U);
  const OptionStyle styles[]{OptionStyle::european, OptionStyle::geometric_asian,
                             OptionStyle::arithmetic_asian};
  const OptionType types[]{OptionType::call, OptionType::put};
  const double strikes[]{80.0, 100.0, 120.0};
  const double maturities[]{0.50, 1.00, 1.75};
  const double volatilities[]{0.18, 0.25, 0.32};
  const std::size_t observations[]{12U, 24U, 52U};
  for (const auto style : styles) {
    for (const auto type : types) {
      for (std::size_t variant = 0U; variant < 3U; ++variant) {
        const std::string id = to_string(style) + "_" + to_string(type) + "_" +
                               std::to_string(variant + 1U);
        result.push_back({
            .id = id,
            .contract = {.type = type,
                         .style = style,
                         .strike = strikes[variant],
                         .maturity_years = maturities[variant],
                         .observations = style == OptionStyle::european ? 1U
                                                                       : observations[variant]},
            .base_market = {.spot = 100.0,
                            .volatility = volatilities[variant],
                            .risk_free_rate = 0.03,
                            .dividend_yield = 0.01},
        });
      }
    }
  }
  return result;
}

std::vector<MarketScenario> m10_market_scenarios() {
  return {
      {.id = "spot_down_20_vol_down_5", .spot_multiplier = 0.80, .volatility_shift = -0.05},
      {.id = "spot_down_20_vol_up_10", .spot_multiplier = 0.80, .volatility_shift = 0.10},
      {.id = "spot_flat_vol_down_5", .spot_multiplier = 1.00, .volatility_shift = -0.05},
      {.id = "base", .spot_multiplier = 1.00, .volatility_shift = 0.00},
      {.id = "spot_flat_vol_up_10", .spot_multiplier = 1.00, .volatility_shift = 0.10},
      {.id = "spot_up_20_vol_down_5", .spot_multiplier = 1.20, .volatility_shift = -0.05},
      {.id = "spot_up_20_vol_up_10", .spot_multiplier = 1.20, .volatility_shift = 0.10},
      {.id = "ood_spot_down_45", .spot_multiplier = 0.55, .volatility_shift = 0.00},
      {.id = "ood_vol_up_45", .spot_multiplier = 1.00, .volatility_shift = 0.45},
  };
}

std::vector<PortfolioWorkItem> expand_portfolio_scenarios(
    const std::vector<PortfolioPosition>& portfolio,
    const std::vector<MarketScenario>& scenarios) {
  std::vector<PortfolioWorkItem> result;
  result.reserve(portfolio.size() * scenarios.size());
  for (const auto& position : portfolio) {
    if (position.id.empty() || !validate(position.contract).empty() ||
        !validate(position.base_market).empty()) {
      throw std::invalid_argument("portfolio contains an invalid position");
    }
    for (const auto& scenario : scenarios) {
      if (scenario.id.empty() || !std::isfinite(scenario.spot_multiplier) ||
          scenario.spot_multiplier <= 0.0 || !std::isfinite(scenario.volatility_shift)) {
        throw std::invalid_argument("portfolio contains an invalid scenario");
      }
      auto market = position.base_market;
      market.spot *= scenario.spot_multiplier;
      market.volatility += scenario.volatility_shift;
      if (!validate(market).empty()) {
        throw std::invalid_argument("scenario produces an invalid market state");
      }
      result.push_back({.ordinal = result.size(),
                        .position_id = position.id,
                        .scenario_id = scenario.id,
                        .contract = position.contract,
                        .market = market});
    }
  }
  return result;
}

std::vector<PricingRequest> make_portfolio_fallback_requests(
    const std::vector<PortfolioWorkItem>& items, std::size_t effective_paths,
    std::size_t pilot_paths, std::size_t thread_count, std::uint64_t master_seed) {
  if (effective_paths < 2U || pilot_paths < 2U || thread_count == 0U) {
    throw std::invalid_argument("portfolio fallback budgets and thread count must be valid");
  }
  std::vector<PricingRequest> requests;
  requests.reserve(items.size());
  for (std::size_t index = 0U; index < items.size(); ++index) {
    const auto& item = items[index];
    if (item.ordinal != index) {
      throw std::invalid_argument("portfolio work-item ordinals must be contiguous and ordered");
    }
    const std::uint64_t pricing_seed = splitmix64(master_seed + 2U * index);
    const std::uint64_t pilot_seed = splitmix64(master_seed + 2U * index + 1U);
    PricingRequest request{
        .contract = item.contract,
        .market = item.market,
        .backend = PricingBackend::monte_carlo,
        .estimator = item.contract.style == OptionStyle::arithmetic_asian
                         ? PricingEstimator::geometric_control_variate
                         : PricingEstimator::plain,
        .monte_carlo_config = std::nullopt,
        .control_variate_config = std::nullopt,
    };
    const MonteCarloConfig pricing{.seed = pricing_seed,
                                   .path_count = effective_paths,
                                   .thread_count = thread_count};
    if (item.contract.style == OptionStyle::arithmetic_asian) {
      request.control_variate_config = ControlVariateConfig{
          .pricing = pricing, .pilot_seed = pilot_seed, .pilot_path_count = pilot_paths};
    } else {
      request.monte_carlo_config = pricing;
    }
    requests.push_back(std::move(request));
  }
  return requests;
}

ErrorMetrics calculate_error_metrics(const std::vector<UnifiedPricingResult>& results,
                                     const std::vector<UnifiedPricingResult>& references,
                                     double price_floor) {
  if (results.size() != references.size()) {
    throw std::invalid_argument("result and reference counts differ");
  }
  std::vector<std::size_t> indices(results.size());
  for (std::size_t index = 0U; index < indices.size(); ++index) {
    indices[index] = index;
  }
  return metrics_from_indices(results, references, indices, price_floor);
}

ErrorMetrics calculate_accepted_neural_error_metrics(
    const GuardedBatchResult& routed, const std::vector<UnifiedPricingResult>& references,
    double price_floor) {
  if (routed.items.size() != references.size()) {
    throw std::invalid_argument("routed and reference counts differ");
  }
  std::vector<UnifiedPricingResult> results;
  std::vector<std::size_t> indices;
  results.reserve(routed.items.size());
  indices.reserve(routed.items.size());
  for (std::size_t index = 0U; index < routed.items.size(); ++index) {
    results.push_back(routed.items[index].result);
    if (routed.items[index].neural_accepted) {
      indices.push_back(index);
    }
  }
  return metrics_from_indices(results, references, indices, price_floor);
}

bool meets_error_tolerance(const ErrorMetrics& measured,
                           const ErrorMetrics& tolerance) noexcept {
  return measured.median_normalized_price_error <=
             tolerance.median_normalized_price_error &&
         measured.p99_normalized_price_error <= tolerance.p99_normalized_price_error &&
         measured.delta_rmse <= tolerance.delta_rmse;
}

MatchedToleranceCandidate select_matched_tolerance(
    const std::vector<MatchedToleranceCandidate>& candidates, const ErrorMetrics& tolerance) {
  std::size_t prior_paths = 0U;
  for (const auto& candidate : candidates) {
    if (candidate.effective_paths <= prior_paths) {
      throw std::invalid_argument("matched-tolerance candidates must have increasing path counts");
    }
    prior_paths = candidate.effective_paths;
    if (meets_error_tolerance(candidate.metrics, tolerance)) {
      return candidate;
    }
  }
  throw std::runtime_error("no Monte Carlo candidate meets the guarded-neural error tolerance");
}

std::vector<std::size_t> aggregate_fallback_reason_counts(const GuardedBatchResult& routed) {
  std::vector<std::size_t> result(
      static_cast<std::size_t>(NeuralRejectionReason::count), 0U);
  for (const auto& item : routed.items) {
    if (item.neural_accepted != (item.rejection_reason == NeuralRejectionReason::none)) {
      throw std::invalid_argument("routed item acceptance and reason disagree");
    }
    ++result[static_cast<std::size_t>(item.rejection_reason)];
  }
  for (std::size_t index = 0U; index < result.size(); ++index) {
    if (result[index] != routed.counters[index]) {
      throw std::invalid_argument("routed counters do not match item-level reasons");
    }
  }
  return result;
}

}  // namespace nre

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nre/domain.hpp"
#include "nre/neural_router.hpp"
#include "nre/pricing.hpp"

namespace nre {

// M10 uses a deliberately synthetic, deterministic portfolio. It is a benchmark workload, not a
// representation of client positions or calibrated market data.
struct PortfolioPosition {
  std::string id;
  OptionContract contract;
  MarketState base_market;
};

struct MarketScenario {
  std::string id;
  double spot_multiplier;
  double volatility_shift;
};

struct PortfolioWorkItem {
  std::size_t ordinal;
  std::string position_id;
  std::string scenario_id;
  OptionContract contract;
  MarketState market;
};

struct ErrorMetrics {
  std::size_t count{};
  double median_normalized_price_error{};
  double p99_normalized_price_error{};
  double delta_rmse{};
};

struct MatchedToleranceCandidate {
  std::size_t effective_paths{};
  ErrorMetrics metrics{};
};

[[nodiscard]] std::vector<PortfolioPosition> m10_synthetic_portfolio();
[[nodiscard]] std::vector<MarketScenario> m10_market_scenarios();

// Ordering is position-major then scenario-minor and ordinals are contiguous from zero.
[[nodiscard]] std::vector<PortfolioWorkItem> expand_portfolio_scenarios(
    const std::vector<PortfolioPosition>& portfolio,
    const std::vector<MarketScenario>& scenarios);

// European/geometric requests use plain Monte Carlo. Arithmetic Asians use the geometric control
// variate with a separate deterministic pilot stream. Seeds are stable functions of ordinal.
[[nodiscard]] std::vector<PricingRequest> make_portfolio_fallback_requests(
    const std::vector<PortfolioWorkItem>& items, std::size_t effective_paths,
    std::size_t pilot_paths, std::size_t thread_count, std::uint64_t master_seed);

[[nodiscard]] ErrorMetrics calculate_error_metrics(
    const std::vector<UnifiedPricingResult>& results,
    const std::vector<UnifiedPricingResult>& references, double price_floor = 1.0);

[[nodiscard]] ErrorMetrics calculate_accepted_neural_error_metrics(
    const GuardedBatchResult& routed, const std::vector<UnifiedPricingResult>& references,
    double price_floor = 1.0);

[[nodiscard]] bool meets_error_tolerance(const ErrorMetrics& measured,
                                         const ErrorMetrics& tolerance) noexcept;

// Candidates must be ordered by increasing work. Returns the first candidate meeting all three
// price/Delta criteria or throws when the grid contains no matched candidate.
[[nodiscard]] MatchedToleranceCandidate select_matched_tolerance(
    const std::vector<MatchedToleranceCandidate>& candidates, const ErrorMetrics& tolerance);

[[nodiscard]] std::vector<std::size_t> aggregate_fallback_reason_counts(
    const GuardedBatchResult& routed);

}  // namespace nre

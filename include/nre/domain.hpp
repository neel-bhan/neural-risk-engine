#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nre {

enum class OptionType { call, put };

enum class OptionStyle { european, geometric_asian, arithmetic_asian };

struct OptionContract {
  OptionType type{OptionType::call};
  OptionStyle style{OptionStyle::european};
  double strike{100.0};
  double maturity_years{1.0};
  std::size_t observations{1};
};

struct MarketState {
  double spot{100.0};
  double volatility{0.20};
  double risk_free_rate{0.05};
  double dividend_yield{0.0};
};

// Returns all validation failures so callers can report useful diagnostics at boundaries.
[[nodiscard]] std::vector<std::string> validate(const OptionContract& contract);
[[nodiscard]] std::vector<std::string> validate(const MarketState& market);

[[nodiscard]] std::string to_string(OptionType type);
[[nodiscard]] std::string to_string(OptionStyle style);

}  // namespace nre


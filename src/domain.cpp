#include "nre/domain.hpp"

#include <cmath>

namespace nre {

std::vector<std::string> validate(const OptionContract& contract) {
  std::vector<std::string> errors;

  if (!std::isfinite(contract.strike) || contract.strike <= 0.0) {
    errors.emplace_back("strike must be finite and greater than zero");
  }
  if (!std::isfinite(contract.maturity_years) || contract.maturity_years <= 0.0) {
    errors.emplace_back("maturity_years must be finite and greater than zero");
  }
  if (contract.style == OptionStyle::european && contract.observations != 1) {
    errors.emplace_back("a European option must have exactly one maturity observation");
  }
  if (contract.style != OptionStyle::european && contract.observations < 2) {
    errors.emplace_back("an Asian option must have at least two observations");
  }

  return errors;
}

std::vector<std::string> validate(const MarketState& market) {
  std::vector<std::string> errors;

  if (!std::isfinite(market.spot) || market.spot <= 0.0) {
    errors.emplace_back("spot must be finite and greater than zero");
  }
  if (!std::isfinite(market.volatility) || market.volatility < 0.0) {
    errors.emplace_back("volatility must be finite and non-negative");
  }
  if (!std::isfinite(market.risk_free_rate)) {
    errors.emplace_back("risk_free_rate must be finite");
  }
  if (!std::isfinite(market.dividend_yield)) {
    errors.emplace_back("dividend_yield must be finite");
  }

  return errors;
}

std::string to_string(OptionType type) {
  switch (type) {
    case OptionType::call:
      return "call";
    case OptionType::put:
      return "put";
  }
  return "unknown";
}

std::string to_string(OptionStyle style) {
  switch (style) {
    case OptionStyle::european:
      return "European";
    case OptionStyle::geometric_asian:
      return "geometric Asian";
    case OptionStyle::arithmetic_asian:
      return "arithmetic Asian";
  }
  return "unknown";
}

}  // namespace nre


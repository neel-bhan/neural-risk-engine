#pragma once

#include "nre/domain.hpp"

namespace nre {

struct PricingResult {
  double price;
  double delta;
};

// The contract and market must have passed validate before this function is called.
// Throws std::invalid_argument when the contract is not European.
[[nodiscard]] PricingResult black_scholes_european(const OptionContract& contract,
                                                   const MarketState& market);

// The contract and market must have passed validate before this function is called.
// Throws std::invalid_argument when the contract is not a geometric Asian option.
[[nodiscard]] PricingResult geometric_asian_analytical(const OptionContract& contract,
                                                       const MarketState& market);

}  // namespace nre

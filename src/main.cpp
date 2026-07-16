#include <iostream>

#include "nre/domain.hpp"

int main() {
  const nre::OptionContract contract{
      .type = nre::OptionType::call,
      .style = nre::OptionStyle::arithmetic_asian,
      .strike = 100.0,
      .maturity_years = 1.0,
      .observations = 12,
  };
  const nre::MarketState market{};

  const auto contract_errors = nre::validate(contract);
  const auto market_errors = nre::validate(market);
  if (!contract_errors.empty() || !market_errors.empty()) {
    std::cerr << "Invalid example configuration\n";
    return 1;
  }

  std::cout << "Neural Risk Engine analytical references are ready.\n"
            << "Example: " << nre::to_string(contract.style) << ' '
            << nre::to_string(contract.type) << ", strike=" << contract.strike
            << ", observations=" << contract.observations << '\n'
            << "Next milestone: correct scalar Monte Carlo pricing.\n";
}

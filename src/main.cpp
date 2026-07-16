#include <iostream>

#include "nre/domain.hpp"
#include "nre/monte_carlo.hpp"

int main() {
  const nre::OptionContract contract{
      .type = nre::OptionType::call,
      .style = nre::OptionStyle::european,
      .strike = 100.0,
      .maturity_years = 1.0,
      .observations = 1,
  };
  const nre::MarketState market{};
  const nre::MonteCarloConfig config{.seed = 42ULL, .path_count = 100000};

  const auto contract_errors = nre::validate(contract);
  const auto market_errors = nre::validate(market);
  if (!contract_errors.empty() || !market_errors.empty()) {
    std::cerr << "Invalid example configuration\n";
    return 1;
  }

  const auto result = nre::price_european_monte_carlo(contract, market, config);
  std::cout << "Neural Risk Engine scalar Monte Carlo is ready.\n"
            << "Example: " << nre::to_string(contract.style) << ' ' << nre::to_string(contract.type)
            << ", strike=" << contract.strike << ", observations=" << contract.observations << '\n'
            << "Estimate=" << result.estimate
            << ", sample standard error=" << result.sample_standard_error << ", 95% CI=["
            << result.confidence_interval_95.lower << ", " << result.confidence_interval_95.upper
            << "]\n"
            << "Paths=" << result.effective_paths << ", seed=" << result.seed << '\n'
            << "Next milestone: backend-neutral pricing and Monte Carlo Delta.\n";
}

#include <onnxruntime_c_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nre/onnx_backend.hpp"
#include "nre/risk.hpp"

namespace {

#ifndef NRE_BUILD_FLAGS
#define NRE_BUILD_FLAGS "not supplied"
#endif

constexpr std::size_t kThreadCount = 10U;
constexpr std::uint64_t kMasterSeed = 202607160010ULL;
constexpr std::size_t kGuardedFallbackPaths = 512U;
constexpr std::size_t kGuardedFallbackPilotPaths = 128U;
constexpr std::size_t kReferencePaths = 131072U;
constexpr std::size_t kReferencePilotPaths = 32768U;
constexpr std::size_t kWarmups = 2U;
constexpr std::size_t kRepetitions = 7U;
constexpr std::size_t kInferenceWarmups = 20U;
constexpr std::size_t kInferenceRepetitions = 100U;
constexpr double kPriceFloor = 1.0;
constexpr std::size_t kCandidatePaths[]{128U, 512U, 2048U, 8192U, 32768U, 131072U};

struct ReferenceEvidence {
  std::vector<nre::UnifiedPricingResult> results;
  std::size_t analytical_count{};
  std::size_t arithmetic_monte_carlo_count{};
  double maximum_arithmetic_price_standard_error{};
  double maximum_arithmetic_delta_standard_error{};
};

struct TimingEvidence {
  std::size_t warmups{};
  std::vector<double> milliseconds;
  double checksum{};
};

struct SliceMetrics {
  std::string id;
  nre::ErrorMetrics metrics;
  std::size_t neural_accepted_count{};
  std::array<std::size_t,
             static_cast<std::size_t>(nre::NeuralRejectionReason::count)>
      counters{};
};

struct WorstExample {
  std::size_t index{};
  double normalized_price_error{};
};

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

std::string hexadecimal(std::uint64_t value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(16U, '0');
  for (std::size_t index = 0U; index < result.size(); ++index) {
    result[result.size() - index - 1U] = digits[value & 0xFU];
    value >>= 4U;
  }
  return result;
}

std::string protocol_checksum(const std::vector<nre::PortfolioPosition>& portfolio,
                              const std::vector<nre::MarketScenario>& scenarios) {
  std::ostringstream canonical;
  canonical << std::setprecision(17);
  for (const auto& position : portfolio) {
    canonical << position.id << '|' << nre::to_string(position.contract.style) << '|'
              << nre::to_string(position.contract.type) << '|' << position.contract.strike << '|'
              << position.contract.maturity_years << '|' << position.contract.observations << '|'
              << position.base_market.spot << '|' << position.base_market.volatility << '|'
              << position.base_market.risk_free_rate << '|'
              << position.base_market.dividend_yield << '\n';
  }
  for (const auto& scenario : scenarios) {
    canonical << scenario.id << '|' << scenario.spot_multiplier << '|'
              << scenario.volatility_shift << '\n';
  }
  std::uint64_t hash = 14695981039346656037ULL;
  for (const char byte : canonical.str()) {
    hash ^= static_cast<unsigned char>(byte);
    hash *= 1099511628211ULL;
  }
  return hexadecimal(hash);
}

ReferenceEvidence make_references(const std::vector<nre::PortfolioWorkItem>& items) {
  ReferenceEvidence evidence{};
  evidence.results.reserve(items.size());
  const auto arithmetic_requests = nre::make_portfolio_fallback_requests(
      items, kReferencePaths, kReferencePilotPaths, kThreadCount, kMasterSeed + 900000U);
  for (std::size_t index = 0U; index < items.size(); ++index) {
    const auto& item = items[index];
    if (item.contract.style != nre::OptionStyle::arithmetic_asian) {
      const nre::PricingRequest analytical{
          .contract = item.contract,
          .market = item.market,
          .backend = nre::PricingBackend::analytical,
          .estimator = nre::PricingEstimator::analytical,
          .monte_carlo_config = std::nullopt,
          .control_variate_config = std::nullopt,
      };
      evidence.results.push_back(nre::price(analytical));
      ++evidence.analytical_count;
    } else {
      auto result = nre::price(arithmetic_requests[index]);
      evidence.maximum_arithmetic_price_standard_error =
          std::max(evidence.maximum_arithmetic_price_standard_error,
                   *result.price.sample_standard_error);
      evidence.maximum_arithmetic_delta_standard_error =
          std::max(evidence.maximum_arithmetic_delta_standard_error,
                   *result.delta.sample_standard_error);
      evidence.results.push_back(std::move(result));
      ++evidence.arithmetic_monte_carlo_count;
    }
  }
  return evidence;
}

std::vector<nre::UnifiedPricingResult> price_all(
    const std::vector<nre::PricingRequest>& requests) {
  std::vector<nre::UnifiedPricingResult> results;
  results.reserve(requests.size());
  for (const auto& request : requests) {
    results.push_back(nre::price(request));
  }
  return results;
}

std::vector<nre::UnifiedPricingResult> routed_results(const nre::GuardedBatchResult& routed) {
  std::vector<nre::UnifiedPricingResult> results;
  results.reserve(routed.items.size());
  for (const auto& item : routed.items) {
    results.push_back(item.result);
  }
  return results;
}

template <class Operation>
TimingEvidence time_operation(std::size_t warmups, std::size_t repetitions,
                              Operation&& operation) {
  double checksum = 0.0;
  for (std::size_t index = 0U; index < warmups; ++index) {
    checksum += operation();
  }
  std::vector<double> times;
  times.reserve(repetitions);
  for (std::size_t index = 0U; index < repetitions; ++index) {
    const auto start = std::chrono::steady_clock::now();
    checksum += operation();
    const auto end = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  return {.warmups = warmups, .milliseconds = std::move(times), .checksum = checksum};
}

double result_checksum(const std::vector<nre::UnifiedPricingResult>& results) {
  double checksum = 0.0;
  for (const auto& result : results) {
    checksum += result.price.estimate + result.delta.estimate;
  }
  return checksum;
}

double routed_checksum(const nre::GuardedBatchResult& routed) {
  double checksum = 0.0;
  for (const auto& item : routed.items) {
    checksum += item.result.price.estimate + item.result.delta.estimate;
  }
  return checksum;
}

std::vector<SliceMetrics> style_type_slices(
    const std::vector<nre::PortfolioWorkItem>& items,
    const std::vector<nre::UnifiedPricingResult>& results,
    const std::vector<nre::UnifiedPricingResult>& references,
    const nre::GuardedBatchResult& routed) {
  std::vector<SliceMetrics> slices;
  const nre::OptionStyle styles[]{nre::OptionStyle::european, nre::OptionStyle::geometric_asian,
                                  nre::OptionStyle::arithmetic_asian};
  const nre::OptionType types[]{nre::OptionType::call, nre::OptionType::put};
  for (const auto style : styles) {
    for (const auto type : types) {
      std::vector<nre::UnifiedPricingResult> selected;
      std::vector<nre::UnifiedPricingResult> selected_references;
      std::size_t accepted_count = 0U;
      std::array<std::size_t,
                 static_cast<std::size_t>(nre::NeuralRejectionReason::count)>
          counters{};
      for (std::size_t index = 0U; index < items.size(); ++index) {
        if (items[index].contract.style == style && items[index].contract.type == type) {
          selected.push_back(results[index]);
          selected_references.push_back(references[index]);
          accepted_count += routed.items[index].neural_accepted ? 1U : 0U;
          ++counters[static_cast<std::size_t>(routed.items[index].rejection_reason)];
        }
      }
      slices.push_back({.id = nre::to_string(style) + "_" + nre::to_string(type),
                        .metrics = nre::calculate_error_metrics(selected, selected_references,
                                                                kPriceFloor),
                        .neural_accepted_count = accepted_count,
                        .counters = counters});
    }
  }
  return slices;
}

std::vector<SliceMetrics> scenario_slices(
    const std::vector<nre::PortfolioWorkItem>& items,
    const std::vector<nre::MarketScenario>& scenarios,
    const std::vector<nre::UnifiedPricingResult>& results,
    const std::vector<nre::UnifiedPricingResult>& references,
    const nre::GuardedBatchResult& routed) {
  std::vector<SliceMetrics> slices;
  for (const auto& scenario : scenarios) {
    std::vector<nre::UnifiedPricingResult> selected;
    std::vector<nre::UnifiedPricingResult> selected_references;
    std::size_t accepted_count = 0U;
    std::array<std::size_t,
               static_cast<std::size_t>(nre::NeuralRejectionReason::count)>
        counters{};
    for (std::size_t index = 0U; index < items.size(); ++index) {
      if (items[index].scenario_id == scenario.id) {
        selected.push_back(results[index]);
        selected_references.push_back(references[index]);
        accepted_count += routed.items[index].neural_accepted ? 1U : 0U;
        ++counters[static_cast<std::size_t>(routed.items[index].rejection_reason)];
      }
    }
    slices.push_back({.id = scenario.id,
                      .metrics = nre::calculate_error_metrics(selected, selected_references,
                                                              kPriceFloor),
                      .neural_accepted_count = accepted_count,
                      .counters = counters});
  }
  return slices;
}

void write_metrics(std::ostream& output, const nre::ErrorMetrics& metrics,
                   const std::string& indent) {
  output << indent << "\"count\": " << metrics.count << ",\n"
         << indent << "\"median_normalized_price_error\": "
         << metrics.median_normalized_price_error << ",\n"
         << indent << "\"p99_normalized_price_error\": "
         << metrics.p99_normalized_price_error << ",\n"
         << indent << "\"delta_rmse\": " << metrics.delta_rmse << '\n';
}

void write_timing(std::ostream& output, const TimingEvidence& timing, std::size_t workload_size,
                  const std::string& indent) {
  const double median_ms = quantile(timing.milliseconds, 0.50);
  output << indent << "\"warmups\": " << timing.warmups << ",\n"
         << indent << "\"repetitions\": " << timing.milliseconds.size() << ",\n"
         << indent << "\"median_milliseconds\": " << median_ms << ",\n"
         << indent << "\"empirical_p99_milliseconds\": "
         << quantile(timing.milliseconds, 0.99) << ",\n"
         << indent << "\"repricings_per_second_at_median\": "
         << 1000.0 * static_cast<double>(workload_size) / median_ms << ",\n"
         << indent << "\"result_checksum\": " << timing.checksum << '\n';
}

void write_slices(std::ostream& output, const std::vector<SliceMetrics>& slices,
                  const std::string& indent) {
  output << "[\n";
  for (std::size_t index = 0U; index < slices.size(); ++index) {
    output << indent << "{\"id\": \"" << slices[index].id << "\", \"metrics\": {\n";
    write_metrics(output, slices[index].metrics, indent + "  ");
    output << indent << "}, \"neural_accepted_count\": "
           << slices[index].neural_accepted_count << ", \"fallback_count\": "
           << slices[index].metrics.count - slices[index].neural_accepted_count
           << ", \"fallback_reason_counts\": {";
    for (std::size_t reason_index = 1U;
         reason_index < static_cast<std::size_t>(nre::NeuralRejectionReason::count);
         ++reason_index) {
      const auto reason = static_cast<nre::NeuralRejectionReason>(reason_index);
      output << "\"" << nre::to_string(reason) << "\":"
             << slices[index].counters[reason_index]
             << (reason_index + 1U ==
                         static_cast<std::size_t>(nre::NeuralRejectionReason::count)
                     ? ""
                     : ",");
    }
    output << "}}" << (index + 1U == slices.size() ? "\n" : ",\n");
  }
  output << indent.substr(0U, indent.size() - 2U) << ']';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7 || std::string(argv[1]) != "--metadata" ||
        std::string(argv[3]) != "--model" || std::string(argv[5]) != "--output") {
      std::cerr << "usage: " << argv[0]
                << " --metadata artifact.json --model model.onnx --output report.json\n";
      return 2;
    }
    const auto portfolio = nre::m10_synthetic_portfolio();
    const auto scenarios = nre::m10_market_scenarios();
    const auto items = nre::expand_portfolio_scenarios(portfolio, scenarios);
    nre::OnnxPricingBackend backend(argv[2], argv[4]);
    const auto references = make_references(items);

    const auto guarded_requests = nre::make_portfolio_fallback_requests(
        items, kGuardedFallbackPaths, kGuardedFallbackPilotPaths, kThreadCount, kMasterSeed);
    const auto routed = nre::price_guarded_neural_batch(guarded_requests, backend,
                                                        backend.guardrail_config());
    static_cast<void>(nre::aggregate_fallback_reason_counts(routed));
    const auto full_results = routed_results(routed);
    const auto full_metrics =
        nre::calculate_error_metrics(full_results, references.results, kPriceFloor);
    const auto accepted_metrics =
        nre::calculate_accepted_neural_error_metrics(routed, references.results, kPriceFloor);

    std::vector<nre::MatchedToleranceCandidate> candidates;
    std::vector<std::vector<nre::PricingRequest>> candidate_requests;
    for (const std::size_t paths : kCandidatePaths) {
      candidate_requests.push_back(nre::make_portfolio_fallback_requests(
          items, paths, std::max<std::size_t>(64U, paths / 4U), kThreadCount,
          kMasterSeed + 100000U));
      const auto candidate_results = price_all(candidate_requests.back());
      candidates.push_back({
          .effective_paths = paths,
          .metrics = nre::calculate_error_metrics(candidate_results, references.results,
                                                  kPriceFloor),
      });
    }
    const auto selected = nre::select_matched_tolerance(candidates, full_metrics);
    const auto selected_iterator =
        std::find(kCandidatePaths, kCandidatePaths + std::size(kCandidatePaths),
                  selected.effective_paths);
    const std::size_t selected_index =
        static_cast<std::size_t>(selected_iterator - kCandidatePaths);
    const auto& matched_requests = candidate_requests[selected_index];

    std::vector<nre::PricingRequest> rejected_requests;
    for (std::size_t index = 0U; index < routed.items.size(); ++index) {
      if (!routed.items[index].neural_accepted) {
        rejected_requests.push_back(guarded_requests[index]);
      }
    }
    std::vector<nre::NeuralInput> neural_inputs;
    neural_inputs.reserve(items.size());
    for (const auto& item : items) {
      neural_inputs.push_back({.contract = item.contract, .market = item.market});
    }

    const auto inference_timing = time_operation(
        kInferenceWarmups, kInferenceRepetitions, [&backend, &neural_inputs]() {
          const auto values = backend.predict(neural_inputs);
          double checksum = 0.0;
          for (const auto& value : values) {
            checksum += value.price + value.delta;
          }
          return checksum;
        });
    const auto routed_timing = time_operation(kWarmups, kRepetitions, [&]() {
      return routed_checksum(nre::price_guarded_neural_batch(
          guarded_requests, backend, backend.guardrail_config()));
    });
    const auto fallback_timing = time_operation(kWarmups, kRepetitions, [&]() {
      return result_checksum(price_all(rejected_requests));
    });
    const auto monte_carlo_timing = time_operation(kWarmups, kRepetitions, [&]() {
      return result_checksum(price_all(matched_requests));
    });

    const double routed_median = quantile(routed_timing.milliseconds, 0.50);
    const double monte_carlo_median = quantile(monte_carlo_timing.milliseconds, 0.50);
    const std::size_t accepted_count = accepted_metrics.count;
    const std::size_t fallback_count = items.size() - accepted_count;
    std::size_t near_zero_reference_count = 0U;
    for (const auto& reference : references.results) {
      if (reference.price.estimate < kPriceFloor) {
        ++near_zero_reference_count;
      }
    }
    const auto style_slices =
        style_type_slices(items, full_results, references.results, routed);
    const auto scenario_metrics =
        scenario_slices(items, scenarios, full_results, references.results, routed);
    std::vector<WorstExample> worst_examples;
    worst_examples.reserve(items.size());
    for (std::size_t index = 0U; index < items.size(); ++index) {
      worst_examples.push_back({
          .index = index,
          .normalized_price_error =
              std::abs(full_results[index].price.estimate - references.results[index].price.estimate) /
              std::max(references.results[index].price.estimate, kPriceFloor),
      });
    }
    std::sort(worst_examples.begin(), worst_examples.end(),
              [](const WorstExample& left, const WorstExample& right) {
                return left.normalized_price_error > right.normalized_price_error;
              });
    worst_examples.resize(std::min<std::size_t>(5U, worst_examples.size()));

    std::ofstream output(argv[6]);
    if (!output) {
      throw std::runtime_error("cannot write M10 portfolio report");
    }
    output << std::setprecision(17);
    output << "{\n"
           << "  \"result_version\": \"nre.portfolio_benchmark.v1\",\n"
           << "  \"source_implementation_commit\": \"PENDING\",\n"
           << "  \"protocol_fnv1a64\": \"" << protocol_checksum(portfolio, scenarios)
           << "\",\n"
           << "  \"artifact_version\": \"" << backend.artifact_version() << "\",\n"
           << "  \"model_fnv1a64\": \"" << backend.model_checksum() << "\",\n"
           << "  \"workload\": {\n"
           << "    \"description\": \"deterministic synthetic portfolio; not client positions or calibrated market data\",\n"
           << "    \"position_count\": " << portfolio.size() << ",\n"
           << "    \"scenario_count\": " << scenarios.size() << ",\n"
           << "    \"repricing_count\": " << items.size() << ",\n"
           << "    \"ordering\": \"position-major then scenario-minor\",\n"
           << "    \"spot_multipliers\": [0.80, 1.00, 1.20, 0.55],\n"
           << "    \"volatility_shifts\": [-0.05, 0.00, 0.10, 0.45],\n"
           << "    \"declared_out_of_domain_scenario_count\": 2,\n"
           << "    \"declared_out_of_domain_repricing_count\": 36,\n"
           << "    \"near_zero_reference_price_count\": " << near_zero_reference_count << "\n"
           << "  },\n"
           << "  \"reference\": {\n"
           << "    \"analytical_repricing_count\": " << references.analytical_count << ",\n"
           << "    \"arithmetic_control_variate_repricing_count\": "
           << references.arithmetic_monte_carlo_count << ",\n"
           << "    \"arithmetic_effective_paths\": " << kReferencePaths << ",\n"
           << "    \"arithmetic_pilot_paths\": " << kReferencePilotPaths << ",\n"
           << "    \"maximum_arithmetic_price_standard_error\": "
           << references.maximum_arithmetic_price_standard_error << ",\n"
           << "    \"maximum_arithmetic_delta_standard_error\": "
           << references.maximum_arithmetic_delta_standard_error << "\n"
           << "  },\n"
           << "  \"matched_tolerance\": {\n"
           << "    \"rule\": \"smallest predeclared Monte Carlo path grid candidate whose median and p99 normalized price error and Delta RMSE are each no worse than the full guarded route on the identical reference grid\",\n"
           << "    \"price_floor\": " << kPriceFloor << ",\n"
           << "    \"candidate_effective_paths\": [128, 512, 2048, 8192, 32768, 131072],\n"
           << "    \"selected_effective_paths\": " << selected.effective_paths << ",\n"
           << "    \"selected_pilot_paths\": "
           << std::max<std::size_t>(64U, selected.effective_paths / 4U) << ",\n"
           << "    \"guarded_route_tolerance\": {\n";
    write_metrics(output, full_metrics, "      ");
    output << "    },\n    \"selected_monte_carlo_metrics\": {\n";
    write_metrics(output, selected.metrics, "      ");
    output << "    },\n    \"candidate_measurements\": [\n";
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
      output << "      {\"effective_paths\": " << candidates[index].effective_paths
             << ", \"metrics\": {\n";
      write_metrics(output, candidates[index].metrics, "        ");
      output << "      }}" << (index + 1U == candidates.size() ? "\n" : ",\n");
    }
    output << "    ]\n  },\n"
           << "  \"routing\": {\n"
           << "    \"neural_accepted_count\": " << accepted_count << ",\n"
           << "    \"monte_carlo_fallback_count\": " << fallback_count << ",\n"
           << "    \"neural_acceptance_rate\": "
           << static_cast<double>(accepted_count) / static_cast<double>(items.size()) << ",\n"
           << "    \"monte_carlo_fallback_rate\": "
           << static_cast<double>(fallback_count) / static_cast<double>(items.size()) << ",\n"
           << "    \"guarded_fallback_effective_paths\": " << kGuardedFallbackPaths << ",\n"
           << "    \"guarded_fallback_pilot_paths\": " << kGuardedFallbackPilotPaths << ",\n"
           << "    \"fallback_reason_counts\": {\n";
    for (std::size_t reason_index = 1U;
         reason_index < static_cast<std::size_t>(nre::NeuralRejectionReason::count);
         ++reason_index) {
      const auto reason = static_cast<nre::NeuralRejectionReason>(reason_index);
      output << "      \"" << nre::to_string(reason) << "\": "
             << routed.counters[reason_index]
             << (reason_index + 1U ==
                         static_cast<std::size_t>(nre::NeuralRejectionReason::count)
                     ? "\n"
                     : ",\n");
    }
    output << "    },\n    \"accepted_neural_metrics\": {\n";
    write_metrics(output, accepted_metrics, "      ");
    output << "    },\n    \"full_routed_metrics\": {\n";
    write_metrics(output, full_metrics, "      ");
    output << "    }\n  },\n"
           << "  \"timing\": {\n"
           << "    \"timer\": \"std::chrono::steady_clock\",\n"
           << "    \"empirical_p99_definition\": \"linear empirical quantile; seven-repeat end-to-end p99 is near the observed maximum and is not a production tail estimate\",\n"
           << "    \"raw_neural_price_delta_batch\": {\n";
    write_timing(output, inference_timing, items.size(), "      ");
    output << "    },\n    \"guarded_route_including_checks_and_fallback\": {\n";
    write_timing(output, routed_timing, items.size(), "      ");
    output << "    },\n    \"trusted_fallback_subset_only\": {\n";
    write_timing(output, fallback_timing, fallback_count, "      ");
    output << "    },\n    \"matched_tolerance_all_monte_carlo\": {\n";
    write_timing(output, monte_carlo_timing, items.size(), "      ");
    output << "    },\n"
           << "    \"guarded_neural_speedup_vs_matched_monte_carlo\": "
           << monte_carlo_median / routed_median << "\n"
           << "  },\n"
           << "  \"full_routed_style_type_slices\": ";
    write_slices(output, style_slices, "    ");
    output << ",\n  \"full_routed_scenario_slices\": ";
    write_slices(output, scenario_metrics, "    ");
    output << ",\n"
           << "  \"worst_full_routed_examples\": [\n";
    for (std::size_t rank = 0U; rank < worst_examples.size(); ++rank) {
      const auto& worst = worst_examples[rank];
      const auto& item = items[worst.index];
      const auto& routed_item = routed.items[worst.index];
      output << "    {\"position_id\": \"" << item.position_id
             << "\", \"scenario_id\": \"" << item.scenario_id
             << "\", \"reference_price\": " << references.results[worst.index].price.estimate
             << ", \"routed_price\": " << full_results[worst.index].price.estimate
             << ", \"normalized_price_error\": " << worst.normalized_price_error
             << ", \"reference_delta\": " << references.results[worst.index].delta.estimate
             << ", \"routed_delta\": " << full_results[worst.index].delta.estimate
             << ", \"neural_accepted\": "
             << (routed_item.neural_accepted ? "true" : "false")
             << ", \"rejection_reason\": \""
             << nre::to_string(routed_item.rejection_reason) << "\"}"
             << (rank + 1U == worst_examples.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"environment\": {\n"
           << "    \"measured_date\": \"2026-07-16\",\n"
           << "    \"hardware\": \"Apple M4 MacBook Air, arm64\",\n"
           << "    \"hardware_concurrency\": " << std::thread::hardware_concurrency() << ",\n"
           << "    \"configured_threads_per_monte_carlo_request\": " << kThreadCount << ",\n"
           << "    \"onnxruntime_cpp\": \"" << OrtGetApiBase()->GetVersionString() << "\",\n"
           << "    \"compiler\": \"" << __VERSION__ << "\",\n"
           << "    \"compiler_flags\": \"" << NRE_BUILD_FLAGS << "\",\n"
           << "    \"cpp_standard\": \"C++20\",\n"
           << "    \"master_seed\": " << kMasterSeed << "\n"
           << "  },\n"
           << "  \"limitations\": [\n"
           << "    \"The portfolio and shocks are synthetic and the model is not calibrated to market data.\",\n"
           << "    \"The arithmetic-Asian reference is a finite high-budget control-variate estimate; its maximum reported sampling errors provide context rather than certainty.\",\n"
           << "    \"Matched-tolerance selection is descriptive on this frozen portfolio and candidate grid, not a universal speedup claim.\",\n"
           << "    \"Guardrails are finite engineering checks, not formal no-arbitrage guarantees, calibrated confidence, or general OOD detection.\",\n"
           << "    \"Timing applies to this machine and includes per-request Monte Carlo thread creation; it is not a production latency claim.\"\n"
           << "  ]\n"
           << "}\n";
    std::cout << "wrote M10 benchmark: " << items.size() << " repricings, "
              << accepted_count << " neural accepted, matched MC " << selected.effective_paths
              << " paths\n";
  } catch (const std::exception& error) {
    std::cerr << "M10 portfolio benchmark failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

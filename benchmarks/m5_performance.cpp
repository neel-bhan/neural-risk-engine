#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "nre/pricing.hpp"

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kWarmups = 1U;
constexpr std::size_t kRepetitions = 7U;
constexpr std::uint64_t kMasterSeed = 2026071601ULL;
constexpr std::uint64_t kPilotSeed = 2026071602ULL;
constexpr std::size_t kPilotPaths = 20000U;
constexpr double kTargetCiWidth = 0.10;
constexpr std::string_view kBuildFlags =
    "-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror";

volatile double benchmark_sink = 0.0;

enum class Workload { european_plain, arithmetic_plain, arithmetic_antithetic, arithmetic_control };

struct TimedResult {
  nre::UnifiedPricingResult result;
  double milliseconds;
};

std::string_view workload_name(Workload workload) {
  switch (workload) {
    case Workload::european_plain:
      return "european_plain";
    case Workload::arithmetic_plain:
      return "arithmetic_plain";
    case Workload::arithmetic_antithetic:
      return "arithmetic_antithetic";
    case Workload::arithmetic_control:
      return "arithmetic_control_variate";
  }
  return "unknown";
}

nre::PricingRequest request_for(Workload workload, std::size_t paths, std::size_t threads) {
  const bool european = workload == Workload::european_plain;
  nre::PricingRequest request{
      .contract =
          {
              .type = nre::OptionType::call,
              .style = european ? nre::OptionStyle::european
                                : nre::OptionStyle::arithmetic_asian,
              .strike = 100.0,
              .maturity_years = 1.0,
              .observations = european ? 1U : 12U,
          },
      .market =
          {
              .spot = 100.0,
              .volatility = 0.20,
              .risk_free_rate = 0.05,
              .dividend_yield = 0.02,
          },
      .backend = nre::PricingBackend::monte_carlo,
      .estimator = nre::PricingEstimator::plain,
  };
  const nre::MonteCarloConfig config{
      .seed = kMasterSeed, .path_count = paths, .thread_count = threads};
  if (workload == Workload::arithmetic_antithetic) {
    request.estimator = nre::PricingEstimator::antithetic;
  } else if (workload == Workload::arithmetic_control) {
    request.estimator = nre::PricingEstimator::geometric_control_variate;
    request.control_variate_config = nre::ControlVariateConfig{
        .pricing = config, .pilot_seed = kPilotSeed, .pilot_path_count = kPilotPaths};
    return request;
  }
  request.monte_carlo_config = config;
  return request;
}

TimedResult time_request(const nre::PricingRequest& request) {
  const auto start = Clock::now();
  auto result = nre::price(request);
  const auto stop = Clock::now();
  benchmark_sink = benchmark_sink + result.price.estimate + result.delta.estimate;
  return {
      .result = std::move(result),
      .milliseconds = std::chrono::duration<double, std::milli>(stop - start).count(),
  };
}

double percentile(std::vector<double> values, double probability) {
  std::sort(values.begin(), values.end());
  const double scaled = probability * static_cast<double>(values.size() - 1U);
  const std::size_t index = static_cast<std::size_t>(std::ceil(scaled));
  return values[index];
}

std::string compiler_name() {
#if defined(__apple_build_version__)
  return "AppleClang " + std::to_string(__clang_major__) + "." +
         std::to_string(__clang_minor__) + "." + std::to_string(__clang_patchlevel__) +
         " build " + std::to_string(__apple_build_version__);
#elif defined(__clang__)
  return "Clang " + std::to_string(__clang_major__) + "." +
         std::to_string(__clang_minor__) + "." + std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
  return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
         std::to_string(__GNUC_PATCHLEVEL__);
#else
  return "unknown";
#endif
}

std::string architecture_name() {
#if defined(__aarch64__) || defined(__arm64__)
  return "arm64";
#elif defined(__x86_64__)
  return "x86_64";
#else
  return "unknown";
#endif
}

void run_profile(Workload workload) {
  const std::size_t paths = workload == Workload::european_plain ? 5000000U : 500000U;
  const auto start = Clock::now();
  for (std::size_t repetition = 0; repetition < 30U; ++repetition) {
    static_cast<void>(time_request(request_for(workload, paths, 1U)));
  }
  const auto stop = Clock::now();
  const double total_milliseconds =
      std::chrono::duration<double, std::milli>(stop - start).count();
  std::cout << "{\"profile_workload\":\"" << workload_name(workload)
            << "\",\"paths_per_call\":" << paths
            << ",\"calls\":30,\"total_ms\":" << total_milliseconds << "}\n";
}

void run_benchmarks() {
  const unsigned int reported_hardware_threads = std::thread::hardware_concurrency();
  std::set<std::size_t> thread_set{1U, 2U, 4U, 8U};
  if (reported_hardware_threads != 0U) {
    thread_set.insert(static_cast<std::size_t>(reported_hardware_threads));
  }
  const std::vector<std::size_t> thread_counts(thread_set.begin(), thread_set.end());
  const std::vector<Workload> workloads{
      Workload::european_plain,
      Workload::arithmetic_plain,
      Workload::arithmetic_antithetic,
      Workload::arithmetic_control,
  };

  std::cout << std::setprecision(10);
  for (const Workload workload : workloads) {
    const std::size_t fixed_paths =
        workload == Workload::european_plain ? 1000000U : 250000U;
    double scalar_median_ms = 0.0;
    for (const std::size_t threads : thread_counts) {
      const auto fixed_request = request_for(workload, fixed_paths, threads);
      for (std::size_t warmup = 0; warmup < kWarmups; ++warmup) {
        static_cast<void>(time_request(fixed_request));
      }

      std::vector<double> latencies;
      latencies.reserve(kRepetitions);
      nre::UnifiedPricingResult final_result{};
      for (std::size_t repetition = 0; repetition < kRepetitions; ++repetition) {
        auto timed = time_request(fixed_request);
        latencies.push_back(timed.milliseconds);
        final_result = std::move(timed.result);
      }
      const double median_ms = percentile(latencies, 0.50);
      const double p99_ms = percentile(latencies, 0.99);
      if (threads == 1U) {
        scalar_median_ms = median_ms;
      }
      const double raw_paths = static_cast<double>(*final_result.metadata.raw_paths);
      const double million_raw_paths_per_second = raw_paths / (median_ms * 1000.0);

      std::size_t target_paths = 0U;
      double target_latency_ms = 0.0;
      double achieved_width = 0.0;
      for (std::size_t candidate = 1000U; candidate <= 1024000U; candidate *= 4U) {
        const auto timed = time_request(request_for(workload, candidate, threads));
        const auto interval = *timed.result.price.confidence_interval_95;
        achieved_width = interval.upper - interval.lower;
        if (achieved_width <= kTargetCiWidth) {
          target_paths = candidate;
          target_latency_ms = timed.milliseconds;
          break;
        }
      }

      std::cout << "{\"schema\":\"nre.m5.performance.v1\",\"workload\":\""
                << workload_name(workload) << "\",\"compiler\":\"" << compiler_name()
                << "\",\"flags\":\"" << kBuildFlags << "\",\"architecture\":\""
                << architecture_name() << "\",\"hardware_threads\":"
                << reported_hardware_threads << ",\"requested_threads\":" << threads
                << ",\"active_threads\":" << *final_result.metadata.active_threads
                << ",\"master_seed\":" << kMasterSeed
                << ",\"seed_policy\":\"scalar master seed; SplitMix64(master,worker_index) for multiworker mt19937_64 streams\""
                << ",\"effective_paths\":" << *final_result.metadata.effective_paths
                << ",\"raw_paths\":" << *final_result.metadata.raw_paths
                << ",\"observations\":" << fixed_request.contract.observations
                << ",\"warmups\":" << kWarmups << ",\"repetitions\":" << kRepetitions
                << ",\"timer\":\"std::chrono::steady_clock\",\"median_latency_ms\":"
                << median_ms << ",\"p99_latency_ms\":" << p99_ms
                << ",\"million_raw_paths_per_second\":" << million_raw_paths_per_second
                << ",\"speedup_vs_scalar\":" << scalar_median_ms / median_ms
                << ",\"target_ci_full_width\":" << kTargetCiWidth
                << ",\"target_grid_effective_paths\":" << target_paths
                << ",\"time_to_target_ci_ms\":" << target_latency_ms
                << ",\"target_achieved_width\":" << achieved_width << "}\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2) {
    const std::string_view argument(argv[1]);
    if (argument == "--profile-european") {
      run_profile(Workload::european_plain);
      return EXIT_SUCCESS;
    }
    if (argument == "--profile-arithmetic") {
      run_profile(Workload::arithmetic_plain);
      return EXIT_SUCCESS;
    }
  }
  if (argc != 1) {
    std::cerr << "usage: nre_m5_performance [--profile-european|--profile-arithmetic]\n";
    return EXIT_FAILURE;
  }
  run_benchmarks();
  return EXIT_SUCCESS;
}

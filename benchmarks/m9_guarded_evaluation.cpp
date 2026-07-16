#include <onnxruntime_c_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "nre/neural_router.hpp"
#include "nre/onnx_backend.hpp"

namespace {

#ifndef NRE_BUILD_FLAGS
#define NRE_BUILD_FLAGS "not supplied"
#endif

struct EvaluationRow {
  nre::PricingRequest request;
  double reference_price;
  double reference_delta;
};

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t begin = 0U;
  while (begin <= line.size()) {
    const std::size_t comma = line.find(',', begin);
    fields.push_back(line.substr(begin, comma - begin));
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1U;
  }
  return fields;
}

nre::OptionStyle parse_style(const std::string& value) {
  if (value == "european") {
    return nre::OptionStyle::european;
  }
  if (value == "geometric_asian") {
    return nre::OptionStyle::geometric_asian;
  }
  if (value == "arithmetic_asian") {
    return nre::OptionStyle::arithmetic_asian;
  }
  throw std::runtime_error("unknown option style in evaluation dataset");
}

nre::OptionType parse_type(const std::string& value) {
  if (value == "call") {
    return nre::OptionType::call;
  }
  if (value == "put") {
    return nre::OptionType::put;
  }
  throw std::runtime_error("unknown option type in evaluation dataset");
}

EvaluationRow parse_row(const std::vector<std::string>& fields) {
  if (fields.size() < 44U) {
    throw std::runtime_error("M9 evaluation encountered a truncated labels.csv row");
  }
  const auto style = parse_style(fields[6]);
  nre::PricingRequest request{
      .contract = {.type = parse_type(fields[7]),
                   .style = style,
                   .strike = std::stod(fields[9]),
                   .maturity_years = std::stod(fields[10]),
                   .observations = static_cast<std::size_t>(std::stoull(fields[14]))},
      .market = {.spot = std::stod(fields[8]),
                 .volatility = std::stod(fields[11]),
                 .risk_free_rate = std::stod(fields[12]),
                 .dividend_yield = std::stod(fields[13])},
      .backend = nre::PricingBackend::monte_carlo,
      .estimator = style == nre::OptionStyle::arithmetic_asian
                       ? nre::PricingEstimator::geometric_control_variate
                       : nre::PricingEstimator::plain,
      .monte_carlo_config = std::nullopt,
      .control_variate_config = std::nullopt,
  };
  const auto pricing = nre::MonteCarloConfig{
      .seed = static_cast<std::uint64_t>(std::stoull(fields[28])),
      .path_count = static_cast<std::size_t>(std::stoull(fields[26])),
      .thread_count = static_cast<std::size_t>(std::stoull(fields[29])),
  };
  if (style == nre::OptionStyle::arithmetic_asian) {
    request.control_variate_config = nre::ControlVariateConfig{
        .pricing = pricing,
        .pilot_seed = static_cast<std::uint64_t>(std::stoull(fields[32])),
        .pilot_path_count = static_cast<std::size_t>(std::stoull(fields[31])),
    };
  } else {
    request.monte_carlo_config = pricing;
  }
  return {
      .request = request,
      .reference_price = std::stod(fields[18]),
      .reference_delta = std::stod(fields[22]),
  };
}

std::vector<EvaluationRow> load_test_rows(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open M9 evaluation labels: " + path.string());
  }
  std::string line;
  static_cast<void>(std::getline(input, line));
  std::vector<EvaluationRow> rows;
  while (std::getline(input, line)) {
    const auto fields = split_csv(line);
    if (fields.size() >= 4U && fields[2] == "test" && fields[3] == "true") {
      rows.push_back(parse_row(fields));
    }
  }
  if (rows.empty()) {
    throw std::runtime_error("M9 evaluation dataset contains no accepted held-out rows");
  }
  return rows;
}

std::string fnv1a64_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot checksum evaluation dataset");
  }
  std::uint64_t hash = 14695981039346656037ULL;
  char byte = 0;
  while (input.get(byte)) {
    hash ^= static_cast<unsigned char>(byte);
    hash *= 1099511628211ULL;
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
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

struct Metrics {
  std::size_t count{};
  double median_normalized_price_error{};
  double p99_normalized_price_error{};
  double delta_rmse{};
};

Metrics metrics(const std::vector<double>& prices, const std::vector<double>& deltas,
                const std::vector<double>& reference_prices,
                const std::vector<double>& reference_deltas) {
  if (prices.size() != deltas.size() || prices.size() != reference_prices.size() ||
      prices.size() != reference_deltas.size()) {
    throw std::runtime_error("M9 metric input sizes differ");
  }
  std::vector<double> normalized_errors;
  normalized_errors.reserve(prices.size());
  double squared_delta_error = 0.0;
  for (std::size_t index = 0U; index < prices.size(); ++index) {
    normalized_errors.push_back(std::abs(prices[index] - reference_prices[index]) /
                                std::max(reference_prices[index], 1.0));
    const double delta_error = deltas[index] - reference_deltas[index];
    squared_delta_error += delta_error * delta_error;
  }
  return {
      .count = prices.size(),
      .median_normalized_price_error = quantile(normalized_errors, 0.50),
      .p99_normalized_price_error = quantile(normalized_errors, 0.99),
      .delta_rmse = prices.empty()
                        ? 0.0
                        : std::sqrt(squared_delta_error / static_cast<double>(prices.size())),
  };
}

void write_metrics(std::ostream& output, const Metrics& value, const std::string& indent) {
  output << indent << "\"count\": " << value.count << ",\n"
         << indent << "\"median_normalized_price_error\": "
         << value.median_normalized_price_error << ",\n"
         << indent << "\"p99_normalized_price_error\": "
         << value.p99_normalized_price_error << ",\n"
         << indent << "\"delta_rmse\": " << value.delta_rmse << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9 || std::string(argv[1]) != "--dataset" ||
        std::string(argv[3]) != "--metadata" || std::string(argv[5]) != "--model" ||
        std::string(argv[7]) != "--output") {
      std::cerr << "usage: " << argv[0]
                << " --dataset labels.csv --metadata artifact.json --model model.onnx --output report.json\n";
      return 2;
    }
    const auto rows = load_test_rows(argv[2]);
    const std::string labels_checksum = fnv1a64_file(argv[2]);
    nre::OnnxPricingBackend backend(argv[4], argv[6]);
    std::vector<nre::PricingRequest> requests;
    std::vector<nre::NeuralInput> timing_inputs;
    requests.reserve(rows.size());
    timing_inputs.reserve(rows.size());
    for (const auto& row : rows) {
      requests.push_back(row.request);
      timing_inputs.push_back({row.request.contract, row.request.market});
    }

    const auto routing_start = std::chrono::steady_clock::now();
    const auto routed =
        nre::price_guarded_neural_batch(requests, backend, backend.guardrail_config());
    const auto routing_end = std::chrono::steady_clock::now();

    std::vector<double> full_prices;
    std::vector<double> full_deltas;
    std::vector<double> full_reference_prices;
    std::vector<double> full_reference_deltas;
    std::vector<double> accepted_prices;
    std::vector<double> accepted_deltas;
    std::vector<double> accepted_reference_prices;
    std::vector<double> accepted_reference_deltas;
    for (std::size_t index = 0U; index < routed.items.size(); ++index) {
      full_prices.push_back(routed.items[index].result.price.estimate);
      full_deltas.push_back(routed.items[index].result.delta.estimate);
      full_reference_prices.push_back(rows[index].reference_price);
      full_reference_deltas.push_back(rows[index].reference_delta);
      if (routed.items[index].neural_accepted) {
        accepted_prices.push_back(routed.items[index].result.price.estimate);
        accepted_deltas.push_back(routed.items[index].result.delta.estimate);
        accepted_reference_prices.push_back(rows[index].reference_price);
        accepted_reference_deltas.push_back(rows[index].reference_delta);
      }
    }
    const Metrics full =
        metrics(full_prices, full_deltas, full_reference_prices, full_reference_deltas);
    const Metrics accepted = metrics(accepted_prices, accepted_deltas, accepted_reference_prices,
                                     accepted_reference_deltas);

    constexpr std::size_t warmups = 20U;
    constexpr std::size_t repetitions = 300U;
    for (std::size_t index = 0U; index < warmups; ++index) {
      static_cast<void>(backend.predict(timing_inputs));
    }
    std::vector<double> timing_microseconds;
    timing_microseconds.reserve(repetitions);
    for (std::size_t index = 0U; index < repetitions; ++index) {
      const auto start = std::chrono::steady_clock::now();
      static_cast<void>(backend.predict(timing_inputs));
      const auto end = std::chrono::steady_clock::now();
      timing_microseconds.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::ofstream output(argv[8]);
    if (!output) {
      throw std::runtime_error("cannot write M9 C++ evaluation report");
    }
    output << std::setprecision(17);
    const std::size_t accepted_count = accepted.count;
    const std::size_t fallback_count = rows.size() - accepted_count;
    output << "{\n"
           << "  \"result_version\": \"nre.onnx.cpp_guarded.v1\",\n"
           << "  \"source_implementation_commit\": \"PENDING\",\n"
           << "  \"artifact_version\": \"" << backend.artifact_version() << "\",\n"
           << "  \"model_fnv1a64\": \"" << backend.model_checksum() << "\",\n"
           << "  \"labels_fnv1a64\": \"" << labels_checksum << "\",\n"
           << "  \"heldout_count\": " << rows.size() << ",\n"
           << "  \"neural_accepted_count\": " << accepted_count << ",\n"
           << "  \"monte_carlo_fallback_count\": " << fallback_count << ",\n"
           << "  \"neural_acceptance_rate\": "
           << static_cast<double>(accepted_count) / static_cast<double>(rows.size()) << ",\n"
           << "  \"monte_carlo_fallback_rate\": "
           << static_cast<double>(fallback_count) / static_cast<double>(rows.size()) << ",\n"
           << "  \"fallback_reason_counts\": {\n";
    bool first = true;
    for (std::size_t reason_index = 1U;
         reason_index < static_cast<std::size_t>(nre::NeuralRejectionReason::count);
         ++reason_index) {
      if (!first) {
        output << ",\n";
      }
      first = false;
      const auto reason = static_cast<nre::NeuralRejectionReason>(reason_index);
      output << "    \"" << nre::to_string(reason) << "\": " << routed.counters[reason_index];
    }
    output << "\n  },\n  \"accepted_neural_metrics\": {\n";
    write_metrics(output, accepted, "    ");
    output << "  },\n  \"full_routed_metrics\": {\n";
    write_metrics(output, full, "    ");
    output << "  },\n"
           << "  \"timing\": {\n"
           << "    \"scope\": \"C++ ONNX price plus centered-bump Delta for the full held-out batch; excludes guardrail probes and Monte Carlo fallback\",\n"
           << "    \"batch_size\": " << timing_inputs.size() << ",\n"
           << "    \"warmups\": " << warmups << ",\n"
           << "    \"repetitions\": " << repetitions << ",\n"
           << "    \"median_microseconds\": " << quantile(timing_microseconds, 0.50) << ",\n"
           << "    \"empirical_p99_microseconds\": " << quantile(timing_microseconds, 0.99)
           << ",\n"
           << "    \"single_routed_evaluation_seconds\": "
           << std::chrono::duration<double>(routing_end - routing_start).count() << "\n"
           << "  },\n"
           << "  \"environment\": {\n"
           << "    \"onnxruntime_cpp\": \"" << OrtGetApiBase()->GetVersionString() << "\",\n"
           << "    \"compiler\": \"" << __VERSION__ << "\",\n"
           << "    \"compiler_flags\": \"" << NRE_BUILD_FLAGS << "\",\n"
           << "    \"cpp_standard\": \"C++20\",\n"
           << "    \"hardware_concurrency\": " << std::thread::hardware_concurrency() << "\n"
           << "  },\n"
           << "  \"limitations\": [\n"
           << "    \"Guardrails are sampled engineering checks, not formal no-arbitrage guarantees or general OOD detection.\",\n"
           << "    \"Accepted-set metrics are reported beside full routed metrics so selective acceptance is visible.\",\n"
           << "    \"Timing is descriptive for this machine and is not a matched-error speedup or production latency claim.\"\n"
           << "  ]\n"
           << "}\n";
    std::cout << "wrote guarded M9 evaluation for " << rows.size() << " held-out rows\n";
  } catch (const std::exception& error) {
    std::cerr << "M9 guarded evaluation failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

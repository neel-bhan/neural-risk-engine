#include "nre/dataset.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "nre/pricing.hpp"

namespace nre {
namespace {

constexpr std::string_view kSchemaVersion = "nre.dataset.v1";
constexpr std::size_t kCsvColumnCount = 44U;

struct LabelRow {
  DatasetParameterPoint point;
  bool included_for_training;
  std::string quality_status;
  std::string quality_flags;
  PricingEstimator estimator;
  std::string label_tier;
  UnifiedPricingResult result;
  std::optional<double> analytical_price;
  std::optional<double> analytical_delta;
  std::optional<double> analytical_price_absolute_error;
  std::optional<double> analytical_delta_absolute_error;
};

[[nodiscard]] std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1U);
}

[[nodiscard]] std::uint64_t parse_uint64(const std::string& value, const std::string& key) {
  std::size_t consumed = 0U;
  std::uint64_t parsed = 0U;
  try {
    parsed = std::stoull(value, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid unsigned integer for " + key);
  }
  if (consumed != value.size()) {
    throw std::invalid_argument("invalid unsigned integer for " + key);
  }
  return parsed;
}

[[nodiscard]] std::size_t parse_size(const std::string& value, const std::string& key) {
  const auto parsed = parse_uint64(value, key);
  if (parsed > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument("value for " + key + " exceeds size_t");
  }
  return static_cast<std::size_t>(parsed);
}

[[nodiscard]] double parse_double(const std::string& value, const std::string& key) {
  std::size_t consumed = 0U;
  double parsed = 0.0;
  try {
    parsed = std::stod(value, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid floating-point value for " + key);
  }
  if (consumed != value.size() || !std::isfinite(parsed)) {
    throw std::invalid_argument("invalid floating-point value for " + key);
  }
  return parsed;
}

[[nodiscard]] std::string checksum_bytes(std::istream& input) {
  constexpr std::uint64_t kOffsetBasis = 14695981039346656037ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  std::uint64_t hash = kOffsetBasis;
  char byte = '\0';
  while (input.get(byte)) {
    hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(byte));
    hash *= kPrime;
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

[[nodiscard]] std::string checksum_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open file for checksum: " + path.string());
  }
  return checksum_bytes(input);
}

void require_range(const ParameterRange& range, const std::string& name, bool positive_minimum) {
  if (!std::isfinite(range.minimum) || !std::isfinite(range.maximum) ||
      range.minimum >= range.maximum || (positive_minimum && range.minimum <= 0.0)) {
    throw std::invalid_argument("invalid dataset range: " + name);
  }
}

void validate_config(const DatasetConfig& config) {
  if (config.schema_version != kSchemaVersion) {
    throw std::invalid_argument("unsupported dataset schema_version");
  }
  if (config.preset_name.empty() || config.points_per_contract < 2U ||
      config.training_path_count < 2U || config.heldout_path_count < 2U ||
      config.training_pilot_path_count < 2U || config.heldout_pilot_path_count < 2U ||
      config.thread_count == 0U) {
    throw std::invalid_argument("dataset counts and thread_count must satisfy their minimums");
  }
  require_range(config.domain.spot, "spot", true);
  require_range(config.domain.strike, "strike", true);
  require_range(config.domain.maturity_years, "maturity_years", true);
  require_range(config.domain.volatility, "volatility", false);
  require_range(config.domain.risk_free_rate, "risk_free_rate", false);
  require_range(config.domain.dividend_yield, "dividend_yield", false);
  if (config.domain.volatility.minimum < 0.0 || config.domain.minimum_asian_observations < 2U ||
      config.domain.maximum_asian_observations < config.domain.minimum_asian_observations) {
    throw std::invalid_argument("invalid volatility or Asian observation domain");
  }
  const std::array<double, 8> quality_values{
      config.quality.training_max_price_standard_error,
      config.quality.heldout_max_price_standard_error,
      config.quality.training_max_delta_standard_error,
      config.quality.heldout_max_delta_standard_error,
      config.quality.analytical_standard_error_multiplier,
      config.quality.analytical_price_absolute_tolerance,
      config.quality.analytical_delta_absolute_tolerance,
      config.quality.bounds_standard_error_multiplier,
  };
  if (std::any_of(quality_values.begin(), quality_values.end(),
                  [](double value) { return !std::isfinite(value) || value < 0.0; }) ||
      config.quality.training_max_price_standard_error <= 0.0 ||
      config.quality.heldout_max_price_standard_error <= 0.0 ||
      config.quality.training_max_delta_standard_error <= 0.0 ||
      config.quality.heldout_max_delta_standard_error <= 0.0) {
    throw std::invalid_argument("invalid dataset quality policy");
  }
  if (config.source_finalization_commit.empty() || config.build_configuration.empty() ||
      config.generation_command.empty() || config.config_source.empty() ||
      config.config_checksum.empty()) {
    throw std::invalid_argument("dataset provenance fields must not be empty");
  }
}

[[nodiscard]] double interpolate(const ParameterRange& range, double fraction) noexcept {
  return range.minimum + fraction * (range.maximum - range.minimum);
}

[[nodiscard]] double radical_inverse(std::size_t index, std::size_t base) noexcept {
  double result = 0.0;
  double factor = 1.0 / static_cast<double>(base);
  while (index != 0U) {
    result += factor * static_cast<double>(index % base);
    index /= base;
    factor /= static_cast<double>(base);
  }
  return result;
}

[[nodiscard]] double design_fraction(std::size_t local_index, std::size_t base) noexcept {
  if (local_index == 0U) {
    return 0.0;
  }
  if (local_index == 1U) {
    return 1.0;
  }
  return radical_inverse(local_index - 1U, base);
}

[[nodiscard]] DatasetSplit split_for_index(std::size_t index) noexcept {
  const auto bucket = index % 20U;
  if (bucket < 14U) {
    return DatasetSplit::train;
  }
  if (bucket < 17U) {
    return DatasetSplit::validation;
  }
  return DatasetSplit::test;
}

[[nodiscard]] std::string parameter_id(std::size_t index) {
  std::ostringstream output;
  output << 'p' << std::setfill('0') << std::setw(6) << index;
  return output.str();
}

[[nodiscard]] std::string canonical_point_key(const DatasetParameterPoint& point) {
  std::ostringstream output;
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << static_cast<int>(point.contract.style) << '|' << static_cast<int>(point.contract.type)
         << '|' << point.market.spot << '|' << point.contract.strike << '|'
         << point.contract.maturity_years << '|' << point.market.volatility << '|'
         << point.market.risk_free_rate << '|' << point.market.dividend_yield << '|'
         << point.contract.observations;
  return output.str();
}

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] std::string csv_style(OptionStyle style) {
  switch (style) {
    case OptionStyle::european:
      return "european";
    case OptionStyle::geometric_asian:
      return "geometric_asian";
    case OptionStyle::arithmetic_asian:
      return "arithmetic_asian";
  }
  throw std::invalid_argument("unknown option style");
}

[[nodiscard]] std::string csv_estimator(PricingEstimator estimator) {
  switch (estimator) {
    case PricingEstimator::analytical:
      return "analytical";
    case PricingEstimator::plain:
      return "plain";
    case PricingEstimator::antithetic:
      return "antithetic";
    case PricingEstimator::geometric_control_variate:
      return "geometric_control_variate";
  }
  throw std::invalid_argument("unknown pricing estimator");
}

template <typename T>
[[nodiscard]] const T& require_optional(const std::optional<T>& value, const char* name) {
  if (!value.has_value()) {
    throw std::runtime_error(std::string("dataset pricing result is missing ") + name);
  }
  return *value;
}

[[nodiscard]] double discounted_underlying_factor(const DatasetParameterPoint& point) {
  const auto observations = point.contract.observations;
  double sum = 0.0;
  for (std::size_t index = 1U; index <= observations; ++index) {
    const double time = point.contract.maturity_years * static_cast<double>(index) /
                        static_cast<double>(observations);
    sum += std::exp((point.market.risk_free_rate - point.market.dividend_yield) * time);
  }
  const double expected_average_factor = sum / static_cast<double>(observations);
  return std::exp(-point.market.risk_free_rate * point.contract.maturity_years) *
         expected_average_factor;
}

void add_flag(std::vector<std::string>& flags, std::string flag) {
  flags.push_back(std::move(flag));
}

[[nodiscard]] std::string join_flags(const std::vector<std::string>& flags) {
  if (flags.empty()) {
    return "none";
  }
  std::string joined;
  for (std::size_t index = 0U; index < flags.size(); ++index) {
    if (index != 0U) {
      joined += ';';
    }
    joined += flags[index];
  }
  return joined;
}

[[nodiscard]] LabelRow generate_row(const DatasetConfig& config,
                                    const DatasetParameterPoint& point) {
  const bool is_training = point.split == DatasetSplit::train;
  const std::size_t paths = is_training ? config.training_path_count : config.heldout_path_count;
  const std::size_t pilot_paths =
      is_training ? config.training_pilot_path_count : config.heldout_pilot_path_count;
  const std::uint64_t point_offset = static_cast<std::uint64_t>(point.index) * 2ULL;
  const std::uint64_t pricing_seed = splitmix64(config.master_seed + point_offset);
  const std::uint64_t pilot_seed = splitmix64(config.master_seed + point_offset + 1ULL);
  const PricingEstimator estimator = point.contract.style == OptionStyle::arithmetic_asian
                                         ? PricingEstimator::geometric_control_variate
                                         : PricingEstimator::plain;

  PricingRequest request{
      .contract = point.contract,
      .market = point.market,
      .backend = PricingBackend::monte_carlo,
      .estimator = estimator,
  };
  if (estimator == PricingEstimator::plain) {
    request.monte_carlo_config = MonteCarloConfig{
        .seed = pricing_seed,
        .path_count = paths,
        .thread_count = config.thread_count,
    };
  } else {
    request.control_variate_config = ControlVariateConfig{
        .pricing =
            MonteCarloConfig{
                .seed = pricing_seed,
                .path_count = paths,
                .thread_count = config.thread_count,
            },
        .pilot_seed = pilot_seed,
        .pilot_path_count = pilot_paths,
    };
  }

  auto result = price(request);
  std::optional<double> analytical_price;
  std::optional<double> analytical_delta;
  std::optional<double> price_error;
  std::optional<double> delta_error;
  if (point.contract.style != OptionStyle::arithmetic_asian) {
    const auto analytical = price({
        .contract = point.contract,
        .market = point.market,
        .backend = PricingBackend::analytical,
        .estimator = PricingEstimator::analytical,
    });
    analytical_price = analytical.price.estimate;
    analytical_delta = analytical.delta.estimate;
    price_error = std::abs(result.price.estimate - *analytical_price);
    delta_error = std::abs(result.delta.estimate - *analytical_delta);
  }

  const double price_se = require_optional(result.price.sample_standard_error, "price SE");
  const double delta_se = require_optional(result.delta.sample_standard_error, "Delta SE");
  const auto& price_ci =
      require_optional(result.price.confidence_interval_95, "price confidence interval");
  const auto& delta_ci =
      require_optional(result.delta.confidence_interval_95, "Delta confidence interval");
  static_cast<void>(require_optional(result.metadata.effective_paths, "effective paths"));
  static_cast<void>(require_optional(result.metadata.raw_paths, "raw paths"));
  static_cast<void>(require_optional(result.metadata.seed, "pricing seed"));
  static_cast<void>(require_optional(result.metadata.requested_threads, "requested threads"));
  static_cast<void>(require_optional(result.metadata.active_threads, "active threads"));

  std::vector<std::string> flags;
  const std::array<double, 10> required_finite{
      result.price.estimate, price_se,
      price_ci.lower,        price_ci.upper,
      result.delta.estimate, delta_se,
      delta_ci.lower,        delta_ci.upper,
      point.market.spot,     point.contract.strike,
  };
  if (std::any_of(required_finite.begin(), required_finite.end(),
                  [](double value) { return !std::isfinite(value); })) {
    add_flag(flags, "non_finite_value");
  }
  if (price_se < 0.0 || delta_se < 0.0 || price_ci.lower > result.price.estimate ||
      price_ci.upper < result.price.estimate || delta_ci.lower > result.delta.estimate ||
      delta_ci.upper < result.delta.estimate) {
    add_flag(flags, "invalid_sampling_diagnostics");
  }

  const double max_price_se = is_training ? config.quality.training_max_price_standard_error
                                          : config.quality.heldout_max_price_standard_error;
  const double max_delta_se = is_training ? config.quality.training_max_delta_standard_error
                                          : config.quality.heldout_max_delta_standard_error;
  if (price_se > max_price_se) {
    add_flag(flags, "price_uncertainty_exceeds_tolerance");
  }
  if (delta_se > max_delta_se) {
    add_flag(flags, "delta_uncertainty_exceeds_tolerance");
  }

  const double discount = std::exp(-point.market.risk_free_rate * point.contract.maturity_years);
  const double underlying_factor = discounted_underlying_factor(point);
  const double price_upper = point.contract.type == OptionType::call
                                 ? point.market.spot * underlying_factor
                                 : point.contract.strike * discount;
  const double delta_lower = point.contract.type == OptionType::call ? 0.0 : -underlying_factor;
  const double delta_upper = point.contract.type == OptionType::call ? underlying_factor : 0.0;
  const double price_margin = config.quality.bounds_standard_error_multiplier * price_se + 1.0e-12;
  const double delta_margin = config.quality.bounds_standard_error_multiplier * delta_se + 1.0e-12;
  if (result.price.estimate < -price_margin || result.price.estimate > price_upper + price_margin) {
    add_flag(flags, "price_bound_violation");
  }
  if (result.delta.estimate < delta_lower - delta_margin ||
      result.delta.estimate > delta_upper + delta_margin) {
    add_flag(flags, "delta_sign_or_range_violation");
  }

  if (analytical_price.has_value()) {
    const double price_tolerance = config.quality.analytical_price_absolute_tolerance +
                                   config.quality.analytical_standard_error_multiplier * price_se;
    const double delta_tolerance = config.quality.analytical_delta_absolute_tolerance +
                                   config.quality.analytical_standard_error_multiplier * delta_se;
    if (*price_error > price_tolerance) {
      add_flag(flags, "analytical_price_cross_check_failed");
    }
    if (*delta_error > delta_tolerance) {
      add_flag(flags, "analytical_delta_cross_check_failed");
    }
  }

  const bool accepted = flags.empty();
  return {
      .point = point,
      .included_for_training = accepted,
      .quality_status = accepted ? "accepted" : "rejected",
      .quality_flags = join_flags(flags),
      .estimator = estimator,
      .label_tier = is_training ? "bulk_training" : "heldout_reference",
      .result = std::move(result),
      .analytical_price = analytical_price,
      .analytical_delta = analytical_delta,
      .analytical_price_absolute_error = price_error,
      .analytical_delta_absolute_error = delta_error,
  };
}

[[nodiscard]] std::string optional_number(const std::optional<double>& value) {
  if (!value.has_value()) {
    return {};
  }
  std::ostringstream output;
  output << std::setprecision(std::numeric_limits<double>::max_digits10) << *value;
  return output.str();
}

template <typename T>
[[nodiscard]] std::string optional_integer(const std::optional<T>& value) {
  return value.has_value() ? std::to_string(*value) : std::string{};
}

[[nodiscard]] std::string optional_bool(const std::optional<bool>& value) {
  if (!value.has_value()) {
    return {};
  }
  return *value ? "true" : "false";
}

void write_csv_header(std::ostream& output) {
  output << "schema_version,parameter_id,split,included_for_training,quality_status,quality_flags,"
            "option_style,option_type,spot,strike,maturity_years,volatility,risk_free_rate,"
            "dividend_yield,observations,backend,estimator,label_tier,price,price_standard_error,"
            "price_ci_95_lower,price_ci_95_upper,delta,delta_standard_error,delta_ci_95_lower,"
            "delta_ci_95_upper,effective_paths,raw_paths,pricing_seed,requested_threads,active_"
            "threads,"
            "pilot_paths,pilot_seed,pilot_active_threads,price_control_coefficient,"
            "price_control_expectation,price_control_applied,delta_control_coefficient,"
            "delta_control_expectation,delta_control_applied,analytical_price,analytical_delta,"
            "analytical_price_absolute_error,analytical_delta_absolute_error\n";
}

void write_csv_row(std::ostream& output, const DatasetConfig& config, const LabelRow& row) {
  const auto& result = row.result;
  const auto& price_ci = *result.price.confidence_interval_95;
  const auto& delta_ci = *result.delta.confidence_interval_95;
  output << std::setprecision(std::numeric_limits<double>::max_digits10) << config.schema_version
         << ',' << row.point.parameter_id << ',' << to_string(row.point.split) << ','
         << (row.included_for_training ? "true" : "false") << ',' << row.quality_status << ','
         << row.quality_flags << ',' << csv_style(row.point.contract.style) << ','
         << nre::to_string(row.point.contract.type) << ',' << row.point.market.spot << ','
         << row.point.contract.strike << ',' << row.point.contract.maturity_years << ','
         << row.point.market.volatility << ',' << row.point.market.risk_free_rate << ','
         << row.point.market.dividend_yield << ',' << row.point.contract.observations
         << ",monte_carlo," << csv_estimator(row.estimator) << ',' << row.label_tier << ','
         << result.price.estimate << ',' << *result.price.sample_standard_error << ','
         << price_ci.lower << ',' << price_ci.upper << ',' << result.delta.estimate << ','
         << *result.delta.sample_standard_error << ',' << delta_ci.lower << ',' << delta_ci.upper
         << ',' << optional_integer(result.metadata.effective_paths) << ','
         << optional_integer(result.metadata.raw_paths) << ','
         << optional_integer(result.metadata.seed) << ','
         << optional_integer(result.metadata.requested_threads) << ','
         << optional_integer(result.metadata.active_threads) << ','
         << optional_integer(result.metadata.pilot_paths) << ','
         << optional_integer(result.metadata.pilot_seed) << ','
         << optional_integer(result.metadata.pilot_active_threads) << ','
         << optional_number(result.metadata.price_control_coefficient) << ','
         << optional_number(result.metadata.price_control_expectation) << ','
         << optional_bool(result.metadata.price_control_applied) << ','
         << optional_number(result.metadata.delta_control_coefficient) << ','
         << optional_number(result.metadata.delta_control_expectation) << ','
         << optional_bool(result.metadata.delta_control_applied) << ','
         << optional_number(row.analytical_price) << ',' << optional_number(row.analytical_delta)
         << ',' << optional_number(row.analytical_price_absolute_error) << ','
         << optional_number(row.analytical_delta_absolute_error) << '\n';
}

[[nodiscard]] std::string json_escape(std::string_view value) {
  std::string escaped;
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += character;
        break;
    }
  }
  return escaped;
}

[[nodiscard]] std::string platform_name() {
#if defined(__APPLE__)
  return "Apple platform";
#elif defined(__linux__)
  return "Linux";
#elif defined(_WIN32)
  return "Windows";
#else
  return "unknown platform";
#endif
}

[[nodiscard]] std::string build_mode() {
#ifdef NDEBUG
  return "NDEBUG/release";
#else
  return "assertions-enabled";
#endif
}

void write_range(std::ostream& output, const char* name, const ParameterRange& range,
                 bool trailing_comma) {
  output << "      \"" << name << "\": {\"minimum\": " << range.minimum
         << ", \"maximum\": " << range.maximum << '}';
  if (trailing_comma) {
    output << ',';
  }
  output << '\n';
}

void write_manifest(const std::filesystem::path& path, const DatasetConfig& config,
                    const DatasetSummary& summary) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create dataset manifest: " + path.string());
  }
  output << std::setprecision(std::numeric_limits<double>::max_digits10) << "{\n"
         << "  \"schema_version\": \"" << json_escape(config.schema_version) << "\",\n"
         << "  \"source_finalization_commit\": \"" << json_escape(config.source_finalization_commit)
         << "\",\n"
         << "  \"engine\": {\n"
         << "    \"label_api\": \"nre::price backend-neutral interface\",\n"
         << "    \"trusted_backend\": \"C++20 Monte Carlo\",\n"
         << "    \"compiler\": \"" << json_escape(__VERSION__) << "\",\n"
         << "    \"cpp_standard\": " << __cplusplus << ",\n"
         << "    \"build_configuration\": \"" << json_escape(config.build_configuration) << "; "
         << build_mode() << "\",\n"
         << "    \"platform\": \"" << platform_name() << "\",\n"
         << "    \"hardware_concurrency\": " << std::thread::hardware_concurrency() << "\n"
         << "  },\n"
         << "  \"config\": {\n"
         << "    \"preset_name\": \"" << json_escape(config.preset_name) << "\",\n"
         << "    \"source\": \"" << json_escape(config.config_source) << "\",\n"
         << "    \"fnv1a64\": \"" << config.config_checksum << "\",\n"
         << "    \"generation_command\": \"" << json_escape(config.generation_command) << "\"\n"
         << "  },\n"
         << "  \"parameter_design\": {\n"
         << "    \"method\": \"declared bounds plus deterministic radical-inverse interiors\",\n"
         << "    \"points_per_style_type\": " << config.points_per_contract << ",\n"
         << "    \"split_rule\": \"parameter_index modulo 20: 0-13 train, 14-16 validation, "
            "17-19 test\",\n"
         << "    \"monitoring\": \"t_i=iT/m for i=1..m; t=0 excluded\",\n"
         << "    \"domain\": {\n";
  write_range(output, "spot", config.domain.spot, true);
  write_range(output, "strike", config.domain.strike, true);
  write_range(output, "maturity_years", config.domain.maturity_years, true);
  write_range(output, "volatility", config.domain.volatility, true);
  write_range(output, "risk_free_rate", config.domain.risk_free_rate, true);
  write_range(output, "dividend_yield", config.domain.dividend_yield, true);
  output << "      \"asian_observations\": {\"minimum\": "
         << config.domain.minimum_asian_observations
         << ", \"maximum\": " << config.domain.maximum_asian_observations << "}\n"
         << "    }\n"
         << "  },\n"
         << "  \"label_configuration\": {\n"
         << "    \"master_seed\": " << config.master_seed << ",\n"
         << "    \"seed_rule\": \"SplitMix64(master_seed + 2*parameter_index); pilot uses +1\",\n"
         << "    \"requested_threads\": " << config.thread_count << ",\n"
         << "    \"training_paths\": " << config.training_path_count << ",\n"
         << "    \"heldout_paths\": " << config.heldout_path_count << ",\n"
         << "    \"training_pilot_paths\": " << config.training_pilot_path_count << ",\n"
         << "    \"heldout_pilot_paths\": " << config.heldout_pilot_path_count << ",\n"
         << "    \"european_estimator\": \"plain\",\n"
         << "    \"geometric_asian_estimator\": \"plain\",\n"
         << "    \"arithmetic_asian_estimator\": \"geometric_control_variate\"\n"
         << "  },\n"
         << "  \"quality_policy\": {\n"
         << "    \"training_max_price_standard_error\": "
         << config.quality.training_max_price_standard_error << ",\n"
         << "    \"heldout_max_price_standard_error\": "
         << config.quality.heldout_max_price_standard_error << ",\n"
         << "    \"training_max_delta_standard_error\": "
         << config.quality.training_max_delta_standard_error << ",\n"
         << "    \"heldout_max_delta_standard_error\": "
         << config.quality.heldout_max_delta_standard_error << ",\n"
         << "    \"analytical_standard_error_multiplier\": "
         << config.quality.analytical_standard_error_multiplier << ",\n"
         << "    \"analytical_price_absolute_tolerance\": "
         << config.quality.analytical_price_absolute_tolerance << ",\n"
         << "    \"analytical_delta_absolute_tolerance\": "
         << config.quality.analytical_delta_absolute_tolerance << ",\n"
         << "    \"bounds_standard_error_multiplier\": "
         << config.quality.bounds_standard_error_multiplier << ",\n"
         << "    \"failure_handling\": \"retain row; set included_for_training=false and record "
            "quality_flags\"\n"
         << "  },\n"
         << "  \"counts\": {\n"
         << "    \"total_rows\": " << summary.total_rows << ",\n"
         << "    \"accepted_rows\": " << summary.accepted_rows << ",\n"
         << "    \"rejected_rows\": " << summary.rejected_rows << ",\n"
         << "    \"train_rows\": " << summary.train_rows << ",\n"
         << "    \"validation_rows\": " << summary.validation_rows << ",\n"
         << "    \"test_rows\": " << summary.test_rows << ",\n"
         << "    \"analytical_cross_checks\": " << summary.analytical_cross_checks << ",\n"
         << "    \"analytical_cross_check_failures\": " << summary.analytical_cross_check_failures
         << "\n"
         << "  },\n"
         << "  \"measured_quality\": {\n"
         << "    \"maximum_price_standard_error\": " << summary.maximum_price_standard_error
         << ",\n"
         << "    \"maximum_delta_standard_error\": " << summary.maximum_delta_standard_error
         << ",\n"
         << "    \"maximum_analytical_price_absolute_error\": "
         << summary.maximum_analytical_price_absolute_error << ",\n"
         << "    \"maximum_analytical_delta_absolute_error\": "
         << summary.maximum_analytical_delta_absolute_error << "\n"
         << "  },\n"
         << "  \"artifacts\": {\n"
         << "    \"labels_file\": \"labels.csv\",\n"
         << "    \"labels_fnv1a64\": \"" << summary.labels_checksum << "\",\n"
         << "    \"checksum_note\": \"FNV-1a-64 detects regeneration drift; it is not a "
            "cryptographic integrity claim\"\n"
         << "  }\n"
         << "}\n";
}

[[nodiscard]] std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t begin = 0U;
  while (true) {
    const auto comma = line.find(',', begin);
    if (comma == std::string::npos) {
      fields.push_back(line.substr(begin));
      break;
    }
    fields.push_back(line.substr(begin, comma - begin));
    begin = comma + 1U;
  }
  return fields;
}

[[nodiscard]] std::string manifest_string(const std::string& manifest, const std::string& key) {
  const std::string marker = "\"" + key + "\": \"";
  const auto begin = manifest.find(marker);
  if (begin == std::string::npos) {
    throw std::runtime_error("manifest is missing " + key);
  }
  const auto value_begin = begin + marker.size();
  const auto end = manifest.find('"', value_begin);
  if (end == std::string::npos) {
    throw std::runtime_error("manifest has malformed " + key);
  }
  return manifest.substr(value_begin, end - value_begin);
}

[[nodiscard]] std::size_t manifest_size(const std::string& manifest, const std::string& key) {
  const std::string marker = "\"" + key + "\": ";
  const auto begin = manifest.find(marker);
  if (begin == std::string::npos) {
    throw std::runtime_error("manifest is missing " + key);
  }
  const auto value_begin = begin + marker.size();
  const auto end = manifest.find_first_not_of("0123456789", value_begin);
  return parse_size(manifest.substr(value_begin, end - value_begin), key);
}

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read file: " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

}  // namespace

DatasetConfig load_dataset_config(const std::filesystem::path& config_path) {
  DatasetConfig config{};
  std::ifstream input(config_path);
  if (!input) {
    throw std::runtime_error("cannot open dataset config: " + config_path.string());
  }
  std::set<std::string> keys;
  std::string line;
  std::size_t line_number = 0U;
  while (std::getline(input, line)) {
    ++line_number;
    line = trim(line);
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto equals = line.find('=');
    if (equals == std::string::npos) {
      throw std::invalid_argument("malformed dataset config line " + std::to_string(line_number));
    }
    const std::string key = trim(line.substr(0U, equals));
    const std::string value = trim(line.substr(equals + 1U));
    if (key.empty() || value.empty() || !keys.insert(key).second) {
      throw std::invalid_argument("empty or duplicate dataset config key on line " +
                                  std::to_string(line_number));
    }

    if (key == "schema_version") {
      config.schema_version = value;
    } else if (key == "preset_name") {
      config.preset_name = value;
    } else if (key == "points_per_contract") {
      config.points_per_contract = parse_size(value, key);
    } else if (key == "master_seed") {
      config.master_seed = parse_uint64(value, key);
    } else if (key == "training_path_count") {
      config.training_path_count = parse_size(value, key);
    } else if (key == "heldout_path_count") {
      config.heldout_path_count = parse_size(value, key);
    } else if (key == "training_pilot_path_count") {
      config.training_pilot_path_count = parse_size(value, key);
    } else if (key == "heldout_pilot_path_count") {
      config.heldout_pilot_path_count = parse_size(value, key);
    } else if (key == "thread_count") {
      config.thread_count = parse_size(value, key);
    } else if (key == "spot_min") {
      config.domain.spot.minimum = parse_double(value, key);
    } else if (key == "spot_max") {
      config.domain.spot.maximum = parse_double(value, key);
    } else if (key == "strike_min") {
      config.domain.strike.minimum = parse_double(value, key);
    } else if (key == "strike_max") {
      config.domain.strike.maximum = parse_double(value, key);
    } else if (key == "maturity_years_min") {
      config.domain.maturity_years.minimum = parse_double(value, key);
    } else if (key == "maturity_years_max") {
      config.domain.maturity_years.maximum = parse_double(value, key);
    } else if (key == "volatility_min") {
      config.domain.volatility.minimum = parse_double(value, key);
    } else if (key == "volatility_max") {
      config.domain.volatility.maximum = parse_double(value, key);
    } else if (key == "risk_free_rate_min") {
      config.domain.risk_free_rate.minimum = parse_double(value, key);
    } else if (key == "risk_free_rate_max") {
      config.domain.risk_free_rate.maximum = parse_double(value, key);
    } else if (key == "dividend_yield_min") {
      config.domain.dividend_yield.minimum = parse_double(value, key);
    } else if (key == "dividend_yield_max") {
      config.domain.dividend_yield.maximum = parse_double(value, key);
    } else if (key == "asian_observations_min") {
      config.domain.minimum_asian_observations = parse_size(value, key);
    } else if (key == "asian_observations_max") {
      config.domain.maximum_asian_observations = parse_size(value, key);
    } else if (key == "training_max_price_standard_error") {
      config.quality.training_max_price_standard_error = parse_double(value, key);
    } else if (key == "heldout_max_price_standard_error") {
      config.quality.heldout_max_price_standard_error = parse_double(value, key);
    } else if (key == "training_max_delta_standard_error") {
      config.quality.training_max_delta_standard_error = parse_double(value, key);
    } else if (key == "heldout_max_delta_standard_error") {
      config.quality.heldout_max_delta_standard_error = parse_double(value, key);
    } else if (key == "analytical_standard_error_multiplier") {
      config.quality.analytical_standard_error_multiplier = parse_double(value, key);
    } else if (key == "analytical_price_absolute_tolerance") {
      config.quality.analytical_price_absolute_tolerance = parse_double(value, key);
    } else if (key == "analytical_delta_absolute_tolerance") {
      config.quality.analytical_delta_absolute_tolerance = parse_double(value, key);
    } else if (key == "bounds_standard_error_multiplier") {
      config.quality.bounds_standard_error_multiplier = parse_double(value, key);
    } else if (key == "source_finalization_commit") {
      config.source_finalization_commit = value;
    } else if (key == "build_configuration") {
      config.build_configuration = value;
    } else if (key == "generation_command") {
      config.generation_command = value;
    } else {
      throw std::invalid_argument("unknown dataset config key: " + key);
    }
  }

  constexpr std::size_t kRequiredKeyCount = 34U;
  if (keys.size() != kRequiredKeyCount) {
    throw std::invalid_argument("dataset config is missing required keys");
  }
  config.config_source = config_path.generic_string();
  config.config_checksum = checksum_file(config_path);
  validate_config(config);
  return config;
}

std::vector<DatasetParameterPoint> make_dataset_parameter_points(const DatasetConfig& config) {
  validate_config(config);
  const std::array<OptionStyle, 3> styles{OptionStyle::european, OptionStyle::geometric_asian,
                                          OptionStyle::arithmetic_asian};
  const std::array<OptionType, 2> types{OptionType::call, OptionType::put};
  std::vector<DatasetParameterPoint> points;
  points.reserve(styles.size() * types.size() * config.points_per_contract);

  std::size_t global_index = 0U;
  for (const auto style : styles) {
    for (const auto type : types) {
      for (std::size_t local_index = 0U; local_index < config.points_per_contract; ++local_index) {
        const double observations_fraction = design_fraction(local_index, 13U);
        const double observation_value =
            static_cast<double>(config.domain.minimum_asian_observations) +
            observations_fraction * static_cast<double>(config.domain.maximum_asian_observations -
                                                        config.domain.minimum_asian_observations);
        const auto observations = style == OptionStyle::european
                                      ? 1U
                                      : static_cast<std::size_t>(std::llround(observation_value));
        points.push_back({
            .index = global_index,
            .parameter_id = parameter_id(global_index),
            .split = split_for_index(global_index),
            .contract =
                OptionContract{
                    .type = type,
                    .style = style,
                    .strike = interpolate(config.domain.strike, design_fraction(local_index, 3U)),
                    .maturity_years =
                        interpolate(config.domain.maturity_years, design_fraction(local_index, 5U)),
                    .observations = observations,
                },
            .market =
                MarketState{
                    .spot = interpolate(config.domain.spot, design_fraction(local_index, 2U)),
                    .volatility =
                        interpolate(config.domain.volatility, design_fraction(local_index, 7U)),
                    .risk_free_rate = interpolate(config.domain.risk_free_rate,
                                                  design_fraction(local_index, 11U)),
                    .dividend_yield = interpolate(config.domain.dividend_yield,
                                                  design_fraction(local_index, 17U)),
                },
        });
        ++global_index;
      }
    }
  }

  std::set<std::string> unique_points;
  for (const auto& point : points) {
    if (!unique_points.insert(canonical_point_key(point)).second) {
      throw std::runtime_error("deterministic parameter design produced a duplicate point");
    }
  }
  return points;
}

DatasetSummary generate_dataset(const DatasetConfig& config,
                                const std::filesystem::path& output_directory) {
  validate_config(config);
  const auto points = make_dataset_parameter_points(config);
  std::filesystem::create_directories(output_directory);
  const auto labels_path = output_directory / "labels.csv";
  const auto manifest_path = output_directory / "manifest.json";
  std::filesystem::remove(labels_path);
  std::filesystem::remove(manifest_path);

  std::ofstream labels(labels_path, std::ios::binary | std::ios::trunc);
  if (!labels) {
    throw std::runtime_error("cannot create dataset labels: " + labels_path.string());
  }
  write_csv_header(labels);

  DatasetSummary summary{};
  summary.total_rows = points.size();
  for (const auto& point : points) {
    const auto row = generate_row(config, point);
    write_csv_row(labels, config, row);
    if (row.included_for_training) {
      ++summary.accepted_rows;
    } else {
      ++summary.rejected_rows;
    }
    switch (point.split) {
      case DatasetSplit::train:
        ++summary.train_rows;
        break;
      case DatasetSplit::validation:
        ++summary.validation_rows;
        break;
      case DatasetSplit::test:
        ++summary.test_rows;
        break;
    }
    summary.maximum_price_standard_error =
        std::max(summary.maximum_price_standard_error, *row.result.price.sample_standard_error);
    summary.maximum_delta_standard_error =
        std::max(summary.maximum_delta_standard_error, *row.result.delta.sample_standard_error);
    if (row.analytical_price.has_value()) {
      ++summary.analytical_cross_checks;
      summary.maximum_analytical_price_absolute_error = std::max(
          summary.maximum_analytical_price_absolute_error, *row.analytical_price_absolute_error);
      summary.maximum_analytical_delta_absolute_error = std::max(
          summary.maximum_analytical_delta_absolute_error, *row.analytical_delta_absolute_error);
      if (row.quality_flags.find("analytical_") != std::string::npos) {
        ++summary.analytical_cross_check_failures;
      }
    }
  }
  labels.close();
  if (!labels) {
    throw std::runtime_error("failed while writing dataset labels");
  }
  summary.labels_checksum = checksum_file(labels_path);
  write_manifest(manifest_path, config, summary);

  const auto verified = verify_generated_dataset(output_directory);
  if (verified.labels_checksum != summary.labels_checksum ||
      verified.total_rows != summary.total_rows ||
      verified.accepted_rows != summary.accepted_rows ||
      verified.rejected_rows != summary.rejected_rows) {
    throw std::runtime_error("fresh dataset verification disagrees with generation summary");
  }
  return summary;
}

DatasetSummary verify_generated_dataset(const std::filesystem::path& output_directory) {
  const auto labels_path = output_directory / "labels.csv";
  const auto manifest_path = output_directory / "manifest.json";
  const std::string manifest = read_file(manifest_path);
  const std::string expected_checksum = manifest_string(manifest, "labels_fnv1a64");
  const std::string actual_checksum = checksum_file(labels_path);
  if (actual_checksum != expected_checksum) {
    throw std::runtime_error("labels checksum does not match manifest");
  }

  DatasetSummary summary{};
  summary.labels_checksum = actual_checksum;
  std::ifstream labels(labels_path);
  std::string line;
  if (!std::getline(labels, line)) {
    throw std::runtime_error("labels file is empty");
  }
  if (split_csv_line(line).size() != kCsvColumnCount) {
    throw std::runtime_error("labels header does not match schema column count");
  }
  std::set<std::string> parameter_ids;
  while (std::getline(labels, line)) {
    const auto fields = split_csv_line(line);
    if (fields.size() != kCsvColumnCount) {
      throw std::runtime_error("labels row does not match schema column count");
    }
    if (fields[0] != kSchemaVersion || !parameter_ids.insert(fields[1]).second) {
      throw std::runtime_error("labels contain a schema mismatch or duplicate parameter point");
    }
    const bool accepted = fields[3] == "true" && fields[4] == "accepted";
    const bool rejected = fields[3] == "false" && fields[4] == "rejected";
    if (!accepted && !rejected) {
      throw std::runtime_error("labels contain inconsistent quality inclusion fields");
    }
    if (accepted) {
      ++summary.accepted_rows;
    } else {
      ++summary.rejected_rows;
    }
    if (fields[2] == "train") {
      ++summary.train_rows;
    } else if (fields[2] == "validation") {
      ++summary.validation_rows;
    } else if (fields[2] == "test") {
      ++summary.test_rows;
    } else {
      throw std::runtime_error("labels contain an unknown split");
    }
    for (const std::size_t field_index : {18U, 19U, 20U, 21U, 22U, 23U, 24U, 25U}) {
      static_cast<void>(parse_double(fields[field_index], "labels numeric field"));
    }
    ++summary.total_rows;
  }
  if (!labels.eof()) {
    throw std::runtime_error("failed while reading labels file");
  }

  if (summary.total_rows != manifest_size(manifest, "total_rows") ||
      summary.accepted_rows != manifest_size(manifest, "accepted_rows") ||
      summary.rejected_rows != manifest_size(manifest, "rejected_rows") ||
      summary.train_rows != manifest_size(manifest, "train_rows") ||
      summary.validation_rows != manifest_size(manifest, "validation_rows") ||
      summary.test_rows != manifest_size(manifest, "test_rows")) {
    throw std::runtime_error("manifest counts do not match labels.csv");
  }
  return summary;
}

std::string to_string(DatasetSplit split) {
  switch (split) {
    case DatasetSplit::train:
      return "train";
    case DatasetSplit::validation:
      return "validation";
    case DatasetSplit::test:
      return "test";
  }
  return "unknown";
}

}  // namespace nre

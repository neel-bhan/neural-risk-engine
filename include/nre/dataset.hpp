#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "nre/domain.hpp"

namespace nre {

enum class DatasetSplit { train, validation, test };

struct ParameterRange {
  double minimum;
  double maximum;
};

struct DatasetDomain {
  ParameterRange spot;
  ParameterRange strike;
  ParameterRange maturity_years;
  ParameterRange volatility;
  ParameterRange risk_free_rate;
  ParameterRange dividend_yield;
  std::size_t minimum_asian_observations;
  std::size_t maximum_asian_observations;
};

struct DatasetQualityPolicy {
  double training_max_price_standard_error;
  double heldout_max_price_standard_error;
  double training_max_delta_standard_error;
  double heldout_max_delta_standard_error;
  double analytical_standard_error_multiplier;
  double analytical_price_absolute_tolerance;
  double analytical_delta_absolute_tolerance;
  double bounds_standard_error_multiplier;
};

struct DatasetConfig {
  std::string schema_version;
  std::string preset_name;
  std::size_t points_per_contract;
  std::uint64_t master_seed;
  std::size_t training_path_count;
  std::size_t heldout_path_count;
  std::size_t training_pilot_path_count;
  std::size_t heldout_pilot_path_count;
  std::size_t thread_count;
  DatasetDomain domain;
  DatasetQualityPolicy quality;
  std::string source_finalization_commit;
  std::string build_configuration;
  std::string generation_command;
  std::string config_source;
  std::string config_checksum;
};

struct DatasetParameterPoint {
  std::size_t index;
  std::string parameter_id;
  DatasetSplit split;
  OptionContract contract;
  MarketState market;
};

struct DatasetSummary {
  std::size_t total_rows;
  std::size_t accepted_rows;
  std::size_t rejected_rows;
  std::size_t train_rows;
  std::size_t validation_rows;
  std::size_t test_rows;
  std::size_t analytical_cross_checks;
  std::size_t analytical_cross_check_failures;
  double maximum_price_standard_error;
  double maximum_delta_standard_error;
  double maximum_analytical_price_absolute_error;
  double maximum_analytical_delta_absolute_error;
  std::string labels_checksum;
};

// Loads the versioned standard-library key/value configuration used by the M6 commands.
[[nodiscard]] DatasetConfig load_dataset_config(const std::filesystem::path& config_path);

// Produces all six style/type combinations. Split assignment is a deterministic 70/15/15 mapping
// by unique parameter-point index; no stochastic observation is split independently.
[[nodiscard]] std::vector<DatasetParameterPoint> make_dataset_parameter_points(
    const DatasetConfig& config);

// Generates labels through the backend-neutral C++ pricing interface, writes labels.csv and
// manifest.json, and returns the measured quality summary. Rejected rows remain in labels.csv with
// included_for_training=false so failures are auditable rather than silently dropped.
[[nodiscard]] DatasetSummary generate_dataset(const DatasetConfig& config,
                                              const std::filesystem::path& output_directory);

// Recomputes the labels checksum and validates the manifest row count. Throws on mismatch.
[[nodiscard]] DatasetSummary verify_generated_dataset(
    const std::filesystem::path& output_directory);

[[nodiscard]] std::string to_string(DatasetSplit split);

}  // namespace nre

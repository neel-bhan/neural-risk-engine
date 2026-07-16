#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "nre/dataset.hpp"

#ifndef NRE_SOURCE_DIR
#error "NRE_SOURCE_DIR must identify the repository root"
#endif

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_throws(Function&& function, const std::string& message) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  std::cerr << "FAIL: " << message << " (no exception)\n";
  ++failures;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

nre::DatasetConfig test_config() {
  auto config =
      nre::load_dataset_config(std::filesystem::path(NRE_SOURCE_DIR) / "data/config/m6-small.cfg");
  config.points_per_contract = 4U;
  config.training_path_count = 256U;
  config.heldout_path_count = 512U;
  config.training_pilot_path_count = 64U;
  config.heldout_pilot_path_count = 128U;
  config.thread_count = 2U;
  config.quality.training_max_price_standard_error = 1.0e6;
  config.quality.heldout_max_price_standard_error = 1.0e6;
  config.quality.training_max_delta_standard_error = 1.0e6;
  config.quality.heldout_max_delta_standard_error = 1.0e6;
  config.quality.analytical_standard_error_multiplier = 1.0e6;
  config.quality.analytical_price_absolute_tolerance = 1.0e6;
  config.quality.analytical_delta_absolute_tolerance = 1.0e6;
  config.quality.bounds_standard_error_multiplier = 1.0e6;
  return config;
}

void test_parameter_design_and_split_integrity() {
  const auto config = test_config();
  const auto first = nre::make_dataset_parameter_points(config);
  const auto second = nre::make_dataset_parameter_points(config);
  expect(first.size() == 24U, "all six style/type combinations are generated");
  expect(first.size() == second.size(), "parameter design is deterministic in size");

  std::set<std::string> ids;
  std::set<std::string> combinations;
  bool saw_spot_minimum = false;
  bool saw_spot_maximum = false;
  bool saw_observation_minimum = false;
  bool saw_observation_maximum = false;
  std::size_t train = 0U;
  std::size_t validation = 0U;
  std::size_t test = 0U;
  for (std::size_t index = 0U; index < first.size(); ++index) {
    const auto& point = first[index];
    const auto& repeated = second[index];
    expect(point.parameter_id == repeated.parameter_id &&
               point.market.spot == repeated.market.spot &&
               point.contract.strike == repeated.contract.strike,
           "parameter design repeats exactly");
    expect(ids.insert(point.parameter_id).second, "parameter ids are unique");
    combinations.insert(std::to_string(static_cast<int>(point.contract.style)) + ":" +
                        std::to_string(static_cast<int>(point.contract.type)));
    saw_spot_minimum = saw_spot_minimum || point.market.spot == config.domain.spot.minimum;
    saw_spot_maximum = saw_spot_maximum || point.market.spot == config.domain.spot.maximum;
    if (point.contract.style != nre::OptionStyle::european) {
      saw_observation_minimum =
          saw_observation_minimum ||
          point.contract.observations == config.domain.minimum_asian_observations;
      saw_observation_maximum =
          saw_observation_maximum ||
          point.contract.observations == config.domain.maximum_asian_observations;
    }
    if (point.split == nre::DatasetSplit::train) {
      ++train;
    } else if (point.split == nre::DatasetSplit::validation) {
      ++validation;
    } else {
      ++test;
    }
  }
  expect(combinations.size() == 6U, "all three styles and both option types are covered");
  expect(saw_spot_minimum && saw_spot_maximum, "declared continuous boundaries are included");
  expect(saw_observation_minimum && saw_observation_maximum,
         "declared Asian observation boundaries are included");
  expect(train == 18U && validation == 3U && test == 3U,
         "point-level split rule has deterministic counts");
}

void test_generation_and_checksum_reproduction() {
  const auto config = test_config();
  const auto root = std::filesystem::temp_directory_path() / "nre_m6_dataset_tests";
  const auto first_dir = root / "first";
  const auto second_dir = root / "second";
  std::filesystem::remove_all(root);

  const auto first = nre::generate_dataset(config, first_dir);
  const auto second = nre::generate_dataset(config, second_dir);
  expect(first.total_rows == 24U && second.total_rows == 24U,
         "integration generation writes every parameter point");
  expect(first.accepted_rows + first.rejected_rows == first.total_rows,
         "integration quality policy accounts for every generated row");
  expect(first.analytical_cross_checks == 16U,
         "European and geometric-Asian rows receive analytical cross-checks");
  expect(first.analytical_cross_check_failures == 0U,
         "fixed analytical subsets pass the configured cross-check gate");
  expect(read_file(first_dir / "labels.csv").find("geometric_control_variate") != std::string::npos,
         "arithmetic labels record the trusted control-variate estimator");
  expect(first.labels_checksum == second.labels_checksum,
         "fixed configuration regenerates the same labels checksum");
  expect(read_file(first_dir / "labels.csv") == read_file(second_dir / "labels.csv"),
         "fixed configuration regenerates byte-identical labels");
  expect(read_file(first_dir / "manifest.json") == read_file(second_dir / "manifest.json"),
         "fixed configuration regenerates a byte-identical manifest");

  const auto verified = nre::verify_generated_dataset(first_dir);
  expect(
      verified.total_rows == first.total_rows && verified.labels_checksum == first.labels_checksum,
      "independent verifier confirms row count and checksum");

  {
    std::ofstream tampered(first_dir / "labels.csv", std::ios::app);
    tampered << "tampered\n";
  }
  expect_throws([&] { static_cast<void>(nre::verify_generated_dataset(first_dir)); },
                "verifier rejects a checksum mismatch");

  auto rejecting_config = config;
  rejecting_config.quality.training_max_price_standard_error = 1.0e-15;
  rejecting_config.quality.heldout_max_price_standard_error = 1.0e-15;
  const auto rejected = nre::generate_dataset(rejecting_config, root / "rejected");
  expect(rejected.rejected_rows > 0U,
         "rows that miss a predeclared uncertainty tolerance are rejected");
  const auto rejected_csv = read_file(root / "rejected/labels.csv");
  expect(
      rejected_csv.find("false,rejected,price_uncertainty_exceeds_tolerance") != std::string::npos,
      "rejected rows remain auditable with an explicit reason");
  std::filesystem::remove_all(root);
}

}  // namespace

int main() {
  test_parameter_design_and_split_integrity();
  test_generation_and_checksum_reproduction();
  if (failures != 0) {
    std::cerr << failures << " dataset test(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "dataset tests passed\n";
  return EXIT_SUCCESS;
}

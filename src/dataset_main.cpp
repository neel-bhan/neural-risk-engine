#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nre/dataset.hpp"

namespace {

void print_usage(const char* program) {
  std::cerr << "usage: " << program << " --config <file> --output <directory>\n       " << program
            << " --verify <directory>\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--verify") {
      const auto summary = nre::verify_generated_dataset(std::filesystem::path(argv[2]));
      std::cout << "{\"status\":\"verified\",\"rows\":" << summary.total_rows
                << ",\"accepted\":" << summary.accepted_rows
                << ",\"rejected\":" << summary.rejected_rows << ",\"labels_fnv1a64\":\""
                << summary.labels_checksum << "\"}\n";
      return 0;
    }
    if (argc != 5 || std::string(argv[1]) != "--config" || std::string(argv[3]) != "--output") {
      print_usage(argv[0]);
      return 2;
    }
    const auto config = nre::load_dataset_config(std::filesystem::path(argv[2]));
    const auto summary = nre::generate_dataset(config, std::filesystem::path(argv[4]));
    std::cout << "{\"status\":\"generated\",\"preset\":\"" << config.preset_name
              << "\",\"rows\":" << summary.total_rows << ",\"accepted\":" << summary.accepted_rows
              << ",\"rejected\":" << summary.rejected_rows << ",\"train\":" << summary.train_rows
              << ",\"validation\":" << summary.validation_rows << ",\"test\":" << summary.test_rows
              << ",\"analytical_failures\":" << summary.analytical_cross_check_failures
              << ",\"labels_fnv1a64\":\"" << summary.labels_checksum << "\"}\n";
  } catch (const std::exception& error) {
    std::cerr << "dataset command failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

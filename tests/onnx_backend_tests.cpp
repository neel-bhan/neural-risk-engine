#include "nre/onnx_backend.hpp"

#include <cmath>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

nre::NeuralInput european_call() {
  return {
      .contract = {.type = nre::OptionType::call,
                   .style = nre::OptionStyle::european,
                   .strike = 100.0,
                   .maturity_years = 1.0,
                   .observations = 1U},
      .market = {.spot = 100.0,
                 .volatility = 0.20,
                 .risk_free_rate = 0.03,
                 .dividend_yield = 0.01},
  };
}

struct GoldenFixture {
  nre::NeuralInput input;
  double price;
  double delta;
};

std::array<GoldenFixture, 6U> boundary_fixtures() {
  using nre::OptionStyle;
  using nre::OptionType;
  const auto make = [](OptionType type, OptionStyle style, bool low, double price, double delta) {
    const nre::NeuralInput input{
        .contract = {.type = type,
                     .style = style,
                     .strike = low ? 140.0 : 60.0,
                     .maturity_years = low ? 0.25 : 2.0,
                     .observations = style == OptionStyle::european ? 1U : (low ? 2U : 52U)},
        .market = {.spot = low ? 60.0 : 140.0,
                   .volatility = low ? 0.05 : 0.60,
                   .risk_free_rate = low ? -0.02 : 0.10,
                   .dividend_yield = low ? 0.08 : -0.01},
    };
    return GoldenFixture{.input = input, .price = price, .delta = delta};
  };
  return {
      make(OptionType::call, OptionStyle::european, true, 7.633753301993212,
           0.02694790569806223),
      make(OptionType::put, OptionStyle::european, false, 5.279478955555811,
           0.013989702621104121),
      make(OptionType::call, OptionStyle::geometric_asian, true, 7.170529378485079,
           0.09571532094359199),
      make(OptionType::put, OptionStyle::geometric_asian, false, 2.78519285204447,
           0.022339458356739655),
      make(OptionType::call, OptionStyle::arithmetic_asian, true, 10.786200650227459,
           -0.0625401311468247),
      make(OptionType::put, OptionStyle::arithmetic_asian, false, 3.39623871990801,
           0.020932896169064845),
  };
}

}  // namespace

int main() {
  const std::filesystem::path root{NRE_SOURCE_DIR};
  const auto metadata = root / "models/m9/scalar-price-v1.json";
  const auto model = root / "models/m9/scalar-price-v1.onnx";
  nre::OnnxPricingBackend backend(metadata, model);
  require(backend.artifact_version() == "nre.onnx.scalar_price.v1",
          "artifact version must be exposed");

  const auto input = european_call();
  const auto single = backend.predict({input});
  require(single.size() == 1U && std::isfinite(single[0].price) &&
              std::isfinite(single[0].delta),
          "batch-one result must be finite");
  // Golden values are emitted from the frozen Python ONNX Runtime artifact. Cross-language
  // tolerances allow only ordinary libm/runtime rounding, not model drift.
  require(std::abs(single[0].price - 9.481667379340559) <= 1.0e-10,
          "C++ physical price must match Python ONNX golden value");
  require(std::abs(single[0].delta - 0.5617866408563765) <= 1.0e-8,
          "C++ centered-bump Delta must match Python ONNX golden value");

  const auto fixtures = boundary_fixtures();
  std::vector<nre::NeuralInput> inputs;
  inputs.reserve(fixtures.size());
  for (const auto& fixture : fixtures) {
    inputs.push_back(fixture.input);
  }
  const auto batch = backend.predict(inputs);
  require(batch.size() == fixtures.size(), "larger batch size and ordering must be retained");
  for (std::size_t index = 0U; index < fixtures.size(); ++index) {
    require(std::abs(batch[index].price - fixtures[index].price) <= 1.0e-10,
            "all-style/type C++ prices must match Python boundary goldens");
    require(std::abs(batch[index].delta - fixtures[index].delta) <= 1.0e-8,
            "all-style/type C++ Deltas must match Python boundary goldens");
  }

  bool rejected = false;
  try {
    static_cast<void>(nre::OnnxPricingBackend(metadata, root / "models/m9/missing.onnx"));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "missing or mismatched model must fail clearly");

  const auto invalid_metadata =
      std::filesystem::temp_directory_path() / "nre-m9-invalid-artifact-version.json";
  {
    std::ifstream input_stream(metadata);
    std::string text{std::istreambuf_iterator<char>(input_stream),
                     std::istreambuf_iterator<char>()};
    const std::string valid_version = "nre.onnx.scalar_price.v1";
    const std::size_t position = text.find(valid_version);
    require(position != std::string::npos, "test metadata must contain the artifact version");
    text.replace(position, valid_version.size(), "nre.onnx.invalid.v1");
    std::ofstream output_stream(invalid_metadata);
    output_stream << text;
  }
  rejected = false;
  try {
    static_cast<void>(nre::OnnxPricingBackend(invalid_metadata, model));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  std::filesystem::remove(invalid_metadata);
  require(rejected, "unsupported artifact version must fail clearly");

  std::cout << "ONNX C++ parity tests passed\n";
  return 0;
}

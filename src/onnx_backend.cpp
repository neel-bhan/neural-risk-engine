#include "nre/onnx_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <onnxruntime_cxx_api.h>

namespace nre {
namespace {

constexpr std::size_t kFeatureCount = 9U;
constexpr std::array<std::string_view, kFeatureCount> kFeatureNames{
    "log_spot_over_strike", "maturity_years", "volatility", "risk_free_rate",
    "dividend_yield", "log_observations", "style_geometric_asian",
    "style_arithmetic_asian", "type_put"};

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::invalid_argument("cannot open ONNX metadata: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::size_t value_position(std::string_view text, std::string_view key) {
  const std::string quoted = std::string("\"") + std::string(key) + "\"";
  const std::size_t key_position = text.find(quoted);
  if (key_position == std::string_view::npos) {
    throw std::invalid_argument("ONNX metadata is missing key: " + std::string(key));
  }
  const std::size_t colon = text.find(':', key_position + quoted.size());
  if (colon == std::string_view::npos) {
    throw std::invalid_argument("ONNX metadata has malformed key: " + std::string(key));
  }
  return text.find_first_not_of(" \t\r\n", colon + 1U);
}

std::string string_value(std::string_view text, std::string_view key) {
  const std::size_t begin = value_position(text, key);
  if (begin == std::string_view::npos || text[begin] != '"') {
    throw std::invalid_argument("ONNX metadata key is not a string: " + std::string(key));
  }
  const std::size_t end = text.find('"', begin + 1U);
  if (end == std::string_view::npos) {
    throw std::invalid_argument("ONNX metadata contains an unterminated string");
  }
  return std::string(text.substr(begin + 1U, end - begin - 1U));
}

double number_value(std::string_view text, std::string_view key) {
  const std::size_t begin = value_position(text, key);
  const std::size_t end = text.find_first_of(",}\r\n", begin);
  const auto token = text.substr(begin, end - begin);
  std::size_t consumed = 0U;
  double value = 0.0;
  try {
    value = std::stod(std::string(token), &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("ONNX metadata key is not numeric: " + std::string(key));
  }
  if (token.find_first_not_of(" \t\r\n", consumed) != std::string_view::npos) {
    throw std::invalid_argument("ONNX metadata key is not numeric: " + std::string(key));
  }
  return value;
}

std::string_view object_value(std::string_view text, std::string_view key) {
  const std::size_t begin = value_position(text, key);
  if (begin == std::string_view::npos || text[begin] != '{') {
    throw std::invalid_argument("ONNX metadata key is not an object: " + std::string(key));
  }
  std::size_t depth = 0U;
  for (std::size_t index = begin; index < text.size(); ++index) {
    if (text[index] == '{') {
      ++depth;
    } else if (text[index] == '}') {
      --depth;
      if (depth == 0U) {
        return text.substr(begin, index - begin + 1U);
      }
    }
  }
  throw std::invalid_argument("ONNX metadata contains an unterminated object");
}

std::vector<double> number_array(std::string_view text, std::string_view key) {
  const std::size_t begin = value_position(text, key);
  if (begin == std::string_view::npos || text[begin] != '[') {
    throw std::invalid_argument("ONNX metadata key is not an array: " + std::string(key));
  }
  const std::size_t end = text.find(']', begin + 1U);
  if (end == std::string_view::npos) {
    throw std::invalid_argument("ONNX metadata contains an unterminated array");
  }
  std::vector<double> values;
  std::size_t cursor = begin + 1U;
  while (cursor < end) {
    cursor = text.find_first_not_of(" \t\r\n,", cursor);
    if (cursor == std::string_view::npos || cursor >= end) {
      break;
    }
    const std::size_t token_end = text.find_first_of(",]", cursor);
    const auto token = text.substr(cursor, token_end - cursor);
    std::size_t consumed = 0U;
    double value = 0.0;
    try {
      value = std::stod(std::string(token), &consumed);
    } catch (const std::exception&) {
      throw std::invalid_argument("ONNX metadata contains a nonnumeric array value");
    }
    if (token.find_first_not_of(" \t\r\n", consumed) != std::string_view::npos) {
      throw std::invalid_argument("ONNX metadata contains a nonnumeric array value");
    }
    values.push_back(value);
    cursor = token_end + 1U;
  }
  return values;
}

std::vector<std::string> string_array(std::string_view text, std::string_view key) {
  const std::size_t begin = value_position(text, key);
  if (begin == std::string_view::npos || text[begin] != '[') {
    throw std::invalid_argument("ONNX metadata key is not an array: " + std::string(key));
  }
  const std::size_t end = text.find(']', begin + 1U);
  if (end == std::string_view::npos) {
    throw std::invalid_argument("ONNX metadata contains an unterminated array");
  }
  std::vector<std::string> values;
  std::size_t cursor = begin + 1U;
  while (cursor < end) {
    cursor = text.find('"', cursor);
    if (cursor == std::string_view::npos || cursor >= end) {
      break;
    }
    const std::size_t close = text.find('"', cursor + 1U);
    if (close == std::string_view::npos || close > end) {
      throw std::invalid_argument("ONNX metadata contains an unterminated string array");
    }
    values.emplace_back(text.substr(cursor + 1U, close - cursor - 1U));
    cursor = close + 1U;
  }
  return values;
}

std::uint64_t fnv1a64_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("cannot open ONNX model: " + path.string());
  }
  std::uint64_t hash = 14695981039346656037ULL;
  char byte = 0;
  while (input.get(byte)) {
    hash ^= static_cast<unsigned char>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
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

struct Metadata {
  std::string artifact_version;
  std::string model_checksum;
  std::string input_name;
  std::string output_name;
  std::array<double, kFeatureCount> means{};
  std::array<double, kFeatureCount> scales{};
  NeuralGuardrailConfig guardrails{};
  double delta_relative_bump{};
  double delta_absolute_floor{};
};

Metadata load_metadata(const std::filesystem::path& path,
                       const std::filesystem::path& model_path) {
  const std::string text = read_text(path);
  Metadata metadata{};
  metadata.artifact_version = string_value(text, "artifact_version");
  if (metadata.artifact_version != "nre.onnx.scalar_price.v1") {
    throw std::invalid_argument("unsupported ONNX artifact version");
  }
  metadata.input_name = string_value(text, "input_name");
  metadata.output_name = string_value(text, "output_name");
  if (metadata.input_name != "scaled_features" || metadata.output_name != "normalized_price" ||
      string_value(text, "input_dtype") != "float64") {
    throw std::invalid_argument("unsupported ONNX tensor contract");
  }
  const auto feature_names = string_array(text, "feature_names");
  if (feature_names.size() != kFeatureNames.size()) {
    throw std::invalid_argument("ONNX feature count mismatch");
  }
  for (std::size_t index = 0U; index < kFeatureNames.size(); ++index) {
    if (feature_names[index] != kFeatureNames[index]) {
      throw std::invalid_argument("ONNX feature order mismatch");
    }
  }
  const auto preprocessing = object_value(text, "preprocessing");
  const auto means = number_array(preprocessing, "means");
  const auto scales = number_array(preprocessing, "scales");
  if (means.size() != kFeatureCount || scales.size() != kFeatureCount) {
    throw std::invalid_argument("ONNX preprocessing shape mismatch");
  }
  std::copy(means.begin(), means.end(), metadata.means.begin());
  std::copy(scales.begin(), scales.end(), metadata.scales.begin());
  for (double scale : metadata.scales) {
    if (!std::isfinite(scale) || scale <= 0.0) {
      throw std::invalid_argument("ONNX preprocessing scale must be positive and finite");
    }
  }

  const auto domain = object_value(text, "deployment_domain");
  const auto spot = number_array(domain, "spot");
  const auto strike = number_array(domain, "strike");
  const auto maturity = number_array(domain, "maturity_years");
  const auto volatility = number_array(domain, "volatility");
  const auto rate = number_array(domain, "risk_free_rate");
  const auto dividend = number_array(domain, "dividend_yield");
  const auto observations = number_array(domain, "asian_observations");
  if (spot.size() != 2U || strike.size() != 2U || maturity.size() != 2U ||
      volatility.size() != 2U || rate.size() != 2U || dividend.size() != 2U ||
      observations.size() != 2U) {
    throw std::invalid_argument("ONNX deployment-domain metadata mismatch");
  }
  metadata.guardrails.spot_min = spot[0];
  metadata.guardrails.spot_max = spot[1];
  metadata.guardrails.strike_min = strike[0];
  metadata.guardrails.strike_max = strike[1];
  metadata.guardrails.maturity_min = maturity[0];
  metadata.guardrails.maturity_max = maturity[1];
  metadata.guardrails.volatility_min = volatility[0];
  metadata.guardrails.volatility_max = volatility[1];
  metadata.guardrails.rate_min = rate[0];
  metadata.guardrails.rate_max = rate[1];
  metadata.guardrails.dividend_min = dividend[0];
  metadata.guardrails.dividend_max = dividend[1];
  metadata.guardrails.asian_observations_min = static_cast<std::size_t>(observations[0]);
  metadata.guardrails.asian_observations_max = static_cast<std::size_t>(observations[1]);

  const auto delta_policy = object_value(text, "delta_policy");
  if (string_value(delta_policy, "name") != "centered_spot_bump") {
    throw std::invalid_argument("unsupported ONNX Delta policy");
  }
  metadata.delta_relative_bump = number_value(delta_policy, "relative_bump");
  metadata.delta_absolute_floor = number_value(delta_policy, "absolute_floor");
  const auto guardrails = object_value(text, "guardrails");
  metadata.guardrails.bound_tolerance = number_value(guardrails, "bound_tolerance");
  metadata.guardrails.monotonicity_tolerance =
      number_value(guardrails, "monotonicity_tolerance");
  metadata.guardrails.spot_probe_relative =
      number_value(guardrails, "spot_probe_relative");
  metadata.guardrails.volatility_probe_absolute =
      number_value(guardrails, "volatility_probe_absolute");

  metadata.model_checksum = string_value(text, "model_fnv1a64");
  if (hexadecimal(fnv1a64_file(model_path)) != metadata.model_checksum) {
    throw std::invalid_argument("ONNX model checksum does not match metadata");
  }
  return metadata;
}

std::array<double, kFeatureCount> raw_features(const NeuralInput& input) {
  return {
      std::log(input.market.spot / input.contract.strike),
      input.contract.maturity_years,
      input.market.volatility,
      input.market.risk_free_rate,
      input.market.dividend_yield,
      std::log(static_cast<double>(input.contract.observations)),
      input.contract.style == OptionStyle::geometric_asian ? 1.0 : 0.0,
      input.contract.style == OptionStyle::arithmetic_asian ? 1.0 : 0.0,
      input.contract.type == OptionType::put ? 1.0 : 0.0,
  };
}

}  // namespace

class OnnxPricingBackend::Impl {
 public:
  Impl(const std::filesystem::path& metadata_path, const std::filesystem::path& model_path)
      : metadata_(load_metadata(metadata_path, model_path)),
        environment_(ORT_LOGGING_LEVEL_WARNING, "nre"),
        session_options_(),
        session_(nullptr) {
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(1);
    session_ = Ort::Session(environment_, model_path.c_str(), session_options_);
    const auto input_info = session_.GetInputTypeInfo(0U);
    const auto output_info = session_.GetOutputTypeInfo(0U);
    const auto input_type = input_info.GetTensorTypeAndShapeInfo();
    const auto output_type = output_info.GetTensorTypeAndShapeInfo();
    if (session_.GetInputCount() != 1U || session_.GetOutputCount() != 1U ||
        input_type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
        output_type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
      throw std::invalid_argument(
          "ONNX graph does not match the scalar float64 tensor contract: inputs=" +
          std::to_string(session_.GetInputCount()) + ", outputs=" +
          std::to_string(session_.GetOutputCount()) + ", input_type=" +
          std::to_string(static_cast<int>(input_type.GetElementType())) + ", output_type=" +
          std::to_string(static_cast<int>(output_type.GetElementType())));
    }
  }

  std::vector<NeuralCandidate> predict(const std::vector<NeuralInput>& inputs) {
    if (inputs.empty()) {
      return {};
    }
    const std::size_t graph_rows = inputs.size() * 3U;
    feature_buffer_.resize(graph_rows * kFeatureCount);
    strike_buffer_.resize(inputs.size());
    bump_buffer_.resize(inputs.size());
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
      strike_buffer_[index] = inputs[index].contract.strike;
      const double bump = std::max(metadata_.delta_relative_bump * inputs[index].market.spot,
                                   metadata_.delta_absolute_floor);
      if (!std::isfinite(bump) || bump <= 0.0 || bump >= inputs[index].market.spot) {
        throw std::invalid_argument("invalid deployment Delta bump");
      }
      bump_buffer_[index] = bump;
      write_features(index * 3U, inputs[index]);
      auto up = inputs[index];
      up.market.spot += bump;
      write_features(index * 3U + 1U, up);
      auto down = inputs[index];
      down.market.spot -= bump;
      write_features(index * 3U + 2U, down);
    }

    const std::array<std::int64_t, 2U> shape{
        static_cast<std::int64_t>(graph_rows), static_cast<std::int64_t>(kFeatureCount)};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<double>(memory, feature_buffer_.data(),
                                                   feature_buffer_.size(), shape.data(),
                                                   shape.size());
    const char* input_names[] = {metadata_.input_name.c_str()};
    const char* output_names[] = {metadata_.output_name.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1U, output_names, 1U);
    if (outputs.size() != 1U || !outputs[0].IsTensor()) {
      throw std::runtime_error("ONNX Runtime returned an invalid output");
    }
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    if (output_info.GetElementCount() != graph_rows) {
      throw std::runtime_error("ONNX Runtime output shape mismatch");
    }
    const double* normalized_prices = outputs[0].GetTensorData<double>();
    std::vector<NeuralCandidate> candidates(inputs.size());
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
      const double strike = strike_buffer_[index];
      const double price = strike * normalized_prices[index * 3U];
      const double up = strike * normalized_prices[index * 3U + 1U];
      const double down = strike * normalized_prices[index * 3U + 2U];
      candidates[index] = {.price = price, .delta = (up - down) / (2.0 * bump_buffer_[index])};
    }
    return candidates;
  }

  const NeuralGuardrailConfig& guardrail_config() const noexcept { return metadata_.guardrails; }
  const std::string& artifact_version() const noexcept { return metadata_.artifact_version; }
  const std::string& model_checksum() const noexcept { return metadata_.model_checksum; }

 private:
  void write_features(std::size_t row, const NeuralInput& input) {
    const auto values = raw_features(input);
    for (std::size_t feature = 0U; feature < kFeatureCount; ++feature) {
      feature_buffer_[row * kFeatureCount + feature] =
          (values[feature] - metadata_.means[feature]) / metadata_.scales[feature];
    }
  }

  Metadata metadata_;
  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  std::vector<double> feature_buffer_;
  std::vector<double> strike_buffer_;
  std::vector<double> bump_buffer_;
};

OnnxPricingBackend::OnnxPricingBackend(const std::filesystem::path& metadata_path,
                                       const std::filesystem::path& model_path)
    : impl_(std::make_unique<Impl>(metadata_path, model_path)) {}

OnnxPricingBackend::~OnnxPricingBackend() = default;
OnnxPricingBackend::OnnxPricingBackend(OnnxPricingBackend&&) noexcept = default;
OnnxPricingBackend& OnnxPricingBackend::operator=(OnnxPricingBackend&&) noexcept = default;

std::vector<NeuralCandidate> OnnxPricingBackend::predict(
    const std::vector<NeuralInput>& inputs) {
  return impl_->predict(inputs);
}

const NeuralGuardrailConfig& OnnxPricingBackend::guardrail_config() const noexcept {
  return impl_->guardrail_config();
}

const std::string& OnnxPricingBackend::artifact_version() const noexcept {
  return impl_->artifact_version();
}

const std::string& OnnxPricingBackend::model_checksum() const noexcept {
  return impl_->model_checksum();
}

}  // namespace nre

CXX ?= c++
CPPFLAGS := -Iinclude
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
BUILD_DIR := build/make

.PHONY: all run test check convergence variance delta-validation performance dataset-small \
	dataset-large dataset-m7 dataset-verify dataset-reproduce python-bootstrap python-test \
	baseline-train baseline-evaluate baseline-reproduce neural-train neural-evaluate \
	neural-reproduce onnx-export onnx-evaluate onnx-check onnx-evaluate-cpp clean

PYTHON := .venv/bin/python
M7_DATASET := data/generated/m7-baseline
M7_CONFIG := data/config/m7-baseline.cfg
M7_ARTIFACT := models/m7/polynomial-ridge-v1.json
M7_RESULTS := benchmarks/m7-polynomial-ridge-v1.json
M8_EXPERIMENT := python/config/m8-neural-v1.json
M8_ARTIFACT_DIR := models/m8
M8_RESULTS := benchmarks/m8-neural-v1.json
M9_METADATA := models/m9/scalar-price-v1.json
M9_MODEL := models/m9/scalar-price-v1.onnx
M9_RESULTS := benchmarks/m9-onnx-python-v1.json
ONNXRUNTIME_PREFIX ?= $(shell brew --prefix onnxruntime 2>/dev/null)
ONNX_CPPFLAGS := -isystem $(ONNXRUNTIME_PREFIX)/include/onnxruntime
ONNX_LDFLAGS := -L$(ONNXRUNTIME_PREFIX)/lib -lonnxruntime \
	-Wl,-rpath,$(ONNXRUNTIME_PREFIX)/lib

all: $(BUILD_DIR)/nre_cli

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/domain.o: src/domain.cpp include/nre/domain.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/analytics.o: src/analytics.cpp include/nre/analytics.hpp include/nre/domain.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/statistics.o: src/statistics.cpp include/nre/statistics.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/random.o: src/random.cpp include/nre/random.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/monte_carlo.o: src/monte_carlo.cpp include/nre/monte_carlo.hpp \
		include/nre/analytics.hpp include/nre/domain.hpp include/nre/random.hpp \
		include/nre/statistics.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/pricing.o: src/pricing.cpp include/nre/pricing.hpp include/nre/analytics.hpp \
		include/nre/domain.hpp include/nre/monte_carlo.hpp include/nre/statistics.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/neural_router.o: src/neural_router.cpp include/nre/neural_router.hpp \
		include/nre/pricing.hpp include/nre/domain.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/dataset.o: src/dataset.cpp include/nre/dataset.hpp include/nre/domain.hpp \
		include/nre/pricing.hpp include/nre/monte_carlo.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/nre_cli: src/main.cpp $(BUILD_DIR)/domain.o $(BUILD_DIR)/analytics.o \
		$(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o $(BUILD_DIR)/monte_carlo.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/pricing_tests: tests/pricing_tests.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o $(BUILD_DIR)/pricing.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/nre_tests: tests/domain_tests.cpp $(BUILD_DIR)/domain.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/analytics_tests: tests/analytics_tests.cpp $(BUILD_DIR)/domain.o $(BUILD_DIR)/analytics.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/statistics_tests: tests/statistics_tests.cpp $(BUILD_DIR)/statistics.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/random_tests: tests/random_tests.cpp $(BUILD_DIR)/random.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/monte_carlo_tests: tests/monte_carlo_tests.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/threading_tests: tests/threading_tests.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o $(BUILD_DIR)/pricing.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/neural_router_tests: tests/neural_router_tests.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o $(BUILD_DIR)/pricing.o $(BUILD_DIR)/neural_router.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/onnx_backend.o: src/onnx_backend.cpp include/nre/onnx_backend.hpp \
		include/nre/neural_router.hpp | $(BUILD_DIR)
	@test -n "$(ONNXRUNTIME_PREFIX)" || \
		(echo "ONNX Runtime not found; install with: brew install onnxruntime" && false)
	$(CXX) $(CPPFLAGS) $(ONNX_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/onnx_backend_tests: tests/onnx_backend_tests.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/onnx_backend.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(ONNX_CPPFLAGS) $(CXXFLAGS) -DNRE_SOURCE_DIR='"$(CURDIR)"' \
		$^ $(ONNX_LDFLAGS) -o $@

$(BUILD_DIR)/m9_guarded_evaluation: benchmarks/m9_guarded_evaluation.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o $(BUILD_DIR)/pricing.o $(BUILD_DIR)/neural_router.o \
		$(BUILD_DIR)/onnx_backend.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(ONNX_CPPFLAGS) $(CXXFLAGS) \
		-DNRE_BUILD_FLAGS='"$(CXXFLAGS)"' $^ $(ONNX_LDFLAGS) -o $@

$(BUILD_DIR)/dataset_tests: tests/dataset_tests.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o $(BUILD_DIR)/pricing.o $(BUILD_DIR)/dataset.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DNRE_SOURCE_DIR='"$(CURDIR)"' $^ -o $@

$(BUILD_DIR)/nre_dataset: src/dataset_main.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o $(BUILD_DIR)/pricing.o $(BUILD_DIR)/dataset.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/m2_convergence: benchmarks/m2_convergence.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/m3_variance_reduction: benchmarks/m3_variance_reduction.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/m4_delta_validation: benchmarks/m4_delta_validation.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/m5_performance: benchmarks/m5_performance.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o $(BUILD_DIR)/pricing.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

run: $(BUILD_DIR)/nre_cli
	./$(BUILD_DIR)/nre_cli

test: $(BUILD_DIR)/nre_tests $(BUILD_DIR)/analytics_tests $(BUILD_DIR)/statistics_tests \
		$(BUILD_DIR)/random_tests $(BUILD_DIR)/monte_carlo_tests $(BUILD_DIR)/pricing_tests \
		$(BUILD_DIR)/threading_tests $(BUILD_DIR)/dataset_tests $(BUILD_DIR)/neural_router_tests
	./$(BUILD_DIR)/nre_tests
	./$(BUILD_DIR)/analytics_tests
	./$(BUILD_DIR)/statistics_tests
	./$(BUILD_DIR)/random_tests
	./$(BUILD_DIR)/monte_carlo_tests
	./$(BUILD_DIR)/pricing_tests
	./$(BUILD_DIR)/threading_tests
	./$(BUILD_DIR)/dataset_tests
	./$(BUILD_DIR)/neural_router_tests

convergence: $(BUILD_DIR)/m2_convergence
	./$(BUILD_DIR)/m2_convergence

variance: $(BUILD_DIR)/m3_variance_reduction
	./$(BUILD_DIR)/m3_variance_reduction

delta-validation: $(BUILD_DIR)/m4_delta_validation
	./$(BUILD_DIR)/m4_delta_validation

performance: $(BUILD_DIR)/m5_performance
	./$(BUILD_DIR)/m5_performance

dataset-small: $(BUILD_DIR)/nre_dataset
	./$(BUILD_DIR)/nre_dataset --config data/config/m6-small.cfg --output data/generated/m6-small

dataset-large: $(BUILD_DIR)/nre_dataset
	./$(BUILD_DIR)/nre_dataset --config $(or $(DATASET_CONFIG),data/config/m6-large.cfg) \
		--output $(or $(DATASET_OUTPUT),data/generated/m6-large)

dataset-m7: $(BUILD_DIR)/nre_dataset
	./$(BUILD_DIR)/nre_dataset --config $(M7_CONFIG) --output $(M7_DATASET)

dataset-verify: $(BUILD_DIR)/nre_dataset
	./$(BUILD_DIR)/nre_dataset --verify $(or $(DATASET_OUTPUT),data/generated/m6-small)

dataset-reproduce: $(BUILD_DIR)/nre_dataset
	rm -rf data/generated/m6-small data/generated/m6-small-reproduction
	./$(BUILD_DIR)/nre_dataset --config data/config/m6-small.cfg --output data/generated/m6-small
	./$(BUILD_DIR)/nre_dataset --config data/config/m6-small.cfg \
		--output data/generated/m6-small-reproduction
	cmp data/generated/m6-small/labels.csv data/generated/m6-small-reproduction/labels.csv
	cmp data/generated/m6-small/manifest.json data/generated/m6-small-reproduction/manifest.json

.venv/.m9-ready: python/requirements-m9.txt
	python3 -m venv .venv
	$(PYTHON) -m pip install --disable-pip-version-check -r python/requirements-m9.txt
	touch $@

python-bootstrap: .venv/.m9-ready

python-test: python-bootstrap
	PYTHONPATH=python $(PYTHON) -m unittest discover -s python/tests -v

baseline-train: python-bootstrap
	PYTHONPATH=python $(PYTHON) -m nre_baseline.cli train --dataset $(M7_DATASET) \
		--config $(M7_CONFIG) --artifact $(M7_ARTIFACT)

baseline-evaluate: python-bootstrap
	PYTHONPATH=python $(PYTHON) -m nre_baseline.cli evaluate --dataset $(M7_DATASET) \
		--config $(M7_CONFIG) --artifact $(M7_ARTIFACT) --output $(M7_RESULTS)

baseline-reproduce: python-bootstrap
	rm -f models/generated/m7-reproduction-a.json models/generated/m7-reproduction-b.json
	PYTHONPATH=python $(PYTHON) -m nre_baseline.cli train --dataset $(M7_DATASET) \
		--config $(M7_CONFIG) --artifact models/generated/m7-reproduction-a.json
	PYTHONPATH=python $(PYTHON) -m nre_baseline.cli train --dataset $(M7_DATASET) \
		--config $(M7_CONFIG) --artifact models/generated/m7-reproduction-b.json
	cmp models/generated/m7-reproduction-a.json models/generated/m7-reproduction-b.json

neural-train: python-bootstrap
	PYTHONPATH=python $(PYTHON) -m nre_neural.cli train --dataset $(M7_DATASET) \
		--config $(M7_CONFIG) --experiment $(M8_EXPERIMENT) --output-dir $(M8_ARTIFACT_DIR)

neural-evaluate: python-bootstrap
	PYTHONPATH=python $(PYTHON) -m nre_neural.cli evaluate --dataset $(M7_DATASET) \
		--config $(M7_CONFIG) --price-only $(M8_ARTIFACT_DIR)/price-only-v1.json \
		--derivative-supervised $(M8_ARTIFACT_DIR)/derivative-supervised-v1.json \
		--baseline $(M7_ARTIFACT) --output $(M8_RESULTS)

neural-reproduce: python-bootstrap
	rm -rf models/generated/m8-reproduction
	PYTHONPATH=python $(PYTHON) -m nre_neural.cli train --dataset $(M7_DATASET) \
		--config $(M7_CONFIG) --experiment $(M8_EXPERIMENT) \
		--output-dir models/generated/m8-reproduction
	PYTHONPATH=python $(PYTHON) -m nre_neural.cli compare-reproduction \
		--reference-dir $(M8_ARTIFACT_DIR) --reproduction-dir models/generated/m8-reproduction

onnx-export: python-bootstrap
	PYTHONPATH=python $(PYTHON) -m nre_onnx.cli export \
		--neural $(M8_ARTIFACT_DIR)/derivative-supervised-v1.json \
		--model $(M9_MODEL) --metadata $(M9_METADATA)

onnx-evaluate: python-bootstrap
	PYTHONPATH=python $(PYTHON) -m nre_onnx.cli evaluate --metadata $(M9_METADATA) \
		--neural $(M8_ARTIFACT_DIR)/derivative-supervised-v1.json \
		--dataset $(M7_DATASET) --config $(M7_CONFIG) --output $(M9_RESULTS)

onnx-check: $(BUILD_DIR)/onnx_backend_tests
	./$(BUILD_DIR)/onnx_backend_tests

onnx-evaluate-cpp: $(BUILD_DIR)/m9_guarded_evaluation
	./$(BUILD_DIR)/m9_guarded_evaluation --dataset $(M7_DATASET)/labels.csv \
		--metadata $(M9_METADATA) --model $(M9_MODEL) \
		--output benchmarks/m9-onnx-cpp-guarded-v1.json

check: test python-test

clean:
	rm -rf build

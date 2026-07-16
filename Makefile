CXX ?= c++
CPPFLAGS := -Iinclude
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
BUILD_DIR := build/make

.PHONY: all run test check convergence variance delta-validation performance dataset-small \
	dataset-large dataset-verify dataset-reproduce clean

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
		$(BUILD_DIR)/threading_tests $(BUILD_DIR)/dataset_tests
	./$(BUILD_DIR)/nre_tests
	./$(BUILD_DIR)/analytics_tests
	./$(BUILD_DIR)/statistics_tests
	./$(BUILD_DIR)/random_tests
	./$(BUILD_DIR)/monte_carlo_tests
	./$(BUILD_DIR)/pricing_tests
	./$(BUILD_DIR)/threading_tests
	./$(BUILD_DIR)/dataset_tests

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

dataset-verify: $(BUILD_DIR)/nre_dataset
	./$(BUILD_DIR)/nre_dataset --verify $(or $(DATASET_OUTPUT),data/generated/m6-small)

dataset-reproduce: $(BUILD_DIR)/nre_dataset
	rm -rf data/generated/m6-small data/generated/m6-small-reproduction
	./$(BUILD_DIR)/nre_dataset --config data/config/m6-small.cfg --output data/generated/m6-small
	./$(BUILD_DIR)/nre_dataset --config data/config/m6-small.cfg \
		--output data/generated/m6-small-reproduction
	cmp data/generated/m6-small/labels.csv data/generated/m6-small-reproduction/labels.csv
	cmp data/generated/m6-small/manifest.json data/generated/m6-small-reproduction/manifest.json

check: test

clean:
	rm -rf build

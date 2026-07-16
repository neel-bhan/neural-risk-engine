CXX ?= c++
CPPFLAGS := -Iinclude
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
BUILD_DIR := build/make

.PHONY: all run test check convergence clean

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
		include/nre/domain.hpp include/nre/random.hpp include/nre/statistics.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/nre_cli: src/main.cpp $(BUILD_DIR)/domain.o $(BUILD_DIR)/analytics.o \
		$(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o $(BUILD_DIR)/monte_carlo.o | $(BUILD_DIR)
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

$(BUILD_DIR)/m2_convergence: benchmarks/m2_convergence.cpp $(BUILD_DIR)/domain.o \
		$(BUILD_DIR)/analytics.o $(BUILD_DIR)/statistics.o $(BUILD_DIR)/random.o \
		$(BUILD_DIR)/monte_carlo.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

run: $(BUILD_DIR)/nre_cli
	./$(BUILD_DIR)/nre_cli

test: $(BUILD_DIR)/nre_tests $(BUILD_DIR)/analytics_tests $(BUILD_DIR)/statistics_tests \
		$(BUILD_DIR)/random_tests $(BUILD_DIR)/monte_carlo_tests
	./$(BUILD_DIR)/nre_tests
	./$(BUILD_DIR)/analytics_tests
	./$(BUILD_DIR)/statistics_tests
	./$(BUILD_DIR)/random_tests
	./$(BUILD_DIR)/monte_carlo_tests

convergence: $(BUILD_DIR)/m2_convergence
	./$(BUILD_DIR)/m2_convergence

check: test

clean:
	rm -rf build

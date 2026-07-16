CXX ?= c++
CPPFLAGS := -Iinclude
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
BUILD_DIR := build/make

.PHONY: all run test check clean

all: $(BUILD_DIR)/nre_cli

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/domain.o: src/domain.cpp include/nre/domain.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/nre_cli: src/main.cpp $(BUILD_DIR)/domain.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/nre_tests: tests/domain_tests.cpp $(BUILD_DIR)/domain.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

run: $(BUILD_DIR)/nre_cli
	./$(BUILD_DIR)/nre_cli

test: $(BUILD_DIR)/nre_tests
	./$(BUILD_DIR)/nre_tests

check: test

clean:
	rm -rf build


cmake_minimum_required(VERSION 3.24)

project(neural_risk_engine VERSION 0.1.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

option(NRE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(NRE_ENABLE_ONNX "Build the optional ONNX Runtime pricing backend" OFF)

add_library(nre_core
    src/analytics.cpp
    src/dataset.cpp
    src/domain.cpp
    src/monte_carlo.cpp
    src/neural_router.cpp
    src/pricing.cpp
    src/random.cpp
    src/statistics.cpp
)
add_library(nre::core ALIAS nre_core)

target_include_directories(nre_core PUBLIC include)
target_compile_features(nre_core PUBLIC cxx_std_20)
target_link_libraries(nre_core PUBLIC Threads::Threads)

if(NRE_ENABLE_ONNX)
  find_package(onnxruntime 1.27 CONFIG REQUIRED)
  add_library(nre_onnx src/onnx_backend.cpp)
  add_library(nre::onnx ALIAS nre_onnx)
  target_include_directories(nre_onnx PUBLIC include)
  target_compile_features(nre_onnx PUBLIC cxx_std_20)
  target_link_libraries(nre_onnx PUBLIC nre::core onnxruntime::onnxruntime)

  add_executable(nre_m9_guarded_evaluation benchmarks/m9_guarded_evaluation.cpp)
  target_link_libraries(nre_m9_guarded_evaluation PRIVATE nre::onnx)
  target_compile_definitions(nre_m9_guarded_evaluation PRIVATE
      NRE_BUILD_FLAGS="CMake ${CMAKE_BUILD_TYPE} build")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
  target_compile_options(nre_core PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow
  )
  if(NRE_WARNINGS_AS_ERRORS)
    target_compile_options(nre_core PRIVATE -Werror)
  endif()
  if(NRE_ENABLE_ONNX)
    target_compile_options(nre_onnx PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )
    if(NRE_WARNINGS_AS_ERRORS)
      target_compile_options(nre_onnx PRIVATE -Werror)
    endif()
  endif()
elseif(MSVC)
  target_compile_options(nre_core PRIVATE /W4)
  if(NRE_WARNINGS_AS_ERRORS)
    target_compile_options(nre_core PRIVATE /WX)
  endif()
  if(NRE_ENABLE_ONNX)
    target_compile_options(nre_onnx PRIVATE /W4)
    if(NRE_WARNINGS_AS_ERRORS)
      target_compile_options(nre_onnx PRIVATE /WX)
    endif()
  endif()
endif()

add_executable(nre_cli src/main.cpp)
target_link_libraries(nre_cli PRIVATE nre::core)

add_executable(nre_m2_convergence benchmarks/m2_convergence.cpp)
target_link_libraries(nre_m2_convergence PRIVATE nre::core)

add_executable(nre_m3_variance_reduction benchmarks/m3_variance_reduction.cpp)
target_link_libraries(nre_m3_variance_reduction PRIVATE nre::core)

add_executable(nre_m4_delta_validation benchmarks/m4_delta_validation.cpp)
target_link_libraries(nre_m4_delta_validation PRIVATE nre::core)

add_executable(nre_m5_performance benchmarks/m5_performance.cpp)
target_link_libraries(nre_m5_performance PRIVATE nre::core)

add_executable(nre_dataset src/dataset_main.cpp)
target_link_libraries(nre_dataset PRIVATE nre::core)

include(CTest)
if(BUILD_TESTING)
  add_executable(nre_tests tests/domain_tests.cpp)
  target_link_libraries(nre_tests PRIVATE nre::core)
  add_test(NAME domain_tests COMMAND nre_tests)

  add_executable(nre_analytics_tests tests/analytics_tests.cpp)
  target_link_libraries(nre_analytics_tests PRIVATE nre::core)
  add_test(NAME analytics_tests COMMAND nre_analytics_tests)

  add_executable(nre_statistics_tests tests/statistics_tests.cpp)
  target_link_libraries(nre_statistics_tests PRIVATE nre::core)
  add_test(NAME statistics_tests COMMAND nre_statistics_tests)

  add_executable(nre_random_tests tests/random_tests.cpp)
  target_link_libraries(nre_random_tests PRIVATE nre::core)
  add_test(NAME random_tests COMMAND nre_random_tests)

  add_executable(nre_monte_carlo_tests tests/monte_carlo_tests.cpp)
  target_link_libraries(nre_monte_carlo_tests PRIVATE nre::core)
  add_test(NAME monte_carlo_tests COMMAND nre_monte_carlo_tests)

  add_executable(nre_pricing_tests tests/pricing_tests.cpp)
  target_link_libraries(nre_pricing_tests PRIVATE nre::core)
  add_test(NAME pricing_tests COMMAND nre_pricing_tests)

  add_executable(nre_threading_tests tests/threading_tests.cpp)
  target_link_libraries(nre_threading_tests PRIVATE nre::core)
  add_test(NAME threading_tests COMMAND nre_threading_tests)

  add_executable(nre_dataset_tests tests/dataset_tests.cpp)
  target_link_libraries(nre_dataset_tests PRIVATE nre::core)
  target_compile_definitions(nre_dataset_tests PRIVATE NRE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME dataset_tests COMMAND nre_dataset_tests)

  add_executable(nre_neural_router_tests tests/neural_router_tests.cpp)
  target_link_libraries(nre_neural_router_tests PRIVATE nre::core)
  add_test(NAME neural_router_tests COMMAND nre_neural_router_tests)

  if(NRE_ENABLE_ONNX)
    add_executable(nre_onnx_backend_tests tests/onnx_backend_tests.cpp)
    target_link_libraries(nre_onnx_backend_tests PRIVATE nre::onnx)
    target_compile_definitions(nre_onnx_backend_tests PRIVATE NRE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    add_test(NAME onnx_backend_tests COMMAND nre_onnx_backend_tests)
  endif()
endif()

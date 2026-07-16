cmake_minimum_required(VERSION 3.24)

project(neural_risk_engine VERSION 0.1.0 LANGUAGES CXX)

option(NRE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(nre_core
    src/analytics.cpp
    src/domain.cpp
)
add_library(nre::core ALIAS nre_core)

target_include_directories(nre_core PUBLIC include)
target_compile_features(nre_core PUBLIC cxx_std_20)

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
elseif(MSVC)
  target_compile_options(nre_core PRIVATE /W4)
  if(NRE_WARNINGS_AS_ERRORS)
    target_compile_options(nre_core PRIVATE /WX)
  endif()
endif()

add_executable(nre_cli src/main.cpp)
target_link_libraries(nre_cli PRIVATE nre::core)

include(CTest)
if(BUILD_TESTING)
  add_executable(nre_tests tests/domain_tests.cpp)
  target_link_libraries(nre_tests PRIVATE nre::core)
  add_test(NAME domain_tests COMMAND nre_tests)

  add_executable(nre_analytics_tests tests/analytics_tests.cpp)
  target_link_libraries(nre_analytics_tests PRIVATE nre::core)
  add_test(NAME analytics_tests COMMAND nre_analytics_tests)
endif()

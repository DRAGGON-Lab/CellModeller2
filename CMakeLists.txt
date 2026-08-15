cmake_minimum_required(VERSION 3.25)

project(
  CellModeller2
  VERSION 0.1.0
  DESCRIPTION "Native Metal and CUDA multicellular simulation engine"
  LANGUAGES CXX
)

option(CM2_BUILD_PYTHON "Build the nanobind Python extension" ON)
option(CM2_BUILD_TESTS "Build C++ tests" ON)

set(CM2_METAL_DEFAULT OFF)
if(APPLE)
  set(CM2_METAL_DEFAULT ON)
endif()
option(CM2_ENABLE_METAL "Build the native Apple Metal backend" ${CM2_METAL_DEFAULT})

add_library(cm2_core STATIC
  cpp/core/world_state.cpp
  cpp/core/simulation.cpp
  cpp/cpu/cpu_backend.cpp
)
add_library(cm2::core ALIAS cm2_core)

target_compile_features(cm2_core PUBLIC cxx_std_23)
target_include_directories(cm2_core PUBLIC cpp/include)
set_target_properties(cm2_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CM2_ENABLE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "CM2_ENABLE_METAL requires an Apple platform")
  endif()

  enable_language(OBJCXX)
  find_library(CM2_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(CM2_METAL_FRAMEWORK Metal REQUIRED)

  set(CM2_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(CM2_METAL_GROWTH_HEADER "${CM2_GENERATED_DIR}/cm2/metal/growth_source.hpp")
  add_custom_command(
    OUTPUT "${CM2_METAL_GROWTH_HEADER}"
    COMMAND "${CMAKE_COMMAND}"
      "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/cpp/metal/kernels/growth.metal"
      "-DOUTPUT=${CM2_METAL_GROWTH_HEADER}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalSource.cmake"
    DEPENDS
      cpp/metal/kernels/growth.metal
      cmake/EmbedMetalSource.cmake
    VERBATIM
  )

  target_sources(cm2_core PRIVATE
    cpp/metal/metal_backend.mm
    "${CM2_METAL_GROWTH_HEADER}"
  )
  target_include_directories(cm2_core PRIVATE "${CM2_GENERATED_DIR}")
  target_compile_definitions(cm2_core PRIVATE CM2_HAS_METAL=1)
  target_compile_options(cm2_core PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
  target_link_libraries(cm2_core PRIVATE
    "${CM2_FOUNDATION_FRAMEWORK}"
    "${CM2_METAL_FRAMEWORK}"
  )
  set_target_properties(cm2_core PROPERTIES OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON)
endif()

if(MSVC)
  target_compile_options(cm2_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(cm2_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

if(CM2_BUILD_PYTHON)
  find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_core cpp/python/bindings.cpp)
  target_link_libraries(_core PRIVATE cm2::core)
  install(TARGETS _core LIBRARY DESTINATION cellmodeller2)
endif()

if(CM2_BUILD_TESTS)
  include(CTest)
  add_executable(cm2_world_state_test tests/cpp/world_state_test.cpp)
  target_link_libraries(cm2_world_state_test PRIVATE cm2::core)
  add_test(NAME world_state COMMAND cm2_world_state_test)

  if(CM2_ENABLE_METAL)
    add_executable(cm2_metal_growth_test tests/conformance/metal_growth_test.cpp)
    target_link_libraries(cm2_metal_growth_test PRIVATE cm2::core)
    add_test(NAME metal_growth COMMAND cm2_metal_growth_test)
  endif()
endif()

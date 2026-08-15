cmake_minimum_required(VERSION 3.25)

project(
  CellModeller2
  VERSION 0.1.0
  DESCRIPTION "Native Metal and CUDA multicellular simulation engine"
  LANGUAGES CXX
)

option(CM2_BUILD_PYTHON "Build the nanobind Python extension" ON)
option(CM2_BUILD_TESTS "Build C++ tests" ON)
option(CM2_ENABLE_CUDA "Build the native NVIDIA CUDA backend" OFF)

set(CM2_METAL_DEFAULT OFF)
if(APPLE)
  set(CM2_METAL_DEFAULT ON)
endif()
option(CM2_ENABLE_METAL "Build the native Apple Metal backend" ${CM2_METAL_DEFAULT})

add_library(cm2_core STATIC
  cpp/core/checkpoint.cpp
  cpp/core/contact_graph.cpp
  cpp/core/constraints.cpp
  cpp/core/mechanics_integration.cpp
  cpp/core/signals.cpp
  cpp/core/species.cpp
  cpp/core/world_state.cpp
  cpp/core/simulation.cpp
  cpp/cpu/cpu_backend.cpp
  cpp/cpu/cpu_contacts.cpp
  cpp/cpu/cpu_constraints.cpp
  cpp/cpu/cpu_mechanics.cpp
  cpp/cpu/cpu_species.cpp
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
  set(CM2_METAL_CONTACTS_HEADER "${CM2_GENERATED_DIR}/cm2/metal/contacts_source.hpp")
  set(CM2_METAL_MECHANICS_HEADER "${CM2_GENERATED_DIR}/cm2/metal/mechanics_source.hpp")
  set(CM2_METAL_SPECIES_HEADER "${CM2_GENERATED_DIR}/cm2/metal/species_source.hpp")
  add_custom_command(
    OUTPUT "${CM2_METAL_GROWTH_HEADER}"
    COMMAND "${CMAKE_COMMAND}"
      "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/cpp/metal/kernels/growth.metal"
      "-DOUTPUT=${CM2_METAL_GROWTH_HEADER}"
      "-DSYMBOL=growth_source"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalSource.cmake"
    DEPENDS
      cpp/metal/kernels/growth.metal
      cmake/EmbedMetalSource.cmake
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${CM2_METAL_CONTACTS_HEADER}"
    COMMAND "${CMAKE_COMMAND}"
      "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/cpp/metal/kernels/contacts.metal"
      "-DOUTPUT=${CM2_METAL_CONTACTS_HEADER}"
      "-DSYMBOL=contacts_source"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalSource.cmake"
    DEPENDS
      cpp/metal/kernels/contacts.metal
      cmake/EmbedMetalSource.cmake
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${CM2_METAL_MECHANICS_HEADER}"
    COMMAND "${CMAKE_COMMAND}"
      "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/cpp/metal/kernels/mechanics.metal"
      "-DOUTPUT=${CM2_METAL_MECHANICS_HEADER}"
      "-DSYMBOL=mechanics_source"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalSource.cmake"
    DEPENDS
      cpp/metal/kernels/mechanics.metal
      cmake/EmbedMetalSource.cmake
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${CM2_METAL_SPECIES_HEADER}"
    COMMAND "${CMAKE_COMMAND}"
      "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/cpp/metal/kernels/species.metal"
      "-DOUTPUT=${CM2_METAL_SPECIES_HEADER}"
      "-DSYMBOL=species_source"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalSource.cmake"
    DEPENDS
      cpp/metal/kernels/species.metal
      cmake/EmbedMetalSource.cmake
    VERBATIM
  )

  target_sources(cm2_core PRIVATE
    cpp/metal/metal_backend.mm
    "${CM2_METAL_CONTACTS_HEADER}"
    "${CM2_METAL_GROWTH_HEADER}"
    "${CM2_METAL_MECHANICS_HEADER}"
    "${CM2_METAL_SPECIES_HEADER}"
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

if(CM2_ENABLE_CUDA)
  if(APPLE)
    message(FATAL_ERROR "CM2_ENABLE_CUDA requires a platform supported by the CUDA Toolkit")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(cm2_core PRIVATE
    cpp/cuda/cuda_backend.cu
    cpp/cuda/kernels/contacts.cu
    cpp/cuda/kernels/growth.cu
    cpp/cuda/kernels/mechanics.cu
    cpp/cuda/kernels/species.cu
  )
  target_compile_definitions(cm2_core PRIVATE CM2_HAS_CUDA=1)
  target_link_libraries(cm2_core PRIVATE CUDA::cudart)
  set_target_properties(cm2_core PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )
endif()

if(MSVC)
  target_compile_options(cm2_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(cm2_core PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>"
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>"
  )
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

  add_executable(cm2_contact_geometry_test tests/cpp/contact_geometry_test.cpp)
  target_link_libraries(cm2_contact_geometry_test PRIVATE cm2::core)
  add_test(NAME contact_geometry COMMAND cm2_contact_geometry_test)

  add_executable(cm2_constraint_geometry_test tests/cpp/constraint_geometry_test.cpp)
  target_link_libraries(cm2_constraint_geometry_test PRIVATE cm2::core)
  add_test(NAME constraint_geometry COMMAND cm2_constraint_geometry_test)

  add_executable(cm2_mechanics_operator_test tests/cpp/mechanics_operator_test.cpp)
  target_link_libraries(cm2_mechanics_operator_test PRIVATE cm2::core)
  add_test(NAME mechanics_operator COMMAND cm2_mechanics_operator_test)

  add_executable(cm2_mechanics_integration_test tests/cpp/mechanics_integration_test.cpp)
  target_link_libraries(cm2_mechanics_integration_test PRIVATE cm2::core)
  add_test(NAME mechanics_integration COMMAND cm2_mechanics_integration_test)

  add_executable(cm2_species_test tests/cpp/species_test.cpp)
  target_link_libraries(cm2_species_test PRIVATE cm2::core)
  add_test(NAME species COMMAND cm2_species_test)

  add_executable(cm2_signal_grid_test tests/cpp/signal_grid_test.cpp)
  target_link_libraries(cm2_signal_grid_test PRIVATE cm2::core)
  add_test(NAME signal_grid COMMAND cm2_signal_grid_test)

  add_executable(cm2_checkpoint_test tests/cpp/checkpoint_test.cpp)
  target_link_libraries(cm2_checkpoint_test PRIVATE cm2::core)
  add_test(NAME checkpoint COMMAND cm2_checkpoint_test)
  set_tests_properties(checkpoint PROPERTIES LABELS conformance)

  add_executable(cm2_growth_conformance_test tests/conformance/growth_conformance_test.cpp)
  target_link_libraries(cm2_growth_conformance_test PRIVATE cm2::core)
  add_test(NAME growth_conformance COMMAND cm2_growth_conformance_test)
  set_tests_properties(growth_conformance PROPERTIES LABELS conformance)

  add_executable(cm2_species_conformance_test
    tests/conformance/species_conformance_test.cpp
  )
  target_link_libraries(cm2_species_conformance_test PRIVATE cm2::core)
  add_test(NAME species_conformance COMMAND cm2_species_conformance_test)
  set_tests_properties(species_conformance PROPERTIES LABELS conformance)

  add_executable(cm2_signal_grid_conformance_test
    tests/conformance/signal_grid_conformance_test.cpp
  )
  target_link_libraries(cm2_signal_grid_conformance_test PRIVATE cm2::core)
  add_test(NAME signal_grid_conformance COMMAND cm2_signal_grid_conformance_test)
  set_tests_properties(signal_grid_conformance PROPERTIES LABELS conformance)

  add_executable(cm2_lifecycle_conformance_test
    tests/conformance/lifecycle_conformance_test.cpp
  )
  target_link_libraries(cm2_lifecycle_conformance_test PRIVATE cm2::core)
  add_test(NAME lifecycle_conformance COMMAND cm2_lifecycle_conformance_test)
  set_tests_properties(lifecycle_conformance PROPERTIES LABELS conformance)

  add_executable(cm2_contact_conformance_test
    tests/conformance/contact_conformance_test.cpp
  )
  target_link_libraries(cm2_contact_conformance_test PRIVATE cm2::core)
  add_test(NAME contact_conformance COMMAND cm2_contact_conformance_test)
  set_tests_properties(contact_conformance PROPERTIES LABELS conformance)

  add_executable(cm2_constraint_conformance_test
    tests/conformance/constraint_conformance_test.cpp
  )
  target_link_libraries(cm2_constraint_conformance_test PRIVATE cm2::core)
  add_test(NAME constraint_conformance COMMAND cm2_constraint_conformance_test)
  set_tests_properties(constraint_conformance PROPERTIES LABELS conformance)

  add_executable(cm2_mechanics_conformance_test
    tests/conformance/mechanics_conformance_test.cpp
  )
  target_link_libraries(cm2_mechanics_conformance_test PRIVATE cm2::core)
  add_test(NAME mechanics_conformance COMMAND cm2_mechanics_conformance_test)
  set_tests_properties(mechanics_conformance PROPERTIES LABELS conformance)

  add_executable(cm2_constraint_mechanics_conformance_test
    tests/conformance/constraint_mechanics_conformance_test.cpp
  )
  target_link_libraries(cm2_constraint_mechanics_conformance_test PRIVATE cm2::core)
  add_test(NAME constraint_mechanics_conformance COMMAND cm2_constraint_mechanics_conformance_test)
  set_tests_properties(constraint_mechanics_conformance PROPERTIES LABELS conformance)
endif()

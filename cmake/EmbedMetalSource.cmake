if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
  message(FATAL_ERROR "EmbedMetalSource.cmake requires INPUT, OUTPUT, and SYMBOL")
endif()
if(NOT SYMBOL MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
  message(FATAL_ERROR "EmbedMetalSource.cmake received an invalid C++ symbol")
endif()

file(READ "${INPUT}" CM2_METAL_SOURCE)
get_filename_component(CM2_OUTPUT_DIRECTORY "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${CM2_OUTPUT_DIRECTORY}")
file(WRITE "${OUTPUT}" "#pragma once\n\nnamespace cm2::metal {\ninline constexpr char ${SYMBOL}[] = R\"CM2_METAL(")
file(APPEND "${OUTPUT}" "${CM2_METAL_SOURCE}")
file(APPEND "${OUTPUT}" [=[)CM2_METAL";
}  // namespace cm2::metal
]=])

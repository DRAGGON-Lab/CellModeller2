if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
  message(FATAL_ERROR "EmbedMetalSource.cmake requires INPUT, OUTPUT, and SYMBOL")
endif()
if(NOT SYMBOL MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
  message(FATAL_ERROR "EmbedMetalSource.cmake received an invalid C++ symbol")
endif()

file(READ "${INPUT}" CM_METAL_SOURCE)
get_filename_component(CM_OUTPUT_DIRECTORY "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${CM_OUTPUT_DIRECTORY}")
file(WRITE "${OUTPUT}" "#pragma once\n\nnamespace cm::metal {\ninline constexpr char ${SYMBOL}[] = R\"CM_METAL(")
file(APPEND "${OUTPUT}" "${CM_METAL_SOURCE}")
file(APPEND "${OUTPUT}" [=[)CM_METAL";
}  // namespace cm::metal
]=])

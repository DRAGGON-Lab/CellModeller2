if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "EmbedMetalSource.cmake requires INPUT and OUTPUT")
endif()

file(READ "${INPUT}" CM2_METAL_SOURCE)
get_filename_component(CM2_OUTPUT_DIRECTORY "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${CM2_OUTPUT_DIRECTORY}")
file(WRITE "${OUTPUT}" [=[#pragma once

namespace cm2::metal {
inline constexpr char growth_source[] = R"CM2_METAL(]=])
file(APPEND "${OUTPUT}" "${CM2_METAL_SOURCE}")
file(APPEND "${OUTPUT}" [=[)CM2_METAL";
}  // namespace cm2::metal
]=])

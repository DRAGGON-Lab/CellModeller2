#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cm2::cuda {

void launch_growth(float* lengths, const float* growth_rates, float dt, std::uint32_t count,
                   cudaStream_t stream);

}  // namespace cm2::cuda

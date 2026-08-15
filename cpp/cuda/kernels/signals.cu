#include <cmath>

#include "signals.cuh"

namespace cm2::cuda {
namespace {

__device__ std::uint32_t site_index(SignalGridShapeGpu shape, std::uint32_t x, std::uint32_t y,
                                    std::uint32_t z) {
  return x * shape.y * shape.z + y * shape.z + z;
}

__device__ float grid_level(const float* levels, SignalGridShapeGpu shape, std::uint32_t signal,
                            std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return levels[signal * shape.sites + site_index(shape, x, y, z)];
}

__device__ float exterior_value(std::uint32_t kind, const float* fixed_values, std::uint32_t face,
                                std::uint32_t signal, std::uint32_t signal_count, float current,
                                float periodic) {
  if (kind == 0) {
    return current;
  }
  if (kind == 1) {
    return periodic;
  }
  return fixed_values[face * signal_count + signal];
}

__global__ void advance_signal_grid(const float* levels, float* output, const float* diffusion,
                                    const float4* advection, const float* fixed_values,
                                    std::uint32_t* error, SignalGridBoundariesGpu boundaries,
                                    SignalGridShapeGpu shape, float4 spacing, float dt,
                                    std::uint32_t signal_count, std::uint32_t level_count) {
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= level_count) {
    return;
  }

  const auto signal = index / shape.sites;
  const auto site = index - signal * shape.sites;
  const auto x = site / (shape.y * shape.z);
  const auto yz = site - x * shape.y * shape.z;
  const auto y = yz / shape.z;
  const auto z = yz - y * shape.z;
  const auto current = levels[index];

  float lower[3];
  float upper[3];
  lower[0] = x == 0 ? exterior_value(boundaries.x_lower, fixed_values, 0, signal, signal_count,
                                     current, grid_level(levels, shape, signal, shape.x - 1, y, z))
                    : grid_level(levels, shape, signal, x - 1, y, z);
  upper[0] = x + 1 == shape.x
                 ? exterior_value(boundaries.x_upper, fixed_values, 1, signal, signal_count,
                                  current, grid_level(levels, shape, signal, 0, y, z))
                 : grid_level(levels, shape, signal, x + 1, y, z);
  lower[1] = y == 0 ? exterior_value(boundaries.y_lower, fixed_values, 2, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, shape.y - 1, z))
                    : grid_level(levels, shape, signal, x, y - 1, z);
  upper[1] = y + 1 == shape.y
                 ? exterior_value(boundaries.y_upper, fixed_values, 3, signal, signal_count,
                                  current, grid_level(levels, shape, signal, x, 0, z))
                 : grid_level(levels, shape, signal, x, y + 1, z);
  lower[2] = z == 0 ? exterior_value(boundaries.z_lower, fixed_values, 4, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, y, shape.z - 1))
                    : grid_level(levels, shape, signal, x, y, z - 1);
  upper[2] = z + 1 == shape.z
                 ? exterior_value(boundaries.z_upper, fixed_values, 5, signal, signal_count,
                                  current, grid_level(levels, shape, signal, x, y, 0))
                 : grid_level(levels, shape, signal, x, y, z + 1);

  const std::uint32_t dimensions[3]{shape.x, shape.y, shape.z};
  const bool at_lower[3]{x == 0, y == 0, z == 0};
  const bool at_upper[3]{x + 1 == shape.x, y + 1 == shape.y, z + 1 == shape.z};
  const std::uint32_t lower_kinds[3]{boundaries.x_lower, boundaries.y_lower, boundaries.z_lower};
  const std::uint32_t upper_kinds[3]{boundaries.x_upper, boundaries.y_upper, boundaries.z_upper};
  const float velocity[3]{advection[signal].x, advection[signal].y, advection[signal].z};
  const float grid_spacing[3]{spacing.x, spacing.y, spacing.z};
  float rate = 0.0F;
  for (std::uint32_t axis = 0; axis < 3; ++axis) {
    if (dimensions[axis] == 1) {
      continue;
    }
    const auto inverse_spacing = 1.0F / grid_spacing[axis];
    rate += diffusion[signal] * (lower[axis] - 2.0F * current + upper[axis]) * inverse_spacing *
            inverse_spacing;
    auto lower_flux =
        velocity[axis] >= 0.0F ? velocity[axis] * lower[axis] : velocity[axis] * current;
    auto upper_flux =
        velocity[axis] >= 0.0F ? velocity[axis] * current : velocity[axis] * upper[axis];
    if (at_lower[axis] && lower_kinds[axis] == 0) {
      lower_flux = 0.0F;
    }
    if (at_upper[axis] && upper_kinds[axis] == 0) {
      upper_flux = 0.0F;
    }
    rate -= (upper_flux - lower_flux) * inverse_spacing;
  }

  const auto candidate = current + dt * rate;
  output[index] = candidate;
  if (!isfinite(candidate) || candidate < 0.0F) {
    atomicOr(error, 1U);
  }
}

}  // namespace

void launch_advance_signal_grid(const float* levels, float* output, const float* diffusion,
                                const float4* advection, const float* fixed_values,
                                std::uint32_t* error, SignalGridBoundariesGpu boundaries,
                                SignalGridShapeGpu shape, float4 spacing, float dt,
                                std::uint32_t signal_count, std::uint32_t level_count,
                                cudaStream_t stream) {
  constexpr std::uint32_t threads_per_block = 256;
  const auto block_count = ((level_count - 1) / threads_per_block) + 1;
  advance_signal_grid<<<block_count, threads_per_block, 0, stream>>>(
      levels, output, diffusion, advection, fixed_values, error, boundaries, shape, spacing, dt,
      signal_count, level_count);
}

}  // namespace cm2::cuda

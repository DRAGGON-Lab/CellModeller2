#include <metal_stdlib>

using namespace metal;

struct GridShape {
  uint x;
  uint y;
  uint z;
  uint sites;
};

uint site_index(GridShape shape, uint x, uint y, uint z) {
  return x * shape.y * shape.z + y * shape.z + z;
}

float grid_level(device const float* levels, GridShape shape, uint signal, uint x, uint y, uint z) {
  return levels[signal * shape.sites + site_index(shape, x, y, z)];
}

float exterior_value(uint kind, device const float* fixed_values, uint face, uint signal,
                     uint signal_count, float current, float periodic) {
  if (kind == 0u) {
    return current;
  }
  if (kind == 1u) {
    return periodic;
  }
  return fixed_values[face * signal_count + signal];
}

kernel void advance_signal_grid(
    device const float* levels [[buffer(0)]], device float* output [[buffer(1)]],
    device const float* diffusion [[buffer(2)]], device const float4* advection [[buffer(3)]],
    device const float* fixed_values [[buffer(4)]], device atomic_uint* error [[buffer(5)]],
    constant uint* boundary_kinds [[buffer(6)]], constant GridShape& shape [[buffer(7)]],
    constant float4& spacing [[buffer(8)]], constant float& dt [[buffer(9)]],
    constant uint& signal_count [[buffer(10)]], constant uint& level_count [[buffer(11)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= level_count) {
    return;
  }

  uint signal = index / shape.sites;
  uint site = index - signal * shape.sites;
  uint x = site / (shape.y * shape.z);
  uint yz = site - x * shape.y * shape.z;
  uint y = yz / shape.z;
  uint z = yz - y * shape.z;
  float current = levels[index];

  float3 lower;
  float3 upper;
  lower.x = x == 0u ? exterior_value(boundary_kinds[0], fixed_values, 0u, signal, signal_count,
                                     current, grid_level(levels, shape, signal, shape.x - 1u, y, z))
                    : grid_level(levels, shape, signal, x - 1u, y, z);
  upper.x = x + 1u == shape.x
                ? exterior_value(boundary_kinds[1], fixed_values, 1u, signal, signal_count, current,
                                 grid_level(levels, shape, signal, 0u, y, z))
                : grid_level(levels, shape, signal, x + 1u, y, z);
  lower.y = y == 0u ? exterior_value(boundary_kinds[2], fixed_values, 2u, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, shape.y - 1u, z))
                    : grid_level(levels, shape, signal, x, y - 1u, z);
  upper.y = y + 1u == shape.y
                ? exterior_value(boundary_kinds[3], fixed_values, 3u, signal, signal_count, current,
                                 grid_level(levels, shape, signal, x, 0u, z))
                : grid_level(levels, shape, signal, x, y + 1u, z);
  lower.z = z == 0u ? exterior_value(boundary_kinds[4], fixed_values, 4u, signal, signal_count,
                                     current, grid_level(levels, shape, signal, x, y, shape.z - 1u))
                    : grid_level(levels, shape, signal, x, y, z - 1u);
  upper.z = z + 1u == shape.z
                ? exterior_value(boundary_kinds[5], fixed_values, 5u, signal, signal_count, current,
                                 grid_level(levels, shape, signal, x, y, 0u))
                : grid_level(levels, shape, signal, x, y, z + 1u);

  uint3 dimensions = uint3(shape.x, shape.y, shape.z);
  bool3 at_lower = bool3(x == 0u, y == 0u, z == 0u);
  bool3 at_upper = bool3(x + 1u == shape.x, y + 1u == shape.y, z + 1u == shape.z);
  float3 velocity = advection[signal].xyz;
  float3 grid_spacing = spacing.xyz;
  float rate = 0.0f;
  for (uint axis = 0; axis < 3u; ++axis) {
    if (dimensions[axis] == 1u) {
      continue;
    }
    float inverse_spacing = 1.0f / grid_spacing[axis];
    rate += diffusion[signal] * (lower[axis] - 2.0f * current + upper[axis]) * inverse_spacing *
            inverse_spacing;
    float lower_flux =
        velocity[axis] >= 0.0f ? velocity[axis] * lower[axis] : velocity[axis] * current;
    float upper_flux =
        velocity[axis] >= 0.0f ? velocity[axis] * current : velocity[axis] * upper[axis];
    if (at_lower[axis] && boundary_kinds[axis * 2u] == 0u) {
      lower_flux = 0.0f;
    }
    if (at_upper[axis] && boundary_kinds[axis * 2u + 1u] == 0u) {
      upper_flux = 0.0f;
    }
    rate -= (upper_flux - lower_flux) * inverse_spacing;
  }

  float candidate = current + dt * rate;
  output[index] = candidate;
  if (!isfinite(candidate) || candidate < 0.0f) {
    atomic_fetch_or_explicit(error, 1u, memory_order_relaxed);
  }
}

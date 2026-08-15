#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace cm2 {

using CellId = std::uint64_t;
using Slot = std::uint32_t;

inline constexpr CellId invalid_cell_id = 0;
inline constexpr Slot invalid_slot = std::numeric_limits<Slot>::max();

struct Vec3 {
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};

  [[nodiscard]] constexpr Vec3 operator+(const Vec3& other) const noexcept {
    return {x + other.x, y + other.y, z + other.z};
  }

  [[nodiscard]] constexpr Vec3 operator-(const Vec3& other) const noexcept {
    return {x - other.x, y - other.y, z - other.z};
  }

  [[nodiscard]] constexpr Vec3 operator*(float scale) const noexcept {
    return {x * scale, y * scale, z * scale};
  }
};

[[nodiscard]] inline float norm(const Vec3& value) noexcept {
  return std::sqrt((value.x * value.x) + (value.y * value.y) + (value.z * value.z));
}

[[nodiscard]] inline Vec3 normalized(const Vec3& value) {
  const auto magnitude = norm(value);
  if (!std::isfinite(magnitude) || magnitude <= 0.0F) {
    throw std::invalid_argument("cell direction must be finite and non-zero");
  }
  return value * (1.0F / magnitude);
}

enum class BackendKind : std::uint8_t {
  cpu,
  metal,
  cuda,
};

struct BackendInfo {
  BackendKind kind{BackendKind::cpu};
  std::string name;
  std::string device;
  bool native{false};
};

}  // namespace cm2

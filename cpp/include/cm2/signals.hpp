#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "cm2/types.hpp"

namespace cm2 {

enum class GridBoundaryKind : std::uint8_t {
  no_flux,
  periodic,
  fixed,
};

struct GridBoundary {
  GridBoundaryKind kind{GridBoundaryKind::no_flux};
  std::vector<float> values;

  void validate(std::size_t signal_count) const;
};

struct GridShape {
  std::uint32_t x{1};
  std::uint32_t y{1};
  std::uint32_t z{1};
};

struct SignalGridSpec {
  std::uint32_t signal_count{0};
  GridShape shape;
  Vec3 origin;
  Vec3 spacing{1.0F, 1.0F, 1.0F};
  std::vector<float> diffusion;
  std::vector<Vec3> advection;
  GridBoundary x_lower;
  GridBoundary x_upper;
  GridBoundary y_lower;
  GridBoundary y_upper;
  GridBoundary z_lower;
  GridBoundary z_upper;

  [[nodiscard]] std::size_t site_count() const;
  [[nodiscard]] std::size_t level_count() const;
  [[nodiscard]] float voxel_volume() const noexcept;
  void validate() const;
};

struct SignalGridCheckpoint {
  SignalGridSpec spec;
  std::vector<float> levels;

  void validate() const;
};

class SignalGrid {
 public:
  explicit SignalGrid(const SignalGridSpec& spec, std::vector<float> levels = {});
  explicit SignalGrid(const SignalGridCheckpoint& checkpoint);

  [[nodiscard]] const SignalGridSpec& spec() const noexcept;
  [[nodiscard]] std::span<const float> levels() const& noexcept;
  [[nodiscard]] std::span<const float> levels() && = delete;
  [[nodiscard]] std::vector<float> sample(Vec3 position) const;
  [[nodiscard]] SignalGridCheckpoint checkpoint() const;
  void set_levels(std::span<const float> levels);
  void replace_levels(std::vector<float> levels);
  void validate_step(float dt) const;
  void validate() const;

 private:
  SignalGridSpec spec_;
  std::vector<float> levels_;
};

void advance_signal_grid_cpu(SignalGrid& grid, float dt);

}  // namespace cm2

#include "cm2/signals.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cm2 {
namespace {

std::size_t checked_multiply(std::size_t left, std::size_t right, const char* name) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(std::string("signal grid ") + name + " exceeds address space");
  }
  return left * right;
}

bool finite(Vec3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void validate_periodic_pair(const GridBoundary& lower, const GridBoundary& upper,
                            const char* axis) {
  const auto lower_periodic = lower.kind == GridBoundaryKind::periodic;
  const auto upper_periodic = upper.kind == GridBoundaryKind::periodic;
  if (lower_periodic != upper_periodic) {
    throw std::invalid_argument(std::string("signal grid periodic ") + axis +
                                " boundaries must be paired");
  }
}

struct AxisWeights {
  std::array<std::uint32_t, 2> indices{};
  std::array<float, 2> weights{1.0F, 0.0F};
  std::size_t count{1};
};

AxisWeights interpolation_axis(float position, float origin, float spacing, std::uint32_t dimension,
                               const char* axis) {
  if (!std::isfinite(position)) {
    throw std::invalid_argument("signal sample position must be finite");
  }
  if (dimension == 1) {
    return {};
  }

  const auto coordinate =
      (static_cast<double>(position) - static_cast<double>(origin)) / static_cast<double>(spacing);
  const auto upper_bound = static_cast<double>(dimension - 1);
  if (coordinate < 0.0 || coordinate > upper_bound) {
    throw std::out_of_range(std::string("signal sample is outside the ") + axis + " grid bound");
  }
  const auto lower = static_cast<std::uint32_t>(std::floor(coordinate));
  if (lower == dimension - 1) {
    return {.indices = {lower, lower}, .weights = {1.0F, 0.0F}, .count = 1};
  }
  const auto fraction = static_cast<float>(coordinate - static_cast<double>(lower));
  return {
      .indices = {lower, lower + 1},
      .weights = {1.0F - fraction, fraction},
      .count = 2,
  };
}

std::size_t flat_site(const GridShape& shape, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return (static_cast<std::size_t>(x) * shape.y * shape.z) +
         (static_cast<std::size_t>(y) * shape.z) + z;
}

float boundary_value(const GridBoundary& boundary, std::size_t signal, float current,
                     float periodic) {
  switch (boundary.kind) {
    case GridBoundaryKind::no_flux:
      return current;
    case GridBoundaryKind::periodic:
      return periodic;
    case GridBoundaryKind::fixed:
      return boundary.values[signal];
  }
  throw std::logic_error("unknown signal grid boundary kind");
}

}  // namespace

void GridBoundary::validate(std::size_t signal_count) const {
  switch (kind) {
    case GridBoundaryKind::no_flux:
    case GridBoundaryKind::periodic:
      if (!values.empty()) {
        throw std::invalid_argument("non-fixed signal grid boundaries cannot have values");
      }
      return;
    case GridBoundaryKind::fixed:
      if (values.size() != signal_count) {
        throw std::invalid_argument("fixed signal grid boundary values must match signal count");
      }
      for (const auto value : values) {
        if (!std::isfinite(value) || value < 0.0F) {
          throw std::invalid_argument(
              "fixed signal grid boundary values must be finite and non-negative");
        }
      }
      return;
  }
  throw std::invalid_argument("unknown signal grid boundary kind");
}

std::size_t SignalGridSpec::site_count() const {
  const auto xy = checked_multiply(shape.x, shape.y, "site count");
  return checked_multiply(xy, shape.z, "site count");
}

std::size_t SignalGridSpec::level_count() const {
  return checked_multiply(signal_count, site_count(), "level count");
}

float SignalGridSpec::voxel_volume() const noexcept { return spacing.x * spacing.y * spacing.z; }

void SignalGridSpec::validate() const {
  if (signal_count == 0) {
    throw std::invalid_argument("signal grid must contain at least one signal");
  }
  if (shape.x == 0 || shape.y == 0 || shape.z == 0) {
    throw std::invalid_argument("signal grid dimensions must be positive");
  }
  if (!finite(origin)) {
    throw std::invalid_argument("signal grid origin must be finite");
  }
  if (!finite(spacing) || spacing.x <= 0.0F || spacing.y <= 0.0F || spacing.z <= 0.0F) {
    throw std::invalid_argument("signal grid spacing must be finite and positive");
  }
  if (diffusion.size() != signal_count || advection.size() != signal_count) {
    throw std::invalid_argument("signal grid transport arrays must match signal count");
  }
  for (const auto value : diffusion) {
    if (!std::isfinite(value) || value < 0.0F) {
      throw std::invalid_argument("signal diffusion must be finite and non-negative");
    }
  }
  for (const auto velocity : advection) {
    if (!finite(velocity)) {
      throw std::invalid_argument("signal advection must be finite");
    }
  }
  for (const auto* boundary : {&x_lower, &x_upper, &y_lower, &y_upper, &z_lower, &z_upper}) {
    boundary->validate(signal_count);
  }
  validate_periodic_pair(x_lower, x_upper, "x");
  validate_periodic_pair(y_lower, y_upper, "y");
  validate_periodic_pair(z_lower, z_upper, "z");
  const auto levels = level_count();
  if (levels > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("signal grid level count exceeds the uint32 index space");
  }
}

void SignalGridCheckpoint::validate() const {
  spec.validate();
  if (levels.size() != spec.level_count()) {
    throw std::invalid_argument("signal grid level count does not match its specification");
  }
  for (const auto level : levels) {
    if (!std::isfinite(level) || level < 0.0F) {
      throw std::invalid_argument("signal grid levels must be finite and non-negative");
    }
  }
}

SignalGrid::SignalGrid(const SignalGridSpec& spec, std::vector<float> levels)
    : spec_(spec), levels_(std::move(levels)) {
  spec_.validate();
  if (levels_.empty()) {
    levels_.resize(spec_.level_count(), 0.0F);
  }
  validate();
}

SignalGrid::SignalGrid(const SignalGridCheckpoint& checkpoint)
    : spec_(checkpoint.spec), levels_(checkpoint.levels) {
  validate();
}

const SignalGridSpec& SignalGrid::spec() const noexcept { return spec_; }

std::span<const float> SignalGrid::levels() const& noexcept { return levels_; }

std::vector<float> SignalGrid::sample(Vec3 position) const {
  const auto x =
      interpolation_axis(position.x, spec_.origin.x, spec_.spacing.x, spec_.shape.x, "x");
  const auto y =
      interpolation_axis(position.y, spec_.origin.y, spec_.spacing.y, spec_.shape.y, "y");
  const auto z =
      interpolation_axis(position.z, spec_.origin.z, spec_.spacing.z, spec_.shape.z, "z");
  const auto sites = spec_.site_count();
  std::vector<float> result(spec_.signal_count, 0.0F);
  for (std::size_t xi = 0; xi < x.count; ++xi) {
    for (std::size_t yi = 0; yi < y.count; ++yi) {
      for (std::size_t zi = 0; zi < z.count; ++zi) {
        const auto weight = x.weights[xi] * y.weights[yi] * z.weights[zi];
        const auto site = flat_site(spec_.shape, x.indices[xi], y.indices[yi], z.indices[zi]);
        for (std::size_t signal = 0; signal < spec_.signal_count; ++signal) {
          result[signal] += weight * levels_[(signal * sites) + site];
        }
      }
    }
  }
  return result;
}

SignalGridCheckpoint SignalGrid::checkpoint() const {
  validate();
  return {.spec = spec_, .levels = levels_};
}

void SignalGrid::set_levels(std::span<const float> levels) {
  replace_levels(std::vector<float>(levels.begin(), levels.end()));
}

void SignalGrid::replace_levels(std::vector<float> levels) {
  SignalGridCheckpoint candidate{.spec = spec_, .levels = levels};
  candidate.validate();
  levels_ = std::move(levels);
}

void SignalGrid::validate_step(float dt) const {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  const std::array<std::uint32_t, 3> dimensions{spec_.shape.x, spec_.shape.y, spec_.shape.z};
  const std::array<float, 3> spacing{spec_.spacing.x, spec_.spacing.y, spec_.spacing.z};
  for (std::size_t signal = 0; signal < spec_.signal_count; ++signal) {
    const std::array<float, 3> velocity{spec_.advection[signal].x, spec_.advection[signal].y,
                                        spec_.advection[signal].z};
    double inverse_square_sum = 0.0;
    double courant_sum = 0.0;
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
      if (dimensions[axis] == 1) {
        continue;
      }
      const auto inverse_spacing = 1.0 / static_cast<double>(spacing[axis]);
      inverse_square_sum += inverse_spacing * inverse_spacing;
      courant_sum += std::abs(static_cast<double>(velocity[axis])) * inverse_spacing;
    }
    const auto factor =
        static_cast<double>(dt) *
        ((2.0 * static_cast<double>(spec_.diffusion[signal]) * inverse_square_sum) + courant_sum);
    if (!std::isfinite(factor) || factor > 1.0) {
      throw std::invalid_argument("signal grid time step violates the explicit stability bound");
    }
  }
}

void SignalGrid::validate() const {
  SignalGridCheckpoint{.spec = spec_, .levels = levels_}.validate();
}

void advance_signal_grid_cpu(SignalGrid& grid, float dt) {
  grid.validate();
  grid.validate_step(dt);
  if (dt == 0.0F) {
    return;
  }
  const auto& spec = grid.spec();
  const auto levels = grid.levels();
  const auto sites = spec.site_count();
  std::vector<float> updated(levels.begin(), levels.end());
  const std::array<const GridBoundary*, 3> lower_boundaries{&spec.x_lower, &spec.y_lower,
                                                            &spec.z_lower};
  const std::array<const GridBoundary*, 3> upper_boundaries{&spec.x_upper, &spec.y_upper,
                                                            &spec.z_upper};

  const auto level = [&](std::size_t signal, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return levels[(signal * sites) + flat_site(spec.shape, x, y, z)];
  };

  for (std::size_t signal = 0; signal < spec.signal_count; ++signal) {
    const auto diffusion = spec.diffusion[signal];
    const std::array<float, 3> velocity{spec.advection[signal].x, spec.advection[signal].y,
                                        spec.advection[signal].z};
    const std::array<float, 3> spacing{spec.spacing.x, spec.spacing.y, spec.spacing.z};
    for (std::uint32_t x = 0; x < spec.shape.x; ++x) {
      for (std::uint32_t y = 0; y < spec.shape.y; ++y) {
        for (std::uint32_t z = 0; z < spec.shape.z; ++z) {
          const auto current = level(signal, x, y, z);
          std::array<float, 3> lower{};
          std::array<float, 3> upper{};
          lower[0] = x == 0 ? boundary_value(spec.x_lower, signal, current,
                                             level(signal, spec.shape.x - 1, y, z))
                            : level(signal, x - 1, y, z);
          upper[0] = x + 1 == spec.shape.x
                         ? boundary_value(spec.x_upper, signal, current, level(signal, 0, y, z))
                         : level(signal, x + 1, y, z);
          lower[1] = y == 0 ? boundary_value(spec.y_lower, signal, current,
                                             level(signal, x, spec.shape.y - 1, z))
                            : level(signal, x, y - 1, z);
          upper[1] = y + 1 == spec.shape.y
                         ? boundary_value(spec.y_upper, signal, current, level(signal, x, 0, z))
                         : level(signal, x, y + 1, z);
          lower[2] = z == 0 ? boundary_value(spec.z_lower, signal, current,
                                             level(signal, x, y, spec.shape.z - 1))
                            : level(signal, x, y, z - 1);
          upper[2] = z + 1 == spec.shape.z
                         ? boundary_value(spec.z_upper, signal, current, level(signal, x, y, 0))
                         : level(signal, x, y, z + 1);

          float rate = 0.0F;
          const std::array<std::uint32_t, 3> dimensions{spec.shape.x, spec.shape.y, spec.shape.z};
          const std::array<bool, 3> at_lower{x == 0, y == 0, z == 0};
          const std::array<bool, 3> at_upper{x + 1 == spec.shape.x, y + 1 == spec.shape.y,
                                             z + 1 == spec.shape.z};
          for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
            if (dimensions[axis] == 1) {
              continue;
            }
            const auto inverse_spacing = 1.0F / spacing[axis];
            rate += diffusion * (lower[axis] - (2.0F * current) + upper[axis]) * inverse_spacing *
                    inverse_spacing;
            auto lower_flux =
                velocity[axis] >= 0.0F ? velocity[axis] * lower[axis] : velocity[axis] * current;
            auto upper_flux =
                velocity[axis] >= 0.0F ? velocity[axis] * current : velocity[axis] * upper[axis];
            if (at_lower[axis] && lower_boundaries[axis]->kind == GridBoundaryKind::no_flux) {
              lower_flux = 0.0F;
            }
            if (at_upper[axis] && upper_boundaries[axis]->kind == GridBoundaryKind::no_flux) {
              upper_flux = 0.0F;
            }
            rate -= (upper_flux - lower_flux) * inverse_spacing;
          }
          const auto candidate = current + (dt * rate);
          if (!std::isfinite(candidate) || candidate < 0.0F) {
            throw std::runtime_error(
                "signal grid update produced a non-finite or negative concentration");
          }
          updated[(signal * sites) + flat_site(spec.shape, x, y, z)] = candidate;
        }
      }
    }
  }
  grid.replace_levels(std::move(updated));
}

}  // namespace cm2

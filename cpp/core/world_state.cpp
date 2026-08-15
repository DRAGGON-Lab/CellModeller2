#include "cm2/world_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cm2 {
namespace {

void validate_cell(const CellInit& cell) {
  const std::array values{
      cell.position.x,  cell.position.y, cell.position.z, cell.direction.x, cell.direction.y,
      cell.direction.z, cell.length,     cell.radius,     cell.growth_rate,
  };
  if (!std::ranges::all_of(values, [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("cell fields must be finite");
  }
  if (cell.length < 0.0F) {
    throw std::invalid_argument("cell length must be non-negative");
  }
  if (cell.radius <= 0.0F) {
    throw std::invalid_argument("cell radius must be positive");
  }
  static_cast<void>(normalized(cell.direction));
}

}  // namespace

WorldState::WorldState(std::size_t reserved_capacity) {
  ids_.reserve(reserved_capacity);
  position_x_.reserve(reserved_capacity);
  position_y_.reserve(reserved_capacity);
  position_z_.reserve(reserved_capacity);
  direction_x_.reserve(reserved_capacity);
  direction_y_.reserve(reserved_capacity);
  direction_z_.reserve(reserved_capacity);
  length_.reserve(reserved_capacity);
  radius_.reserve(reserved_capacity);
  growth_rate_.reserve(reserved_capacity);
  cell_type_.reserve(reserved_capacity);
  id_to_slot_.reserve(reserved_capacity);
  lineage_.reserve(reserved_capacity);
}

std::size_t WorldState::size() const noexcept { return ids_.size(); }

bool WorldState::empty() const noexcept { return ids_.empty(); }

bool WorldState::contains(CellId id) const noexcept { return id_to_slot_.contains(id); }

CellId WorldState::allocate_id() {
  if (next_id_ == invalid_cell_id || next_id_ == std::numeric_limits<CellId>::max()) {
    throw std::overflow_error("cell identifier space exhausted");
  }
  return next_id_++;
}

Slot WorldState::slot_for(CellId id) const {
  const auto found = id_to_slot_.find(id);
  if (found == id_to_slot_.end()) {
    throw std::out_of_range("unknown cell id " + std::to_string(id));
  }
  return found->second;
}

void WorldState::append(CellId id, const CellInit& cell) {
  validate_cell(cell);
  if (ids_.size() >= static_cast<std::size_t>(invalid_slot)) {
    throw std::overflow_error("cell slot space exhausted");
  }
  const auto direction = normalized(cell.direction);
  const auto slot = static_cast<Slot>(ids_.size());
  ids_.push_back(id);
  position_x_.push_back(cell.position.x);
  position_y_.push_back(cell.position.y);
  position_z_.push_back(cell.position.z);
  direction_x_.push_back(direction.x);
  direction_y_.push_back(direction.y);
  direction_z_.push_back(direction.z);
  length_.push_back(cell.length);
  radius_.push_back(cell.radius);
  growth_rate_.push_back(cell.growth_rate);
  cell_type_.push_back(cell.cell_type);
  id_to_slot_.emplace(id, slot);
}

void WorldState::replace(Slot slot, CellId id, const CellInit& cell) {
  validate_cell(cell);
  const auto index = static_cast<std::size_t>(slot);
  if (index >= size()) {
    throw std::out_of_range("cell slot is out of range");
  }
  const auto direction = normalized(cell.direction);
  ids_[index] = id;
  position_x_[index] = cell.position.x;
  position_y_[index] = cell.position.y;
  position_z_[index] = cell.position.z;
  direction_x_[index] = direction.x;
  direction_y_[index] = direction.y;
  direction_z_[index] = direction.z;
  length_[index] = cell.length;
  radius_[index] = cell.radius;
  growth_rate_[index] = cell.growth_rate;
  cell_type_[index] = cell.cell_type;
  id_to_slot_[id] = slot;
}

CellId WorldState::add_cell(const CellInit& cell) {
  const auto id = allocate_id();
  append(id, cell);
  return id;
}

std::pair<CellId, CellId> WorldState::divide_equal(CellId parent_id) {
  const auto parent = cell(parent_id);
  const auto daughter_length = (parent.length * 0.5F) - parent.radius;
  if (!(daughter_length >= 0.0F)) {
    throw std::domain_error("parent is too short to divide into valid daughters");
  }

  const auto offset = parent.direction * ((daughter_length * 0.5F) + parent.radius);
  CellInit daughter{
      .position = parent.position - offset,
      .direction = parent.direction,
      .length = daughter_length,
      .radius = parent.radius,
      .growth_rate = parent.growth_rate,
      .cell_type = parent.cell_type,
  };

  const auto first_id = allocate_id();
  const auto second_id = allocate_id();
  const auto parent_slot = parent.slot;

  id_to_slot_.erase(parent_id);
  replace(parent_slot, first_id, daughter);
  daughter.position = parent.position + offset;
  append(second_id, daughter);
  lineage_[first_id] = parent_id;
  lineage_[second_id] = parent_id;
  return {first_id, second_id};
}

void WorldState::advance_growth(float dt) {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  for (std::size_t index = 0; index < size(); ++index) {
    length_[index] += growth_rate_[index] * length_[index] * dt;
  }
}

void WorldState::set_cell_geometry(Slot slot, Vec3 position, Vec3 direction, float length) {
  const auto index = static_cast<std::size_t>(slot);
  if (index >= size()) {
    throw std::out_of_range("cell geometry slot is out of range");
  }
  const CellInit candidate{
      .position = position,
      .direction = direction,
      .length = length,
      .radius = radius_[index],
      .growth_rate = growth_rate_[index],
      .cell_type = cell_type_[index],
  };
  validate_cell(candidate);
  const auto unit_direction = normalized(direction);
  position_x_[index] = position.x;
  position_y_[index] = position.y;
  position_z_[index] = position.z;
  direction_x_[index] = unit_direction.x;
  direction_y_[index] = unit_direction.y;
  direction_z_[index] = unit_direction.z;
  length_[index] = length;
}

GrowthStateView WorldState::growth_state() noexcept {
  return {
      .lengths = length_,
      .growth_rates = growth_rate_,
  };
}

CellGeometryView WorldState::geometry_state() const noexcept {
  return {
      .ids = ids_,
      .position_x = position_x_,
      .position_y = position_y_,
      .position_z = position_z_,
      .direction_x = direction_x_,
      .direction_y = direction_y_,
      .direction_z = direction_z_,
      .lengths = length_,
      .radii = radius_,
  };
}

CellSnapshot WorldState::cell(CellId id) const {
  const auto slot = slot_for(id);
  const auto index = static_cast<std::size_t>(slot);
  return {
      .id = ids_[index],
      .slot = slot,
      .position = {position_x_[index], position_y_[index], position_z_[index]},
      .direction = {direction_x_[index], direction_y_[index], direction_z_[index]},
      .length = length_[index],
      .radius = radius_[index],
      .growth_rate = growth_rate_[index],
      .cell_type = cell_type_[index],
  };
}

std::vector<CellSnapshot> WorldState::cells() const {
  std::vector<CellSnapshot> result;
  result.reserve(size());
  for (const auto id : ids_) {
    result.push_back(cell(id));
  }
  return result;
}

std::optional<CellId> WorldState::lineage_parent(CellId id) const noexcept {
  const auto found = lineage_.find(id);
  if (found == lineage_.end()) {
    return std::nullopt;
  }
  return found->second;
}

void WorldState::validate() const {
  const auto expected = ids_.size();
  const std::array sizes{
      position_x_.size(),  position_y_.size(),  position_z_.size(), direction_x_.size(),
      direction_y_.size(), direction_z_.size(), length_.size(),     radius_.size(),
      growth_rate_.size(), cell_type_.size(),
  };
  if (!std::ranges::all_of(sizes, [expected](std::size_t size) { return size == expected; })) {
    throw std::logic_error("world state arrays have inconsistent lengths");
  }
  if (id_to_slot_.size() != expected) {
    throw std::logic_error("cell id index has the wrong size");
  }
  for (std::size_t index = 0; index < expected; ++index) {
    const auto id = ids_[index];
    if (id == invalid_cell_id) {
      throw std::logic_error("active cell has an invalid identifier");
    }
    const auto found = id_to_slot_.find(id);
    if (found == id_to_slot_.end() || found->second != static_cast<Slot>(index)) {
      throw std::logic_error("cell id and slot index disagree");
    }
    const CellInit value{
        .position = {position_x_[index], position_y_[index], position_z_[index]},
        .direction = {direction_x_[index], direction_y_[index], direction_z_[index]},
        .length = length_[index],
        .radius = radius_[index],
        .growth_rate = growth_rate_[index],
        .cell_type = cell_type_[index],
    };
    validate_cell(value);
    if (std::abs(norm(value.direction) - 1.0F) > 1.0e-5F) {
      throw std::logic_error("cell direction is not normalized");
    }
  }
}

}  // namespace cm2

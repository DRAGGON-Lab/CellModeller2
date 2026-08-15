#include <cassert>
#include <cmath>
#include <stdexcept>

#include "cm2/simulation.hpp"

namespace {

bool close(float left, float right, float tolerance = 1.0e-6F) {
  return std::abs(left - right) <= tolerance;
}

void test_growth_uses_stable_ids() {
  cm2::Simulation simulation;
  cm2::CellInit initial;
  initial.length = 4.0F;
  initial.radius = 0.5F;
  initial.growth_rate = 0.25F;

  const auto id = simulation.add_cell(initial);
  simulation.step(0.5F);

  assert(id == 1);
  assert(simulation.cell_count() == 1);
  assert(close(simulation.cell(id).length, 4.5F));
  assert(std::abs(simulation.time() - 0.5) <= 1.0e-12);
  simulation.validate();
}

void test_division_reuses_slot_but_not_identity() {
  cm2::Simulation simulation;
  cm2::CellInit initial;
  initial.position = {2.0F, 3.0F, 0.0F};
  initial.direction = {2.0F, 0.0F, 0.0F};
  initial.length = 4.0F;
  initial.radius = 0.5F;
  initial.growth_rate = 0.75F;
  initial.cell_type = 7;

  const auto parent = simulation.add_cell(initial);
  const auto [first, second] = simulation.divide_equal(parent);
  const auto first_cell = simulation.cell(first);
  const auto second_cell = simulation.cell(second);

  assert(simulation.cell_count() == 2);
  assert(first == 2);
  assert(second == 3);
  assert(first_cell.slot == 0);
  assert(second_cell.slot == 1);
  assert(close(first_cell.length, 1.5F));
  assert(close(second_cell.length, 1.5F));
  assert(close(first_cell.position.x, 0.75F));
  assert(close(second_cell.position.x, 3.25F));
  assert(first_cell.cell_type == 7);
  assert(close(first_cell.growth_rate, 0.75F));
  assert(simulation.lineage_parent(first) == parent);
  assert(simulation.lineage_parent(second) == parent);

  bool parent_is_gone = false;
  try {
    static_cast<void>(simulation.cell(parent));
  } catch (const std::out_of_range&) {
    parent_is_gone = true;
  }
  assert(parent_is_gone);
  simulation.validate();
}

void test_invalid_state_fails_explicitly() {
  cm2::Simulation simulation;
  cm2::CellInit invalid;
  invalid.radius = 0.0F;

  bool rejected = false;
  try {
    static_cast<void>(simulation.add_cell(invalid));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    simulation.step(-0.1F);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void test_unavailable_backends_do_not_fall_back() {
  bool rejected = false;
  try {
    cm2::Simulation simulation(cm2::BackendKind::metal);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);

  rejected = false;
  try {
    cm2::Simulation simulation(cm2::BackendKind::cuda);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}

}  // namespace

int main() {
  test_growth_uses_stable_ids();
  test_division_reuses_slot_but_not_identity();
  test_invalid_state_fails_explicitly();
  test_unavailable_backends_do_not_fall_back();
  return 0;
}

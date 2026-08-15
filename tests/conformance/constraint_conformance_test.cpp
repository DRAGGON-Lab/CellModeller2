#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "backend_devices.hpp"
#include "cm2/simulation.hpp"

namespace {

constexpr float absolute_tolerance = 2.0e-5F;
constexpr float relative_tolerance = 2.0e-5F;

bool close(float actual, float expected) {
  const auto tolerance = absolute_tolerance + relative_tolerance * std::abs(expected);
  return std::abs(actual - expected) <= tolerance;
}

void add_cell(cm2::Simulation& simulation, cm2::Vec3 position, cm2::Vec3 direction, float length,
              float radius = 0.5F) {
  cm2::CellInit cell;
  cell.position = position;
  cell.direction = direction;
  cell.length = length;
  cell.radius = radius;
  simulation.add_cell(cell);
}

void populate_mixed_constraints(cm2::Simulation& simulation) {
  add_cell(simulation, {0.0F, 0.4F, 0.0F}, {1.0F, 0.0F, 0.0F}, 2.0F);
  add_cell(simulation, {1.2F, 0.6F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  add_cell(simulation, {4.8F, 2.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.0F);

  cm2::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 2.0F, 0.0F};
  plane.coefficient = 1.25F;
  assert(simulation.add_plane_constraint(plane) == 1);

  cm2::SphereConstraintInit outside;
  outside.radius = 1.0F;
  outside.coefficient = 0.75F;
  assert(simulation.add_sphere_constraint(outside) == 2);

  cm2::SphereConstraintInit inside;
  inside.radius = 5.0F;
  inside.coefficient = 1.5F;
  inside.allowed_region = cm2::SphereRegion::inside;
  assert(simulation.add_sphere_constraint(inside) == 3);

  cm2::PlaneConstraintInit second_plane;
  second_plane.point = {6.0F, 0.0F, 0.0F};
  second_plane.inward_normal = {-1.0F, 0.0F, 0.0F};
  assert(simulation.add_plane_constraint(second_plane) == 4);
}

void populate_degenerate_sphere(cm2::Simulation& simulation) {
  add_cell(simulation, {}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm2::SphereConstraintInit sphere;
  sphere.radius = 2.0F;
  simulation.add_sphere_constraint(sphere);
}

void compare_graphs(const cm2::ExternalContactGraph& actual,
                    const cm2::ExternalContactGraph& expected) {
  assert(actual.cell_count() == expected.cell_count());
  assert(actual.size() == expected.size());
  const auto actual_contacts = actual.contacts();
  const auto expected_contacts = expected.contacts();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto& left = actual_contacts[index];
    const auto& right = expected_contacts[index];
    assert(left.cell_id == right.cell_id);
    assert(left.cell_slot == right.cell_slot);
    assert(left.constraint_id == right.constraint_id);
    assert(left.constraint_kind == right.constraint_kind);
    assert(left.endpoint == right.endpoint);
    assert(close(left.point_on_cell.x, right.point_on_cell.x));
    assert(close(left.point_on_cell.y, right.point_on_cell.y));
    assert(close(left.point_on_cell.z, right.point_on_cell.z));
    assert(close(left.normal.x, right.normal.x));
    assert(close(left.normal.y, right.normal.y));
    assert(close(left.normal.z, right.normal.z));
    assert(close(left.signed_separation, right.signed_separation));
    assert(close(left.weight, right.weight));
  }
  for (std::size_t slot = 0; slot < expected.cell_count(); ++slot) {
    const auto actual_incidence = actual.incident_contact_indices(static_cast<cm2::Slot>(slot));
    const auto expected_incidence = expected.incident_contact_indices(static_cast<cm2::Slot>(slot));
    assert(actual_incidence.size() == expected_incidence.size());
    for (std::size_t index = 0; index < expected_incidence.size(); ++index) {
      assert(actual_incidence[index] == expected_incidence[index]);
    }
  }
}

void run_fixture(cm2::BackendKind backend, std::uint32_t device_index,
                 void (*populate)(cm2::Simulation&)) {
  cm2::Simulation reference(cm2::BackendKind::cpu);
  cm2::Simulation candidate(backend, 0, 0, device_index);
  populate(reference);
  populate(candidate);
  compare_graphs(candidate.find_external_contacts(), reference.find_external_contacts());
}

void run_empty_inputs(cm2::BackendKind backend, std::uint32_t device_index) {
  cm2::Simulation empty(backend, 0, 0, device_index);
  const auto no_cells = empty.find_external_contacts();
  assert(no_cells.empty());
  assert(no_cells.cell_count() == 0);

  cm2::Simulation no_constraints(backend, 0, 0, device_index);
  add_cell(no_constraints, {}, {1.0F, 0.0F, 0.0F}, 1.0F);
  const auto graph = no_constraints.find_external_contacts();
  assert(graph.empty());
  assert(graph.cell_count() == 1);
}

}  // namespace

int main() {
  cm2::test::for_each_backend_device([](cm2::BackendKind backend, std::uint32_t device_index) {
    cm2::Simulation capability_probe(backend, 0, 0, device_index);
    if (!capability_probe.supports(cm2::BackendFeature::external_constraints)) {
      return;
    }
    run_empty_inputs(backend, device_index);
    run_fixture(backend, device_index, populate_mixed_constraints);
    run_fixture(backend, device_index, populate_degenerate_sphere);
  });
  return 0;
}

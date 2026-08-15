#include <cassert>
#include <cmath>
#include <stdexcept>

#include "cm/constraints.hpp"
#include "cm/simulation.hpp"

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

cm::CellId add_capsule(cm::WorldState& state, cm::Vec3 center, cm::Vec3 axis,
                        float length = 2.0F, float radius = 0.5F) {
  cm::CellInit cell;
  cell.position = center;
  cell.direction = axis;
  cell.length = length;
  cell.radius = radius;
  return state.add_cell(cell);
}

void test_constraint_ids_and_validation() {
  cm::ConstraintSet constraints;
  cm::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 2.0F, 0.0F};
  const auto plane_id = constraints.add_plane(plane);
  cm::SphereConstraintInit sphere;
  const auto sphere_id = constraints.add_sphere(sphere);
  assert(plane_id == 1);
  assert(sphere_id == 2);
  assert(constraints.size() == 2);
  assert(close(constraints.planes()[0].inward_normal.y, 1.0F));

  plane.coefficient = 0.0F;
  bool rejected = false;
  try {
    static_cast<void>(constraints.add_plane(plane));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  plane.coefficient = 1.0F;
  plane.inward_normal = {};
  rejected = false;
  try {
    static_cast<void>(constraints.add_plane(plane));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void test_parallel_plane_contact_uses_two_weighted_endpoints() {
  cm::WorldState state;
  add_capsule(state, {0.0F, 0.4F, 0.0F}, {1.0F, 0.0F, 0.0F});
  cm::ConstraintSet constraints;
  cm::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  plane.coefficient = 2.0F;
  const auto plane_id = constraints.add_plane(plane);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  assert(graph.incident_contact_indices(0).size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(contact.constraint_id == plane_id);
    assert(contact.constraint_kind == cm::ExternalConstraintKind::plane);
    assert(close(contact.signed_separation, -0.1F));
    assert(close(contact.normal.y, -1.0F));
    assert(close(contact.point_on_cell.y, -0.1F));
    assert(close(contact.weight, std::sqrt(2.0F)));
  }
}

void test_perpendicular_plane_emits_one_endpoint() {
  cm::WorldState state;
  add_capsule(state, {0.0F, 1.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  cm::ConstraintSet constraints;
  cm::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  constraints.add_plane(plane);

  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 1);
  assert(graph.contacts()[0].endpoint == cm::RodEndpoint::negative);
  assert(close(graph.contacts()[0].signed_separation, -0.5F));
  assert(close(graph.contacts()[0].weight, 1.0F));
}

void test_outside_and_inside_spheres_have_typed_orientation() {
  cm::WorldState outside_state;
  add_capsule(outside_state, {1.2F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet outside_constraints;
  cm::SphereConstraintInit outside;
  outside.radius = 1.0F;
  const auto outside_id = outside_constraints.add_sphere(outside);
  const auto outside_graph = cm::find_external_contacts_cpu(outside_state, outside_constraints);
  assert(outside_graph.size() == 2);
  for (const auto& contact : outside_graph.contacts()) {
    assert(contact.constraint_id == outside_id);
    assert(contact.constraint_kind == cm::ExternalConstraintKind::sphere);
    assert(close(contact.signed_separation, -0.3F));
    assert(close(contact.normal.x, -1.0F));
    assert(close(contact.point_on_cell.x, 0.7F));
    assert(close(contact.weight, std::sqrt(0.5F)));
  }

  cm::WorldState inside_state;
  add_capsule(inside_state, {4.8F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet inside_constraints;
  cm::SphereConstraintInit inside;
  inside.radius = 5.0F;
  inside.allowed_region = cm::SphereRegion::inside;
  inside_constraints.add_sphere(inside);
  const auto inside_graph = cm::find_external_contacts_cpu(inside_state, inside_constraints);
  assert(inside_graph.size() == 2);
  for (const auto& contact : inside_graph.contacts()) {
    assert(close(contact.signed_separation, -0.3F));
    assert(close(contact.normal.x, 1.0F));
    assert(close(contact.point_on_cell.x, 5.3F));
  }
}

void test_degenerate_sphere_normal_is_finite_and_deterministic() {
  cm::WorldState state;
  add_capsule(state, {}, {0.0F, 1.0F, 0.0F}, 0.0F);
  cm::ConstraintSet constraints;
  cm::SphereConstraintInit sphere;
  sphere.radius = 2.0F;
  constraints.add_sphere(sphere);
  const auto graph = cm::find_external_contacts_cpu(state, constraints);
  assert(graph.size() == 2);
  for (const auto& contact : graph.contacts()) {
    assert(close(cm::norm(contact.normal), 1.0F));
    assert(close(contact.normal.x, -1.0F));
  }
}

void test_simulation_exposes_cpu_constraint_graph() {
  cm::Simulation simulation;
  cm::CellInit cell;
  cell.position.y = 0.4F;
  cell.length = 2.0F;
  simulation.add_cell(cell);
  cm::PlaneConstraintInit plane;
  plane.inward_normal = {0.0F, 1.0F, 0.0F};
  simulation.add_plane_constraint(plane);
  assert(simulation.supports(cm::BackendFeature::external_constraints));
  const auto graph = simulation.find_external_contacts();
  assert(graph.size() == 2);
}

}  // namespace

int main() {
  test_constraint_ids_and_validation();
  test_parallel_plane_contact_uses_two_weighted_endpoints();
  test_perpendicular_plane_emits_one_endpoint();
  test_outside_and_inside_spheres_have_typed_orientation();
  test_degenerate_sphere_normal_is_finite_and_deterministic();
  test_simulation_exposes_cpu_constraint_graph();
  return 0;
}

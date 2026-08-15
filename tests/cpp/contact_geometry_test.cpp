#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>

#include "cm2/contact_graph.hpp"

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5F) {
  return std::abs(actual - expected) <= tolerance;
}

cm2::CellId add_capsule(cm2::WorldState& state, cm2::Vec3 center, cm2::Vec3 axis,
                        float length = 2.0F, float radius = 0.5F) {
  cm2::CellInit cell;
  cell.position = center;
  cell.direction = axis;
  cell.length = length;
  cell.radius = radius;
  return state.add_cell(cell);
}

void test_separated_capsules_have_no_contact() {
  cm2::WorldState state;
  add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(state, {4.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});

  const auto graph = cm2::find_cell_contacts_cpu(state);
  assert(graph.cell_count() == 2);
  assert(graph.empty());
  assert(graph.incident_contact_indices(0).empty());
  assert(graph.incident_contact_indices(1).empty());
}

void test_end_on_contact_has_signed_separation() {
  cm2::WorldState state;
  add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(state, {2.9F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});

  const auto graph = cm2::find_cell_contacts_cpu(state);
  assert(graph.size() == 1);
  const auto& contact = graph.contacts().front();
  assert(contact.first_id == 1);
  assert(contact.second_id == 2);
  assert(contact.ordinal == 0);
  assert(close(contact.signed_separation, -0.1F));
  assert(close(contact.normal.x, 1.0F));
  assert(close(contact.point_on_first.x, 1.5F));
  assert(close(contact.weight, 1.0F));
}

void test_parallel_overlap_uses_two_weighted_contacts() {
  cm2::WorldState state;
  add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(state, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);

  const auto graph = cm2::find_cell_contacts_cpu(state);
  assert(graph.size() == 2);
  const auto contacts = graph.contacts();
  assert(contacts[0].ordinal == 0);
  assert(contacts[1].ordinal == 1);
  assert(close(contacts[0].point_on_first.x, -2.0F));
  assert(close(contacts[1].point_on_first.x, 2.0F));
  for (const auto& contact : contacts) {
    assert(close(contact.signed_separation, -0.2F));
    assert(close(contact.normal.y, 1.0F));
    assert(close(contact.weight, std::sqrt(0.5F)));
  }
  assert(graph.incident_contact_indices(0).size() == 2);
  assert(graph.incident_contact_indices(1).size() == 2);
}

void test_skew_and_coincident_contacts_have_finite_normals() {
  cm2::WorldState skew;
  add_capsule(skew, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(skew, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  const auto skew_graph = cm2::find_cell_contacts_cpu(skew);
  const auto skew_contact = skew_graph.contacts().front();
  assert(close(skew_contact.normal.z, 1.0F));
  assert(close(cm2::norm(skew_contact.normal), 1.0F));

  cm2::WorldState coincident;
  add_capsule(coincident, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(coincident, {0.0F, 0.0F, 0.0F}, {-1.0F, 0.0F, 0.0F}, 4.0F);
  const auto graph = cm2::find_cell_contacts_cpu(coincident);
  const auto contacts = graph.contacts();
  assert(contacts.size() == 2);
  for (const auto& contact : contacts) {
    assert(std::isfinite(contact.normal.x));
    assert(std::isfinite(contact.normal.y));
    assert(std::isfinite(contact.normal.z));
    assert(close(cm2::norm(contact.normal), 1.0F));
    assert(close(contact.signed_separation, -1.0F));
  }
}

void test_contact_graph_has_no_per_cell_limit() {
  cm2::WorldState state;
  constexpr std::size_t capsule_count = 31;
  for (std::size_t index = 0; index < capsule_count; ++index) {
    add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  }

  const auto graph = cm2::find_cell_contacts_cpu(state);
  const auto pair_count = capsule_count * (capsule_count - 1) / 2;
  assert(graph.size() == pair_count * 2);
  assert(graph.incident_contact_indices(0).size() == (capsule_count - 1) * 2);
  assert(graph.incident_contact_indices(0).size() > 24);
}

void test_contacts_are_sorted_by_stable_identity() {
  cm2::WorldState state;
  const auto old = add_capsule(state, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(state, {0.0F, 0.2F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  static_cast<void>(state.divide_equal(old));

  const auto graph = cm2::find_cell_contacts_cpu(state);
  const auto contacts = graph.contacts();
  auto previous = std::tuple{cm2::invalid_cell_id, cm2::invalid_cell_id, std::uint8_t{0}};
  for (const auto& contact : contacts) {
    const auto key = std::tuple{contact.first_id, contact.second_id, contact.ordinal};
    assert(contact.first_id < contact.second_id);
    assert(previous < key);
    previous = key;
  }
}

void test_pair_order_reverses_the_contact_normal() {
  cm2::WorldState forward;
  add_capsule(forward, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(forward, {0.0F, 0.8F, 0.0F}, {-1.0F, 0.0F, 0.0F}, 4.0F);
  const auto forward_graph = cm2::find_cell_contacts_cpu(forward);

  cm2::WorldState reversed;
  add_capsule(reversed, {0.0F, 0.8F, 0.0F}, {-1.0F, 0.0F, 0.0F}, 4.0F);
  add_capsule(reversed, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, 4.0F);
  const auto reversed_graph = cm2::find_cell_contacts_cpu(reversed);

  assert(forward_graph.size() == reversed_graph.size());
  for (std::size_t index = 0; index < forward_graph.size(); ++index) {
    const auto& forward_contact = forward_graph.contacts()[index];
    const auto& reversed_contact = reversed_graph.contacts()[index];
    assert(close(forward_contact.signed_separation, reversed_contact.signed_separation));
    assert(close(forward_contact.weight, reversed_contact.weight));
    assert(close(forward_contact.normal.x, -reversed_contact.normal.x));
    assert(close(forward_contact.normal.y, -reversed_contact.normal.y));
    assert(close(forward_contact.normal.z, -reversed_contact.normal.z));
  }
}

void test_invalid_parameters_are_rejected() {
  cm2::WorldState state;
  cm2::ContactParameters parameters;
  parameters.parallel_sine_threshold = 2.0F;
  bool rejected = false;
  try {
    static_cast<void>(cm2::find_cell_contacts_cpu(state, parameters));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

}  // namespace

int main() {
  test_separated_capsules_have_no_contact();
  test_end_on_contact_has_signed_separation();
  test_parallel_overlap_uses_two_weighted_contacts();
  test_skew_and_coincident_contacts_have_finite_normals();
  test_contact_graph_has_no_per_cell_limit();
  test_contacts_are_sorted_by_stable_identity();
  test_pair_order_reverses_the_contact_normal();
  test_invalid_parameters_are_rejected();
  return 0;
}

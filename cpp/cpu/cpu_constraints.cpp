#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <vector>

#include "cm2/constraints.hpp"

namespace cm2 {
namespace {

constexpr float inverse_sqrt_two = 0.7071067811865475F;

struct EndpointGeometry {
  RodEndpoint endpoint;
  Vec3 centerline_point;
};

std::array<EndpointGeometry, 2> endpoints(const CellGeometryView& geometry, std::size_t slot) {
  const Vec3 center{geometry.position_x[slot], geometry.position_y[slot],
                    geometry.position_z[slot]};
  const Vec3 axis{geometry.direction_x[slot], geometry.direction_y[slot],
                  geometry.direction_z[slot]};
  const auto half_length = geometry.lengths[slot] * 0.5F;
  return {
      EndpointGeometry{RodEndpoint::negative, center - axis * half_length},
      EndpointGeometry{RodEndpoint::positive, center + axis * half_length},
  };
}

void append_plane_contacts(std::vector<ExternalContact>& contacts, const CellGeometryView& geometry,
                           std::size_t slot, const PlaneConstraint& plane,
                           const ConstraintContactParameters& parameters) {
  const auto cell_endpoints = endpoints(geometry, slot);
  std::array<float, 2> separations{};
  std::array<bool, 2> active{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    separations[index] =
        dot(cell_endpoints[index].centerline_point - plane.point, plane.inward_normal) -
        geometry.radii[slot];
    active[index] = separations[index] < parameters.activation_margin;
  }
  const auto active_count = static_cast<unsigned>(active[0]) + static_cast<unsigned>(active[1]);
  const auto weight = plane.coefficient * (active_count == 2 ? inverse_sqrt_two : 1.0F);
  const auto normal = plane.inward_normal * -1.0F;
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    contacts.push_back({
        .cell_id = geometry.ids[slot],
        .cell_slot = static_cast<Slot>(slot),
        .constraint_id = plane.id,
        .constraint_kind = ExternalConstraintKind::plane,
        .endpoint = cell_endpoints[index].endpoint,
        .point_on_cell = cell_endpoints[index].centerline_point + normal * geometry.radii[slot],
        .normal = normal,
        .signed_separation = separations[index],
        .weight = weight,
    });
  }
}

void append_sphere_contacts(std::vector<ExternalContact>& contacts,
                            const CellGeometryView& geometry, std::size_t slot,
                            const SphereConstraint& sphere,
                            const ConstraintContactParameters& parameters) {
  const auto cell_endpoints = endpoints(geometry, slot);
  std::array<float, 2> separations{};
  std::array<Vec3, 2> normals{};
  std::array<bool, 2> active{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    const auto center_delta = cell_endpoints[index].centerline_point - sphere.center;
    const auto distance = norm(center_delta);
    const auto radial = distance > parameters.degeneracy_epsilon ? center_delta * (1.0F / distance)
                                                                 : Vec3{1.0F, 0.0F, 0.0F};
    if (sphere.allowed_region == SphereRegion::outside) {
      separations[index] = distance - sphere.radius - geometry.radii[slot];
      normals[index] = radial * -1.0F;
    } else {
      separations[index] = sphere.radius - distance - geometry.radii[slot];
      normals[index] = radial;
    }
    active[index] = separations[index] < parameters.activation_margin;
  }
  const auto active_count = static_cast<unsigned>(active[0]) + static_cast<unsigned>(active[1]);
  const auto weight = sphere.coefficient * (active_count == 2 ? inverse_sqrt_two : 1.0F);
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    contacts.push_back({
        .cell_id = geometry.ids[slot],
        .cell_slot = static_cast<Slot>(slot),
        .constraint_id = sphere.id,
        .constraint_kind = ExternalConstraintKind::sphere,
        .endpoint = cell_endpoints[index].endpoint,
        .point_on_cell =
            cell_endpoints[index].centerline_point + normals[index] * geometry.radii[slot],
        .normal = normals[index],
        .signed_separation = separations[index],
        .weight = weight,
    });
  }
}

}  // namespace

ExternalContactGraph find_external_contacts_cpu(const WorldState& state,
                                                const ConstraintSet& constraints,
                                                const ConstraintContactParameters& parameters) {
  validate_constraint_contact_parameters(parameters);
  state.validate();
  const auto geometry = state.geometry_state();
  std::vector<ExternalContact> contacts;
  for (std::size_t slot = 0; slot < geometry.size(); ++slot) {
    for (const auto& plane : constraints.planes()) {
      append_plane_contacts(contacts, geometry, slot, plane, parameters);
    }
    for (const auto& sphere : constraints.spheres()) {
      append_sphere_contacts(contacts, geometry, slot, sphere, parameters);
    }
  }
  std::ranges::sort(contacts, {}, [](const ExternalContact& contact) {
    return std::tuple{contact.cell_id, contact.constraint_id, contact.endpoint};
  });
  return ExternalContactGraph(geometry.size(), std::move(contacts));
}

}  // namespace cm2

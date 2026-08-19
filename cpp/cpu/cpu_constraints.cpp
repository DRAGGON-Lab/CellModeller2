#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <vector>

#include "cm/constraints.hpp"

namespace cm {
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

struct SurfacePoint {
  float signed_distance;
  Vec3 outward;
};

// One contact pass for every constraint kind. The kind supplies the surface a
// point sees - its signed distance and outward direction - and the permitted
// region orients the contact: the normal always points from the permitted
// region toward the boundary, and negative separation means penetration.
template <typename Surface>
void append_surface_contacts(std::vector<ExternalContact>& contacts,
                             const CellGeometryView& geometry, std::size_t slot, ConstraintId id,
                             ExternalConstraintKind kind, float coefficient,
                             ConstraintRegion region,
                             const ConstraintContactParameters& parameters,
                             const Surface& surface_at) {
  const auto cell_endpoints = endpoints(geometry, slot);
  std::array<float, 2> separations{};
  std::array<Vec3, 2> normals{};
  std::array<bool, 2> active{};
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    const auto surface = surface_at(cell_endpoints[index].centerline_point);
    if (region == ConstraintRegion::outside) {
      separations[index] = surface.signed_distance - geometry.radii[slot];
      normals[index] = surface.outward * -1.0F;
    } else {
      separations[index] = -surface.signed_distance - geometry.radii[slot];
      normals[index] = surface.outward;
    }
    active[index] = separations[index] < parameters.activation_margin;
  }
  const auto active_count = static_cast<unsigned>(active[0]) + static_cast<unsigned>(active[1]);
  const auto weight = coefficient * (active_count == 2 ? inverse_sqrt_two : 1.0F);
  for (std::size_t index = 0; index < cell_endpoints.size(); ++index) {
    if (!active[index]) {
      continue;
    }
    contacts.push_back({
        .cell_id = geometry.ids[slot],
        .cell_slot = static_cast<Slot>(slot),
        .constraint_id = id,
        .constraint_kind = kind,
        .endpoint = cell_endpoints[index].endpoint,
        .point_on_cell =
            cell_endpoints[index].centerline_point + normals[index] * geometry.radii[slot],
        .normal = normals[index],
        .signed_separation = separations[index],
        .weight = weight,
    });
  }
}

// A plane's permitted region is the half-space its inward normal points into,
// which is the outside of the surface it defines.
SurfacePoint plane_surface(const Vec3& point, const PlaneConstraint& plane) {
  return {dot(point - plane.point, plane.inward_normal), plane.inward_normal};
}

SurfacePoint sphere_surface(const Vec3& point, const SphereConstraint& sphere,
                            float degeneracy_epsilon) {
  const auto center_delta = point - sphere.center;
  const auto distance = norm(center_delta);
  const auto radial =
      distance > degeneracy_epsilon ? center_delta * (1.0F / distance) : Vec3{1.0F, 0.0F, 0.0F};
  return {distance - sphere.radius, radial};
}

SurfacePoint box_surface(const Vec3& point, const BoxConstraint& box, float degeneracy_epsilon) {
  const auto delta = point - box.center;
  const Vec3 clamped{
      std::clamp(delta.x, -box.half_extents.x, box.half_extents.x),
      std::clamp(delta.y, -box.half_extents.y, box.half_extents.y),
      std::clamp(delta.z, -box.half_extents.z, box.half_extents.z),
  };
  const auto outside_vector = delta - clamped;
  const auto outside_distance = norm(outside_vector);
  if (outside_distance > degeneracy_epsilon) {
    return {outside_distance, outside_vector * (1.0F / outside_distance)};
  }
  const std::array<float, 3> clearances{
      box.half_extents.x - std::abs(delta.x),
      box.half_extents.y - std::abs(delta.y),
      box.half_extents.z - std::abs(delta.z),
  };
  std::size_t nearest_axis = 0;
  for (std::size_t axis = 1; axis < clearances.size(); ++axis) {
    if (clearances[axis] < clearances[nearest_axis]) {
      nearest_axis = axis;
    }
  }
  const std::array<float, 3> offsets{delta.x, delta.y, delta.z};
  const auto sign = offsets[nearest_axis] >= 0.0F ? 1.0F : -1.0F;
  Vec3 outward{};
  if (nearest_axis == 0) {
    outward = {sign, 0.0F, 0.0F};
  } else if (nearest_axis == 1) {
    outward = {0.0F, sign, 0.0F};
  } else {
    outward = {0.0F, 0.0F, sign};
  }
  return {-clearances[nearest_axis], outward};
}

SurfacePoint cylinder_surface(const Vec3& point, const CylinderConstraint& cylinder,
                                 float degeneracy_epsilon) {
  const Vec3 delta{point.x - cylinder.center.x, point.y - cylinder.center.y, 0.0F};
  const auto radial_distance = norm(delta);
  const auto radial = radial_distance > degeneracy_epsilon ? delta * (1.0F / radial_distance)
                                                           : Vec3{1.0F, 0.0F, 0.0F};
  const auto z_offset = point.z - cylinder.center.z;
  const auto z_sign = z_offset >= 0.0F ? 1.0F : -1.0F;
  const Vec3 axial{0.0F, 0.0F, z_sign};
  const auto radial_excess = radial_distance - cylinder.radius;
  const auto axial_excess = std::abs(z_offset) - cylinder.half_height;
  if (radial_excess > 0.0F && axial_excess > 0.0F) {
    const auto distance = std::sqrt((radial_excess * radial_excess) + (axial_excess * axial_excess));
    return {distance, (radial * radial_excess + axial * axial_excess) * (1.0F / distance)};
  }
  if (radial_excess > 0.0F) {
    return {radial_excess, radial};
  }
  if (axial_excess > 0.0F) {
    return {axial_excess, axial};
  }
  if (-radial_excess <= -axial_excess) {
    return {radial_excess, radial};
  }
  return {axial_excess, axial};
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
      append_surface_contacts(
          contacts, geometry, slot, plane.id, ExternalConstraintKind::plane, plane.coefficient,
          ConstraintRegion::outside, parameters,
          [&plane](const Vec3& point) { return plane_surface(point, plane); });
    }
    for (const auto& sphere : constraints.spheres()) {
      append_surface_contacts(
          contacts, geometry, slot, sphere.id, ExternalConstraintKind::sphere, sphere.coefficient,
          sphere.allowed_region, parameters, [&sphere, &parameters](const Vec3& point) {
            return sphere_surface(point, sphere, parameters.degeneracy_epsilon);
          });
    }
    for (const auto& box : constraints.boxes()) {
      append_surface_contacts(contacts, geometry, slot, box.id, ExternalConstraintKind::box,
                              box.coefficient, box.allowed_region, parameters,
                              [&box, &parameters](const Vec3& point) {
                                return box_surface(point, box, parameters.degeneracy_epsilon);
                              });
    }
    for (const auto& cylinder : constraints.cylinders()) {
      append_surface_contacts(
          contacts, geometry, slot, cylinder.id, ExternalConstraintKind::cylinder,
          cylinder.coefficient, cylinder.allowed_region, parameters,
          [&cylinder, &parameters](const Vec3& point) {
            return cylinder_surface(point, cylinder, parameters.degeneracy_epsilon);
          });
    }
  }
  std::ranges::sort(contacts, {}, [](const ExternalContact& contact) {
    return std::tuple{contact.cell_id, contact.constraint_id, contact.endpoint};
  });
  return ExternalContactGraph(geometry.size(), std::move(contacts));
}

}  // namespace cm

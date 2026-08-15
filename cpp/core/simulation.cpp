#include "cm2/simulation.hpp"

#include <cmath>
#include <stdexcept>

namespace cm2 {
namespace {

std::unique_ptr<ComputeBackend> make_backend(BackendKind kind) {
  switch (kind) {
    case BackendKind::cpu:
      return make_cpu_backend();
    case BackendKind::metal:
#if CM2_HAS_METAL
      return make_metal_backend();
#else
      throw std::runtime_error("Metal backend is not implemented in this build");
#endif
    case BackendKind::cuda:
#if CM2_HAS_CUDA
      return make_cuda_backend();
#else
      throw std::runtime_error("CUDA backend is not implemented in this build");
#endif
  }
  throw std::runtime_error("unknown compute backend");
}

}  // namespace

bool backend_available(BackendKind kind) noexcept {
  if (kind == BackendKind::cpu) {
    return true;
  }
#if CM2_HAS_METAL
  if (kind == BackendKind::metal) {
    return metal_backend_available();
  }
#endif
#if CM2_HAS_CUDA
  if (kind == BackendKind::cuda) {
    return cuda_backend_available();
  }
#endif
  return false;
}

Simulation::Simulation(BackendKind backend, std::size_t reserved_capacity)
    : state_(reserved_capacity), backend_(make_backend(backend)) {}

BackendInfo Simulation::backend_info() const { return backend_->info(); }

bool Simulation::supports(BackendFeature feature) const noexcept {
  return backend_->supports(feature);
}

double Simulation::time() const noexcept { return time_; }

std::size_t Simulation::cell_count() const noexcept { return state_.size(); }

CellId Simulation::add_cell(const CellInit& cell) { return state_.add_cell(cell); }

ConstraintId Simulation::add_plane_constraint(const PlaneConstraintInit& plane) {
  return constraints_.add_plane(plane);
}

ConstraintId Simulation::add_sphere_constraint(const SphereConstraintInit& sphere) {
  return constraints_.add_sphere(sphere);
}

std::pair<CellId, CellId> Simulation::divide_equal(CellId parent_id) {
  return state_.divide_equal(parent_id);
}

void Simulation::step(float dt) {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  backend_->advance_growth(state_, dt);
  time_ += static_cast<double>(dt);
}

ContactGraph Simulation::find_cell_contacts(const ContactParameters& parameters) {
  return backend_->find_cell_contacts(state_, parameters);
}

ExternalContactGraph Simulation::find_external_contacts(
    const ConstraintContactParameters& parameters) {
  if (!backend_->supports(BackendFeature::external_constraints)) {
    throw std::runtime_error("selected backend does not implement external constraints");
  }
  return backend_->find_external_contacts(state_, constraints_, parameters);
}

MechanicsSolveResult Simulation::solve_cell_mechanics(
    const MechanicsParameters& mechanics_parameters, const ContactParameters& contact_parameters,
    const ConstraintContactParameters& constraint_parameters) {
  if (!backend_->supports(BackendFeature::cell_mechanics)) {
    throw std::runtime_error("selected backend does not implement cell mechanics");
  }
  validate_constraint_contact_parameters(constraint_parameters);
  if (!constraints_.empty() && !backend_->supports(BackendFeature::external_constraints)) {
    throw std::runtime_error("selected backend does not implement external constraints");
  }
  const auto contacts = backend_->find_cell_contacts(state_, contact_parameters);
  ExternalContactGraph external_contacts(state_.size(), {});
  if (!constraints_.empty()) {
    external_contacts =
        backend_->find_external_contacts(state_, constraints_, constraint_parameters);
  }
  return backend_->solve_cell_mechanics(state_, contacts, external_contacts, mechanics_parameters);
}

MechanicsSolveResult Simulation::relax_cell_mechanics(
    const MechanicsParameters& mechanics_parameters, const ContactParameters& contact_parameters,
    const MechanicsIntegrationParameters& integration_parameters,
    const ConstraintContactParameters& constraint_parameters) {
  auto result =
      solve_cell_mechanics(mechanics_parameters, contact_parameters, constraint_parameters);
  integrate_mechanics_result(state_, result, integration_parameters);
  return result;
}

CellSnapshot Simulation::cell(CellId id) const { return state_.cell(id); }

std::vector<CellSnapshot> Simulation::cells() const { return state_.cells(); }

std::optional<CellId> Simulation::lineage_parent(CellId id) const noexcept {
  return state_.lineage_parent(id);
}

void Simulation::validate() const { state_.validate(); }

}  // namespace cm2

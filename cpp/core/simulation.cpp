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

Simulation::Simulation(BackendKind backend, std::size_t reserved_capacity,
                       std::size_t species_count)
    : state_(reserved_capacity, species_count),
      backend_(make_backend(backend)),
      species_rate_plan_(SpeciesRatePlan::zero(species_count)) {}

BackendInfo Simulation::backend_info() const { return backend_->info(); }

bool Simulation::supports(BackendFeature feature) const noexcept {
  return backend_->supports(feature);
}

double Simulation::time() const noexcept { return time_; }

std::size_t Simulation::cell_count() const noexcept { return state_.size(); }

std::size_t Simulation::species_count() const noexcept { return state_.species_count(); }

CellId Simulation::add_cell(const CellInit& cell) { return state_.add_cell(cell); }

ConstraintId Simulation::add_plane_constraint(const PlaneConstraintInit& plane) {
  return constraints_.add_plane(plane);
}

ConstraintId Simulation::add_sphere_constraint(const SphereConstraintInit& sphere) {
  return constraints_.add_sphere(sphere);
}

void Simulation::set_species(CellId id, std::span<const float> levels) {
  state_.set_species(id, levels);
}

void Simulation::set_species_rate_plan(const SpeciesRatePlan& plan) {
  plan.validate();
  if (plan.species_count() != state_.species_count()) {
    throw std::invalid_argument("species rate plan and simulation species counts disagree");
  }
  species_rate_plan_ = plan;
}

std::pair<CellId, CellId> Simulation::divide_equal(CellId parent_id) {
  return state_.divide_equal(parent_id);
}

void Simulation::step(float dt) {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  if (state_.species_count() != 0 && !backend_->supports(BackendFeature::species)) {
    throw std::runtime_error("selected backend does not implement species integration");
  }
  const auto geometry = state_.geometry_state();
  const std::vector<float> previous_lengths(geometry.lengths.begin(), geometry.lengths.end());
  backend_->advance_growth(state_, dt);
  if (state_.species_count() != 0) {
    backend_->advance_species(state_, species_rate_plan_, previous_lengths, dt);
  }
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

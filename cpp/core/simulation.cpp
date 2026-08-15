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
    return true;
  }
#endif
#if CM2_HAS_CUDA
  if (kind == BackendKind::cuda) {
    return true;
  }
#endif
  return false;
}

Simulation::Simulation(BackendKind backend, std::size_t reserved_capacity)
    : state_(reserved_capacity), backend_(make_backend(backend)) {}

BackendInfo Simulation::backend_info() const { return backend_->info(); }

double Simulation::time() const noexcept { return time_; }

std::size_t Simulation::cell_count() const noexcept { return state_.size(); }

CellId Simulation::add_cell(const CellInit& cell) { return state_.add_cell(cell); }

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

CellSnapshot Simulation::cell(CellId id) const { return state_.cell(id); }

std::vector<CellSnapshot> Simulation::cells() const { return state_.cells(); }

std::optional<CellId> Simulation::lineage_parent(CellId id) const noexcept {
  return state_.lineage_parent(id);
}

void Simulation::validate() const { state_.validate(); }

}  // namespace cm2

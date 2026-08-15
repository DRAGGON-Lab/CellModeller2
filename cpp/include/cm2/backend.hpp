#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "cm2/constraints.hpp"
#include "cm2/contact_graph.hpp"
#include "cm2/mechanics.hpp"
#include "cm2/signals.hpp"
#include "cm2/species.hpp"
#include "cm2/types.hpp"
#include "cm2/world_state.hpp"

namespace cm2 {

class ComputeBackend {
 public:
  virtual ~ComputeBackend() = default;

  [[nodiscard]] virtual BackendInfo info() const = 0;
  [[nodiscard]] virtual bool supports(BackendFeature feature) const noexcept = 0;
  virtual void advance_growth(WorldState& state, float dt) = 0;
  virtual void advance_species(WorldState& state, const SpeciesRatePlan& plan,
                               std::span<const float> previous_lengths, float dt) = 0;
  virtual void advance_signal_grid(SignalGrid& grid, float dt) = 0;
  [[nodiscard]] virtual ContactGraph find_cell_contacts(const WorldState& state,
                                                        const ContactParameters& parameters) = 0;
  [[nodiscard]] virtual ExternalContactGraph find_external_contacts(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) = 0;
  [[nodiscard]] virtual MechanicsSolveResult solve_cell_mechanics(
      const WorldState& state, const ContactGraph& contacts,
      const ExternalContactGraph& external_contacts, const MechanicsParameters& parameters) = 0;
};

[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend(std::uint32_t device_index = 0);
[[nodiscard]] std::unique_ptr<ComputeBackend> make_metal_backend(std::uint32_t device_index = 0);
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend(std::uint32_t device_index = 0);
[[nodiscard]] std::size_t metal_backend_device_count() noexcept;
[[nodiscard]] std::size_t cuda_backend_device_count() noexcept;
[[nodiscard]] std::size_t backend_device_count(BackendKind kind) noexcept;
[[nodiscard]] bool backend_available(BackendKind kind, std::uint32_t device_index = 0) noexcept;

}  // namespace cm2

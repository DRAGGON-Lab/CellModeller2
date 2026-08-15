#include <memory>
#include <stdexcept>

#include "cm2/backend.hpp"

namespace cm2 {
namespace {

class CpuBackend final : public ComputeBackend {
 public:
  [[nodiscard]] BackendInfo info() const override {
    return {
        .kind = BackendKind::cpu,
        .name = "cpu-reference",
        .device = "host",
        .device_index = 0,
        .native = true,
    };
  }

  [[nodiscard]] bool supports(BackendFeature feature) const noexcept override {
    return feature == BackendFeature::growth || feature == BackendFeature::species ||
           feature == BackendFeature::cell_contacts || feature == BackendFeature::cell_mechanics ||
           feature == BackendFeature::external_constraints || feature == BackendFeature::signals ||
           feature == BackendFeature::coupled_rates;
  }

  void advance_growth(WorldState& state, float dt) override { state.advance_growth(dt); }

  void advance_species(WorldState& state, const SpeciesRatePlan& plan,
                       std::span<const float> previous_lengths, float dt) override {
    advance_species_cpu(state, plan, previous_lengths, dt);
  }

  void advance_signal_grid(SignalGrid& grid, float dt) override {
    advance_signal_grid_cpu(grid, dt);
  }

  void advance_coupled(WorldState& state, SignalGrid& grid, const CoupledRatePlan& plan,
                       std::span<const float> previous_lengths, float dt) override {
    advance_coupled_cpu(state, grid, plan, previous_lengths, dt);
  }

  [[nodiscard]] ContactGraph find_cell_contacts(const WorldState& state,
                                                const ContactParameters& parameters) override {
    return find_cell_contacts_cpu(state, parameters);
  }

  [[nodiscard]] ExternalContactGraph find_external_contacts(
      const WorldState& state, const ConstraintSet& constraints,
      const ConstraintContactParameters& parameters) override {
    return find_external_contacts_cpu(state, constraints, parameters);
  }

  [[nodiscard]] MechanicsSolveResult solve_cell_mechanics(
      const WorldState& state, const ContactGraph& contacts,
      const ExternalContactGraph& external_contacts,
      const MechanicsParameters& parameters) override {
    return solve_cell_mechanics_cpu(state, contacts, external_contacts, parameters);
  }
};

}  // namespace

std::unique_ptr<ComputeBackend> make_cpu_backend(std::uint32_t device_index) {
  if (device_index != 0) {
    throw std::out_of_range("CPU backend exposes only device index 0");
  }
  return std::make_unique<CpuBackend>();
}

}  // namespace cm2

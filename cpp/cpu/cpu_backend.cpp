#include <memory>

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
        .native = true,
    };
  }

  [[nodiscard]] bool supports(BackendFeature feature) const noexcept override {
    return feature == BackendFeature::growth || feature == BackendFeature::cell_contacts ||
           feature == BackendFeature::cell_mechanics ||
           feature == BackendFeature::external_constraints;
  }

  void advance_growth(WorldState& state, float dt) override { state.advance_growth(dt); }

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
      const MechanicsParameters& parameters) override {
    return solve_cell_mechanics_cpu(state, contacts, parameters);
  }
};

}  // namespace

std::unique_ptr<ComputeBackend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

}  // namespace cm2

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "cm2/backend.hpp"
#include "cm2/checkpoint.hpp"

namespace cm2 {

class Simulation {
 public:
  explicit Simulation(BackendKind backend = BackendKind::cpu, std::size_t reserved_capacity = 0,
                      std::size_t species_count = 0, std::uint32_t device_index = 0);
  Simulation(BackendKind backend, const SimulationCheckpoint& checkpoint,
             std::uint32_t device_index = 0);

  [[nodiscard]] BackendInfo backend_info() const;
  [[nodiscard]] bool supports(BackendFeature feature) const noexcept;
  [[nodiscard]] double time() const noexcept;
  [[nodiscard]] std::size_t cell_count() const noexcept;
  [[nodiscard]] std::size_t species_count() const noexcept;
  [[nodiscard]] std::size_t signal_count() const noexcept;
  [[nodiscard]] bool has_signal_grid() const noexcept;

  CellId add_cell(const CellInit& cell);
  ConstraintId add_plane_constraint(const PlaneConstraintInit& plane);
  ConstraintId add_sphere_constraint(const SphereConstraintInit& sphere);
  void set_species(CellId id, std::span<const float> levels);
  void set_species_rate_plan(const SpeciesRatePlan& plan);
  void configure_signal_grid(const SignalGridSpec& spec, std::vector<float> levels = {});
  void set_signal_levels(std::span<const float> levels);
  std::pair<CellId, CellId> divide_equal(CellId parent_id);
  void step(float dt);
  [[nodiscard]] ContactGraph find_cell_contacts(
      const ContactParameters& parameters = ContactParameters{});
  [[nodiscard]] ExternalContactGraph find_external_contacts(
      const ConstraintContactParameters& parameters = ConstraintContactParameters{});
  [[nodiscard]] MechanicsSolveResult solve_cell_mechanics(
      const MechanicsParameters& mechanics_parameters = MechanicsParameters{},
      const ContactParameters& contact_parameters = ContactParameters{},
      const ConstraintContactParameters& constraint_parameters = ConstraintContactParameters{});
  [[nodiscard]] MechanicsSolveResult relax_cell_mechanics(
      const MechanicsParameters& mechanics_parameters = MechanicsParameters{},
      const ContactParameters& contact_parameters = ContactParameters{},
      const MechanicsIntegrationParameters& integration_parameters =
          MechanicsIntegrationParameters{},
      const ConstraintContactParameters& constraint_parameters = ConstraintContactParameters{});

  [[nodiscard]] CellSnapshot cell(CellId id) const;
  [[nodiscard]] std::vector<CellSnapshot> cells() const;
  [[nodiscard]] std::optional<CellId> lineage_parent(CellId id) const noexcept;
  [[nodiscard]] std::vector<float> signal_levels() const;
  [[nodiscard]] std::vector<float> sample_signals(Vec3 position) const;
  [[nodiscard]] SimulationCheckpoint checkpoint() const;
  void validate() const;

 private:
  WorldState state_;
  ConstraintSet constraints_;
  std::unique_ptr<ComputeBackend> backend_;
  SpeciesRatePlan species_rate_plan_;
  std::optional<SignalGrid> signal_grid_;
  double time_{0.0};
};

}  // namespace cm2

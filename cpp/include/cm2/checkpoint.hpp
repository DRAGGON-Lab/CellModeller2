#pragma once

#include <cstdint>
#include <optional>

#include "cm2/constraints.hpp"
#include "cm2/signals.hpp"
#include "cm2/species.hpp"
#include "cm2/world_state.hpp"

namespace cm2 {

inline constexpr std::uint32_t checkpoint_schema_version = 2;

struct SimulationCheckpoint {
  std::uint32_t schema_version{checkpoint_schema_version};
  double time{0.0};
  WorldStateCheckpoint world;
  ConstraintSetCheckpoint constraints;
  SpeciesRatePlan species_rate_plan;
  std::optional<SignalGridCheckpoint> signal_grid;

  void validate() const;
};

}  // namespace cm2

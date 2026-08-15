#include "cm2/checkpoint.hpp"

#include <cmath>
#include <stdexcept>

namespace cm2 {

void SimulationCheckpoint::validate() const {
  if (schema_version != checkpoint_schema_version) {
    throw std::invalid_argument("unsupported CellModeller2 checkpoint schema version");
  }
  if (!std::isfinite(time) || time < 0.0) {
    throw std::invalid_argument("checkpoint time must be finite and non-negative");
  }
  world.validate();
  constraints.validate();
  species_rate_plan.validate();
  if (species_rate_plan.species_count() != world.species_count) {
    throw std::invalid_argument("checkpoint rate plan and world species counts disagree");
  }
}

}  // namespace cm2

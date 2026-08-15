#include <array>
#include <cassert>
#include <cstdint>

#include "backend_devices.hpp"
#include "cm2/simulation.hpp"

namespace {

constexpr std::array required_features{
    cm2::BackendFeature::growth,
    cm2::BackendFeature::species,
    cm2::BackendFeature::cell_contacts,
    cm2::BackendFeature::cell_mechanics,
    cm2::BackendFeature::external_constraints,
    cm2::BackendFeature::signals,
    cm2::BackendFeature::coupled_rates,
};

void require_complete_backend(cm2::BackendKind backend, std::uint32_t device_index) {
  cm2::Simulation simulation(backend, 0, 0, device_index);
  for (const auto feature : required_features) {
    assert(simulation.supports(feature));
  }
}

}  // namespace

int main() {
  cm2::test::for_each_backend_device(require_complete_backend);
  return 0;
}

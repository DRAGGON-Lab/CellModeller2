#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cm2/simulation.hpp"

namespace {

constexpr std::size_t cell_count = 513;
constexpr std::size_t species_count = 3;
constexpr std::array time_steps{0.0F, 0.01F, 0.025F, 0.1F};

bool close(float actual, float expected) {
  constexpr float absolute_tolerance = 2.0e-5F;
  constexpr float relative_tolerance = 2.0e-5F;
  return std::abs(actual - expected) <=
         absolute_tolerance + relative_tolerance * std::abs(expected);
}

cm2::SpeciesRatePlan make_plan() {
  using enum cm2::RateOp;
  return cm2::SpeciesRatePlan(species_count,
                              {
                                  {.operation = species, .first = 0},
                                  {.operation = species, .first = 1},
                                  {.operation = constant, .value = 0.25F},
                                  {.operation = multiply, .first = 0, .second = 1},
                                  {.operation = add, .first = 3, .second = 2},
                                  {.operation = cell_type},
                                  {.operation = constant, .value = 1.0F},
                                  {.operation = equal, .first = 5, .second = 6},
                                  {.operation = growth_rate},
                                  {.operation = negate, .first = 8},
                                  {.operation = constant, .value = 0.0F},
                                  {.operation = select, .first = 7, .second = 9, .third = 10},
                                  {.operation = cell_volume},
                                  {.operation = cell_surface_area},
                                  {.operation = divide, .first = 13, .second = 12},
                                  {.operation = position_x},
                                  {.operation = add, .first = 14, .second = 15},
                              },
                              {4, 11, 16});
}

void populate(cm2::Simulation& simulation) {
  simulation.set_species_rate_plan(make_plan());
  for (std::size_t index = 0; index < cell_count; ++index) {
    cm2::CellInit cell;
    cell.position = {static_cast<float>(index % 19) * 0.03F, static_cast<float>(index % 7) * -0.02F,
                     static_cast<float>(index % 5) * 0.01F};
    cell.length = 1.0F + static_cast<float>(index % 13) * 0.1F;
    cell.radius = 0.25F + static_cast<float>(index % 3) * 0.05F;
    cell.growth_rate = static_cast<float>(index % 11) * 0.025F;
    cell.cell_type = static_cast<std::int32_t>(index % 4);
    cell.species = {
        0.1F + static_cast<float>(index % 17) * 0.02F,
        0.2F + static_cast<float>(index % 23) * 0.01F,
        static_cast<float>(index % 9) * -0.03F,
    };
    assert(simulation.add_cell(cell) == static_cast<cm2::CellId>(index + 1));
  }
}

void compare(const cm2::Simulation& actual, const cm2::Simulation& expected) {
  assert(actual.cell_count() == expected.cell_count());
  assert(actual.species_count() == expected.species_count());
  assert(std::abs(actual.time() - expected.time()) <= 1.0e-12);
  const auto actual_cells = actual.cells();
  const auto expected_cells = expected.cells();
  for (std::size_t cell = 0; cell < expected_cells.size(); ++cell) {
    assert(actual_cells[cell].id == expected_cells[cell].id);
    assert(actual_cells[cell].slot == expected_cells[cell].slot);
    assert(close(actual_cells[cell].length, expected_cells[cell].length));
    assert(actual_cells[cell].species.size() == species_count);
    for (std::size_t species = 0; species < species_count; ++species) {
      assert(close(actual_cells[cell].species[species], expected_cells[cell].species[species]));
    }
  }
}

void run_scenario(cm2::BackendKind backend) {
  cm2::Simulation reference(cm2::BackendKind::cpu, cell_count, species_count);
  cm2::Simulation candidate(backend, cell_count, species_count);
  populate(reference);
  populate(candidate);
  for (const auto dt : time_steps) {
    reference.step(dt);
    candidate.step(dt);
  }
  compare(candidate, reference);
  candidate.validate();
}

}  // namespace

int main() {
  for (const auto backend :
       {cm2::BackendKind::cpu, cm2::BackendKind::metal, cm2::BackendKind::cuda}) {
    if (!cm2::backend_available(backend)) {
      continue;
    }
    cm2::Simulation probe(backend);
    if (probe.supports(cm2::BackendFeature::species)) {
      run_scenario(backend);
    }
  }
  return 0;
}

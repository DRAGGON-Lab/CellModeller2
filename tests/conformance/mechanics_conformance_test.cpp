#include <cassert>
#include <cmath>
#include <cstddef>

#include "cm2/simulation.hpp"

namespace {

constexpr float absolute_tolerance = 3.0e-4F;
constexpr float relative_tolerance = 3.0e-4F;

bool close(float actual, float expected) {
  const auto tolerance = absolute_tolerance + relative_tolerance * std::abs(expected);
  return std::abs(actual - expected) <= tolerance;
}

cm2::CellId add_capsule(cm2::Simulation& simulation, cm2::Vec3 center, cm2::Vec3 axis,
                        float length = 4.0F, float radius = 0.5F) {
  cm2::CellInit cell;
  cell.position = center;
  cell.direction = axis;
  cell.length = length;
  cell.radius = radius;
  return simulation.add_cell(cell);
}

void populate_mixed_colony(cm2::Simulation& simulation) {
  add_capsule(simulation, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  add_capsule(simulation, {0.2F, 0.8F, 0.0F}, {1.0F, 0.2F, 0.0F});
  add_capsule(simulation, {-0.3F, 0.1F, 0.7F}, {0.0F, 1.0F, 0.3F});
  add_capsule(simulation, {4.8F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
}

void compare_corrections(const cm2::MechanicsSolveResult& actual,
                         const cm2::MechanicsSolveResult& expected) {
  assert(actual.report.status == expected.report.status);
  assert(actual.report.breakdown == expected.report.breakdown);
  assert(close(actual.report.initial_residual_rms, expected.report.initial_residual_rms));
  assert(actual.corrections.size() == expected.corrections.size());
  for (std::size_t index = 0; index < expected.corrections.size(); ++index) {
    const auto& left = actual.corrections[index];
    const auto& right = expected.corrections[index];
    assert(close(left.translation.x, right.translation.x));
    assert(close(left.translation.y, right.translation.y));
    assert(close(left.translation.z, right.translation.z));
    assert(close(left.rotation.x, right.rotation.x));
    assert(close(left.rotation.y, right.rotation.y));
    assert(close(left.rotation.z, right.rotation.z));
    assert(close(left.length, right.length));
  }
}

void run_converged_colony(cm2::BackendKind backend) {
  cm2::Simulation reference(cm2::BackendKind::cpu);
  cm2::Simulation candidate(backend);
  populate_mixed_colony(reference);
  populate_mixed_colony(candidate);

  cm2::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 2.0e-5F;
  const auto expected = reference.solve_cell_mechanics(parameters);
  const auto actual = candidate.solve_cell_mechanics(parameters);
  assert(expected.report.status == cm2::SolverStatus::converged);
  assert(actual.report.final_residual_rms <= parameters.residual_rms_tolerance);
  compare_corrections(actual, expected);
}

void run_iteration_limit(cm2::BackendKind backend) {
  cm2::Simulation reference(cm2::BackendKind::cpu);
  cm2::Simulation candidate(backend);
  populate_mixed_colony(reference);
  populate_mixed_colony(candidate);

  cm2::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 0.0F;
  parameters.max_iterations = 1;
  const auto expected = reference.solve_cell_mechanics(parameters);
  const auto actual = candidate.solve_cell_mechanics(parameters);
  assert(expected.report.status == cm2::SolverStatus::iteration_limit);
  assert(actual.report.iterations == 1);
  compare_corrections(actual, expected);
}

void run_empty_systems(cm2::BackendKind backend) {
  cm2::Simulation empty(backend);
  const auto empty_result = empty.solve_cell_mechanics();
  assert(empty_result.report.status == cm2::SolverStatus::converged);
  assert(empty_result.corrections.empty());

  cm2::Simulation separated(backend);
  add_capsule(separated, {}, {1.0F, 0.0F, 0.0F});
  add_capsule(separated, {10.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto separated_result = separated.solve_cell_mechanics();
  assert(separated_result.report.status == cm2::SolverStatus::converged);
  assert(separated_result.report.iterations == 0);
  for (const auto& correction : separated_result.corrections) {
    assert(correction.translation.x == 0.0F);
    assert(correction.translation.y == 0.0F);
    assert(correction.translation.z == 0.0F);
  }
}

void run_buffer_growth(cm2::BackendKind backend) {
  cm2::Simulation simulation(backend);
  add_capsule(simulation, {}, {1.0F, 0.0F, 0.0F});
  add_capsule(simulation, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto first = simulation.solve_cell_mechanics();
  assert(first.report.status == cm2::SolverStatus::converged);

  for (std::size_t index = 0; index < 7; ++index) {
    add_capsule(simulation, {0.1F * static_cast<float>(index), 0.2F, 0.1F}, {1.0F, 0.1F, 0.0F});
  }
  const auto grown = simulation.solve_cell_mechanics();
  assert(grown.report.status == cm2::SolverStatus::converged);
  assert(grown.corrections.size() == 9);
}

void run_integrated_relaxation(cm2::BackendKind backend) {
  cm2::Simulation reference(cm2::BackendKind::cpu);
  cm2::Simulation candidate(backend);
  populate_mixed_colony(reference);
  populate_mixed_colony(candidate);

  cm2::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 2.0e-5F;
  const auto expected = reference.relax_cell_mechanics(parameters);
  const auto actual = candidate.relax_cell_mechanics(parameters);
  compare_corrections(actual, expected);
  const auto expected_cells = reference.cells();
  const auto actual_cells = candidate.cells();
  assert(actual_cells.size() == expected_cells.size());
  for (std::size_t index = 0; index < expected_cells.size(); ++index) {
    assert(actual_cells[index].id == expected_cells[index].id);
    assert(close(actual_cells[index].position.x, expected_cells[index].position.x));
    assert(close(actual_cells[index].position.y, expected_cells[index].position.y));
    assert(close(actual_cells[index].position.z, expected_cells[index].position.z));
    assert(close(actual_cells[index].direction.x, expected_cells[index].direction.x));
    assert(close(actual_cells[index].direction.y, expected_cells[index].direction.y));
    assert(close(actual_cells[index].direction.z, expected_cells[index].direction.z));
    assert(close(actual_cells[index].length, expected_cells[index].length));
  }
}

void run_fixed_cell_relaxation(cm2::BackendKind backend) {
  cm2::Simulation reference(cm2::BackendKind::cpu);
  cm2::Simulation candidate(backend);
  const auto reference_fixed =
      add_capsule(reference, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto candidate_fixed =
      add_capsule(candidate, {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto reference_free =
      add_capsule(reference, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F});
  const auto candidate_free =
      add_capsule(candidate, {0.0F, 0.8F, 0.0F}, {1.0F, 0.0F, 0.0F});
  reference.set_cell_fixed(reference_fixed, true);
  candidate.set_cell_fixed(candidate_fixed, true);

  cm2::MechanicsParameters parameters;
  parameters.residual_rms_tolerance = 2.0e-5F;
  const auto expected = reference.solve_cell_mechanics(parameters);
  const auto actual = candidate.solve_cell_mechanics(parameters);
  compare_corrections(actual, expected);
  assert(actual.report.status == cm2::SolverStatus::converged);
  assert(actual.corrections[0].translation.x == 0.0F);
  assert(actual.corrections[0].translation.y == 0.0F);
  assert(actual.corrections[0].translation.z == 0.0F);
  assert(actual.corrections[0].rotation.x == 0.0F);
  assert(actual.corrections[0].rotation.y == 0.0F);
  assert(actual.corrections[0].rotation.z == 0.0F);
  assert(actual.corrections[0].length == 0.0F);
  assert(actual.corrections[1].translation.y > 0.0F);

  const auto expected_relaxation = reference.relax_cell_mechanics(parameters);
  const auto actual_relaxation = candidate.relax_cell_mechanics(parameters);
  compare_corrections(actual_relaxation, expected_relaxation);
  assert(candidate.cell(candidate_fixed).position.x == 0.0F);
  assert(candidate.cell(candidate_fixed).position.y == 0.0F);
  assert(candidate.cell(candidate_fixed).position.z == 0.0F);
  assert(candidate.cell(candidate_free).position.y > 0.8F);
  assert(close(candidate.cell(candidate_free).position.y,
               reference.cell(reference_free).position.y));
}

}  // namespace

int main() {
  for (const auto backend :
       {cm2::BackendKind::cpu, cm2::BackendKind::metal, cm2::BackendKind::cuda}) {
    if (!cm2::backend_available(backend)) {
      continue;
    }
    cm2::Simulation capability_probe(backend);
    if (!capability_probe.supports(cm2::BackendFeature::cell_mechanics)) {
      continue;
    }
    run_empty_systems(backend);
    run_converged_colony(backend);
    run_iteration_limit(backend);
    run_buffer_growth(backend);
    run_integrated_relaxation(backend);
    run_fixed_cell_relaxation(backend);
  }
  return 0;
}

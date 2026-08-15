#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cm2/simulation.hpp"

int main() {
  assert(cm2::backend_available(cm2::BackendKind::cuda));

  cm2::Simulation cpu(cm2::BackendKind::cpu, 513);
  cm2::Simulation cuda(cm2::BackendKind::cuda, 513);
  std::vector<cm2::CellId> ids;
  ids.reserve(513);

  for (std::size_t index = 0; index < 513; ++index) {
    cm2::CellInit cell;
    cell.position.x = static_cast<float>(index) * 0.25F;
    cell.length = 1.0F + static_cast<float>(index % 17) * 0.1F;
    cell.radius = 0.25F + static_cast<float>(index % 3) * 0.05F;
    cell.growth_rate = static_cast<float>(index % 11) * 0.025F;
    ids.push_back(cpu.add_cell(cell));
    assert(cuda.add_cell(cell) == ids.back());
  }

  for (const auto dt : {0.01F, 0.025F, 0.1F, 0.04F}) {
    cpu.step(dt);
    cuda.step(dt);
  }

  for (const auto id : ids) {
    const auto cpu_cell = cpu.cell(id);
    const auto cuda_cell = cuda.cell(id);
    assert(std::abs(cpu_cell.length - cuda_cell.length) <= 1.0e-6F);
    assert(cpu_cell.id == cuda_cell.id);
    assert(cpu_cell.slot == cuda_cell.slot);
  }

  const auto info = cuda.backend_info();
  assert(info.kind == cm2::BackendKind::cuda);
  assert(info.native);
  assert(!info.device.empty());
  cuda.validate();
  return 0;
}

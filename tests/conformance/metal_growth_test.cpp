#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cm2/simulation.hpp"

int main() {
  assert(cm2::backend_available(cm2::BackendKind::metal));

  cm2::Simulation cpu(cm2::BackendKind::cpu, 513);
  cm2::Simulation metal(cm2::BackendKind::metal, 513);
  std::vector<cm2::CellId> ids;
  ids.reserve(513);

  for (std::size_t index = 0; index < 513; ++index) {
    cm2::CellInit cell;
    cell.position.x = static_cast<float>(index) * 0.25F;
    cell.length = 1.0F + static_cast<float>(index % 17) * 0.1F;
    cell.radius = 0.25F + static_cast<float>(index % 3) * 0.05F;
    cell.growth_rate = static_cast<float>(index % 11) * 0.025F;
    ids.push_back(cpu.add_cell(cell));
    assert(metal.add_cell(cell) == ids.back());
  }

  for (const auto dt : {0.01F, 0.025F, 0.1F, 0.04F}) {
    cpu.step(dt);
    metal.step(dt);
  }

  for (const auto id : ids) {
    const auto cpu_cell = cpu.cell(id);
    const auto metal_cell = metal.cell(id);
    assert(std::abs(cpu_cell.length - metal_cell.length) <= 1.0e-6F);
    assert(cpu_cell.id == metal_cell.id);
    assert(cpu_cell.slot == metal_cell.slot);
  }

  const auto info = metal.backend_info();
  assert(info.kind == cm2::BackendKind::metal);
  assert(info.native);
  assert(!info.device.empty());
  metal.validate();
  return 0;
}

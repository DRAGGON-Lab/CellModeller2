from __future__ import annotations

import math

import pytest
from cellmodeller2 import BackendKind, CellInit, Simulation, Vec3


def test_growth_and_division_preserve_declared_semantics() -> None:
    simulation = Simulation(BackendKind.CPU)
    initial = CellInit()
    initial.position = Vec3(2.0, 3.0, 0.0)
    initial.direction = Vec3(2.0, 0.0, 0.0)
    initial.length = 4.0
    initial.radius = 0.5
    initial.growth_rate = 0.25
    initial.cell_type = 7

    parent = simulation.add_cell(initial)
    simulation.step(0.5)
    assert math.isclose(simulation.cell(parent).length, 4.5)

    first, second = simulation.divide_equal(parent)
    assert simulation.cell_count == 2
    assert simulation.lineage_parent(first) == parent
    assert simulation.lineage_parent(second) == parent
    assert [cell.slot for cell in simulation.cells()] == [0, 1]
    assert all(math.isclose(cell.direction.x, 1.0) for cell in simulation.cells())
    simulation.validate()


@pytest.mark.parametrize("backend", [BackendKind.METAL, BackendKind.CUDA])
def test_unavailable_backend_fails_instead_of_falling_back(backend: BackendKind) -> None:
    with pytest.raises(RuntimeError, match="not implemented"):
        Simulation(backend)


def test_invalid_time_step_is_rejected() -> None:
    simulation = Simulation()
    with pytest.raises(ValueError, match="time step"):
        simulation.step(-0.1)

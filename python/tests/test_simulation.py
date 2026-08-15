from __future__ import annotations

import math

import pytest
from cellmodeller2 import (
    BackendKind,
    CellInit,
    ContactParameters,
    Simulation,
    Vec3,
    backend_available,
)


@pytest.mark.parametrize("backend", list(BackendKind))
def test_growth_and_division_preserve_declared_semantics(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")

    simulation = Simulation(backend)
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


@pytest.mark.parametrize("backend", list(BackendKind))
def test_unavailable_backend_fails_instead_of_falling_back(backend: BackendKind) -> None:
    if backend_available(backend):
        Simulation(backend)
        return
    with pytest.raises(RuntimeError, match="not implemented"):
        Simulation(backend)


def test_invalid_time_step_is_rejected() -> None:
    simulation = Simulation()
    with pytest.raises(ValueError, match="time step"):
        simulation.step(-0.1)


def test_cpu_contact_graph_is_available_through_the_public_api() -> None:
    simulation = Simulation()
    first = CellInit()
    first.length = 4.0
    first.radius = 0.5
    second = CellInit()
    second.position = Vec3(0.0, 0.8, 0.0)
    second.length = 4.0
    second.radius = 0.5
    simulation.add_cell(first)
    simulation.add_cell(second)

    parameters = ContactParameters()
    graph = simulation.find_cell_contacts(parameters)
    assert graph.cell_count == 2
    assert not graph.empty
    assert len(graph) == 2
    assert graph.incident_contact_indices(0) == [0, 1]
    assert [contact.ordinal for contact in graph.contacts] == [0, 1]
    assert all(
        math.isclose(contact.signed_separation, -0.2, abs_tol=1.0e-6)
        for contact in graph.contacts
    )


@pytest.mark.parametrize("backend", [BackendKind.METAL, BackendKind.CUDA])
def test_native_growth_matches_cpu(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")

    cpu = Simulation(BackendKind.CPU)
    native = Simulation(backend)
    for index in range(33):
        cell = CellInit()
        cell.length = 1.0 + index * 0.1
        cell.growth_rate = (index % 7) * 0.025
        assert cpu.add_cell(cell) == native.add_cell(cell)

    for dt in (0.01, 0.025, 0.1):
        cpu.step(dt)
        native.step(dt)

    for cpu_cell, native_cell in zip(cpu.cells(), native.cells(), strict=True):
        assert math.isclose(cpu_cell.length, native_cell.length, abs_tol=1.0e-6)

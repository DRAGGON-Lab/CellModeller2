from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
from cellmodeller2 import (
    BackendKind,
    CellInit,
    LegacyCell,
    LegacyCompatibilityError,
    LegacyModelAdapter,
    Simulation,
    backend_available,
    load_checkpoint_bundle,
    save_checkpoint,
)


class _ArrayView(Protocol):
    @property
    def dtype(self) -> object: ...

    def tolist(self) -> list[float]: ...


@pytest.mark.parametrize("backend", list(BackendKind))
def test_legacy_callbacks_drive_native_growth_and_division(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    divided: list[tuple[int, int, int]] = []

    def initialize(cell: LegacyCell) -> None:
        cell.growthRate = 1.0
        cell.targetVol = 4.5
        cell.marker = "founder"

    def update(cells: dict[int, LegacyCell]) -> None:
        for cell in cells.values():
            cell.growthRate = 0.5
            if cell.volume >= cell.targetVol:
                cell.divideFlag = True

    def divide(parent: LegacyCell, first: LegacyCell, second: LegacyCell) -> None:
        divided.append((parent.id, first.id, second.id))
        first.marker = "left"
        second.marker = "right"
        first.targetVol = 10.0
        second.targetVol = 10.0

    simulation = Simulation(backend)
    adapter = LegacyModelAdapter(
        simulation,
        init=initialize,
        update=update,
        divide=divide,
        mechanics=False,
        compute_neighbors=True,
    )
    initial = CellInit()
    initial.length = 5.0
    parent_id = adapter.add_cell(initial)

    adapter.step(0.2)

    assert parent_id == 1
    assert divided == [(1, 2, 3)]
    assert list(adapter.cells) == [2, 3]
    first, second = adapter.cells.values()
    assert first.idx == 0
    assert second.idx == 1
    assert first.marker == "left"
    assert second.marker == "right"
    assert first.cellAge == 1
    assert second.cellAge == 1
    assert first.neighbours == [second.id]
    assert second.neighbours == [first.id]
    assert math.isclose(first.length, 2.2, abs_tol=1.0e-6)
    assert math.isclose(second.length, 2.2, abs_tol=1.0e-6)
    assert simulation.lineage_parent(first.id) == parent_id
    assert simulation.lineage_parent(second.id) == parent_id


def test_legacy_callback_changes_are_validated_before_native_updates() -> None:
    def initialize(cell: LegacyCell) -> None:
        cell.label = "arbitrary state is retained"

    def update(cells: dict[int, LegacyCell]) -> None:
        cell = next(iter(cells.values()))
        cell.growthRate = math.nan

    simulation = Simulation()
    adapter = LegacyModelAdapter(simulation, init=initialize, update=update, mechanics=False)
    cell_id = adapter.add_cell(CellInit())

    with pytest.raises(LegacyCompatibilityError, match="growthRate"):
        adapter.step(0.1)
    assert simulation.cell(cell_id).growth_rate == 1.0


def test_legacy_adapter_rejects_geometry_mutation() -> None:
    def initialize(cell: LegacyCell) -> None:
        cell.targetVol = 1.0

    def mutate_geometry(cells: dict[int, LegacyCell]) -> None:
        next(iter(cells.values())).length = 12.0

    simulation = Simulation()
    adapter = LegacyModelAdapter(
        simulation, init=initialize, update=mutate_geometry, mechanics=False
    )
    adapter.add_cell(CellInit())
    with pytest.raises(LegacyCompatibilityError, match="may not mutate"):
        adapter.step(0.1)


def test_legacy_division_jitter_is_explicit_and_seeded() -> None:
    def initialize(cell: LegacyCell) -> None:
        cell.divideFlag = False

    def divide_immediately(cells: dict[int, LegacyCell]) -> None:
        next(iter(cells.values())).divideFlag = True

    simulation = Simulation()
    adapter = LegacyModelAdapter(
        simulation,
        init=initialize,
        update=divide_immediately,
        mechanics=False,
        division_jitter_z=False,
        rng=random.Random(17),
    )
    initial = CellInit()
    initial.length = 5.0
    adapter.add_cell(initial)
    adapter.step(0.0)

    directions = [cell.dir for cell in adapter.cells.values()]
    assert directions[0] != [1.0, 0.0, 0.0]
    assert directions[1] != [1.0, 0.0, 0.0]
    assert directions[0][2] == 0.0
    assert directions[1][2] == 0.0


def test_legacy_controller_state_resumes_attributes_and_random_stream(tmp_path: Path) -> None:
    def initialize(cell: LegacyCell) -> None:
        cell.metadata = {"line": (1, 2), "weights": [0.25, 0.75]}
        cell.color = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    def divide_every_step(cells: dict[int, LegacyCell]) -> None:
        for cell in cells.values():
            cell.divideFlag = True

    simulation = Simulation()
    adapter = LegacyModelAdapter(
        simulation,
        init=initialize,
        update=divide_every_step,
        mechanics=False,
        division_jitter_z=True,
        rng=random.Random(91),
    )
    initial = CellInit()
    initial.length = 9.0
    adapter.add_cell(initial)
    adapter.step(0.0)

    path = tmp_path / "legacy.cm2.json"
    save_checkpoint(simulation, path, controller=adapter.controller_state())
    bundle = load_checkpoint_bundle(path)
    restored = LegacyModelAdapter.from_controller_state(
        bundle.simulation,
        bundle.controller,
        init=initialize,
        update=divide_every_step,
    )

    restored_first = next(iter(restored.cells.values()))
    assert restored_first.metadata == {"line": (1, 2), "weights": [0.25, 0.75]}
    raw_color: object = restored_first.color
    assert isinstance(raw_color, np.ndarray)
    color = cast(_ArrayView, raw_color)
    assert str(color.dtype) == "float32"
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-7)
        for actual, expected in zip(color.tolist(), [0.1, 0.2, 0.3], strict=True)
    )

    adapter.step(0.0)
    restored.step(0.0)
    for original_cell, restored_cell in zip(
        adapter.simulation.cells(), restored.simulation.cells(), strict=True
    ):
        assert original_cell.id == restored_cell.id
        assert original_cell.slot == restored_cell.slot
        assert original_cell.position.x == restored_cell.position.x
        assert original_cell.position.y == restored_cell.position.y
        assert original_cell.position.z == restored_cell.position.z
        assert original_cell.direction.x == restored_cell.direction.x
        assert original_cell.direction.y == restored_cell.direction.y
        assert original_cell.direction.z == restored_cell.direction.z


def test_legacy_asymm_weights_drive_native_division() -> None:
    def initialize(cell: LegacyCell) -> None:
        cell.asymm = [1.0, 3.0]

    def divide_immediately(cells: dict[int, LegacyCell]) -> None:
        next(iter(cells.values())).divideFlag = True

    simulation = Simulation()
    adapter = LegacyModelAdapter(
        simulation,
        init=initialize,
        update=divide_immediately,
        mechanics=False,
    )
    initial = CellInit()
    initial.length = 6.0
    adapter.add_cell(initial)
    adapter.step(0.0)

    daughters = simulation.cells()
    assert [cell.length for cell in daughters] == [1.25, 3.75]
    assert math.isclose(daughters[0].position.x, -2.375)
    assert math.isclose(daughters[1].position.x, 1.125)


def test_legacy_asymm_weights_fail_explicitly() -> None:
    def initialize(cell: LegacyCell) -> None:
        cell.asymm = [0.0, 1.0]

    def divide_immediately(cells: dict[int, LegacyCell]) -> None:
        next(iter(cells.values())).divideFlag = True

    simulation = Simulation()
    adapter = LegacyModelAdapter(
        simulation,
        init=initialize,
        update=divide_immediately,
        mechanics=False,
    )
    initial = CellInit()
    initial.length = 6.0
    adapter.add_cell(initial)

    with pytest.raises(LegacyCompatibilityError, match="positive weights"):
        adapter.step(0.0)
    assert simulation.cell_count == 1

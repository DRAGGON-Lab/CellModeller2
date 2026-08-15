"""Explicit compatibility layer for legacy Python regulation callbacks."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendFeature,
    CellInit,
    CellSnapshot,
    MechanicsParameters,
    Simulation,
    Vec3,
)


class LegacyCompatibilityError(RuntimeError):
    """Raised when a callback requests behavior outside the compatibility contract."""


class LegacyCell:
    """Mutable CellState-shaped view used by legacy callbacks.

    Arbitrary user attributes are intentionally supported. Engine-owned geometry
    is refreshed from the native simulation after every completed step.
    """

    def __init__(self, snapshot: CellSnapshot) -> None:
        self.id = snapshot.id
        self.idx = snapshot.slot
        self.pos = _vector(snapshot.position)
        self.dir = _vector(snapshot.direction)
        self.length = snapshot.length
        self.radius = snapshot.radius
        self.growthRate = snapshot.growth_rate
        self.cellType = snapshot.cell_type
        self.species = list(snapshot.species)
        self.signals: list[float] = []
        self.volume = snapshot.length
        self.oldLen = snapshot.length
        self.strainRate = 0.0
        self.effGrowth = 0.0
        self.cellAge = 0
        self.time = 0.0
        self.neighbours: list[int] = []
        self.cts = 0
        self.divideFlag = False
        self.color: Any = [0.5, 0.5, 0.5]
        self.asymm = [1.0, 1.0]
        self.vel = [0.0, 0.0, 0.0]
        self.ends = _ends(self.pos, self.dir, self.length)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


type InitCallback = Callable[[LegacyCell], None]
type UpdateCallback = Callable[[dict[int, LegacyCell]], None]
type DivideCallback = Callable[[LegacyCell, LegacyCell, LegacyCell], None]


def _vector(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _ends(
    position: list[float], direction: list[float], length: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    half = length * 0.5
    return (
        (
            position[0] - direction[0] * half,
            position[1] - direction[1] * half,
            position[2] - direction[2] * half,
        ),
        (
            position[0] + direction[0] * half,
            position[1] + direction[1] * half,
            position[2] + direction[2] * half,
        ),
    )


class LegacyModelAdapter:
    """Drive maintained ``init/update/divide`` callbacks over a native simulation."""

    def __init__(
        self,
        simulation: Simulation,
        *,
        init: InitCallback,
        update: UpdateCallback,
        divide: DivideCallback | None = None,
        mechanics: bool = True,
        compute_neighbors: bool = False,
        division_jitter_z: bool | None = None,
        rng: random.Random | None = None,
        mechanics_parameters: MechanicsParameters | None = None,
    ) -> None:
        if simulation.cell_count != 0:
            raise LegacyCompatibilityError("legacy adapter requires an empty simulation")
        if mechanics and not simulation.supports(BackendFeature.CELL_MECHANICS):
            raise LegacyCompatibilityError("backend does not implement cell mechanics")
        if compute_neighbors and not simulation.supports(BackendFeature.CELL_CONTACTS):
            raise LegacyCompatibilityError("backend does not implement cell contacts")
        if division_jitter_z is not None and rng is None:
            raise LegacyCompatibilityError(
                "legacy division jitter requires an explicit random stream"
            )
        self.simulation = simulation
        self._init = init
        self._update = update
        self._divide = divide
        self._mechanics = mechanics
        self._compute_neighbors = compute_neighbors
        self._division_jitter_z = division_jitter_z
        self._rng = rng
        self._mechanics_parameters = mechanics_parameters or MechanicsParameters()
        self._cells: dict[int, LegacyCell] = {}

    @property
    def cells(self) -> Mapping[int, LegacyCell]:
        return MappingProxyType(self._cells)

    def add_cell(self, cell: CellInit) -> int:
        """Add a native cell, then run the legacy ``init(cell)`` callback."""

        cell_id = self.simulation.add_cell(cell)
        legacy_cell = LegacyCell(self.simulation.cell(cell_id))
        self._init(legacy_cell)
        self._validate_engine_owned_geometry(legacy_cell, self.simulation.cell(cell_id))
        self._apply_mutable_state(legacy_cell)
        self._cells[cell_id] = legacy_cell
        return cell_id

    def step(self, dt: float) -> None:
        """Run regulation, division, native integration, mechanics, and state refresh."""

        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("time step must be finite and non-negative")

        self._update(self._cells)
        snapshots = {snapshot.id: snapshot for snapshot in self.simulation.cells()}
        for cell_id, cell in self._cells.items():
            self._validate_engine_owned_geometry(cell, snapshots[cell_id])
            self._validate_mutable_state(cell)
        for cell in self._cells.values():
            self._apply_mutable_state(cell)
            cell.time = self.simulation.time

        dividing = [cell_id for cell_id, cell in self._cells.items() if cell.divideFlag]
        for parent_id in dividing:
            self._divide_cell(parent_id)

        self.simulation.step(dt)
        if self._mechanics and self.simulation.cell_count != 0:
            self.simulation.relax_cell_mechanics(self._mechanics_parameters)
        self._refresh_cells()

    def _divide_cell(self, parent_id: int) -> None:
        parent = self._cells[parent_id]
        if list(parent.asymm) != [1.0, 1.0]:
            raise LegacyCompatibilityError("asymmetric legacy division is not implemented")
        parent.divideFlag = False
        first_id, second_id = self.simulation.divide_equal(parent_id)
        self._apply_division_jitter(first_id)
        self._apply_division_jitter(second_id)
        first = copy.deepcopy(parent)
        second = copy.deepcopy(parent)
        first.cellAge = 0
        second.cellAge = 0
        self._set_identity(first, self.simulation.cell(first_id))
        self._set_identity(second, self.simulation.cell(second_id))
        if self._divide is not None:
            self._divide(parent, first, second)
        self._validate_engine_owned_geometry(first, self.simulation.cell(first_id))
        self._validate_engine_owned_geometry(second, self.simulation.cell(second_id))
        self._validate_mutable_state(first)
        self._validate_mutable_state(second)
        self._apply_mutable_state(first)
        self._apply_mutable_state(second)
        del self._cells[parent_id]
        self._cells[first_id] = first
        self._cells[second_id] = second

    def _apply_division_jitter(self, cell_id: int) -> None:
        if self._division_jitter_z is None:
            return
        if self._rng is None:
            raise AssertionError("division jitter random stream is missing")
        snapshot = self.simulation.cell(cell_id)
        jitter = [self._rng.uniform(-0.001, 0.001) for _ in range(3)]
        if not self._division_jitter_z:
            jitter[2] = 0.0
        direction = Vec3(
            snapshot.direction.x + jitter[0],
            snapshot.direction.y + jitter[1],
            snapshot.direction.z + jitter[2],
        )
        self.simulation.set_cell_geometry(cell_id, snapshot.position, direction, snapshot.length)

    @staticmethod
    def _set_identity(cell: LegacyCell, snapshot: CellSnapshot) -> None:
        cell.id = snapshot.id
        cell.idx = snapshot.slot
        cell.pos = _vector(snapshot.position)
        cell.dir = _vector(snapshot.direction)
        cell.length = snapshot.length
        cell.volume = snapshot.length
        cell.oldLen = snapshot.length
        cell.ends = _ends(cell.pos, cell.dir, cell.length)

    @staticmethod
    def _validate_engine_owned_geometry(cell: LegacyCell, snapshot: CellSnapshot) -> None:
        actual = (
            tuple(float(value) for value in cell.pos),
            tuple(float(value) for value in cell.dir),
            float(cell.length),
            float(cell.radius),
        )
        expected = (
            tuple(_vector(snapshot.position)),
            tuple(_vector(snapshot.direction)),
            snapshot.length,
            snapshot.radius,
        )
        if actual != expected:
            raise LegacyCompatibilityError(
                "legacy callbacks may not mutate native position, direction, length, or radius"
            )

    def _validate_mutable_state(self, cell: LegacyCell) -> None:
        growth_rate = float(cell.growthRate)
        if not math.isfinite(growth_rate):
            raise LegacyCompatibilityError("legacy growthRate must be finite")
        cell_type = cast(object, cell.cellType)
        if not isinstance(cell_type, int):
            raise LegacyCompatibilityError("legacy cellType must be an integer")
        if len(cell.species) != self.simulation.species_count:
            raise LegacyCompatibilityError("legacy species count does not match the simulation")
        if not all(math.isfinite(float(value)) for value in cell.species):
            raise LegacyCompatibilityError("legacy species levels must be finite")

    def _apply_mutable_state(self, cell: LegacyCell) -> None:
        self._validate_mutable_state(cell)
        self.simulation.set_cell_attributes(cell.id, float(cell.growthRate), int(cell.cellType))
        if self.simulation.species_count != 0:
            self.simulation.set_species(cell.id, [float(value) for value in cell.species])

    def _refresh_cells(self) -> None:
        graph = self.simulation.find_cell_contacts() if self._compute_neighbors else None
        for snapshot in self.simulation.cells():
            cell = self._cells[snapshot.id]
            previous_position = cell.pos
            previous_length = cell.oldLen
            cell.idx = snapshot.slot
            cell.pos = _vector(snapshot.position)
            cell.dir = _vector(snapshot.direction)
            cell.length = snapshot.length
            cell.radius = snapshot.radius
            cell.growthRate = snapshot.growth_rate
            cell.cellType = snapshot.cell_type
            cell.species = list(snapshot.species)
            cell.signals = (
                self.simulation.sample_signals(snapshot.position)
                if self.simulation.has_signal_grid
                else []
            )
            cell.vel = [cell.pos[index] - previous_position[index] for index in range(3)]
            cell.strainRate = (
                (cell.length - previous_length) / previous_length if previous_length != 0.0 else 0.0
            )
            cell.effGrowth = (
                cell.effGrowth * cell.cellAge + cell.strainRate * previous_length
            ) / (cell.cellAge + 1)
            cell.cellAge += 1
            cell.oldLen = cell.length
            cell.volume = cell.length
            cell.ends = _ends(cell.pos, cell.dir, cell.length)
            cell.neighbours = list(graph.neighbor_ids(snapshot.slot)) if graph is not None else []
            cell.cts = len(cell.neighbours)
            cell.divideFlag = False

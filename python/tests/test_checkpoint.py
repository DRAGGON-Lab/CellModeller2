from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
from cellmodeller2 import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    BackendKind,
    CellInit,
    CheckpointError,
    GridShape,
    PlaneConstraintInit,
    RateInstruction,
    RateOp,
    SignalGridSpec,
    Simulation,
    SpeciesRatePlan,
    SphereConstraintInit,
    SphereRegion,
    Vec3,
    load_checkpoint,
    save_checkpoint,
)


def _instruction(
    operation: RateOp,
    *,
    first: int = 0,
    second: int = 0,
    third: int = 0,
    value: float = 0.0,
) -> RateInstruction:
    instruction = RateInstruction()
    instruction.operation = operation
    instruction.first = first
    instruction.second = second
    instruction.third = third
    instruction.value = value
    return instruction


def _make_simulation() -> tuple[Simulation, int, int]:
    simulation = Simulation(BackendKind.CPU, species_count=2)
    shape = GridShape()
    shape.x = 3
    shape.y = 1
    shape.z = 1
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.diffusion = [0.5]
    grid.advection = [Vec3()]
    simulation.configure_signal_grid(grid, [0.0, 1.0, 0.0])
    simulation.set_species_rate_plan(
        SpeciesRatePlan(
            2,
            [
                _instruction(RateOp.SPECIES, first=0),
                _instruction(RateOp.CONSTANT, value=0.125),
                _instruction(RateOp.ADD, first=0, second=1),
                _instruction(RateOp.SPECIES, first=1),
                _instruction(RateOp.NEGATE, first=3),
            ],
            [2, 4],
        )
    )
    first = CellInit()
    first.position = Vec3(1.25, -2.5, 0.75)
    first.direction = Vec3(1.0, 2.0, 3.0)
    first.length = 4.5
    first.radius = 0.4
    first.growth_rate = 0.2
    first.cell_type = 7
    first.species = [3.0, 1.5]
    first_id = simulation.add_cell(first)

    second = CellInit()
    second.position = Vec3(-0.5, 1.0, 2.0)
    second.length = 2.75
    second.radius = 0.3
    second.growth_rate = 0.05
    second.cell_type = -2
    second.species = [0.25, 4.0]
    simulation.add_cell(second)

    plane = PlaneConstraintInit()
    plane.point = Vec3(0.0, -3.0, 0.0)
    plane.inward_normal = Vec3(0.0, 2.0, 0.0)
    plane.coefficient = 1.25
    assert simulation.add_plane_constraint(plane) == 1

    sphere = SphereConstraintInit()
    sphere.center = Vec3(1.0, 2.0, 3.0)
    sphere.radius = 8.0
    sphere.coefficient = 0.75
    sphere.allowed_region = SphereRegion.INSIDE
    assert simulation.add_sphere_constraint(sphere) == 2

    simulation.step(0.125)
    daughter_a, daughter_b = simulation.divide_equal(first_id)
    simulation.step(0.03125)
    return simulation, daughter_a, daughter_b


def _assert_cells_exact(actual: Simulation, expected: Simulation) -> None:
    assert actual.time == expected.time
    assert actual.species_count == expected.species_count
    assert actual.signal_count == expected.signal_count
    assert actual.has_signal_grid == expected.has_signal_grid
    if actual.has_signal_grid:
        assert actual.signal_levels == expected.signal_levels
    actual_cells = actual.cells()
    expected_cells = expected.cells()
    assert len(actual_cells) == len(expected_cells)
    for left, right in zip(actual_cells, expected_cells, strict=True):
        assert left.id == right.id
        assert left.slot == right.slot
        assert (left.position.x, left.position.y, left.position.z) == (
            right.position.x,
            right.position.y,
            right.position.z,
        )
        assert (left.direction.x, left.direction.y, left.direction.z) == (
            right.direction.x,
            right.direction.y,
            right.direction.z,
        )
        assert left.length == right.length
        assert left.radius == right.radius
        assert left.growth_rate == right.growth_rate
        assert left.cell_type == right.cell_type
        assert left.species == right.species


def _document(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _rewrite_with_state_digest(path: Path, document: dict[str, Any]) -> None:
    simulation = document["simulation"]
    canonical = json.dumps(
        simulation,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    document["integrity"]["simulation"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")


def test_checkpoint_round_trip_resumes_exactly(tmp_path: Path) -> None:
    original, daughter_a, daughter_b = _make_simulation()
    path = tmp_path / "colony.cm2.json"
    save_checkpoint(
        original,
        path,
        provenance={"model": "two-species-test", "parameters": {"seed": 19}},
    )

    document = _document(path)
    assert document["format"] == CHECKPOINT_FORMAT
    assert document["version"] == CHECKPOINT_VERSION
    assert document["provenance"]["model"] == "two-species-test"
    assert document["source_backend"]["kind"] == "cpu"
    assert document["source_backend"]["device_index"] == 0
    assert "module_source" not in path.read_text(encoding="utf-8")
    assert list(tmp_path.glob(".*.tmp")) == []

    restored = load_checkpoint(path)
    _assert_cells_exact(restored, original)
    assert restored.lineage_parent(daughter_a) == 1
    assert restored.lineage_parent(daughter_b) == 1
    assert restored.signal_levels == original.signal_levels

    added = CellInit()
    added.species = [0.5, 0.75]
    assert restored.add_cell(added) == original.add_cell(added)
    plane = PlaneConstraintInit()
    assert restored.add_plane_constraint(plane) == original.add_plane_constraint(plane)
    restored.step(0.0625)
    original.step(0.0625)
    _assert_cells_exact(restored, original)


def test_version_one_checkpoint_migrates_to_an_empty_signal_state(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "legacy-v1.cm2.json"
    save_checkpoint(simulation, path)
    document = _document(path)
    document["version"] = 1
    del document["simulation"]["signal_grid"]
    _rewrite_with_state_digest(path, document)

    restored = load_checkpoint(path)
    assert not restored.has_signal_grid
    assert restored.signal_count == 0
    assert restored.time == simulation.time


def test_checkpoint_rejects_corruption_and_invalid_state(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "colony.cm2.json"
    save_checkpoint(simulation, path)

    corrupted = _document(path)
    corrupted["simulation"]["world"]["cells"][0]["length"] = 99.0
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(CheckpointError, match="digest"):
        load_checkpoint(path)

    save_checkpoint(simulation, path)
    invalid = _document(path)
    invalid["simulation"]["world"]["cells"][0]["slot"] = 7
    _rewrite_with_state_digest(path, invalid)
    with pytest.raises(CheckpointError, match="slots"):
        load_checkpoint(path)

    save_checkpoint(simulation, path)
    unsupported = _document(path)
    unsupported["version"] = CHECKPOINT_VERSION + 1
    path.write_text(json.dumps(unsupported), encoding="utf-8")
    with pytest.raises(CheckpointError, match="unsupported checkpoint version"):
        load_checkpoint(path)

    save_checkpoint(simulation, path)
    unknown = _document(path)
    unknown["simulation"]["world"]["mystery"] = 1
    _rewrite_with_state_digest(path, unknown)
    with pytest.raises(CheckpointError, match="unknown keys"):
        load_checkpoint(path)


def test_checkpoint_rejects_executable_or_non_json_values(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "colony.cm2.json"
    with pytest.raises(CheckpointError, match="provenance"):
        save_checkpoint(simulation, path, provenance={"callback": object()})  # type: ignore[dict-item]
    assert not path.exists()

    path.write_text('{"format":"cellmodeller2-checkpoint","value":NaN}', encoding="utf-8")
    with pytest.raises(CheckpointError, match="non-finite"):
        load_checkpoint(path)

    path.write_text('{"format":"first","format":"second"}', encoding="utf-8")
    with pytest.raises(CheckpointError, match="duplicate key"):
        load_checkpoint(path)

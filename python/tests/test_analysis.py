# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportIndexIssue=false, reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import pytest
import zarr
from cellmodeller2 import (
    BackendKind,
    CellInit,
    GridShape,
    PlaneConstraintInit,
    SignalGridSpec,
    Simulation,
    Vec3,
    save_checkpoint,
)
from cellmodeller2.analysis import (
    ANALYSIS_FORMAT,
    ANALYSIS_VERSION,
    AnalysisError,
    export_dataset,
)
from cellmodeller2.cli import main


def _simulation() -> tuple[Simulation, int]:
    simulation = Simulation(BackendKind.CPU, species_count=2)
    shape = GridShape()
    shape.x = 2
    shape.y = 1
    shape.z = 1
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.diffusion = [0.0]
    grid.advection = [Vec3()]
    simulation.configure_signal_grid(grid, [1.0, 2.0])

    first = CellInit()
    first.position = Vec3(0.0, 0.25, 0.0)
    first.direction = Vec3(1.0, 0.0, 0.0)
    first.length = 2.0
    first.radius = 0.5
    first.growth_rate = 0.1
    first.cell_type = -3
    first.fixed = True
    first.species = [2.0, 3.0]
    parent = simulation.add_cell(first)

    second = CellInit()
    second.position = Vec3(0.0, 0.25, 0.8)
    second.direction = Vec3(1.0, 0.0, 0.0)
    second.length = 2.0
    second.radius = 0.5
    second.species = [5.0, 7.0]
    simulation.add_cell(second)

    plane = PlaneConstraintInit()
    plane.point = Vec3(0.0, 0.0, 0.0)
    plane.inward_normal = Vec3(0.0, 1.0, 0.0)
    simulation.add_plane_constraint(plane)

    return simulation, parent


def _manifest(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((path / "manifest.json").read_text()))


def test_export_dataset_preserves_typed_state_contacts_and_signals(tmp_path: Path) -> None:
    simulation, parent = _simulation()
    first = tmp_path / "frame-0000.cm2.json"
    second = tmp_path / "frame-0001.cm2.json"
    save_checkpoint(simulation, first, provenance={"run_id": "analysis-test"})
    simulation.step(0.25)
    daughters = simulation.divide_equal(parent)
    simulation.set_signal_levels([4.0, 8.0])
    save_checkpoint(simulation, second, provenance={"run_id": "analysis-test"})

    output = tmp_path / "run.cm2.dataset"
    summary = export_dataset(
        [first, second],
        output,
        include_contacts=True,
        include_external_contacts=True,
    )

    assert summary.output == output
    assert summary.frame_count == 2
    assert summary.cell_rows == 5
    assert summary.species_rows == 10
    assert summary.contact_rows > 0
    assert summary.external_contact_rows > 0
    assert summary.signal_epochs == 1

    manifest = _manifest(output)
    assert manifest["format"] == ANALYSIS_FORMAT
    assert manifest["version"] == ANALYSIS_VERSION
    assert len(manifest["dataset_id"]) == 64
    assert [source["path"] for source in manifest["sources"]] == [first.name, second.name]
    assert str(tmp_path) not in (output / "manifest.json").read_text()
    assert manifest["options"]["contact_conformance"] == "cpu_reference"
    assert manifest["tables"]["cells.parquet"]["rows"] == 5

    frames_table = pq.read_table(output / "frames.parquet")
    frames = frames_table.to_pylist()
    assert str(frames_table.schema.field("frame_index").type) == "uint32"
    assert str(frames_table.schema.field("time").type) == "double"
    assert [row["time"] for row in frames] == [0.0, 0.25]
    assert [row["cell_count"] for row in frames] == [2, 3]
    assert all(row["source_backend_kind"] == "cpu" for row in frames)
    assert all(row["reconstruction_backend_kind"] == "cpu" for row in frames)

    cells_table = pq.read_table(output / "cells.parquet")
    cells = cells_table.to_pylist()
    assert str(cells_table.schema.field("id").type) == "uint64"
    assert str(cells_table.schema.field("parent_id").type) == "uint64"
    assert cells_table.schema.field("parent_id").nullable
    assert str(cells_table.schema.field("cylinder_length").type) == "float"
    assert str(cells_table.schema.field("cell_type").type) == "int32"
    initial_parent = next(row for row in cells if row["frame_index"] == 0 and row["id"] == parent)
    assert initial_parent["cylinder_length"] == 2.0
    assert initial_parent["capsule_length"] == 3.0
    assert initial_parent["cell_type"] == -3
    assert initial_parent["fixed"] is True
    children = [row for row in cells if row["id"] in daughters]
    assert {row["parent_id"] for row in children} == {parent}

    species = pq.read_table(output / "species.parquet").to_pylist()
    assert species[:4] == [
        {"frame_index": 0, "cell_id": 1, "channel": 0, "level": 2.0},
        {"frame_index": 0, "cell_id": 1, "channel": 1, "level": 3.0},
        {"frame_index": 0, "cell_id": 2, "channel": 0, "level": 5.0},
        {"frame_index": 0, "cell_id": 2, "channel": 1, "level": 7.0},
    ]
    contacts = pq.read_table(output / "contacts.parquet").to_pylist()
    assert all(row["overlap"] == max(0.0, -row["signed_separation"]) for row in contacts)
    external = pq.read_table(output / "external_contacts.parquet").to_pylist()
    assert {row["constraint_kind"] for row in external} == {"plane"}
    assert {row["endpoint"] for row in external} == {"negative", "positive"}

    signals = zarr.open_group(output / "signals.zarr", mode="r")
    epoch = signals["epoch-0000"]
    assert epoch["levels"].shape == (2, 1, 2, 1, 1)
    assert epoch["levels"][:].tolist() == [[[[[1.0]], [[2.0]]]], [[[[4.0]], [[8.0]]]]]
    assert epoch["frame_index"][:].tolist() == [0, 1]
    assert epoch["time"][:].tolist() == [0.0, 0.25]

    repeated = tmp_path / "repeated.cm2.dataset"
    repeated_summary = export_dataset(
        [first, second],
        repeated,
        include_contacts=True,
        include_external_contacts=True,
    )
    assert repeated_summary.dataset_id == summary.dataset_id
    assert _manifest(repeated) == manifest


def test_export_dataset_rejects_existing_output_and_reverse_time(tmp_path: Path) -> None:
    simulation, _ = _simulation()
    later = tmp_path / "later.cm2.json"
    earlier = tmp_path / "earlier.cm2.json"
    simulation.step(0.5)
    save_checkpoint(simulation, later)
    pristine, _ = _simulation()
    save_checkpoint(pristine, earlier)

    with pytest.raises(AnalysisError, match="before prior time"):
        export_dataset([later, earlier], tmp_path / "reverse.cm2.dataset")

    output = tmp_path / "existing.cm2.dataset"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("mine")
    with pytest.raises(AnalysisError, match="output already exists"):
        export_dataset([earlier], output)
    assert sentinel.read_text() == "mine"

    replaced = export_dataset([earlier], output, replace=True)
    assert replaced.output == output
    assert not sentinel.exists()
    assert (output / "manifest.json").is_file()


def test_signal_geometry_changes_start_a_new_epoch(tmp_path: Path) -> None:
    simulation, _ = _simulation()
    first = tmp_path / "first.cm2.json"
    second = tmp_path / "second.cm2.json"
    save_checkpoint(simulation, first)
    changed_simulation = Simulation(BackendKind.CPU, species_count=2)
    shape = GridShape()
    shape.x = 3
    shape.y = 1
    shape.z = 1
    changed = SignalGridSpec()
    changed.signal_count = 1
    changed.shape = shape
    changed.spacing = Vec3(0.5, 1.0, 1.0)
    changed.diffusion = [0.0]
    changed.advection = [Vec3()]
    changed_simulation.configure_signal_grid(changed, [1.0, 2.0, 3.0])
    save_checkpoint(changed_simulation, second)

    output = tmp_path / "epochs.cm2.dataset"
    summary = export_dataset([first, second], output)

    assert summary.signal_epochs == 2
    signals = zarr.open_group(output / "signals.zarr", mode="r")
    assert signals["epoch-0000"]["levels"].shape == (1, 1, 2, 1, 1)
    assert signals["epoch-0001"]["levels"].shape == (1, 1, 3, 1, 1)


def test_cuda_contact_derivation_stays_gated(tmp_path: Path) -> None:
    simulation, _ = _simulation()
    checkpoint = tmp_path / "frame.cm2.json"
    save_checkpoint(simulation, checkpoint)

    with pytest.raises(AnalysisError, match="NVIDIA hardware conformance"):
        export_dataset(
            [checkpoint],
            tmp_path / "cuda.cm2.dataset",
            backend=BackendKind.CUDA,
            include_contacts=True,
        )


def test_cli_exports_an_analysis_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    simulation, _ = _simulation()
    checkpoint = tmp_path / "frame.cm2.json"
    output = tmp_path / "cli.cm2.dataset"
    save_checkpoint(simulation, checkpoint)

    status = main(["export-analysis", str(checkpoint), "--output", str(output)])

    assert status == 0
    assert (output / "manifest.json").is_file()
    assert "frames=1 cells=2 signal_epochs=1" in capsys.readouterr().out

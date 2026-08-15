from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
from cellmodeller2 import (
    BackendKind,
    BatchError,
    ModelContext,
    build_model,
    load_checkpoint,
    run_simulation,
)
from cellmodeller2.cli import main


def _write_model(path: Path) -> None:
    path.write_text(
        """from cellmodeller2 import CellInit, Vec3

def build(context):
    simulation = context.simulation()
    cell = CellInit()
    cell.position = Vec3(context.rng.random(), 0.0, 0.0)
    cell.length = float(context.parameters.get("length", 2.0))
    cell.growth_rate = 0.25
    cell.cell_type = int(context.parameters.get("cell_type", 0))
    simulation.add_cell(cell)
    return simulation
""",
        encoding="utf-8",
    )


def _document(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_batch_library_is_deterministic_and_preflights_outputs(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    _write_model(model)
    context = ModelContext(
        backend=BackendKind.CPU,
        device_index=0,
        seed=1234,
        parameters={"length": 4.0, "cell_type": 7},
    )
    simulation, provenance = build_model(model, context)
    output = tmp_path / "run.cm2.json"
    progress_steps: list[int] = []
    summary = run_simulation(
        simulation,
        steps=3,
        dt=0.1,
        output=output,
        checkpoint_every=2,
        provenance=provenance,
        progress=lambda progress: progress_steps.append(progress.completed_steps),
    )

    assert summary.output == output
    assert summary.periodic_checkpoints == (tmp_path / "run.step-00000002.cm2.json",)
    assert summary.periodic_checkpoints[0].exists()
    assert progress_steps == [1, 2, 3]
    restored = load_checkpoint(output)
    assert math.isclose(restored.time, simulation.time)
    assert restored.cell(1).cell_type == 7
    assert restored.cell(1).position.x == simulation.cell(1).position.x
    assert restored.cell(1).length == simulation.cell(1).length

    document = _document(output)
    assert document["provenance"]["model"]["seed"] == 1234
    assert document["provenance"]["model"]["sha256"] == hashlib.sha256(
        model.read_bytes()
    ).hexdigest()
    assert document["provenance"]["run"] == {
        "completed_steps": 3,
        "dt": 0.1,
        "requested_steps": 3,
        "status": "complete",
    }

    original_bytes = output.read_bytes()
    with pytest.raises(BatchError, match="already exists"):
        run_simulation(simulation, steps=0, dt=0.1, output=output)
    assert output.read_bytes() == original_bytes

    second, second_provenance = build_model(
        model,
        ModelContext(BackendKind.CPU, 0, 1234, {"length": 4.0, "cell_type": 7}),
    )
    second_output = tmp_path / "second.cm2.json"
    run_simulation(
        second,
        steps=3,
        dt=0.1,
        output=second_output,
        provenance=second_provenance,
    )
    assert _document(second_output)["simulation"] == document["simulation"]


def test_cli_runs_models_resumes_and_lists_devices(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model.py"
    _write_model(model)
    first = tmp_path / "first.cm2.json"
    status = main(
        [
            "run",
            "--model",
            str(model),
            "--backend",
            "cpu",
            "--device-index",
            "0",
            "--seed",
            "99",
            "--parameter",
            "length=3.5",
            "--steps",
            "2",
            "--dt",
            "0.25",
            "--output",
            str(first),
            "--quiet",
        ]
    )
    assert status == 0
    assert first.exists()
    assert "wrote" in capsys.readouterr().out

    resumed = tmp_path / "resumed.cm2.json"
    status = main(
        [
            "run",
            "--resume",
            str(first),
            "--steps",
            "2",
            "--dt",
            "0.25",
            "--output",
            str(resumed),
            "--quiet",
        ]
    )
    assert status == 0
    assert math.isclose(load_checkpoint(resumed).time, 1.0)
    resume_document = _document(resumed)
    assert resume_document["provenance"]["resume"]["sha256"] == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    capsys.readouterr()

    status = main(["devices", "--json"])
    assert status == 0
    devices = cast(list[dict[str, Any]], json.loads(capsys.readouterr().out))
    cpu = next(record for record in devices if record["backend"] == "cpu")
    assert cpu == {
        "available": True,
        "backend": "cpu",
        "devices": [{"index": 0, "name": "host"}],
    }


def test_cli_rejects_bad_parameters_without_creating_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model.py"
    _write_model(model)
    output = tmp_path / "run.cm2.json"
    status = main(
        [
            "run",
            "--model",
            str(model),
            "--parameter",
            "length=NaN",
            "--steps",
            "1",
            "--dt",
            "0.1",
            "--output",
            str(output),
            "--quiet",
        ]
    )
    assert status == 2
    assert "contains NaN" in capsys.readouterr().err
    assert not output.exists()

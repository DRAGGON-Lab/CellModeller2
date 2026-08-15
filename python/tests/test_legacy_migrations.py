from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest
from cellmodeller2 import (
    BackendKind,
    ModelContext,
    NativeController,
    backend_available,
    build_model,
    load_checkpoint_bundle,
    run_simulation,
)

_ROOT = Path(__file__).resolve().parents[2]
_MODELS = (
    ("Tutorial_2/Tutorial_2a.py", (0.02,)),
    ("Tutorial_2/Tutorial_2b.py", (0.02, 0.02)),
    ("ex2_constGene.py", (0.01,)),
    ("ex2a_dilution.py", None),
    ("ex2b_diluteRepression.py", None),
)


def _path(relative: str) -> Path:
    return _ROOT / "examples" / "legacy" / relative


def _simulation_payload(path: Path) -> object:
    document = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    return document["simulation"]


@pytest.mark.parametrize("backend", list(BackendKind))
@pytest.mark.parametrize(("relative", "expected_species"), _MODELS)
def test_species_migrations_run_typed_equations_and_division(
    backend: BackendKind,
    relative: str,
    expected_species: tuple[float, ...] | None,
) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    model, _ = build_model(_path(relative), ModelContext(backend, 0, seed=11))
    assert isinstance(model, NativeController)

    model.step(0.01)

    cells = model.simulation.cells()
    assert len(cells) == 2
    assert len(model.last_mechanics_reports) == 1
    first = cells[0]
    if expected_species is not None:
        assert len(first.species) == len(expected_species)
        for actual, expected in zip(first.species, expected_species, strict=True):
            assert math.isclose(actual, expected, rel_tol=2.0e-5, abs_tol=1.0e-7)
    else:
        dilution = 2.25 / (first.length + 1.0)
        expected_x0 = 10.0 * dilution
        assert math.isclose(first.species[0], expected_x0, rel_tol=2.0e-5)
        if relative == "ex2b_diluteRepression.py":
            expected_x1 = 0.01 * 4.0 / (4.0 + expected_x0 * expected_x0)
            assert math.isclose(first.species[1], expected_x1, rel_tol=2.0e-5)


@pytest.mark.parametrize(("relative", "_"), _MODELS)
def test_species_migrations_resume_exactly(relative: str, _: object, tmp_path: Path) -> None:
    path = _path(relative)
    uninterrupted, provenance = build_model(
        path,
        ModelContext(BackendKind.CPU, 0, seed=73),
    )
    assert isinstance(uninterrupted, NativeController)
    expected_path = tmp_path / "expected.cm2.json"
    run_simulation(
        uninterrupted,
        steps=2,
        dt=0.01,
        output=expected_path,
        provenance=provenance,
    )

    split, provenance = build_model(
        path,
        ModelContext(BackendKind.CPU, 0, seed=73),
    )
    midpoint = tmp_path / "midpoint.cm2.json"
    run_simulation(split, steps=1, dt=0.01, output=midpoint, provenance=provenance)
    resumed, resumed_provenance = build_model(
        path,
        ModelContext(BackendKind.CPU, 0, seed=73),
        checkpoint=load_checkpoint_bundle(midpoint),
    )
    actual_path = tmp_path / "actual.cm2.json"
    run_simulation(
        resumed,
        steps=1,
        dt=0.01,
        output=actual_path,
        provenance=resumed_provenance,
    )

    expected = load_checkpoint_bundle(expected_path)
    actual = load_checkpoint_bundle(actual_path)
    assert _simulation_payload(actual_path) == _simulation_payload(expected_path)
    assert actual.controller == expected.controller

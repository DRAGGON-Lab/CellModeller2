from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from cellmodeller2 import (
    BackendKind,
    LegacyCompatibilityError,
    ModelContext,
    backend_available,
    build_legacy_model,
)
from cellmodeller2.checkpoint import JSONValue

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("backend", list(BackendKind))
def test_unchanged_growth_model_loads_through_setup_facade(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    context = ModelContext(backend, 0, seed=42)
    model, provenance = build_legacy_model(_FIXTURES / "legacy_growth.py", context)

    assert list(model.cells) == [1]
    founder = model.cells[1]
    assert founder.cellType == 2
    assert founder.color.tolist() == [0.1, 0.2, 0.3]
    assert 4.1 <= founder.targetVol <= 4.2
    model_provenance = cast(dict[str, JSONValue], provenance["model"])
    assert model_provenance["compatibility"] == "legacy-python-callbacks-v1"

    model.step(0.2)
    model.step(0.0)
    assert list(model.cells) == [2, 3]
    assert all(cell.dir[2] == 0.0 for cell in model.cells.values())
    assert model.simulation.lineage_parent(2) == 1
    assert model.simulation.lineage_parent(3) == 1


def test_legacy_loader_rejects_opencl_integrators_explicitly() -> None:
    context = ModelContext(BackendKind.CPU, 0, seed=0)
    with pytest.raises(LegacyCompatibilityError, match="OpenCL integrators"):
        build_legacy_model(_FIXTURES / "legacy_opencl_integrator.py", context)

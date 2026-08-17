from __future__ import annotations

import math
from pathlib import Path

import pytest
from cellmodeller2.masks import (
    MaskError,
    MaskPolyline,
    extract_rectangles,
    load_mask_polylines,
    match_rectangles,
)

_PRINDLE = Path(__file__).resolve().parents[2] / "docs" / "tutorials" / "devices" / "prindle.dxf"


def test_prindle_mask_yields_the_biopixel_trap_array() -> None:
    polylines = load_mask_polylines(_PRINDLE)
    rectangles = extract_rectangles(polylines, layer="Layer-2", unit_scale=1000.0)
    traps = match_rectangles(rectangles, 110.0, 100.0, tolerance=1.0)

    assert len(traps) == 496

    xs = sorted({round(trap.center[0], 1) for trap in traps})
    ys = sorted({round(trap.center[1], 1) for trap in traps})
    assert len(xs) == 16
    assert len(ys) == 31
    assert math.isclose(xs[1] - xs[0], 172.5, abs_tol=0.1)
    assert math.isclose(ys[1] - ys[0], 125.0, abs_tol=0.1)
    assert math.isclose(xs[-1] - xs[0], 2400.0, abs_tol=1.0)
    assert math.isclose(ys[-1] - ys[0], 3750.0, abs_tol=1.0)


def test_rectangle_extraction_is_selective() -> None:
    polylines = (
        MaskPolyline("A", True, ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))),
        MaskPolyline("A", False, ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))),
        MaskPolyline("A", True, ((0.0, 0.0), (2.0, 0.5), (2.0, 1.0), (0.0, 1.0))),
        MaskPolyline("B", True, ((0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (0.0, 2.0), (0.0, 0.0))),
    )
    rectangles = extract_rectangles(polylines, unit_scale=10.0)
    assert len(rectangles) == 2
    assert math.isclose(rectangles[0].width, 20.0)
    assert rectangles[0].center == (10.0, 5.0)

    layered = extract_rectangles(polylines, layer="B", unit_scale=10.0)
    assert len(layered) == 1
    assert math.isclose(layered[0].height, 20.0)

    rotated = match_rectangles(rectangles, 10.0, 20.0, tolerance=0.01)
    assert len(rotated) == 2
    strict = match_rectangles(rectangles, 10.0, 20.0, tolerance=0.01, allow_rotated=False)
    assert len(strict) == 1


def test_mask_reader_rejects_unusable_input(tmp_path: Path) -> None:
    empty = tmp_path / "empty.dxf"
    empty.write_text("")
    with pytest.raises(MaskError, match="empty"):
        load_mask_polylines(empty)

    no_entities = tmp_path / "no-entities.dxf"
    no_entities.write_text("  0\nSECTION\n  2\nHEADER\n  0\nENDSEC\n  0\nEOF\n")
    with pytest.raises(MaskError, match="no model-space polylines"):
        load_mask_polylines(no_entities)

    with pytest.raises(MaskError, match="byte limit"):
        load_mask_polylines(_PRINDLE, max_bytes=1024)


def test_block_traversal_reads_the_flow_layer() -> None:
    polylines = load_mask_polylines(_PRINDLE, include_blocks=True)
    blocks = {p.block for p in polylines if p.block is not None}
    assert len(blocks) >= 2

    layer5 = [p for p in polylines if p.layer == "Layer-5" and p.block is not None]
    assert len(layer5) > 3000
    supply = [
        p
        for p in layer5
        if min(
            max(v[0] for v in p.vertices) - min(v[0] for v in p.vertices),
            max(v[1] for v in p.vertices) - min(v[1] for v in p.vertices),
        )
        >= 0.9
    ]
    assert len(supply) >= 40

    modelspace_only = load_mask_polylines(_PRINDLE)
    assert all(p.block is None for p in modelspace_only)

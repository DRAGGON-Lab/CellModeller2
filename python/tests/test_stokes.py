from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from cellmodeller2 import (
    GridBoundaryKind,
    GridShape,
    SignalGridSpec,
    Vec3,
)
from cellmodeller2.flow import FlowError, gap_mobility, solve_flow_field
from cellmodeller2.stokes import colony_drag, solve_stokes_field


def _duct(
    nx: int,
    ny: int,
    nz: int,
    spacing: tuple[float, float, float],
) -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = nx, ny, nz
    spec = SignalGridSpec()
    spec.signal_count = 1
    spec.shape = shape
    spec.spacing = Vec3(*spacing)
    spec.diffusion = [1.0]
    spec.advection = [Vec3()]
    for name in ("y_lower", "y_upper"):
        boundary = getattr(spec, name)
        boundary.kind = GridBoundaryKind.FIXED
        boundary.values = [0.0]
        setattr(spec, name, boundary)
    return spec


def _site(spec: SignalGridSpec, x: int, y: int, z: int) -> int:
    return (x * spec.shape.y + y) * spec.shape.z + z


def _plane_poiseuille_error(nx: int) -> float:
    spec = _duct(nx, 6, 1, (1.0 / nx, 0.25, 1.0))
    field, _ = solve_stokes_field(spec, mean_inlet_speed=1.0, tolerance=1.0e-10)
    profile = np.asarray(field.y_faces).reshape(nx, 7, 1)[:, 3, 0]
    positions = (np.arange(nx) + 0.5) / nx
    exact = 6.0 * positions * (1.0 - positions)
    return float(np.max(np.abs(profile - exact)) / np.max(exact))


def test_plane_poiseuille_profile_converges_at_second_order() -> None:
    coarse = _plane_poiseuille_error(8)
    fine = _plane_poiseuille_error(16)
    assert coarse < 0.02
    assert fine < 0.005
    assert 3.0 < coarse / fine < 5.0


def test_square_duct_peak_to_mean_matches_shah_and_london() -> None:
    # u_max / u_mean = 2.0962 for a square duct (Shah & London 1978).
    n = 16
    spec = _duct(n, 6, n, (1.0 / n, 0.25, 1.0 / n))
    field, report = solve_stokes_field(spec, mean_inlet_speed=1.0, tolerance=1.0e-9)
    cross = np.asarray(field.y_faces).reshape(n, 7, n)[:, 3, :]
    ratio = float(cross.max() / cross.mean())
    assert abs(ratio - 2.0962) / 2.0962 < 0.025
    assert report.divergence_rms < 1.0e-6


def _exact_two_layer(
    drag: float, positions: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    # Unit channel, open below one half, Brinkman drag above; unit pressure
    # gradient. Exact ODE solution: quadratic in the open layer, cosh/sinh in
    # the porous layer, matched in value and slope at the interface.
    root = math.sqrt(drag)
    matrix = np.array(
        [
            [0.0, math.cosh(root), math.sinh(root)],
            [0.5, -math.cosh(root * 0.5), -math.sinh(root * 0.5)],
            [1.0, -root * math.sinh(root * 0.5), -root * math.cosh(root * 0.5)],
        ]
    )
    rhs = np.array([-1.0 / drag, 1.0 / drag + 0.125, 0.5])
    linear, cosh_c, sinh_c = (float(value) for value in np.linalg.solve(matrix, rhs))
    result: npt.NDArray[np.float64] = np.where(
        positions < 0.5,
        -positions * positions / 2.0 + linear * positions,
        1.0 / drag + cosh_c * np.cosh(root * positions) + sinh_c * np.sinh(root * positions),
    )
    return result


def test_two_layer_brinkman_channel_matches_the_exact_solution() -> None:
    drag_value = 200.0
    nz = 32
    spec = _duct(1, 6, nz, (1.0, 0.25, 1.0 / nz))
    drag = [
        0.0 if (z + 0.5) / nz < 0.5 else drag_value
        for _ in range(1)
        for _ in range(6)
        for z in range(nz)
    ]
    field, _ = solve_stokes_field(spec, mean_inlet_speed=1.0, drag=drag, tolerance=1.0e-9)
    profile = np.asarray(field.y_faces).reshape(1, 7, nz)[0, 3, :]
    positions = (np.arange(nz) + 0.5) / nz
    exact = _exact_two_layer(drag_value, positions)
    exact = exact / exact.mean() * profile.mean()
    error = float(np.max(np.abs(profile - exact)) / np.max(np.abs(exact)))
    assert error < 0.01
    # The open layer carries several times the porous layer's flux.
    open_flux = float(profile[positions < 0.5].sum())
    porous_flux = float(profile[positions >= 0.5].sum())
    assert open_flux / porous_flux > 3.0


def test_zero_drag_recovers_pure_stokes() -> None:
    spec = _duct(8, 6, 1, (0.125, 0.25, 1.0))
    plain, _ = solve_stokes_field(spec, mean_inlet_speed=2.0)
    dragged, _ = solve_stokes_field(spec, mean_inlet_speed=2.0, drag=[0.0] * (8 * 6))
    assert plain.y_faces == dragged.y_faces


def test_stokes_field_is_engine_valid_and_conservative_around_a_pillar() -> None:
    spec = _duct(9, 12, 1, (1.0, 1.0, 1.0))
    obstacles = [0] * (9 * 12)
    for y in (5, 6):
        for x in (4, 5):
            obstacles[_site(spec, x, y, 0)] = 1
    spec.obstacles = obstacles
    field, report = solve_stokes_field(spec, mean_inlet_speed=6.0, tolerance=1.0e-9)
    spec.velocity_field = field
    spec.validate()
    assert report.divergence_rms < 1.0e-6

    def y_face(x: int, fy: int) -> float:
        return field.y_faces[x * 13 + fy]

    fluxes = [sum(y_face(x, fy) for x in range(9)) for fy in range(13)]
    for flux in fluxes[1:]:
        assert math.isclose(flux, fluxes[0], rel_tol=1.0e-5)
    assert y_face(4, 6) == 0.0
    assert y_face(1, 6) > 6.0


def test_thin_gap_stokes_depth_averages_to_the_hele_shaw_solution() -> None:
    # A shallow channel with a pillar: depth-averaging the resolved MAC field
    # must reproduce the Hele-Shaw flux split around the pillar.
    nx, ny, nz = 6, 10, 6
    spec = _duct(nx, ny, nz, (1.0, 1.0, 0.05))
    obstacles = [0] * (nx * ny * nz)
    for y in (4, 5):
        for x in (1, 2):
            for z in range(nz):
                obstacles[_site(spec, x, y, z)] = 1
    spec.obstacles = obstacles

    stokes_field, _ = solve_stokes_field(spec, mean_inlet_speed=1.0, tolerance=1.0e-9)
    hele_shaw_field, _ = solve_flow_field(
        spec, mean_inlet_speed=1.0, mobility=gap_mobility(spec)
    )

    def column_flux(field_values: list[float], x: int, fy: int) -> float:
        return sum(field_values[(x * (ny + 1) + fy) * nz + z] for z in range(nz))

    mid = ny // 2
    stokes_split = [column_flux(stokes_field.y_faces, x, mid) for x in range(nx)]
    hele_shaw_split = [column_flux(hele_shaw_field.y_faces, x, mid) for x in range(nx)]
    stokes_total = sum(stokes_split)
    hele_shaw_total = sum(hele_shaw_split)
    for x in range(nx):
        assert math.isclose(
            stokes_split[x] / stokes_total,
            hele_shaw_split[x] / hele_shaw_total,
            abs_tol=0.05,
        )


def test_ill_posed_stokes_problems_are_rejected() -> None:
    spec = _duct(4, 6, 1, (1.0, 1.0, 1.0))
    with pytest.raises(FlowError, match="one of x, y, z"):
        solve_stokes_field(spec, mean_inlet_speed=1.0, axis="w")
    with pytest.raises(FlowError, match="finite and nonzero"):
        solve_stokes_field(spec, mean_inlet_speed=0.0)
    with pytest.raises(FlowError, match="must be FIXED"):
        solve_stokes_field(spec, mean_inlet_speed=1.0, axis="x")
    with pytest.raises(FlowError, match="one value per grid site"):
        solve_stokes_field(spec, mean_inlet_speed=1.0, drag=[1.0])

    blocked = _duct(3, 4, 1, (1.0, 1.0, 1.0))
    obstacles = [0] * 12
    for x in range(3):
        obstacles[_site(blocked, x, 2, 0)] = 1
    blocked.obstacles = obstacles
    with pytest.raises(FlowError, match="no through-flow"):
        solve_stokes_field(blocked, mean_inlet_speed=1.0)


def test_colony_drag_rasterizes_the_colony() -> None:
    spec = _duct(3, 3, 1, (4.0, 4.0, 4.0))
    obstacles = [0] * 9
    obstacles[_site(spec, 2, 2, 0)] = 1
    spec.obstacles = obstacles

    class _Rod:
        def __init__(self, x: float, y: float) -> None:
            self.position = Vec3(x, y, 0.0)
            self.length = 3.0
            self.radius = 0.5

    crowd = [_Rod(0.0, 0.0) for _ in range(40)]
    drag = colony_drag(spec, crowd, drag_coefficient=50.0)
    packed = drag[_site(spec, 0, 0, 0)]
    empty = drag[_site(spec, 1, 1, 0)]
    solid = drag[_site(spec, 2, 2, 0)]
    assert packed > 0.0
    assert empty == 0.0
    assert solid == 0.0
    assert math.isclose(packed, 50.0 * 0.9**2 / (1.0 - 0.9) ** 3, rel_tol=1.0e-9)
    with pytest.raises(FlowError, match="finite and non-negative"):
        colony_drag(spec, [], drag_coefficient=-1.0)

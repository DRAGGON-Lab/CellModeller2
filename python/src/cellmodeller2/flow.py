"""Steady Hele-Shaw-Brinkman flow solve for device grids.

The solver computes the depth-averaged Darcy-Brinkman pressure problem
``div(m grad p) = 0`` over the fluid voxels of a signal grid and returns the
face fluxes ``v = -m_face * dp/dn`` as a face-staggered velocity field. The
per-voxel mobility ``m`` carries the physics: uniform mobility is the Stokes
limit of the closure and resolves flow through arbitrary mask geometry, while
reduced mobility inside a colony (`colony_mobility`) adds Brinkman drag so a
packed trap diverts flow. Mobility is relative - the linear solution is
rescaled to a requested mean inlet speed - so callers never handle pressure or
viscosity units. Discrete conservation and zero velocity on closed faces hold
by construction, and the returned field passes the engine's grid validation
unchanged.

Pressure is fixed on the fluid boundary faces of the flow axis (inlet one,
outlet zero) and every other exterior face carries no flux; the flow axis
boundaries must therefore be `FIXED` and no axis may be periodic. The discrete
operator is symmetric positive definite and is solved matrix-free with
Jacobi-preconditioned conjugate gradient. Side-wall boundary layers, whose
thickness is on the order of the gap height, are outside the closure.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ._core import GridBoundaryKind, SignalGridSpec, SignalGridVelocityField, Vec3

_FloatGrid = NDArray[np.float64]

_AXES = {"x": 0, "y": 1, "z": 2}


class FlowError(ValueError):
    """Raised when a flow problem is ill-posed or its solve fails."""


@dataclass(frozen=True, slots=True)
class FlowSolveReport:
    iterations: int
    residual: float
    mean_inlet_speed: float
    max_speed: float


class _RodLike(Protocol):
    @property
    def position(self) -> Vec3: ...
    @property
    def length(self) -> float: ...
    @property
    def radius(self) -> float: ...


def _slices(axis: int, along: slice) -> tuple[slice, slice, slice]:
    index: list[slice] = [slice(None), slice(None), slice(None)]
    index[axis] = along
    return index[0], index[1], index[2]


def _mobility_grid(spec: SignalGridSpec, mobility: Sequence[float] | None) -> _FloatGrid:
    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    sites = dims[0] * dims[1] * dims[2]
    if mobility is None:
        values = np.ones(dims, dtype=np.float64)
    else:
        if len(mobility) != sites:
            raise FlowError("mobility must hold one value per grid site")
        values = np.asarray(mobility, dtype=np.float64).reshape(dims)
        if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0.0)):
            raise FlowError("mobility values must be finite and non-negative")
    obstacles = spec.obstacles
    if obstacles:
        if len(obstacles) != sites:
            raise FlowError("obstacles must hold one flag per grid site")
        solid = np.asarray(obstacles, dtype=np.uint8).reshape(dims) != 0
        values[solid] = 0.0
    return values


def _harmonic_faces(mobility: _FloatGrid, axis: int, spacing: float) -> _FloatGrid:
    lower = mobility[_slices(axis, slice(None, -1))]
    upper = mobility[_slices(axis, slice(1, None))]
    total = lower + upper
    product = 2.0 * lower * upper
    return np.divide(
        product,
        total,
        out=np.zeros_like(total),
        where=total > 0.0,
    ) / (spacing * spacing)


def _conjugate_gradient(
    apply_operator: Callable[[_FloatGrid], _FloatGrid],
    rhs: _FloatGrid,
    diagonal: _FloatGrid,
    tolerance: float,
    max_iterations: int,
) -> tuple[_FloatGrid, int, float]:
    solution = np.zeros_like(rhs)
    residual = rhs.copy()
    rhs_norm = float(np.sqrt(np.sum(rhs * rhs)))
    if rhs_norm == 0.0:
        return solution, 0, 0.0
    scale = np.where(diagonal > 0.0, diagonal, 1.0)
    preconditioned = residual / scale
    direction = preconditioned.copy()
    rho = float(np.sum(residual * preconditioned))
    relative = 1.0
    for iteration in range(1, max_iterations + 1):
        transformed = apply_operator(direction)
        curvature = float(np.sum(direction * transformed))
        if curvature <= 0.0:
            break
        alpha = rho / curvature
        solution += alpha * direction
        residual -= alpha * transformed
        relative = float(np.sqrt(np.sum(residual * residual))) / rhs_norm
        if relative <= tolerance:
            return solution, iteration, relative
        preconditioned = residual / scale
        next_rho = float(np.sum(residual * preconditioned))
        direction = preconditioned + (next_rho / rho) * direction
        rho = next_rho
    raise FlowError(f"flow solve did not converge: relative residual {relative:.3e}")


def solve_flow_field(
    spec: SignalGridSpec,
    *,
    mean_inlet_speed: float,
    axis: str = "y",
    mobility: Sequence[float] | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 50_000,
) -> tuple[SignalGridVelocityField, FlowSolveReport]:
    """Solve the device flow and return the face-staggered velocity field.

    Flow runs from the lower to the upper boundary of ``axis``; a negative
    ``mean_inlet_speed`` reverses it. The grid's shape, spacing, obstacles,
    and boundary kinds are read from ``spec``; ``mobility`` optionally gives
    one relative mobility per site (default uniform, the Stokes limit).
    """

    if axis not in _AXES:
        raise FlowError("flow axis must be one of x, y, z")
    if not math.isfinite(mean_inlet_speed) or mean_inlet_speed == 0.0:
        raise FlowError("mean inlet speed must be finite and nonzero")
    flow_axis = _AXES[axis]
    boundaries = (
        (spec.x_lower, spec.x_upper),
        (spec.y_lower, spec.y_upper),
        (spec.z_lower, spec.z_upper),
    )
    for lower, upper in boundaries:
        if lower.kind == GridBoundaryKind.PERIODIC or upper.kind == GridBoundaryKind.PERIODIC:
            raise FlowError("the flow solver does not support periodic boundaries")
    for boundary in boundaries[flow_axis]:
        if boundary.kind != GridBoundaryKind.FIXED:
            raise FlowError("the flow axis boundaries must be FIXED to act as inlet and outlet")

    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    spacing = (spec.spacing.x, spec.spacing.y, spec.spacing.z)
    mobility_grid = _mobility_grid(spec, mobility)

    conductances = tuple(
        _harmonic_faces(mobility_grid, index, spacing[index]) for index in range(3)
    )
    step = spacing[flow_axis]
    inlet_conductance = 2.0 * mobility_grid[_slices(flow_axis, slice(0, 1))] / (step * step)
    outlet_conductance = 2.0 * mobility_grid[_slices(flow_axis, slice(-1, None))] / (step * step)
    if not bool(np.any(inlet_conductance > 0.0)):
        raise FlowError("the inlet boundary is entirely blocked")

    diagonal = np.zeros(dims, dtype=np.float64)
    for index in range(3):
        diagonal[_slices(index, slice(1, None))] += conductances[index]
        diagonal[_slices(index, slice(None, -1))] += conductances[index]
    diagonal[_slices(flow_axis, slice(0, 1))] += inlet_conductance
    diagonal[_slices(flow_axis, slice(-1, None))] += outlet_conductance

    def apply_operator(pressure: _FloatGrid) -> _FloatGrid:
        result = diagonal * pressure
        for index in range(3):
            faces = conductances[index]
            result[_slices(index, slice(1, None))] -= (
                faces * pressure[_slices(index, slice(None, -1))]
            )
            result[_slices(index, slice(None, -1))] -= (
                faces * pressure[_slices(index, slice(1, None))]
            )
        return result

    rhs = np.zeros(dims, dtype=np.float64)
    rhs[_slices(flow_axis, slice(0, 1))] += inlet_conductance
    pressure, iterations, residual = _conjugate_gradient(
        apply_operator, rhs, diagonal, tolerance, max_iterations
    )

    face_grids: list[_FloatGrid] = []
    for index in range(3):
        face_dims = list(dims)
        face_dims[index] += 1
        faces = np.zeros(tuple(face_dims), dtype=np.float64)
        gradient = (
            pressure[_slices(index, slice(1, None))] - pressure[_slices(index, slice(None, -1))]
        )
        faces[_slices(index, slice(1, -1))] = -conductances[index] * spacing[index] * gradient
        face_grids.append(faces)
    inlet_faces = -inlet_conductance * step * (pressure[_slices(flow_axis, slice(0, 1))] - 1.0)
    outlet_faces = outlet_conductance * step * pressure[_slices(flow_axis, slice(-1, None))]
    face_grids[flow_axis][_slices(flow_axis, slice(0, 1))] = inlet_faces
    face_grids[flow_axis][_slices(flow_axis, slice(-1, None))] = outlet_faces

    open_inlet = inlet_conductance > 0.0
    solved_mean = float(np.mean(inlet_faces[open_inlet]))
    if solved_mean <= 1.0e-6 * float(np.max(inlet_conductance)) * step:
        raise FlowError("the device carries no through-flow: the outlet is unreachable")
    factor = mean_inlet_speed / solved_mean
    scaled = [faces * factor for faces in face_grids]

    field = SignalGridVelocityField()
    field.x_faces = [float(value) for value in scaled[0].ravel()]
    field.y_faces = [float(value) for value in scaled[1].ravel()]
    field.z_faces = [float(value) for value in scaled[2].ravel()]
    max_speed = max(float(np.max(np.abs(faces))) for faces in scaled)
    report = FlowSolveReport(
        iterations=iterations,
        residual=residual,
        mean_inlet_speed=mean_inlet_speed,
        max_speed=max_speed,
    )
    return field, report


def colony_mobility(
    spec: SignalGridSpec,
    cells: Iterable[_RodLike],
    *,
    base: float = 1.0,
    drag_coefficient: float = 100.0,
    max_volume_fraction: float = 0.9,
) -> list[float]:
    """Build the Brinkman mobility field from the current colony.

    Each cell's capsule volume accumulates into the voxel holding its center;
    the resulting volume fraction ``phi`` adds Kozeny-Carman style drag
    ``drag_coefficient * phi^2 / (1 - phi)^3`` to the base resistance, so
    ``1/m = 1/base + drag``. The drag coefficient is a modeling choice: it
    sets how strongly a packed colony resists through-flow relative to the
    open channel. Solid voxels stay at zero mobility.
    """

    if not math.isfinite(base) or base <= 0.0:
        raise FlowError("base mobility must be finite and positive")
    if not math.isfinite(drag_coefficient) or drag_coefficient < 0.0:
        raise FlowError("drag coefficient must be finite and non-negative")
    if not 0.0 < max_volume_fraction < 1.0:
        raise FlowError("maximum volume fraction must lie strictly between zero and one")

    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    origin = (spec.origin.x, spec.origin.y, spec.origin.z)
    spacing = (spec.spacing.x, spec.spacing.y, spec.spacing.z)
    volume = np.zeros(dims, dtype=np.float64)
    for cell in cells:
        position = (cell.position.x, cell.position.y, cell.position.z)
        indices: list[int] = []
        inside = True
        for component in range(3):
            index = math.floor((position[component] - origin[component]) / spacing[component])
            if not 0 <= index < dims[component]:
                inside = False
                break
            indices.append(index)
        if not inside:
            continue
        radius = cell.radius
        capsule = math.pi * radius * radius * cell.length + (4.0 / 3.0) * math.pi * radius**3
        volume[indices[0], indices[1], indices[2]] += capsule

    fraction = np.minimum(volume / spec.voxel_volume, max_volume_fraction)
    drag = drag_coefficient * fraction * fraction / (1.0 - fraction) ** 3
    mobility = 1.0 / (1.0 / base + drag)
    obstacles = spec.obstacles
    if obstacles:
        solid = np.asarray(obstacles, dtype=np.uint8).reshape(dims) != 0
        mobility[solid] = 0.0
    return [float(value) for value in mobility.ravel()]

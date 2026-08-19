#!/usr/bin/env python3
"""Validate the flow solvers against literature and exact solutions.

Runs the benchmark suite for both flow solvers - the Hele-Shaw closure
(`cellmodeller2.flow`) and the staggered MAC Stokes-Brinkman solver
(`cellmodeller2.stokes`) - and prints a table of computed values against their
references. Exits nonzero if any benchmark exceeds its tolerance, so the
script doubles as a CI gate.

References:
- Plane Poiseuille and rectangular duct series: F. M. White, "Viscous Fluid
  Flow" (3rd ed., ch. 3); duct peak-to-mean ratio tabulated in R. K. Shah and
  A. L. London, "Laminar Flow Forced Convection in Ducts" (1978).
- Two-layer Brinkman channel: exact solution of the Brinkman equation
  (H. C. Brinkman, Appl. Sci. Res. A1, 1949) matched across the fluid-porous
  interface.
- Hele-Shaw consistency: depth-averaged Stokes flow in a thin gap obeys
  Darcy's law with mobility proportional to the squared gap height (H. S.
  Hele-Shaw, 1898); the two solvers must agree on depth-averaged flux splits.

Usage: uv run python scripts/run_flow_benchmarks.py [--fine]
`--fine` doubles the benchmark resolutions to demonstrate mesh convergence.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
from cellmodeller2 import GridBoundaryKind, GridShape, SignalGridSpec, Vec3
from cellmodeller2.flow import gap_mobility, solve_flow_field
from cellmodeller2.stokes import solve_stokes_field


@dataclass(frozen=True)
class Result:
    solver: str
    benchmark: str
    metric: str
    computed: float
    reference: float
    tolerance: float
    seconds: float

    @property
    def error(self) -> float:
        # A zero reference marks a benchmark whose computed value is itself an
        # error measure; the tolerance then bounds it absolutely.
        if self.reference == 0.0:
            return abs(self.computed)
        return abs(self.computed - self.reference) / abs(self.reference)

    @property
    def passed(self) -> bool:
        return self.error <= self.tolerance


def duct_spec(
    nx: int, ny: int, nz: int, spacing: tuple[float, float, float]
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


def site(spec: SignalGridSpec, x: int, y: int, z: int) -> int:
    return (x * spec.shape.y + y) * spec.shape.z + z


def bench_plane_poiseuille_order(coarse: int) -> list[Result]:
    results: list[Result] = []
    errors: list[float] = []
    for n in (coarse, coarse * 2):
        start = time.perf_counter()
        spec = duct_spec(n, 6, 1, (1.0 / n, 0.25, 1.0))
        field, _ = solve_stokes_field(spec, mean_inlet_speed=1.0, tolerance=1.0e-10)
        profile = np.asarray(field.y_faces).reshape(n, 7, 1)[:, 3, 0]
        positions = (np.arange(n) + 0.5) / n
        exact = 6.0 * positions * (1.0 - positions)
        error = float(np.max(np.abs(profile - exact)) / np.max(exact))
        errors.append(error)
        results.append(
            Result(
                "stokes",
                f"plane Poiseuille n={n}",
                "max relative profile error",
                error,
                0.0,
                0.03 if n == coarse else 0.008,
                time.perf_counter() - start,
            )
        )
    order = math.log2(errors[0] / errors[1])
    results.append(
        Result(
            "stokes",
            "plane Poiseuille refinement",
            "observed convergence order",
            order,
            2.0,
            0.25,
            0.0,
        )
    )
    return results


def bench_square_duct(n: int) -> Result:
    start = time.perf_counter()
    spec = duct_spec(n, 6, n, (1.0 / n, 0.25, 1.0 / n))
    field, _ = solve_stokes_field(spec, mean_inlet_speed=1.0, tolerance=1.0e-9)
    cross = np.asarray(field.y_faces).reshape(n, 7, n)[:, 3, :]
    ratio = float(cross.max() / cross.mean())
    return Result(
        "stokes",
        f"square duct (n={n})",
        "u_max / u_mean (Shah & London: 2.0962)",
        ratio,
        2.0962,
        0.03,
        time.perf_counter() - start,
    )


def exact_two_layer(drag: float, positions: np.ndarray) -> np.ndarray:
    root = math.sqrt(drag)
    matrix = np.array(
        [
            [0.0, math.cosh(root), math.sinh(root)],
            [0.5, -math.cosh(root * 0.5), -math.sinh(root * 0.5)],
            [1.0, -root * math.sinh(root * 0.5), -root * math.cosh(root * 0.5)],
        ]
    )
    rhs = np.array([-1.0 / drag, 1.0 / drag + 0.125, 0.5])
    linear, cosh_c, sinh_c = np.linalg.solve(matrix, rhs)
    return np.where(
        positions < 0.5,
        -positions * positions / 2.0 + linear * positions,
        1.0 / drag + cosh_c * np.cosh(root * positions) + sinh_c * np.sinh(root * positions),
    )


def bench_two_layer_brinkman(nz: int) -> Result:
    start = time.perf_counter()
    drag_value = 200.0
    spec = duct_spec(1, 6, nz, (1.0, 0.25, 1.0 / nz))
    drag = [
        0.0 if (z + 0.5) / nz < 0.5 else drag_value
        for _ in range(6)
        for z in range(nz)
    ]
    field, _ = solve_stokes_field(spec, mean_inlet_speed=1.0, drag=drag, tolerance=1.0e-9)
    profile = np.asarray(field.y_faces).reshape(1, 7, nz)[0, 3, :]
    positions = (np.arange(nz) + 0.5) / nz
    exact = exact_two_layer(drag_value, positions)
    exact = exact / exact.mean() * profile.mean()
    error = float(np.max(np.abs(profile - exact)) / np.max(np.abs(exact)))
    return Result(
        "stokes",
        f"two-layer Brinkman (n={nz})",
        "max relative profile error vs exact ODE",
        error,
        0.0,
        0.01,
        time.perf_counter() - start,
    )


def bench_hele_shaw_duct() -> Result:
    start = time.perf_counter()
    spec = duct_spec(4, 8, 3, (1.0, 1.0, 1.0))
    field, _ = solve_flow_field(spec, mean_inlet_speed=5.0)
    error = float(max(abs(v - 5.0) for v in field.y_faces))
    return Result(
        "hele-shaw",
        "uniform duct",
        "max |u - mean| (exact plug flow)",
        error,
        0.0,
        1.0e-6,
        time.perf_counter() - start,
    )


def bench_hele_shaw_mobility_split() -> Result:
    start = time.perf_counter()
    spec = duct_spec(2, 6, 1, (1.0, 1.0, 1.0))
    mobility = [1.0 if x == 0 else 3.0 for x in range(2) for _ in range(6)]
    field, _ = solve_flow_field(spec, mean_inlet_speed=4.0, mobility=mobility)
    slow = field.y_faces[0 * 7 + 3]
    fast = field.y_faces[1 * 7 + 3]
    return Result(
        "hele-shaw",
        "parallel channels",
        "flux ratio at mobility ratio 3 (exact 3)",
        fast / slow,
        3.0,
        1.0e-5,
        time.perf_counter() - start,
    )


def bench_cross_solver_consistency() -> Result:
    start = time.perf_counter()
    nx, ny, nz = 6, 10, 6
    spec = duct_spec(nx, ny, nz, (1.0, 1.0, 0.05))
    obstacles = [0] * (nx * ny * nz)
    for y in (4, 5):
        for x in (1, 2):
            for z in range(nz):
                obstacles[site(spec, x, y, z)] = 1
    spec.obstacles = obstacles
    stokes_field, _ = solve_stokes_field(spec, mean_inlet_speed=1.0, tolerance=1.0e-9)
    hele_shaw_field, _ = solve_flow_field(
        spec, mean_inlet_speed=1.0, mobility=gap_mobility(spec)
    )

    def column_flux(values: list[float], x: int, fy: int) -> float:
        return sum(values[(x * (ny + 1) + fy) * nz + z] for z in range(nz))

    mid = ny // 2
    stokes_split = np.array([column_flux(stokes_field.y_faces, x, mid) for x in range(nx)])
    hele_shaw_split = np.array(
        [column_flux(hele_shaw_field.y_faces, x, mid) for x in range(nx)]
    )
    deviation = float(
        np.max(np.abs(stokes_split / stokes_split.sum() - hele_shaw_split / hele_shaw_split.sum()))
    )
    return Result(
        "cross-check",
        "thin-gap pillar",
        "max |flux-share difference| MAC vs Hele-Shaw",
        deviation,
        0.0,
        0.05,
        time.perf_counter() - start,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fine", action="store_true", help="double the benchmark resolutions"
    )
    arguments = parser.parse_args()
    scale = 2 if arguments.fine else 1

    results: list[Result] = []
    results.extend(bench_plane_poiseuille_order(8 * scale))
    results.append(bench_square_duct(16 * scale))
    results.append(bench_two_layer_brinkman(32 * scale))
    results.append(bench_hele_shaw_duct())
    results.append(bench_hele_shaw_mobility_split())
    results.append(bench_cross_solver_consistency())

    width = max(len(r.benchmark) for r in results)
    print(f"{'solver':<11} {'benchmark':<{width}}  {'computed':>10} {'reference':>10} "
          f"{'error':>9} {'tol':>7} {'time':>7}  status")
    failures = 0
    for r in results:
        status = "pass" if r.passed else "FAIL"
        if not r.passed:
            failures += 1
        print(
            f"{r.solver:<11} {r.benchmark:<{width}}  {r.computed:>10.5f} "
            f"{r.reference:>10.5f} {r.error:>9.5f} {r.tolerance:>7.3g} "
            f"{r.seconds:>6.2f}s  {status}   [{r.metric}]"
        )
    print(f"\n{len(results) - failures}/{len(results)} benchmarks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

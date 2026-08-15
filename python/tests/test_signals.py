from __future__ import annotations

import math

import pytest
from cellmodeller2 import (
    BackendFeature,
    BackendKind,
    GridBoundary,
    GridBoundaryKind,
    GridShape,
    SignalGridSpec,
    Simulation,
    Vec3,
    backend_available,
)


def _line_spec(length: int = 3) -> SignalGridSpec:
    shape = GridShape()
    shape.x = length
    shape.y = 1
    shape.z = 1
    spec = SignalGridSpec()
    spec.signal_count = 1
    spec.shape = shape
    spec.diffusion = [1.0]
    spec.advection = [Vec3()]
    return spec


def _assert_levels(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert math.isclose(left, right, rel_tol=1.0e-6, abs_tol=1.0e-6)


def test_cpu_signal_transport_sampling_and_stability() -> None:
    simulation = Simulation()
    simulation.configure_signal_grid(_line_spec(), [0.0, 1.0, 0.0])

    assert simulation.has_signal_grid
    assert simulation.signal_count == 1
    assert simulation.supports(BackendFeature.SIGNALS)
    assert simulation.sample_signals(Vec3(0.5, 0.0, 0.0)) == [0.5]

    simulation.step(0.25)
    _assert_levels(simulation.signal_levels, [0.25, 0.5, 0.25])
    assert math.isclose(sum(simulation.signal_levels), 1.0)
    before = simulation.signal_levels
    before_time = simulation.time
    with pytest.raises(ValueError, match="stability"):
        simulation.step(0.51)
    assert simulation.signal_levels == before
    assert simulation.time == before_time
    with pytest.raises(IndexError, match="outside"):
        simulation.sample_signals(Vec3(3.0, 0.0, 0.0))


def test_fixed_and_periodic_boundaries_are_explicit() -> None:
    fixed = _line_spec(2)
    lower = GridBoundary()
    lower.kind = GridBoundaryKind.FIXED
    lower.values = [2.0]
    fixed.x_lower = lower
    simulation = Simulation()
    simulation.configure_signal_grid(fixed, [0.0, 0.0])
    simulation.step(0.25)
    _assert_levels(simulation.signal_levels, [0.5, 0.0])

    periodic = _line_spec(4)
    periodic.diffusion = [0.0]
    periodic.advection = [Vec3(1.0, 0.0, 0.0)]
    lower = GridBoundary()
    lower.kind = GridBoundaryKind.PERIODIC
    upper = GridBoundary()
    upper.kind = GridBoundaryKind.PERIODIC
    periodic.x_lower = lower
    periodic.x_upper = upper
    simulation = Simulation()
    simulation.configure_signal_grid(periodic, [1.0, 0.0, 0.0, 0.0])
    simulation.step(0.5)
    _assert_levels(simulation.signal_levels, [0.5, 0.5, 0.0, 0.0])


def test_gpu_signal_paths_are_native_or_fail_before_mutation() -> None:
    spec = _line_spec()
    reference = Simulation()
    reference.configure_signal_grid(spec, [0.0, 1.0, 0.0])
    reference.step(0.25)
    for backend in (BackendKind.METAL, BackendKind.CUDA):
        if not backend_available(backend):
            continue
        simulation = Simulation(backend)
        simulation.configure_signal_grid(spec, [0.0, 1.0, 0.0])
        if simulation.supports(BackendFeature.SIGNALS):
            simulation.step(0.25)
            _assert_levels(simulation.signal_levels, reference.signal_levels)
            assert simulation.time == reference.time
        else:
            with pytest.raises(RuntimeError, match="does not implement signal grid"):
                simulation.step(0.25)
            assert simulation.time == 0.0
            assert simulation.signal_levels == [0.0, 1.0, 0.0]

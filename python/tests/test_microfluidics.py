# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path

import pytest
from cellmodeller2 import (
    BackendKind,
    GridShape,
    ModelContext,
    SignalGridSpec,
    SimulationController,
    Vec3,
)
from cellmodeller2.microfluidics import BiopixelTrapDevice, TrapChannelDevice
from cellmodeller2.runner import build_model

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 64, 72, 4
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.origin = Vec3(-140.0, -144.0, -8.0)
    grid.spacing = Vec3(4.0, 4.0, 4.0)
    grid.diffusion = [40.0]
    grid.advection = [Vec3()]
    return grid


def test_device_grid_projection_is_engine_valid() -> None:
    device = TrapChannelDevice(mean_flow_speed=20.0)
    grid = _grid()
    device.apply_to_grid(grid, inlet_values=[10.0], outlet_values=[0.0])
    grid.validate()

    assert grid.velocity_field is not None
    assert any(value != 0.0 for value in grid.velocity_field.y_faces)
    assert all(value == 0.0 for value in grid.velocity_field.x_faces)
    assert all(value == 0.0 for value in grid.velocity_field.z_faces)
    assert grid.y_lower.values == [10.0]

    solid = sum(grid.obstacles)
    assert 0 < solid < len(grid.obstacles)


def test_channel_profile_is_parabolic_and_zero_in_walls() -> None:
    device = TrapChannelDevice(mean_flow_speed=20.0)
    center_x = (device.channel_far_x + device.trap_open_x) * 0.5
    assert abs(device._channel_speed(center_x, 0.0) - 30.0) < 1.0e-6
    assert device._channel_speed(device.trap_open_x + 1.0, 0.0) == 0.0
    assert device._channel_speed(center_x, device.trap_half_z + 1.0) == 0.0
    assert device._solid(device.trap_back_x + 0.5, 0.0, 0.0)
    assert not device._solid(center_x, 0.0, 0.0)


def test_trap_example_builds_steps_and_transports_nutrient() -> None:
    model, _ = build_model(
        _EXAMPLES / "microfluidic_trap.py",
        ModelContext(BackendKind.CPU, 0, seed=11),
    )
    assert isinstance(model, SimulationController)
    for _ in range(20):
        model.step(0.02)

    simulation = model.simulation
    device = TrapChannelDevice()
    channel_x = (device.channel_far_x + device.trap_open_x) * 0.5
    upstream = simulation.sample_signals(Vec3(channel_x, -100.0, 0.0))[0]
    trap_interior = simulation.sample_signals(Vec3(0.0, 0.0, 0.0))[0]
    assert upstream > 5.0
    assert trap_interior > 5.0
    assert upstream >= trap_interior - 1.0e-3
    with pytest.raises(ValueError, match="inside a grid obstacle"):
        simulation.sample_signals(Vec3(0.0, 100.0, 0.0))
    assert len(simulation.cells()) >= 1


def test_biopixel_trap_matches_fabricated_dimensions() -> None:
    device = BiopixelTrapDevice(mean_flow_speed=20.0)
    assert device.trap_width == 100.0
    assert device.trap_depth == 95.0
    assert device.trap_height == 1.65
    assert device.channel_height == 10.0

    # The cavity is fluid; the surrounding trap layer is solid at cavity depth.
    assert not device._solid(47.5, 0.0, 0.825)
    assert device._solid(-10.0, 0.0, 0.825)
    assert device._solid(47.5, 60.0, 0.825)
    # The channel streams over both the cavity and the trap layer.
    assert not device._solid(47.5, 0.0, 5.0)
    assert not device._solid(-10.0, 0.0, 5.0)
    # Poiseuille across the channel height; still water in the cavity.
    mid = (device.trap_height + device.channel_height) * 0.5
    assert device._channel_speed(47.5, mid) == 30.0
    assert device._channel_speed(47.5, 0.825) == 0.0


def test_biopixel_example_confines_a_monolayer_under_flow() -> None:
    model, _ = build_model(
        _EXAMPLES / "tutorials" / "biopixel_trap.py",
        ModelContext(BackendKind.CPU, 0, seed=5),
    )
    assert isinstance(model, SimulationController)
    for _ in range(60):
        model.step(0.02)

    cells = model.simulation.cells()
    assert len(cells) >= 2
    in_cavity = [cell for cell in cells if cell.position.z < 1.65]
    assert len(in_cavity) >= len(cells) - 2
    for cell in in_cavity:
        assert -50.0 < cell.position.y < 50.0
        assert 0.0 < cell.position.x < 95.0
    for cell in cells:
        assert 0.0 < cell.position.z < 10.0

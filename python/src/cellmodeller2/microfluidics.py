"""Authoring helpers for microfluidic device models.

A device is described once in physical coordinates and then projected into the
engine's typed inputs: box wall constraints for mechanics, a solid mask for the
signal grid, and a numerically solved face-staggered flow field for advection
(the steady Hele-Shaw solve of `cellmodeller2.flow`, so mass is conserved per
voxel through any mask geometry). The runtime receives only materialized data;
every predicate here is authoring-time.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BoxConstraintInit,
    ConstraintRegion,
    GridBoundaryKind,
    PlaneConstraintInit,
    SignalGridSpec,
    Simulation,
    Vec3,
)
from .flow import gap_mobility, solve_flow_field
from .masks import MaskError, extract_rectangles, load_mask_polylines


@dataclass(frozen=True, slots=True)
class TrapChannelDevice:
    """An open-sided cell trap fed by a straight flow channel.

    The channel runs along the y axis between ``channel_far_x`` and
    ``trap_open_x``. The trap cavity spans ``trap_open_x`` to ``trap_back_x``
    in x and ``-trap_half_y`` to ``trap_half_y`` in y, open toward the channel
    and sealed on its other three sides by solid blocks. Media flows along +y
    through the channel with the solved steady device flow; the dead-end trap
    interior carries only the weak circulation at its mouth and exchanges with
    the channel by diffusion through the open face.
    """

    trap_open_x: float = -60.0
    trap_back_x: float = 60.0
    trap_half_y: float = 15.0
    trap_half_z: float = 3.0
    channel_far_x: float = -100.0
    channel_half_length: float = 120.0
    wall_thickness: float = 2.0
    mean_flow_speed: float = 0.0

    def add_constraints(self, simulation: Simulation) -> None:
        """Add the device's wall constraints to a simulation."""

        blocks = (
            (
                (self.trap_open_x, self.trap_half_y, -self.trap_half_z),
                (
                    self.trap_back_x + self.wall_thickness,
                    self.channel_half_length,
                    self.trap_half_z,
                ),
            ),
            (
                (self.trap_open_x, -self.channel_half_length, -self.trap_half_z),
                (
                    self.trap_back_x + self.wall_thickness,
                    -self.trap_half_y,
                    self.trap_half_z,
                ),
            ),
            (
                (
                    self.trap_back_x,
                    -self.trap_half_y - self.wall_thickness,
                    -self.trap_half_z,
                ),
                (
                    self.trap_back_x + self.wall_thickness,
                    self.trap_half_y + self.wall_thickness,
                    self.trap_half_z,
                ),
            ),
        )
        for low, high in blocks:
            box = BoxConstraintInit()
            box.center = Vec3(
                *((left + right) * 0.5 for left, right in zip(low, high, strict=True))
            )
            box.half_extents = Vec3(
                *((right - left) * 0.5 for left, right in zip(low, high, strict=True))
            )
            box.coefficient = 1.0
            box.allowed_region = ConstraintRegion.OUTSIDE
            simulation.add_box_constraint(box)

        chamber = BoxConstraintInit()
        chamber.center = Vec3(
            (self.channel_far_x + self.trap_back_x + self.wall_thickness) * 0.5,
            0.0,
            0.0,
        )
        chamber.half_extents = Vec3(
            (self.trap_back_x + self.wall_thickness - self.channel_far_x) * 0.5,
            self.channel_half_length,
            self.trap_half_z,
        )
        chamber.coefficient = 1.0
        chamber.allowed_region = ConstraintRegion.INSIDE
        simulation.add_box_constraint(chamber)

    def _solid(self, px: float, py: float, pz: float) -> bool:
        if px <= self.channel_far_x or px >= self.trap_back_x + self.wall_thickness:
            return True
        if abs(pz) >= self.trap_half_z:
            return True
        if px >= self.trap_open_x and abs(py) >= self.trap_half_y:
            return True
        return px >= self.trap_back_x

    def apply_to_grid(
        self,
        spec: SignalGridSpec,
        inlet_values: list[float],
        outlet_values: list[float],
    ) -> None:
        """Materialize the device's solid mask, solved flow field, and y inlet and outlet."""

        _project_channel_device(self, spec, inlet_values, outlet_values)


@dataclass(frozen=True, slots=True)
class BiopixelTrapDevice:
    """One trap of a biopixel array: a shallow monolayer cavity beside a tall channel.

    The cavity spans ``0`` to ``trap_depth`` in x with its open face at ``x = 0``
    toward the channel, ``trap_width`` in y, and only ``trap_height`` in z, so
    the colony grows as a monolayer under the cavity ceiling. The flow channel
    runs along y between ``-channel_width`` and ``0`` at the full
    ``channel_height``. The device floor is ``z = 0``. An array device repeats
    this trap along its channels; every trap sees the same fresh-media flow, so
    one simulated trap is representative of each biopixel in the array when
    inter-trap coupling is not modeled.
    """

    trap_depth: float = 95.0
    trap_width: float = 100.0
    trap_height: float = 1.65
    channel_width: float = 100.0
    channel_height: float = 10.0
    channel_half_length: float = 150.0
    wall_thickness: float = 10.0
    mean_flow_speed: float = 0.0

    @classmethod
    def from_mask(
        cls,
        path: str | os.PathLike[str],
        *,
        layer: str = "Layer-2",
        wall_inset: float = 5.0,
        unit_scale: float = 1000.0,
        mean_flow_speed: float = 0.0,
    ) -> BiopixelTrapDevice:
        """Derive the trap footprint from a photomask drawing.

        The mask draws each trap's outer wall outline. The cavity is the
        outline minus ``wall_inset`` per wall: two side walls across the long
        dimension and one back wall across the short dimension, whose remaining
        side is the open face toward the channel. The drawing must contain one
        uniform trap population on the layer.
        """

        polylines = load_mask_polylines(path)
        rectangles = extract_rectangles(polylines, layer=layer, unit_scale=unit_scale)
        if not rectangles:
            raise MaskError(f"mask layer {layer!r} contains no rectangles")
        sizes = Counter(
            (round(max(r.width, r.height), 3), round(min(r.width, r.height), 3))
            for r in rectangles
        )
        (long_side, short_side), count = sizes.most_common(1)[0]
        if count < 2:
            raise MaskError(f"mask layer {layer!r} has no repeated trap outline")
        width = long_side - 2.0 * wall_inset
        depth = short_side - wall_inset
        if width <= 0.0 or depth <= 0.0:
            raise MaskError("wall inset leaves no cavity")
        return cls(trap_width=width, trap_depth=depth, mean_flow_speed=mean_flow_speed)

    def add_constraints(self, simulation: Simulation) -> None:
        """Add the device's wall constraints to a simulation."""

        half_y = self.trap_width * 0.5
        thickness = self.wall_thickness

        floor = PlaneConstraintInit()
        floor.point = Vec3(0.0, 0.0, 0.0)
        floor.inward_normal = Vec3(0.0, 0.0, 1.0)
        simulation.add_plane_constraint(floor)

        ceiling = PlaneConstraintInit()
        ceiling.point = Vec3(0.0, 0.0, self.channel_height)
        ceiling.inward_normal = Vec3(0.0, 0.0, -1.0)
        simulation.add_plane_constraint(ceiling)

        boxes = (
            (
                (0.0, -half_y - thickness, self.trap_height),
                (
                    self.trap_depth + thickness,
                    half_y + thickness,
                    self.channel_height + thickness,
                ),
            ),
            (
                (0.0, half_y, -thickness),
                (
                    self.trap_depth + thickness,
                    self.channel_half_length,
                    self.channel_height + thickness,
                ),
            ),
            (
                (0.0, -self.channel_half_length, -thickness),
                (
                    self.trap_depth + thickness,
                    -half_y,
                    self.channel_height + thickness,
                ),
            ),
            (
                (self.trap_depth, -half_y - thickness, -thickness),
                (
                    self.trap_depth + thickness,
                    half_y + thickness,
                    self.channel_height + thickness,
                ),
            ),
        )
        for low, high in boxes:
            box = BoxConstraintInit()
            box.center = Vec3(
                *((left + right) * 0.5 for left, right in zip(low, high, strict=True))
            )
            box.half_extents = Vec3(
                *((right - left) * 0.5 for left, right in zip(low, high, strict=True))
            )
            box.coefficient = 1.0
            box.allowed_region = ConstraintRegion.OUTSIDE
            simulation.add_box_constraint(box)

        chamber = BoxConstraintInit()
        chamber.center = Vec3(
            (self.trap_depth + self.wall_thickness - self.channel_width) * 0.5,
            0.0,
            self.channel_height * 0.5,
        )
        chamber.half_extents = Vec3(
            (self.channel_width + self.trap_depth + self.wall_thickness) * 0.5,
            self.channel_half_length,
            self.channel_height * 0.5,
        )
        chamber.coefficient = 1.0
        chamber.allowed_region = ConstraintRegion.INSIDE
        simulation.add_box_constraint(chamber)

    def _solid(self, px: float, py: float, pz: float) -> bool:
        if px <= -self.channel_width or px >= self.trap_depth + self.wall_thickness:
            return True
        if pz <= 0.0 or pz >= self.channel_height:
            return True
        if px >= 0.0 and abs(py) >= self.trap_width * 0.5:
            return True
        if px >= self.trap_depth:
            return True
        return px >= 0.0 and pz >= self.trap_height

    def apply_to_grid(
        self,
        spec: SignalGridSpec,
        inlet_values: list[float],
        outlet_values: list[float],
    ) -> None:
        """Materialize the device's solid mask, solved flow field, and y inlet and outlet."""

        _project_channel_device(self, spec, inlet_values, outlet_values)


def _project_channel_device(
    device: TrapChannelDevice | BiopixelTrapDevice,
    spec: SignalGridSpec,
    inlet_values: list[float],
    outlet_values: list[float],
) -> None:
    shape = spec.shape
    origin = spec.origin
    spacing = spec.spacing

    def center(axis_origin: float, axis_spacing: float, index: int) -> float:
        return axis_origin + axis_spacing * index

    obstacles = [0] * (shape.x * shape.y * shape.z)
    for x in range(shape.x):
        px = center(origin.x, spacing.x, x)
        for y in range(shape.y):
            py = center(origin.y, spacing.y, y)
            for z in range(shape.z):
                pz = center(origin.z, spacing.z, z)
                if device._solid(px, py, pz):
                    obstacles[x * shape.y * shape.z + y * shape.z + z] = 1
    spec.obstacles = obstacles

    spec.advection = [Vec3() for _ in range(spec.signal_count)]
    spec.y_lower.kind = GridBoundaryKind.FIXED
    spec.y_lower.values = list(inlet_values)
    spec.y_upper.kind = GridBoundaryKind.FIXED
    spec.y_upper.values = list(outlet_values)

    if device.mean_flow_speed != 0.0:
        field, _ = solve_flow_field(
            spec,
            mean_inlet_speed=device.mean_flow_speed,
            axis="y",
            mobility=gap_mobility(spec),
        )
        spec.velocity_field = field

"""Authoring helpers for microfluidic device models.

A device is described once in physical coordinates and then projected into the
engine's typed inputs: box wall constraints for mechanics, a solid mask for the
signal grid, and a divergence-free face-staggered flow field for advection. The
runtime receives only materialized data; every predicate here is authoring-time.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import dataclass

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BoxConstraintInit,
    ConstraintRegion,
    GridBoundaryKind,
    PlaneConstraintInit,
    SignalGridSpec,
    SignalGridVelocityField,
    Simulation,
    Vec3,
)


@dataclass(frozen=True, slots=True)
class TrapChannelDevice:
    """An open-sided cell trap fed by a straight flow channel.

    The channel runs along the y axis between ``channel_far_x`` and
    ``trap_open_x``. The trap cavity spans ``trap_open_x`` to ``trap_back_x``
    in x and ``-trap_half_y`` to ``trap_half_y`` in y, open toward the channel
    and sealed on its other three sides by solid blocks. Media flows along +y
    with a Poiseuille profile across the channel width; the trap interior
    exchanges with the channel by diffusion through the open face.
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

    def _channel_speed(self, px: float, pz: float) -> float:
        if not (self.channel_far_x < px < self.trap_open_x) or abs(pz) >= self.trap_half_z:
            return 0.0
        center = (self.channel_far_x + self.trap_open_x) * 0.5
        half_width = (self.trap_open_x - self.channel_far_x) * 0.5
        offset = (px - center) / half_width
        return 1.5 * self.mean_flow_speed * (1.0 - offset * offset)

    def apply_to_grid(
        self,
        spec: SignalGridSpec,
        inlet_values: list[float],
        outlet_values: list[float],
    ) -> None:
        """Materialize the device's solid mask, flow field, and y-axis inlet and outlet."""

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

    def _channel_speed(self, px: float, pz: float) -> float:
        if not (-self.channel_width < px < 0.0) or not (0.0 < pz < self.channel_height):
            return 0.0
        center = -self.channel_width * 0.5
        half_width = self.channel_width * 0.5
        offset = (px - center) / half_width
        return 1.5 * self.mean_flow_speed * (1.0 - offset * offset)

    def apply_to_grid(
        self,
        spec: SignalGridSpec,
        inlet_values: list[float],
        outlet_values: list[float],
    ) -> None:
        """Materialize the device's solid mask, flow field, and y-axis inlet and outlet."""

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

    def fluid(x: int, y: int, z: int) -> bool:
        return obstacles[x * shape.y * shape.z + y * shape.z + z] == 0

    y_faces = [0.0] * (shape.x * (shape.y + 1) * shape.z)
    for x in range(shape.x):
        px = center(origin.x, spacing.x, x)
        for z in range(shape.z):
            pz = center(origin.z, spacing.z, z)
            speed = device._channel_speed(px, pz)
            if speed == 0.0:
                continue
            for fy in range(shape.y + 1):
                lower_open = fy == 0 or fluid(x, fy - 1, z)
                upper_open = fy == shape.y or fluid(x, fy, z)
                if lower_open and upper_open:
                    y_faces[x * (shape.y + 1) * shape.z + fy * shape.z + z] = speed
    field = SignalGridVelocityField()
    field.x_faces = [0.0] * ((shape.x + 1) * shape.y * shape.z)
    field.y_faces = y_faces
    field.z_faces = [0.0] * (shape.x * shape.y * (shape.z + 1))
    spec.velocity_field = field

    spec.advection = [Vec3() for _ in range(spec.signal_count)]
    spec.y_lower.kind = GridBoundaryKind.FIXED
    spec.y_lower.values = list(inlet_values)
    spec.y_upper.kind = GridBoundaryKind.FIXED
    spec.y_upper.values = list(outlet_values)

"""One biopixel of the Prindle sensing-array device.

The device photomasks in ``docs/tutorials/devices/prindle.dwg`` and
``prindle.dxf`` lay out 496 identical cell traps in a 16 x 31 grid beside media
channels. This model reads its trap footprint from that DXF at build time:
``BiopixelTrapDevice.from_mask`` extracts the drawn 110 x 100 micrometer outer
wall outline and removes the 5-micrometer walls, giving the 100 x 95 cavity.
The cavity is 1.65 micrometers tall, squeezing the colony into a monolayer,
and opens on one side to the 10-micrometer-tall channel of the flow layer on
top, where media streams past with a Poiseuille profile. Cells that crowd out
of the trap mouth are carried downstream and washed out. Every trap in the
array sees the same flow, so one simulated trap stands for each biopixel when
inter-trap coupling is not modeled.
"""

from __future__ import annotations

from pathlib import Path

from cellmodeller2 import (
    CellInit,
    CellUpdate,
    ControllerStep,
    DivisionEvent,
    GridShape,
    MechanicsConfig,
    ModelContext,
    NativeController,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    StepPlan,
    UniformLengthDivision,
    Vec3,
)
from cellmodeller2.checkpoint import CheckpointBundle, JSONValue
from cellmodeller2.microfluidics import BiopixelTrapDevice

MODEL_ID = "tutorials.biopixel-trap"
MODEL_VERSION = 3
DIVISION = UniformLengthDivision(3.2, 3.8, jitter_z=False)

_MASK = Path(__file__).resolve().parents[2] / "docs" / "tutorials" / "devices" / "prindle.dxf"
DEVICE = BiopixelTrapDevice.from_mask(_MASK, mean_flow_speed=20.0)
CELL_RADIUS = 0.5
WASHOUT_Y = DEVICE.channel_half_length - 10.0

NUTRIENT_INLET = 10.0
BASE_GROWTH_RATE = 1.0
NUTRIENT_K = 5.0


def _grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 41, 60, 6
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.origin = Vec3(-97.5, -147.5, 0.825)
    grid.spacing = Vec3(5.0, 5.0, 1.65)
    grid.diffusion = [40.0]
    grid.advection = [Vec3()]
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    grid.solver.absolute_tolerance = 1.0e-12
    DEVICE.apply_to_grid(grid, inlet_values=[NUTRIENT_INLET], outlet_values=[0.0])
    return grid


def _primed_levels(grid: SignalGridSpec) -> list[float]:
    # The device is loaded flooded with fresh media before flow starts.
    return [NUTRIENT_INLET if solid == 0 else 0.0 for solid in grid.obstacles]


def _nutrient_growth(simulation: Simulation, position: Vec3) -> float:
    nutrient = max(0.0, simulation.sample_signals(position)[0])
    return BASE_GROWTH_RATE * nutrient / (NUTRIENT_K + nutrient)


def _regulate(step: ControllerStep) -> StepPlan:
    divisions = DIVISION.requests(step)
    washed = tuple(cell.id for cell in step.cells if abs(cell.position.y) > WASHOUT_Y)
    if washed:
        DIVISION.forget(step, washed)
        divisions = tuple(request for request in divisions if request.parent_id not in washed)
    return StepPlan(
        updates=tuple(
            CellUpdate(cell.id, growth_rate=_nutrient_growth(step.simulation, cell.position))
            for cell in step.cells
            if cell.id not in washed
        ),
        divisions=divisions,
        removals=washed,
    )


def _divided(step: ControllerStep, event: DivisionEvent) -> None:
    DIVISION.on_division(step, event)


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(reserved_capacity=20_000)
    grid = _grid()
    simulation.configure_signal_grid(grid, _primed_levels(grid))
    DEVICE.add_constraints(simulation)

    founder = CellInit()
    founder.position = Vec3(DEVICE.trap_depth * 0.5, 0.0, DEVICE.trap_height * 0.5)
    founder.length = 3.5
    founder.radius = CELL_RADIUS
    founder.growth_rate = 1.0
    founder_id = simulation.add_cell(founder)
    state: dict[str, JSONValue] = {"scope": "biopixel-trap"}
    DIVISION.initialize(state, context.rng, (founder_id,))
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulate,
        on_division=_divided,
        mechanics=MechanicsConfig(flow_drift=True, passes=2),
        state=state,
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    del context
    return NativeController.from_checkpoint(
        checkpoint,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        regulate=_regulate,
        on_division=_divided,
    )

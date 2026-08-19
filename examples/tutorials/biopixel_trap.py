"""One biopixel of the Prindle sensing-array device.

The device photomasks in ``docs/tutorials/devices/prindle.dwg`` and
``prindle.dxf`` lay out 496 identical cell traps in a 16 x 31 grid beside media
channels. This model reads its trap footprint from that DXF at build time:
``BiopixelTrapDevice.from_mask`` extracts the drawn 110 x 100 micrometer outer
wall outline and removes the 5-micrometer walls, giving the 100 x 95 cavity.
The cavity is 1.65 micrometers tall, squeezing the colony into a monolayer,
and opens on one side to the 10-micrometer-tall channel of the flow layer on
top, where media streams past with the numerically solved steady device flow.
The colony feeds back on that flow: at a fixed cadence the model rasterizes
the packed cells into a Brinkman drag field, re-solves the flow, and swaps the
field into the running simulation, so media diverts around a full trap mouth.
Cells that crowd out of the trap mouth are carried downstream and washed out.
Every trap in the array sees the same flow, so one simulated trap stands for
each biopixel when inter-trap coupling is not modeled.
"""

from __future__ import annotations

from pathlib import Path

from cellmodeller2 import (
    CellInit,
    CellUpdate,
    ControllerStep,
    CoupledRatePlan,
    DivisionEvent,
    GridShape,
    MechanicsConfig,
    ModelContext,
    NativeController,
    RatePlanBuilder,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    StepPlan,
    UniformLengthDivision,
    Vec3,
)
from cellmodeller2.checkpoint import CheckpointBundle, JSONValue
from cellmodeller2.flow import colony_mobility, gap_mobility, solve_flow_field
from cellmodeller2.microfluidics import BiopixelTrapDevice

MODEL_ID = "tutorials.biopixel-trap"
MODEL_VERSION = 5
DIVISION = UniformLengthDivision(3.2, 3.8, jitter_z=False)

_MASK = Path(__file__).resolve().parents[2] / "docs" / "tutorials" / "devices" / "prindle.dxf"
FLOW_SPEED = 20.0
DEVICE = BiopixelTrapDevice.from_mask(_MASK, mean_flow_speed=FLOW_SPEED)
CELL_RADIUS = 0.5
WASHOUT_Y = DEVICE.channel_half_length - 10.0

NUTRIENT_INLET = 10.0
BASE_GROWTH_RATE = 1.0
NUTRIENT_K = 5.0
# Nutrient is one limiting substrate in arbitrary concentration units, fed at
# NUTRIENT_INLET. Uptake is tied to realized growth: a cell consumes
# growth_rate * volume / NUTRIENT_YIELD per unit time, so Monod-limited growth
# and consumption stay consistent. The yield sets the coupling strength, and
# this value makes a packed trap's uptake comparable to the diffusive supply
# through its mouth, so nutrient penetrates a few tens of micrometers and the
# colony behind that front grows more slowly.
NUTRIENT_YIELD = 0.5

# Brinkman feedback: how often the colony's drag re-solves the device flow,
# and how strongly a packed voxel resists through-flow.
RESOLVE_INTERVAL = 100
DRAG_COEFFICIENT = 100.0


def _grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 42, 60, 14
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    # The lattice of site centers covers every position a cell can reach, with
    # a margin of one voxel past the floor and the far channel wall: contact
    # relaxation lets a crowded cell press slightly into a wall, and sampling
    # outside the lattice is an error. Two z layers span the cavity exactly, so
    # the gap-height mobility of the cavity relative to the channel matches the
    # device's true squared gap ratio.
    grid.origin = Vec3(-100.0, -147.5, -0.4125)
    grid.spacing = Vec3(5.0, 5.0, 0.825)
    grid.diffusion = [40.0]
    grid.advection = [Vec3()]
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    # Cell sources are small next to the background level, so convergence is
    # judged on the absolute residual: a relative tolerance scaled by the
    # background would declare a step converged before uptake reaches the
    # field. The absolute bound sits above the float32 residual floor of a
    # grid at this concentration and well below one step of cell uptake.
    grid.solver.absolute_tolerance = 1.0e-6
    grid.solver.relative_tolerance = 0.0
    DEVICE.apply_to_grid(grid, inlet_values=[NUTRIENT_INLET], outlet_values=[0.0])
    return grid


GRID = _grid()
GAP_MOBILITY = gap_mobility(GRID)


def _rate_plan() -> CoupledRatePlan:
    rates = RatePlanBuilder()
    uptake = -(rates.growth_rate() * rates.cell_volume()) / NUTRIENT_YIELD
    return rates.coupled_plan(0, 1, (), (uptake,))


def _primed_levels(grid: SignalGridSpec) -> list[float]:
    # The device is loaded flooded with fresh media before flow starts.
    return [NUTRIENT_INLET if solid == 0 else 0.0 for solid in grid.obstacles]


def _nutrient_growth(simulation: Simulation, position: Vec3) -> float:
    nutrient = max(0.0, simulation.sample_signals(position)[0])
    return BASE_GROWTH_RATE * nutrient / (NUTRIENT_K + nutrient)


def _regulate(step: ControllerStep) -> StepPlan:
    if step.completed_steps and step.completed_steps % RESOLVE_INTERVAL == 0:
        mobility = colony_mobility(
            GRID, step.cells, base=GAP_MOBILITY, drag_coefficient=DRAG_COEFFICIENT
        )
        field, _ = solve_flow_field(GRID, mean_inlet_speed=FLOW_SPEED, mobility=mobility)
        step.simulation.set_velocity_field(field)
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
    simulation.configure_signal_grid(GRID, _primed_levels(GRID))
    simulation.set_coupled_rate_plan(_rate_plan())
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

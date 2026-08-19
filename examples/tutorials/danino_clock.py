"""SimBOL's Danino quorum-sensing clock in a flow-fed microfluidic trap.

Media flows along the channel with the numerically solved steady device flow:
it delivers nutrient, carries secreted AHL downstream, and washes out cells
that escape the trap. The colony feeds back on the flow: at a fixed cadence
the model rasterizes the packed cells into a Brinkman drag field, re-solves
the flow, and swaps the field into the running simulation.
"""

from __future__ import annotations

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
from cellmodeller2.microfluidics import TrapChannelDevice

MODEL_ID = "tutorials.danino-clock"
MODEL_VERSION = 6
DIVISION = UniformLengthDivision(3.2, 3.8, jitter_z=False)

FLOW_SPEED = 20.0
DEVICE = TrapChannelDevice(mean_flow_speed=FLOW_SPEED)
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
    shape.x, shape.y, shape.z = 64, 72, 4
    grid = SignalGridSpec()
    grid.signal_count = 2
    grid.shape = shape
    grid.origin = Vec3(-140.0, -144.0, -8.0)
    grid.spacing = Vec3(4.0, 4.0, 4.0)
    grid.diffusion = [40.0, 20.0]
    grid.advection = [Vec3(), Vec3()]
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    # Cell sources are small next to the background level, so convergence is
    # judged on the absolute residual: a relative tolerance scaled by the
    # background would declare a step converged before uptake reaches the
    # field. The absolute bound sits above the float32 residual floor of a
    # grid at this concentration and well below one step of cell uptake.
    grid.solver.absolute_tolerance = 1.0e-6
    grid.solver.relative_tolerance = 0.0
    DEVICE.apply_to_grid(
        grid,
        inlet_values=[0.0, NUTRIENT_INLET],
        outlet_values=[0.0, 0.0],
    )
    return grid


GRID = _grid()
GAP_MOBILITY = gap_mobility(GRID)


def _primed_levels(grid: SignalGridSpec) -> list[float]:
    # The device is loaded flooded with fresh media before flow starts; AHL
    # starts at zero everywhere.
    site_count = grid.shape.x * grid.shape.y * grid.shape.z
    levels = [0.0] * (2 * site_count)
    for site, solid in enumerate(grid.obstacles):
        if solid == 0:
            levels[site_count + site] = NUTRIENT_INLET
    return levels


def _rate_plan() -> CoupledRatePlan:
    rates = RatePlanBuilder()
    luxi = rates.maximum(rates.species(0), 0.0)
    aiia = rates.maximum(rates.species(1), 0.0)
    gfp = rates.maximum(rates.species(2), 0.0)
    ahl = rates.maximum(rates.signal(0), 0.0)
    ahl_cubed = ahl**3.0
    hill = ahl_cubed / (8.0 + ahl_cubed)
    activated = 0.02 + 8.0 * hill
    return rates.coupled_plan(
        3,
        2,
        (
            activated - 1.2 * luxi,
            activated - 0.3 * aiia,
            activated - 0.5 * gfp,
        ),
        (
            8.0 * luxi - 4.0 * aiia * ahl,
            -(rates.growth_rate() * rates.cell_volume()) / NUTRIENT_YIELD,
        ),
    )


def _nutrient_growth(simulation: Simulation, position: Vec3) -> float:
    nutrient = max(0.0, simulation.sample_signals(position)[1])
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
    for daughter in (event.first, event.second):
        step.simulation.set_species(
            daughter.id,
            [max(0.0, value * step.rng.uniform(0.9, 1.1)) for value in event.parent.species],
        )


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(reserved_capacity=5_000, species_count=3)
    simulation.configure_signal_grid(GRID, _primed_levels(GRID))
    simulation.set_coupled_rate_plan(_rate_plan())
    DEVICE.add_constraints(simulation)

    founder = CellInit()
    founder.position = Vec3(DEVICE.trap_back_x - 5.0, 0.0, 0.0)
    founder.length = 3.5
    founder.radius = CELL_RADIUS
    founder.growth_rate = 1.0
    founder.species = [context.rng.uniform(0.0, 0.2), context.rng.uniform(0.0, 0.2), 0.0]
    founder_id = simulation.add_cell(founder)
    state: dict[str, JSONValue] = {"scope": "clock-nutrient-field-and-trap"}
    DIVISION.initialize(state, context.rng, (founder_id,))
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulate,
        on_division=_divided,
        mechanics=MechanicsConfig(flow_drift=True),
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

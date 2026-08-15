from cellmodeller2 import CellInit, Vec3


def build(context):
    """Construct a reproducible colony for ``cm2 run``."""

    simulation = context.simulation()
    cell = CellInit()
    cell.position = Vec3(context.rng.uniform(-0.1, 0.1), 0.0, 0.0)
    cell.length = float(context.parameters.get("initial_length", 4.0))
    cell.radius = 0.5
    cell.growth_rate = float(context.parameters.get("growth_rate", 0.2))
    simulation.add_cell(cell)
    return simulation

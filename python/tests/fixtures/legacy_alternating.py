from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator


def setup(sim):
    biophys = CLBacterium(sim, alternate_divisions=True)
    sim.init(biophys, ModuleRegulator(sim), None, None)
    sim.addCell(pos=(0, 0, 0), dir=(1, 0, 0), length=5.0)


def init(cell):
    cell.growthRate = 0.0


def update(cells):
    pass

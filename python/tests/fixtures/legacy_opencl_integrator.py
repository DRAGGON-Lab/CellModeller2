from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.Integration.CLEulerIntegrator import CLEulerIntegrator
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator


def setup(sim):
    biophys = CLBacterium(sim)
    integrator = CLEulerIntegrator(sim, 1, 100)
    sim.init(biophys, ModuleRegulator(sim), None, integrator)


def init(cell):
    pass


def update(cells):
    pass

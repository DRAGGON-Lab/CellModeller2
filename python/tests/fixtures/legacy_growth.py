import random

import numpy
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator


def setup(sim):
    biophys = CLBacterium(
        sim,
        jitter_z=False,
        gamma=20.0,
        cgs_tol=1.0e-5,
        compNeighbours=True,
    )
    biophys.addPlane((0, -4, 0), (0, 1, 0), 1.0)
    sim.init(biophys, ModuleRegulator(sim), None, None)
    sim.addCell(cellType=2, pos=(0, 0, 0), dir=(1, 0, 0), length=4.0)
    sim.addRenderer(Renderers.GLBacteriumRenderer(sim))


def init(cell):
    cell.targetVol = 4.1 + random.uniform(0.0, 0.1)
    cell.growthRate = 1.0
    cell.color = numpy.array((0.1, 0.2, 0.3))


def update(cells):
    for cell in cells.values():
        if cell.volume > cell.targetVol:
            cell.divideFlag = True


def divide(parent, d1, d2):
    d1.targetVol = 10.0
    d2.targetVol = 10.0

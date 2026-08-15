# Legacy neighbor-diffusion audit

## Scope

The audit covers `CellModeller/Signalling/NeighbourDiffusion.py` and its
possible integration with the surviving legacy simulator, cell-state,
biophysics, and signal-integrator code.

Repository search finds no bundled model or test that constructs
`NeighbourDiffusion`.

## Intended equation

For every directed neighbor entry and signal `s`, the module accumulates

```text
rate[i, s] += D[s] * (level[j, s] - level[i, s]).
```

If the neighbor relation were symmetric and duplicate-free, this would be an
unweighted conservative graph Laplacian. Production rates are then added to a
cell's corresponding signal row.

## Execution defects

The checked-in module is not a working reference implementation:

1. Python 3 compilation fails with `TabError` because indentation mixes tabs
   and spaces.
2. The transport loop reads `cell.nbs`, but `CellState` defines
   `cell.neighbours` and `CLBacterium.updateCellState` populates that name.
3. Neighbor lists exist only when `CLBacterium` is constructed with neighbor
   computation enabled; the signaling module does not validate that
   precondition.
4. The available signal integrators require grid attributes such as
   `gridDim`, `gridOrig`, `gridSize`, and `initLevels`; `NeighbourDiffusion`
   provides none of them.
5. `dataLen()` depends on the current cell count, while the signal integrators
   allocate their flat signal storage during construction and do not resize it
   when cells are later added.
6. The source itself notes that exchange should be scaled by wall or contact
   area, but no such weight is computed.

Division and deletion semantics, stability constraints, units, and checkpoint
behavior are also unspecified.

## Compatibility judgment

The file provides evidence of an experiment, not an operational CellModeller
feature whose numerical behavior can be reproduced. Porting the visible loop
would freeze known omissions and give an unweighted neighbor count a physical
meaning it never established.

CellModeller2 retires this compatibility item. Any future contact-mediated
transport will be a new typed, conservative graph-flux model built on the
current contact graph and validated independently across native backends.

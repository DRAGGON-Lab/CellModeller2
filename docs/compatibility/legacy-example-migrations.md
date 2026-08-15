# Legacy example migration ledger

This ledger tracks the nine bundled CellModeller examples whose OpenCL
integrator or signaling objects cannot run through the callback compatibility
loader. A migrated row names a self-contained CellModeller2 model with typed
rate equations and restartable native orchestration. It does not by itself
claim agreement with a recorded legacy trajectory.

| Legacy example | Equation family | CellModeller2 model | Status |
|---|---|---|---|
| `ACS2012/EdgeDetectorChamber.py` | five species, one diffusive signal | `examples/legacy/ACS2012/EdgeDetectorChamber.py` | pending |
| `Tutorial_2/Tutorial_2a.py` | one constitutively produced species | `examples/legacy/Tutorial_2/Tutorial_2a.py` | migrated and CPU/Metal exercised |
| `Tutorial_2/Tutorial_2b.py` | two-species nonlinear feedback | `examples/legacy/Tutorial_2/Tutorial_2b.py` | migrated and CPU/Metal exercised |
| `Tutorial_3/Tutorial_3.py` | two species coupled to two diffusive signals | `examples/legacy/Tutorial_3/Tutorial_3.py` | pending |
| `ex2_constGene.py` | one constitutively produced species | `examples/legacy/ex2_constGene.py` | migrated and CPU/Metal exercised |
| `ex2a_dilution.py` | one species with growth dilution only | `examples/legacy/ex2a_dilution.py` | migrated and CPU/Metal exercised |
| `ex2b_diluteRepression.py` | dilution plus Hill repression | `examples/legacy/ex2b_diluteRepression.py` | migrated and CPU/Metal exercised |
| `ex3_simpleSignal.py` | one species coupled to one diffusive signal | `examples/legacy/ex3_simpleSignal.py` | pending |
| `ex4_simpleCellCellSignaling.py` | three species coupled to one diffusive signal | `examples/legacy/ex4_simpleCellCellSignaling.py` | pending |

## Shared migration decisions

The species-only ports preserve the OpenCL equations as `RatePlanBuilder`
graphs and keep the declared initial concentrations and growth rates. The
legacy biophysics assigns `cell.volume = cell.length`, so its `targetVol`
division tests are migrated explicitly as target *lengths* rather than silently
switching to capsule volume. `UniformLengthDivision` stores one threshold per
stable cell ID, samples daughter thresholds from the original uniform ranges,
and uses the controller's checkpointed random stream.

The legacy `jitter_z=False` path used ambient NumPy randomness for small
daughter-axis perturbations. The migrated policy preserves the xy-only
perturbation geometry but deliberately moves its draws into the explicit
controller stream. This is reproducible replacement behavior, not a claim of
bitwise identity with an unseeded legacy NumPy global.

Every migrated model currently requests one exact mechanics pass per biological
step. That matches the existing adapter behavior but does not resolve how
legacy `max_substeps=8` should map to the new solver; the outer-relaxation
decision remains gated on colony-level trajectory evidence. Legacy renderer
colors are also not encoded into simulation state. Their preserve/replace/retire
decision belongs to the renderer-family audit rather than the biochemical
equation port.

# Legacy example migration ledger

This ledger tracks the nine bundled CellModeller examples whose OpenCL integrator or signaling objects cannot run through the callback compatibility loader. A migrated row names a self-contained CellModeller2 model with typed rate equations and restartable native orchestration. It does not by itself claim agreement with a recorded legacy trajectory.

| Legacy example | Equation family | CellModeller2 model | Status |
| --- | --- | --- | --- |
| `ACS2012/EdgeDetectorChamber.py` | five species, one diffusive signal | `examples/legacy/ACS2012/EdgeDetectorChamber.py` | migrated and CPU/Metal exercised |
| `Tutorial_2/Tutorial_2a.py` | one constitutively produced species | `examples/legacy/Tutorial_2/Tutorial_2a.py` | migrated and CPU/Metal exercised |
| `Tutorial_2/Tutorial_2b.py` | two-species nonlinear feedback | `examples/legacy/Tutorial_2/Tutorial_2b.py` | migrated and CPU/Metal exercised |
| `Tutorial_3/Tutorial_3.py` | two species coupled to two diffusive signals | `examples/legacy/Tutorial_3/Tutorial_3.py` | migrated and CPU/Metal exercised |
| `ex2_constGene.py` | one constitutively produced species | `examples/legacy/ex2_constGene.py` | migrated and CPU/Metal exercised |
| `ex2a_dilution.py` | one species with growth dilution only | `examples/legacy/ex2a_dilution.py` | migrated and CPU/Metal exercised |
| `ex2b_diluteRepression.py` | dilution plus Hill repression | `examples/legacy/ex2b_diluteRepression.py` | migrated and CPU/Metal exercised |
| `ex3_simpleSignal.py` | one species coupled to one diffusive signal | `examples/legacy/ex3_simpleSignal.py` | migrated and CPU/Metal exercised |
| `ex4_simpleCellCellSignaling.py` | three species coupled to one diffusive signal | `examples/legacy/ex4_simpleCellCellSignaling.py` | migrated and CPU/Metal exercised |

## Shared migration decisions

The species-only ports preserve the OpenCL equations as `RatePlanBuilder` graphs and keep the declared initial concentrations and growth rates. The legacy biophysics assigns `cell.volume = cell.length`, so its `targetVol` division tests are migrated explicitly as target _lengths_ rather than silently switching to capsule volume. `UniformLengthDivision` stores one threshold per stable cell ID, samples daughter thresholds from the original uniform ranges, and uses the controller's checkpointed random stream.

The legacy `jitter_z=False` path used ambient NumPy randomness for small daughter-axis perturbations. The migrated policy preserves the xy-only perturbation geometry but deliberately moves its draws into the explicit controller stream. This is reproducible replacement behavior, not a claim of bitwise identity with an unseeded legacy NumPy global.

Every migrated model requests one exact mechanics pass per biological step. The callback adapter separately maps legacy `max_substeps` to bounded new-contact-frontier relaxation, and recorded colony trajectories now establish both contracts. Legacy renderer colors are not encoded into simulation state; the renderer-family audit replaces them with typed viewer mappings and records the explicit disposition of every other legacy renderer.

## Coupled-rate translation

The legacy signaling callbacks expose extracellular derivatives as concentration rates and divide cell exchange by the `4 * 4 * 4 = 64` voxel volume before returning them. CellModeller2 coupled plans expose extracellular _amount_ rates; the native scatter operation performs the voxel-volume division. The migrated signal outputs therefore return the unscaled exchange amount. Intracellular equations that explicitly used `area / gridVolume` retain their division by 64.

The migrations use conventional diffusion coefficients rather than preserving the legacy implementation's accidental extra factor of one sixth. They retain the declared no-flux boundary behavior and integration choice: forward Euler for `ex3_simpleSignal.py`, and Crank-Nicolson for the other three models. The Crank-Nicolson ports set an absolute residual tolerance of `1e-12`; the engine default would accept a zero field while these models' initially small signal sources remained below its absolute threshold.

`EdgeDetectorChamber.py` initialized `targetVol` but tested the absent `target_volume` attribute against 3.0. Its migration restores the evident intended uniform division threshold of 3.5 to 4.0. This is an explicit repair, not a claim that the original typo's behavior was reproduced.

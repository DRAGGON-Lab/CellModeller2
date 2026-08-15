# Legacy compatibility ledger

This ledger defines the finish line. `specified` means the intended behavior
has an initial contract; it does not mean a GPU implementation exists.

| Priority | Feature | Legacy source | Intended CellModeller2 behavior | Status |
|---|---|---|---|---|
| P0 | Stable IDs and compact slots | `Simulator` | IDs never alias slots; division records lineage | CPU and Metal conformant; CUDA hardware validation pending |
| P0 | Cell creation and division | `Simulator`, `CLBacterium` | deterministic equal and fractional division with explicit jitter policy | equal and asymmetric division conformant on CPU and Metal; CUDA hardware validation pending |
| P0 | Exponential-Euler length growth | `CLBacterium` | explicit `length += rate * length * dt` reference step | CPU and Metal conformant; CUDA hardware validation pending |
| P0 | 2D/3D rod mechanics | `CLBacterium` | typed contact graph, diagnosed solve, and bounded correction integration | CPU and Metal solve/integration conformant; CUDA builds with hardware validation pending |
| P0 | Cell-cell contacts | `CLBacterium.cl` | dynamic incidence graph; no silent contact cap | deterministic sweep-and-prune staging and native Metal narrow phase conformant; native CUDA candidate kernels implemented with build and hardware validation pending |
| P0 | Plane and sphere constraints | `CLBacterium.cl` | typed constraint records; no sentinel cell IDs | CPU and native Metal geometry/mechanics conformant; native CUDA builds with hardware validation pending |
| P0 | Mechanics solver | `CLBacterium` | regularized matrix-free CG/PCG with residual report | CPU and native Metal CG conformant; native CUDA builds with hardware validation pending |
| P0 | Species Euler integration | `CLEulerIntegrator` | typed rate plan and growth dilution | CPU and native Metal conformant; native CUDA builds with hardware validation pending |
| P0 | Grid signaling | `GridDiffusion` | diffusion, optional advection, declared boundaries | CPU and native Metal conformant and checkpointed; native CUDA builds with hardware validation pending |
| P0 | Coupled cell/grid rates | signal integrators | device-resident sample, rate, scatter, and update | CPU and native Metal conformant with exact checkpoint state; native CUDA builds with hardware validation pending |
| P0 | Checkpoint and exact resume | pickle output | versioned, non-executable checkpoint with provenance | JSON v5 native and authenticated controller state with explicit v1-v4 migration; CPU and Metal continuation conformant; CUDA hardware validation pending |
| P0 | Batch execution | batch scripts | `cm2 run`, backend/device/seed selection | implemented with deterministic construction, collision preflight, periodic checkpoints, and provenance |
| P1 | Legacy Python model adapter | module regulator | adapter for maintained `setup/init/update/divide` models | growth/mechanics callbacks, setup facade, and exact batch resume implemented; signaling models require typed-plan migration |
| P1 | Legacy pickle import | `Simulator` | one-way migration into the new checkpoint schema | trusted native-state importer implemented with explicit loss provenance |
| P1 | Crank-Nicolson signaling | `CLCrankNicIntegrator` | implement intended equation after legacy behavior audit | intended equation audited, specified, and implemented on CPU; native Metal and CUDA pending |
| P1 | Neighbor reporting | `CLBacterium` | stable cell IDs derived from current contact graph | implemented as deterministic backend-neutral graph views |
| P2 | Fixed-position cells | `CLFixedPosition` | reference and native growth-only state | audit needed |
| P2 | Neighbor diffusion | `NeighbourDiffusion` | retain only after model and attribute audit | dormant/ambiguous |
| P2 | SBML import | `SBMLImport` | libSBML to typed reaction model; no generated Python source | redesign needed |
| P2 | Interactive viewer | PyQt/OpenGL GUI | separate consumer of snapshots; no engine ownership | redesign needed |
| P2 | Analysis scripts | `Scripts` | replace with documented Parquet/Zarr workflows | inventory needed |

Each row advances through: `audit` -> `specified` -> `CPU reference` ->
`Metal conformance` and `CUDA conformance` -> `backend-complete`.

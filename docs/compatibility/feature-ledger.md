# Legacy compatibility ledger

This ledger defines the scientific and workflow boundary inherited from CellModeller. CPU and Metal are feature complete across the maintained modeling surface. CUDA is under active development against the same contracts; its progress is tracked by the centralized [validation workflow](../development/validation.md), rather than repeated in every row below.

`Specified` means that CellModeller2 has an explicit replacement contract. `Complete` means the feature is implemented, tested, and included in the CPU/Metal application evidence. A retired feature has an explicit, documented disposition and is not part of backend parity.

| Priority | Feature | Legacy source | CellModeller2 contract | Status |
| --- | --- | --- | --- | --- |
| P0 | Stable IDs and compact slots | `Simulator` | IDs never alias slots; division records lineage | Complete |
| P0 | Cell creation and division | `Simulator`, `CLBacterium` | Deterministic equal and fractional division with explicit jitter policy | Complete |
| P0 | Exponential-Euler length growth | `CLBacterium` | Explicit `length += rate * length * dt` reference step | Complete |
| P0 | 2D/3D rod mechanics | `CLBacterium` | Typed contact graph, diagnosed solve, bounded correction integration, and checkpointed contact-frontier relaxation | Complete |
| P0 | Cell-cell contacts | `CLBacterium.cl` | Deterministic sweep-and-prune staging and dynamic native incidence graph with no silent contact cap | Complete |
| P0 | Plane and sphere constraints | `CLBacterium.cl` | Typed constraint records participating in the native mechanics system; no sentinel cell IDs | Complete |
| P0 | Mechanics solver | `CLBacterium` | Regularized matrix-free CG/PCG with convergence and breakdown diagnostics | Complete |
| P0 | Species Euler integration | `CLEulerIntegrator` | Typed rate plans, legacy-compatible effective-volume dilution, and simultaneous updates | Complete |
| P0 | Grid signaling | `GridDiffusion` | Diffusion, optional advection, declared boundaries, spatial affine source/loss fields, explicit stability checks, and checkpointed state | Complete |
| P0 | Coupled cell/grid rates | Signal integrators | Device-resident sample, rate, transport, scatter, and simultaneous commit | Complete |
| P0 | Checkpoint and exact resume | Pickle output | Versioned non-executable JSON checkpoint with provenance, integrity, schema migration, and authenticated controller state | Complete |
| P0 | Batch execution | Batch scripts | Deterministic `cm run`, explicit backend/device/seed selection, stopping rules, collision safety, and data-only run manifests | Complete |
| P0 | Native model orchestration | Simulator/module lifecycle | Restartable controller with typed regulation, division, exact-pass mechanics, model state, and runtime RNG | Complete |
| P1 | Legacy Python model adapter | Module regulator | Compatibility for maintained setup/init/update/divide models, constraints, neighbors, division, host species, and exact resume | Complete |
| P1 | Legacy pickle import | `Simulator` | Explicitly trusted, lossy, one-way migration into the new checkpoint schema | Complete |
| P1 | Crank-Nicolson signaling | `CLCrankNicIntegrator` | Diagnosed native standalone and coupled solver for the intended equation | Complete |
| P1 | Neighbor reporting | `CLBacterium` | Deterministic stable-ID graph views derived from the current contact graph | Complete |
| P2 | Fixed-position cells | `CLFixedPosition` | Persistent fixed rods with projected mechanics and continuing biological state | Complete |
| Retired | Neighbor diffusion | `NeighbourDiffusion` | Do not preserve the dead, dimensionally unspecified graph loop; require a new typed contact-flux proposal | Retired after audit |
| P2 | SBML import | `SBMLImport` | Bounded libSBML Core subset compiled to typed species-rate IR with no generated source | Complete |
| P2 | Interactive viewer | PyQt/OpenGL GUI | Independent scene consumer with no engine ownership; rod and grid workflows replaced and unused renderer families explicitly retired | Complete |
| P2 | Analysis scripts | `Scripts` | Immutable Parquet/Zarr export, typed provenance, derived contacts, and verified lazy analysis recipes | Complete |

The supporting evidence includes a source-pinned [25-example matrix](legacy-example-matrix.md), five [recorded legacy trajectory contracts](legacy-trajectory-evidence.md), the [mechanics audit](legacy-mechanics-audit.md), the [signaling audit](legacy-signaling-audit.md), and the [viewer](legacy-viewer-audit.md) and [analysis](legacy-analysis-audit.md) dispositions.

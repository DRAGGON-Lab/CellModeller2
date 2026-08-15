# Legacy Python model audit

The legacy repository contains 25 example modules, and all 25 expose the same
regulation lifecycle: `setup(sim)`, `init(cell)`, `update(cells)`, and
`divide(parent, d1, d2)`. The callback layer is ordinary host Python even when
the selected physics and integration implementations use OpenCL.

The maintained examples divide into three migration groups:

- growth and mechanics models whose callbacks set division flags, growth rate,
  cell type, display color, and arbitrary per-cell metadata;
- species or signaling models that additionally return OpenCL source strings;
- models that use optional constraints, neighbor reporting, asymmetric
  division, or GUI renderers.

## Adapter boundary

`LegacyModelAdapter` is the first compatibility boundary for the host callback
lifecycle. It supplies mutable `LegacyCell` objects keyed by stable cell ID,
retains arbitrary Python attributes across equal division, propagates callback
changes to growth rate, cell type, and species, and refreshes geometry,
effective growth, sampled signals, and optional neighbor IDs from native state.
It preserves the legacy order: regulation, division, native integration,
mechanics, then cell-state refresh.

The adapter is deliberately explicit rather than pretending the old OpenCL
objects still exist. It requires construction around an empty native
`Simulation`. `build_legacy_model` supplies an opt-in `setup(sim)` facade and
temporary import shims for `CLBacterium`, `ModuleRegulator`, and renderer
declarations; the shims are removed after construction and do not install a
shadow `CellModeller` package. `controller_state` serializes arbitrary JSON-like
attributes, tuples, numeric NumPy arrays, adapter options, and the exact random
stream into the authenticated v5 controller payload; `from_controller_state`
restores that data without executable content. `cm2 run --legacy-model` writes
the controller payload in periodic and final checkpoints; combining
`--legacy-model` with `--resume` reloads the callbacks only after verifying the
source digest and reuses the recorded seed and parameters. Geometry remains
engine-owned. Setup is replayed from the original seed on resume while the
checkpointed random stream is restored only for subsequent callbacks, so
randomized founder construction cannot consume future runtime draws. Mutable
module globals other than this controlled random binding are not checkpointed;
evolving model state must live on cells or in native arrays. Callback attempts
to mutate position, direction, length, or radius fail explicitly.
Equal and asymmetric division are supported, including an explicitly seeded
compatibility policy for the legacy per-daughter direction jitter. The adapter
normalizes the two positive `cell.asymm` weights into the native daughter
fraction. This implements the legacy API's dormant intent; the old
`CLBacterium` accepted `f1` and `f2` but discarded them before geometry was
updated. `alternate_divisions=True` preserves the legacy 90-degree xy-plane
axis rotation as a backend-neutral topology operation. Controller v3 records
the selected orientation policy; controller v2 migrates to non-alternating
division.

OpenCL strings returned by `specRateCL()` and `sigRateCL()` are not accepted or
translated. Those models must express equations as `SpeciesRatePlan` or
`CoupledRatePlan`, which lets CUDA and Metal compile their own native kernels
without treating OpenCL C as an intermediate language. GUI renderers are also
outside the adapter: a future viewer consumes snapshots without owning engine
state.

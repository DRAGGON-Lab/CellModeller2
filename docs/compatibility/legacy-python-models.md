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
shadow `CellModeller` package. Batch checkpointing of arbitrary Python
attributes remains separate work. Geometry remains engine-owned, and callback
attempts to mutate position, direction, length, or radius fail explicitly.
Equal division is supported, including an explicitly seeded compatibility
policy for the legacy per-daughter direction jitter; asymmetric division fails
explicitly until its native lifecycle contract is implemented.

OpenCL strings returned by `specRateCL()` and `sigRateCL()` are not accepted or
translated. Those models must express equations as `SpeciesRatePlan` or
`CoupledRatePlan`, which lets CUDA and Metal compile their own native kernels
without treating OpenCL C as an intermediate language. GUI renderers are also
outside the adapter: a future viewer consumes snapshots without owning engine
state.

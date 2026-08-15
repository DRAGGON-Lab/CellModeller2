# ADR 0012: headless scene protocol and independent viewer

- Status: accepted
- Date: 2026-08-15

## Context

The legacy PyQt/OpenGL application owns the OpenCL device, simulator lifetime,
model imports, simulation clock, pickle persistence, selection, and rendering.
Models install renderers into the simulation itself. This prevents headless
execution, makes renderer behavior depend on mutable implementation details,
and would force CUDA and Metal concerns into the UI.

CellModeller2 needs interactive inspection on Apple and NVIDIA systems without
creating a fourth simulation implementation or weakening the native backend
contract.

## Decision

The simulation engine remains headless. It exposes an immutable, versioned
scene frame containing only observable presentation data:

- schema name and version;
- simulation time and source backend metadata;
- rods in stable cell-ID order, including geometry and inspectable scalar
  values;
- optional signal-grid geometry, boundaries, and channel-major levels; and
- stable lineage parents for selection context.

The JSON interchange uses finite numbers, closed object schemas, and decimal
strings for 64-bit identifiers. A frame is a projection, not a checkpoint: it
contains no rate plans, solver configuration, executable source, controller
state, or authority to resume a simulation.

The viewer is a separate TypeScript web application using an instanced graphics
renderer. It loads scene files without server authority. Its live controller
publishes the same frames and accepts only typed play, pause, step, reset,
checkpoint, and frame-request commands through an authenticated loopback
protocol. The controller—not the browser—constructs simulations and selects an
explicit backend and device.

CUDA and Metal remain native scientific-compute APIs. Viewer graphics are not
a compute backend and do not participate in numerical conformance. The CPU,
Metal, and CUDA acceptance gate is equality of scene semantics for the same
engine state. Renderer tests cover framing, picking, color mapping, and command
state separately.

Presentation mappings are declarative viewer state. The initial mapping modes
are cell type, species channel, growth rate, and fixed state. Model callbacks do
not mutate RGB fields in the engine. New observable fields require a scene
schema change or a typed extension, not arbitrary object traversal.

## Initial delivery order

1. Implement and test the scene-frame projection and strict JSON reader/writer.
2. Add a read-only viewer for rods, grid slices, camera navigation, picking,
   selection details, and declarative color mapping.
3. Add a local controller using the existing batch model/checkpoint APIs and
   the same scene schema.
4. Profile large colonies before selecting a binary transport. JSON remains
   the inspectable correctness format; a future binary encoding must preserve
   the same logical schema.

## Consequences

- Viewer work is implemented once and consumes CPU, Metal, and CUDA results
  through the same boundary.
- Headless batch execution has no UI dependency.
- Scene files are safe to inspect but cannot resume a run.
- Arbitrary legacy renderers are not automatically portable. Plant, mesh, and
  periodic-domain views wait for typed engine concepts.
- The viewer can evolve or be replaced without changing backend kernels.

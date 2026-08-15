# Legacy interactive viewer audit

## Scope

This audit covers the CellModeller 4 desktop entry point and its rendering
layer:

- `Scripts/CellModellerGUI.py`;
- `CellModeller/GUI/PyGLCMViewer.py`;
- `CellModeller/GUI/PyGLWidget.py`;
- `CellModeller/GUI/Renderers.py`;
- the renderer hooks in `CellModeller/Simulator.py`; and
- model calls to `Simulator.addRenderer`.

The goal is to preserve useful interactive behavior without preserving the
old ownership graph or graphics implementation.

## What the legacy viewer owns

`PyGLCMViewer` does more than display state. It enumerates OpenCL platforms and
devices, constructs and destroys `Simulator` instances, imports and reloads
model modules, advances the simulation from a zero-interval Qt timer, toggles
pickle output, and restores executable pickle payloads. This makes the UI an
owner of compute policy and model execution.

Models also import OpenGL renderer classes and register renderer instances on
the simulator. Those renderers retain the simulator and read mutable
`cellStates`, signaling integrator arrays, meshes, and arbitrary Python cell
attributes directly. The simulator therefore depends on presentation objects,
while presentation code depends on nearly every mutable simulation detail.

The graphics layer uses the compatibility OpenGL API: `QGLWidget`, display
lists, matrix stacks, GLU quadrics, immediate-mode primitives, and the OpenGL
name stack for picking. Rendering and stepping both run on the GUI thread.

## User-visible behavior

The maintained rod-cell workflows use a smaller set of capabilities:

- load a model or saved state;
- run, pause, single-step, and reset;
- save simulation state;
- orbit, pan, zoom, and frame the colony;
- draw 2D or 3D spherocylinders with per-cell color;
- draw a selected-cell outline and inspect its properties; and
- display scalar signaling-grid slices.

The renderer file also contains sphere, plant, periodic-image, static-mesh,
and experimental mesh renderers. These are not general viewer requirements.
They depend on engine concepts that CellModeller2 does not currently expose and
must be specified as engine features before a viewer can represent them.

## Compatibility hazards

The old viewer cannot be used as a compatibility oracle for several reasons:

- it loads arbitrary pickle objects and may execute source stored inside them;
- model files choose concrete renderer classes and mutate presentation state;
- selection assumes a cell ID fits the legacy OpenGL integer name mechanism;
- render refresh depends on mutable `stepNum` and renderer-local caches;
- property inspection walks arbitrary `CellState.__dict__` values; and
- the signaling renderer reaches into integrator-owned arrays rather than a
  declared observation interface.

Visual pixel equality is neither defined nor scientifically meaningful. The
compatibility target is equality of the typed scene data presented to the
viewer, followed by behavioral tests of selection and controls.

## Migration boundary

CellModeller2 will not attach renderers to `Simulation`, expose backend memory
to a UI, or let the UI choose CUDA or Metal devices by reaching into a runtime.
Instead, the engine produces immutable scene frames in stable cell-ID order.
A separate controller owns run policy and produces those frames after complete
simulation steps.

The initial scene vocabulary contains rods and an optional scalar signal grid.
Cell color is a presentation mapping over declared fields such as cell type,
species level, growth rate, and fixed state; it is not a mutable engine field.
Stable 64-bit cell IDs cross the JSON boundary as decimal strings so browser
consumers cannot lose precision.

Legacy pickle viewing goes through the existing trusted, explicit one-way
import command. The new viewer only accepts the non-executable CellModeller2
checkpoint and scene formats.

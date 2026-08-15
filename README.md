# CellModeller2

CellModeller2 is a clean, scientifically testable successor to CellModeller.
It keeps Python as the modeling interface while implementing the simulation
engine in C++23 with independent native CPU, Apple Metal, and NVIDIA CUDA
backends.

The project is being built as vertical, cross-backend slices. A feature is not
considered implemented until its scientific behavior is specified, the CPU
reference is tested, and every advertised GPU backend passes the same
conformance scenarios without silently falling back to the CPU.

Compatibility is evidence-driven rather than file-count-driven. The legacy
`NeighbourDiffusion` experiment is explicitly retired because it is
non-runnable and does not define a physically weighted transport contract; a
future contact-flux model would be a new cross-backend feature.

## Current slice

The initial slice establishes:

- stable cell identifiers separated from compact storage slots;
- structure-of-arrays state owned by the engine;
- deterministic growth and division semantics;
- persistent fixed rod cells with continuing biological state and native projected mechanics;
- fixed-schema species state and typed CPU/Metal/CUDA Euler rate plans with growth dilution;
- versioned JSON checkpoints with exact state restore, provenance, and integrity checks;
- deterministic `cm2` batch execution with explicit backend, device, seed, and parameters;
- immutable Parquet/Zarr analysis exports with typed provenance and stable identities;
- validated CPU and native Metal signal grids with explicit transport boundaries,
  plus a diagnosed CPU Crank-Nicolson reference solver;
- typed coupled cell/grid rate plans with simultaneous CPU and native Metal stages;
- CPU capsule contacts and native Metal/CUDA contact implementations;
- typed CPU and native Metal/CUDA plane and inside/outside sphere constraints;
- matrix-free CPU and native Metal/CUDA rod-mechanics solvers with diagnostics;
- a C++ CPU reference backend;
- a small Python API backed by nanobind;
- C++ and Python conformance tests;
- native Metal growth and CUDA growth implementations;
- explicit backend device enumeration and selection;
- the cross-backend architecture and feature ledger.

Metal growth is validated on Apple GPU hardware. CUDA growth is implemented
with the native CUDA Runtime API and CUDA C++, but remains unadvertised until
the conformance suite passes on NVIDIA hardware. Deterministic equal division
is backend-neutral and conformed on CPU and Metal. Metal mechanics runs native
MSL row assembly, operator, vector, and reduction kernels and matches the CPU
solver on Apple GPU hardware. CUDA now has the equivalent native mechanics
implementation, but both its contact and mechanics conformance still require
execution on NVIDIA hardware. `relax_cell_mechanics` applies a converged result
through the shared bounded-rotation and non-shortening integration contract.
Both GPU contact paths consume deterministic sweep-and-prune capsule candidates
before their native narrow phases. Plane and sphere constraints have native
Metal and CUDA geometry and mechanics paths, where their rows participate in
the same matrix-free solve and relaxation path as cell-cell contacts. Metal is
conformant on Apple GPU hardware; the CUDA implementation compiles against the
12.8 toolkit but still requires execution of the shared fixtures on NVIDIA
hardware. Fixed cells use the same native projection in each mechanics system;
the CUDA fixed-cell cases share that pending hardware gate.

The typed species plan is conformant on CPU and native Metal, including every
declared instruction, legacy-compatible effective-volume dilution, and
simultaneous Euler updates. CUDA has an independent native interpreter that
compiles against the 12.8 toolkit; the same shared scenario still needs to run
on NVIDIA hardware before CUDA species conformance is claimed.

Run `scripts/run_cuda_compile_check.sh` to reproduce the driverless native CUDA
source/link gate in the versioned CUDA development container. Its passing result
is deliberately distinct from `scripts/run_cuda_conformance.sh`, which requires
an NVIDIA GPU and is the only command that can establish CUDA conformance.

Signal-grid state has CPU, native Metal, and native CUDA implementations for
conventional diffusion, conservative vector upwind advection,
no-flux/periodic/fixed boundaries, trilinear sampling, stability checks, and
non-negative concentration updates. The MSL transport kernel is conformant on
Apple GPU hardware. The independent CUDA kernel compiles and links against the
12.8 toolkit; its shared conformance scenario still requires NVIDIA hardware.
The checkpointed Crank-Nicolson option has diagnosed CPU, native Metal, and
native CUDA solvers. The CUDA implementation compiles and links for `sm_75`;
its shared conformance cases still require NVIDIA hardware.

Coupled rate plans add sampled-signal inputs and separate intracellular
concentration-rate and extracellular amount-rate outputs. CPU and native Metal
implement the complete old-grid sample, post-growth dilution, transport,
trilinear scatter, and simultaneous commit contract. Metal runs cell and grid
kernels in one command buffer and passes the shared 513-cell hardware gate on
Apple GPU hardware. Its deterministic grid-thread gather is the correctness
path; a sparse source reduction remains a scaling optimization. Native CUDA
now implements the same two-kernel operation and compiles with the 12.8
toolkit; the shared fixture still requires execution on NVIDIA hardware.

Checkpoints preserve compact slot order, stable identity allocation, complete
lineage, constraints, species rates, signal-grid geometry and levels,
concentrations, cell geometry, coupled rates, and simulation time. Schema v5
adds the signal integration and solver configuration, and schema v6 adds the
persistent fixed-cell attribute. The reader explicitly migrates v1 through v5,
using Forward Euler defaults for older signal grids and movable cells for older
cell records. Files
contain data only: loading never imports a model or evaluates source text.
Writes use an atomic replace and each file carries a SHA-256 digest over the
simulation payload.

```python
from cellmodeller2 import load_checkpoint, save_checkpoint

save_checkpoint(simulation, "run.cm2.json", provenance={"model": "colony-a"})
resumed = load_checkpoint("run.cm2.json")
```

## Import a typed SBML rate model

Install the optional `sbml` dependency, then compile a bounded SBML Level 3
Version 2 Core model into the same data-only rate plan used by native models:

```console
uv sync --extra sbml
```

```python
from cellmodeller2 import CellInit, Simulation, load_sbml

model = load_sbml("model.xml")
simulation = Simulation(species_count=model.species_count)
cell = CellInit()
cell.species = list(model.initial_levels)
simulation.add_cell(cell)
simulation.set_species_rate_plan(model.rate_plan)
```

The importer uses libSBML for parsing and consistency diagnostics. It accepts
the explicit subset in [ADR 0011](docs/architecture/0011-sbml-import.md) and
fails on unsupported semantics; it never generates or evaluates source code.

## Export a viewer scene

Capture presentation data without exposing engine or device memory to the
viewer:

```python
from cellmodeller2 import capture_scene, save_scene

frame = capture_scene(simulation)
save_scene(frame, "colony.cm2.scene.json")
```

Scene v1 preserves rods, stable IDs and active lineage parents, scalar cell
fields, every species channel, and an optional signaling grid. It is strict,
integrity-checked JSON and intentionally cannot resume a simulation. The
[scene format](docs/formats/scene-v1.md) defines its complete wire layout.

Generate a deterministic example and open the standalone viewer:

```console
uv run python examples/viewer_scene.py --output viewer-demo.cm2.scene.json
pnpm --dir viewer install
pnpm --dir viewer dev
```

The TypeScript/Three.js viewer verifies the cross-language digest before it
renders anything. It provides instanced rods, orbit/pan/zoom, cell picking and
inspection, declarative cell coloring, and signal-grid slicing without owning
or importing the simulation engine. See the [viewer README](viewer/README.md).

Build the viewer once, then launch an authenticated live session from a model:

```console
uv sync --group dev --extra viewer
pnpm --dir viewer build
uv run cm2 view \
  --model examples/batch_model.py \
  --backend cpu \
  --seed 42 \
  --parameter growth_rate=0.2 \
  --dt 0.05 \
  --checkpoint-output results/live.cm2.json \
  --open
```

The Python process remains the sole owner of the model, backend, clock, reset,
and checkpoint destination. It prints a per-process tokenized loopback URL;
the browser can request only play, pause, bounded steps, reset, the current
frame, and a checkpoint to that preconfigured path. Every live frame is a full
verified scene-v1 document. The wire contract is documented in the
[live viewer protocol](docs/protocols/live-viewer-v1.md).

## Export analysis datasets

Install the optional analysis dependencies, then export an explicit,
time-ordered checkpoint series:

```console
uv sync --extra analysis
uv run cm2 export-analysis \
  results/step-00025.cm2.json \
  results/step-00050.cm2.json \
  --output results/run.cm2.dataset \
  --contacts \
  --external-contacts
```

The immutable dataset contains explicit-schema Parquet frame, cell, and
species tables. Requested cell and constraint contacts are reconstructed on
the selected native backend and record their device and parameters. Signal
fields are stored as Zarr v3 arrays in named `(frame, channel, x, y, z)`
dimensions; a geometry change starts a new epoch. The manifest records source
checkpoint digests, schema versions, model provenance, table schemas, row
counts, derivations, and output digests without embedding absolute input paths
unless `--path-provenance` is passed. Existing datasets require the explicit
`--overwrite` flag.

Contact reconstruction defaults to the CPU reference. Metal is selectable on
conformant Apple hardware. CUDA state export is available, while CUDA contact
export remains gated until the shared contact fixture passes on NVIDIA
hardware.

Verified lazy Polars recipes cover radial counts and species means, cylinder
or full-capsule length histograms, the legacy length-weighted XY line-density
proxy, unique stable-ID neighbor edges, sister-neighbor counts, and named
signal slices and voxel time courses. Their bin, null, weighting, lineage, and
dimension semantics are documented in the [analysis recipe guide](docs/analysis/recipes.md).

`backend_device_count(kind)` enumerates native devices and every `Simulation`
constructor accepts `device_index`. Invalid indices fail explicitly; CUDA does
not inherit mutable process-thread device selection, and Metal does not
silently substitute the system default for a requested index.

## Run a batch model

List the devices visible to the native runtimes, then run a model on an
explicit backend and device:

```console
uv run cm2 devices
uv run cm2 run \
  --model examples/batch_model.py \
  --backend metal \
  --device-index 0 \
  --seed 42 \
  --parameter growth_rate=0.25 \
  --steps 100 \
  --dt 0.05 \
  --stop-cell-count 10000 \
  --checkpoint-every 25 \
  --output results/colony.cm2.json
```

A model is trusted Python code and must define `build(context)`. Construct the
simulation with `context.simulation()`, use `context.rng` for seeded model
randomness, and read JSON parameters from `context.parameters`. `build` may
return the simulation directly or a structural `SimulationController` owning
it. Controllers implement `step(dt)` and return complete finite, non-null JSON
from `controller_state()`. The runner rejects native state on another backend
or device. Existing final or periodic outputs are never replaced without
`--overwrite`.

Runtime-stochastic controllers should persist dedicated streams with
`capture_random_state` and restore them with `restore_random_state`. To resume a
controller checkpoint, the same explicitly selected model defines
`resume(context, checkpoint)` and must wrap `checkpoint.simulation`:

```console
uv run cm2 run \
  --model models/regulated_colony.py \
  --resume results/colony.cm2.json \
  --backend metal \
  --steps 100 \
  --dt 0.05 \
  --output results/colony-resumed.cm2.json
```

The model digest is checked against checkpoint provenance before its source is
compiled. CellModeller2 never executes a model path merely because it appears
in a checkpoint. Bare native checkpoints with a null controller still resume
with `--resume` alone.

`NativeController` provides the standard lifecycle for new models. Regulation
returns a typed `StepPlan`; the controller validates all cell updates and
division requests before applying them, runs native integration, then performs
the exact number of mechanics passes in `MechanicsConfig`. Its standard payload
also persists model JSON state, controller step count, model identity, random
state, and mechanics configuration. See
[`examples/native_controller.py`](examples/native_controller.py) for a complete
`build`/`resume` model.

`--steps` is always the deterministic maximum. When `--stop-cell-count` is
present, the runner also stops before the first step if the initial colony has
already reached the threshold, or immediately after the first step that does.
The final checkpoint records the requested maximum, actual completed steps,
threshold, and whether `step_limit` or `cell_count` ended the run. This replaces
legacy termination tied indirectly to preallocated cell capacity.

Parameter sweeps and replicates can be declared as strict data rather than
hard-coded shell or Python loops. Each run-manifest job fixes its ID, model
path and digest, backend/device, seed, JSON parameters, complete stopping rule,
checkpoint interval, and output path. Execute one named job per scheduler task:

```console
uv run cm2 run-manifest experiments/gamma.cm2.runs.json \
  --job gamma-0.10-replicate-001
```

Manifest parsing never imports a model, and execution checks the model digest
before compilation. Relative paths resolve from the manifest location, output
names are checked for cross-job periodic collisions, and the manifest/job
identity enters checkpoint provenance. See the
[run manifest v1 format](docs/formats/run-manifest-v1.md).

Resume a data-only checkpoint on any available backend:

```console
uv run cm2 run \
  --resume results/colony.cm2.json \
  --backend cpu \
  --device-index 0 \
  --steps 100 \
  --dt 0.05 \
  --output results/colony-resumed.cm2.json
```

Maintained CellModeller 1 growth/mechanics models can run unchanged through the
explicit compatibility loader. Resuming reloads the same source only after its
digest has been checked, then restores callback attributes and random state from
the authenticated controller payload:

```console
uv run cm2 run \
  --legacy-model ../CellModeller/Examples/ex1a_simpleGrowth2D.py \
  --backend metal \
  --seed 42 \
  --steps 100 \
  --dt 0.05 \
  --output results/legacy.cm2.json

uv run cm2 run \
  --legacy-model ../CellModeller/Examples/ex1a_simpleGrowth2D.py \
  --resume results/legacy.cm2.json \
  --backend metal \
  --steps 100 \
  --dt 0.05 \
  --output results/legacy-resumed.cm2.json
```

Legacy OpenCL rate-source strings are intentionally not accepted. Species and
signaling models must migrate their equations to the typed rate-plan APIs so
Metal and CUDA continue to use native kernels.

Trusted CellModeller 1 snapshots have a separate, explicitly lossy one-way
converter. It preserves native geometry, species, stable identity, and lineage;
the required flag acknowledges that callback attributes, constraints, random
state, and rate equations cannot be reconstructed from the old format:

```console
uv run cm2 import-legacy-pickle data/step-00100.pickle \
  --output results/step-00100.cm2.json \
  --dt 0.05 \
  --trust-legacy-pickle \
  --native-state-only
```

## Build the C++ reference tests

```console
cmake -S . -B build/cpu -G Ninja \
  -DCM2_BUILD_PYTHON=OFF \
  -DCM2_BUILD_TESTS=ON
cmake --build build/cpu
ctest --test-dir build/cpu --output-on-failure
```

## Build and test the Python package

```console
uv sync --group dev
uv run pytest
```

## Repository policy

- `cpp/core` contains backend-neutral orchestration and state semantics.
- `cpp/cpu` is the readable numerical reference.
- `cpp/metal` contains native Metal host code and Metal Shading Language kernels.
- `cpp/cuda` contains CUDA C++ host code and kernels.
- Model equations are shared through a typed model representation; mechanics
  kernels remain explicitly native and independently optimized.

See [ADR 0001](docs/architecture/0001-native-backends.md) and the
[feature ledger](docs/compatibility/feature-ledger.md). The contact/mechanics
redesign is specified in [ADR 0002](docs/architecture/0002-contact-mechanics.md)
and grounded in the [legacy mechanics audit](docs/compatibility/legacy-mechanics-audit.md).
The species state and rate representation is specified in
[ADR 0003](docs/architecture/0003-species-rates.md).
The checkpoint schema and exact-resume boundary are specified in
[ADR 0004](docs/architecture/0004-checkpoints.md).
The command-line model and batch-run contract is specified in
[ADR 0005](docs/architecture/0005-batch-execution.md).
The grid transport and conservative cell-coupling contract is specified in
[ADR 0006](docs/architecture/0006-grid-signaling.md) and grounded in the
[legacy signaling audit](docs/compatibility/legacy-signaling-audit.md).
The equal and asymmetric lifecycle geometry is specified in
[ADR 0007](docs/architecture/0007-division.md).
The host callback compatibility boundary is grounded in the
[legacy Python model audit](docs/compatibility/legacy-python-models.md).
The trusted one-way snapshot boundary is specified in the
[legacy pickle migration audit](docs/compatibility/legacy-pickle-import.md).
The independent interactive-viewer boundary is specified in
[ADR 0012](docs/architecture/0012-viewer-boundary.md) and grounded in the
[legacy viewer audit](docs/compatibility/legacy-viewer-audit.md).
The columnar analysis boundary is specified in
[ADR 0013](docs/architecture/0013-analysis-datasets.md) and grounded in the
[legacy analysis audit](docs/compatibility/legacy-analysis-audit.md). Concrete
queries are documented in the [analysis recipe guide](docs/analysis/recipes.md).

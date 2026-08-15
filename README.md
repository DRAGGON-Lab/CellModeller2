# CellModeller2

CellModeller2 is a clean, scientifically testable successor to CellModeller.
It keeps Python as the modeling interface while implementing the simulation
engine in C++23 with independent native CPU, Apple Metal, and NVIDIA CUDA
backends.

The project is being built as vertical, cross-backend slices. A feature is not
considered implemented until its scientific behavior is specified, the CPU
reference is tested, and every advertised GPU backend passes the same
conformance scenarios without silently falling back to the CPU.

## Current slice

The initial slice establishes:

- stable cell identifiers separated from compact storage slots;
- structure-of-arrays state owned by the engine;
- deterministic growth and division semantics;
- persistent fixed rod cells with continuing biological state and native projected mechanics;
- fixed-schema species state and typed CPU/Metal/CUDA Euler rate plans with growth dilution;
- versioned JSON checkpoints with exact state restore, provenance, and integrity checks;
- deterministic `cm2` batch execution with explicit backend, device, seed, and parameters;
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
Both GPU contact paths use an exhaustive correctness stage while the scalable
broad phase is under construction. Plane and sphere constraints have native
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
  --checkpoint-every 25 \
  --output results/colony.cm2.json
```

A model is trusted Python code and must define `build(context)`. Construct the
simulation with `context.simulation()`, use `context.rng` for seeded model
randomness, and read JSON parameters from `context.parameters`. The runner
rejects a model that substitutes another backend or device. Existing final or
periodic outputs are never replaced without `--overwrite`.

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

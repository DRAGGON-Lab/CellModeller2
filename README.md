# CellModeller2

CellModeller2 is a modern, scientifically testable successor to [CellModeller](https://github.com/cellmodeller/CellModeller) for individual-based multicellular modeling. It keeps Python as the modeling interface and moves the simulation engine to C++23, with independent implementations for the CPU, Apple Metal, and NVIDIA CUDA.

## Project status

| Backend | Status | Role |
| --- | --- | --- |
| CPU | **Feature complete** | Readable numerical reference and portable execution backend |
| Apple Metal | **Feature complete** | Native Apple GPU implementation, validated against the CPU and original CellModeller behavior |
| NVIDIA CUDA | **Under active development** | Native CUDA C++ implementation progressing through compile, hardware, and application conformance |

CPU and Metal cover the complete CellModeller modeling workflow: growth and division, rod mechanics and constraints, species and signaling, model regulation, checkpoint/resume, batch execution, visualization, and analysis. Compatibility is defined by scientific behavior rather than by reproducing the original OpenCL implementation line for line. The [feature ledger](docs/compatibility/feature-ledger.md), [legacy example matrix](docs/compatibility/legacy-example-matrix.md), and [recorded trajectory evidence](docs/compatibility/legacy-trajectory-evidence.md) document that boundary.

CUDA uses the native CUDA Runtime API and CUDA C++; it does not use a portability layer or translate Metal kernels. Its current development and validation workflow is documented in the [CUDA environment guide](environments/cuda/README.md).

## Highlights

- Native CPU, Metal, and CUDA backends behind one explicit device API
- Stable cell identities separated from compact simulation storage
- Deterministic growth, equal and asymmetric division, and lineage tracking
- Capsule contacts, plane and sphere constraints, fixed cells, and matrix-free rod mechanics
- Typed species and coupled cell/grid rate plans with no generated shader source
- Forward Euler and Crank-Nicolson signal integration with declared boundary conditions
- Restartable native controllers with seeded, checkpointed runtime randomness
- Versioned, integrity-checked JSON checkpoints with exact resume
- Reproducible batch runs and data-only run manifests
- A standalone Three.js viewer and authenticated live-viewer protocol
- Immutable Parquet/Zarr analysis datasets with typed provenance
- A compatibility loader for maintained CellModeller Python models
- A one-way migration path for trusted legacy pickle snapshots

## Quick start

CellModeller2 requires Python 3.12, CMake, Ninja, a C++23 compiler, and [uv](https://docs.astral.sh/uv/). The default Python build uses the CPU backend.

```console
uv sync --group dev
uv run cm2 devices
uv run cm2 run \
  --model examples/batch_model.py \
  --backend cpu \
  --seed 42 \
  --parameter growth_rate=0.25 \
  --steps 100 \
  --dt 0.05 \
  --output results/colony.cm2.json
```

Run the test suite with:

```console
uv run pytest
```

## Write a model

A native model is trusted Python code that defines `build(context)`. The context constructs the requested backend, exposes the seeded model random stream, and carries JSON parameters supplied by the CLI.

```python
from cellmodeller2 import CellInit, Vec3


def build(context):
    simulation = context.simulation()
    cell = CellInit()
    cell.position = Vec3(context.rng.uniform(-0.1, 0.1), 0.0, 0.0)
    cell.length = float(context.parameters.get("initial_length", 4.0))
    cell.radius = 0.5
    cell.growth_rate = float(context.parameters.get("growth_rate", 0.2))
    simulation.add_cell(cell)
    return simulation
```

Models that need regulation, division policy, mechanics, or runtime stochastic state can return a `SimulationController`. `NativeController` provides the standard restartable lifecycle: regulation produces a typed `StepPlan`, native integration advances the biological state, division requests are applied deterministically, and `MechanicsConfig` controls relaxation. See [`examples/native_controller.py`](examples/native_controller.py) for a complete `build` and `resume` implementation.

Rate equations are expressed as a small typed instruction representation shared by all backends:

```python
from cellmodeller2 import RatePlanBuilder

rates = RatePlanBuilder()
x = rates.species(0)
production = 2.0 / (1.0 + x * x)
simulation.set_species_rate_plan(rates.species_plan(1, (production,)))
```

Each backend interprets that validated plan with its own native C++, Metal Shading Language, or CUDA C++ implementation. This keeps models inspectable and checkpoints data-only without introducing a shader language or runtime source generation.

## Run and resume simulations

Choose a backend and device explicitly:

```console
uv run cm2 devices
uv run cm2 run \
  --model examples/native_controller.py \
  --backend metal \
  --device-index 0 \
  --seed 42 \
  --steps 100 \
  --dt 0.05 \
  --checkpoint-every 25 \
  --output results/colony.cm2.json
```

Resume on any available backend:

```console
uv run cm2 run \
  --model examples/native_controller.py \
  --resume results/colony.cm2.json \
  --backend cpu \
  --steps 100 \
  --dt 0.05 \
  --output results/colony-resumed.cm2.json
```

Checkpoints contain simulation and controller data, never executable model code. Resume verifies the model digest before compiling the explicitly supplied source. Bare simulation checkpoints can resume without a model when their controller state is null. Existing outputs are not replaced unless `--overwrite` is provided.

For parameter sweeps and scheduler jobs, use the strict [run manifest v1 format](docs/formats/run-manifest-v1.md):

```console
uv run cm2 run-manifest experiments/gamma.cm2.runs.json \
  --job gamma-0.10-replicate-001
```

## Run CellModeller models

Maintained CellModeller growth, mechanics, regulation, constraint, neighbor, and host-species models run through the explicit compatibility loader:

```console
uv run cm2 run \
  --legacy-model ../CellModeller/Examples/ex1a_simpleGrowth2D.py \
  --backend metal \
  --seed 42 \
  --steps 100 \
  --dt 0.05 \
  --output results/legacy.cm2.json
```

The nine bundled examples that previously embedded OpenCL species or signaling equations have typed CellModeller2 ports under [`examples/legacy`](examples/legacy). Legacy OpenCL source strings are not executed: their equations become typed rate plans so CPU, Metal, and CUDA retain independent native implementations.

Trusted CellModeller pickle snapshots use a deliberately explicit, one-way migration command:

```console
uv run cm2 import-legacy-pickle data/step-00100.pickle \
  --output results/step-00100.cm2.json \
  --dt 0.05 \
  --trust-legacy-pickle \
  --native-state-only
```

The converter preserves geometry, species, stable identity, and lineage. It records that callback attributes, constraints, random state, and rate equations cannot be reconstructed from the old executable format.

## Signaling and SBML

Signal grids support diffusion, conservative vector upwind advection, no-flux, periodic, and fixed boundaries, trilinear sampling, non-negative concentration updates, and checkpointed Forward Euler or diagnosed Crank-Nicolson integration. Coupled plans compose intracellular rates, sampled extracellular signals, transport, and trilinear source scatter into a simultaneous update contract.

The optional SBML importer compiles a bounded SBML Level 3 Version 2 Core subset into the same typed species-rate representation:

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

Unsupported semantics fail explicitly; the importer never generates or evaluates source code. [ADR 0011](docs/architecture/0011-sbml-import.md) defines the accepted subset.

## Visualize a simulation

Export an integrity-checked scene without exposing engine or device memory:

```python
from cellmodeller2 import capture_scene, save_scene

frame = capture_scene(simulation)
save_scene(frame, "colony.cm2.scene.json")
```

The standalone TypeScript/Three.js viewer provides instanced rods, orbit controls, cell inspection, declarative coloring, and signal-grid slicing:

```console
uv run python examples/viewer_scene.py --output viewer-demo.cm2.scene.json
pnpm --dir viewer install
pnpm --dir viewer dev
```

For interactive runs, build the viewer and launch an authenticated loopback session:

```console
uv sync --group dev --extra viewer
pnpm --dir viewer build
uv run cm2 view \
  --model examples/batch_model.py \
  --backend cpu \
  --seed 42 \
  --dt 0.05 \
  --checkpoint-output results/live.cm2.json \
  --open
```

The Python process remains the sole owner of the model, backend, clock, reset behavior, and checkpoint destination. The browser owns presentation state and communicates through the bounded [live viewer protocol](docs/protocols/live-viewer-v1.md). See the [viewer guide](viewer/README.md) and [scene format](docs/formats/scene-v1.md) for details.

## Export analysis datasets

Install the analysis dependencies and export an ordered checkpoint series:

```console
uv sync --extra analysis
uv run cm2 export-analysis \
  results/step-00025.cm2.json \
  results/step-00050.cm2.json \
  --output results/run.cm2.dataset \
  --contacts \
  --external-contacts
```

Exports contain explicit-schema Parquet frame, cell, species, and contact tables. Signal fields use named `(frame, channel, x, y, z)` Zarr v3 arrays. The manifest records schemas, row counts, derivations, model provenance, source checkpoint digests, and output digests. The [analysis recipe guide](docs/analysis/recipes.md) provides verified lazy Polars workflows for colony geometry, species, lineage, contacts, and signals.

## Build native tests

CPU:

```console
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

Metal on macOS:

```console
scripts/run_metal_conformance.sh
CM2_LEGACY_ROOT=/path/to/pinned/CellModeller scripts/run_metal_application_conformance.sh
```

CUDA development:

```console
scripts/run_cuda_compile_check.sh
scripts/run_cuda_conformance.sh
CM2_LEGACY_ROOT=/path/to/pinned/CellModeller scripts/run_cuda_application_conformance.sh
```

The conformance runners require clean source trees and emit checksummed evidence containing the exact commit, environment, device inventory, logs, and machine-readable results. See [validation](docs/development/validation.md), the [Metal environment guide](environments/metal/README.md), and the [CUDA environment guide](environments/cuda/README.md).

## Architecture

The engine is deliberately organized around semantic contracts rather than a lowest-common-denominator GPU abstraction:

- [`cpp/core`](cpp/core) owns backend-neutral state, orchestration, checkpoints, and typed plans.
- [`cpp/cpu`](cpp/cpu) is the readable numerical reference.
- [`cpp/metal`](cpp/metal) contains native Metal host code and MSL kernels.
- [`cpp/cuda`](cpp/cuda) contains native CUDA host code and CUDA C++ kernels.
- [`python/src/cellmodeller2`](python/src/cellmodeller2) provides modeling, compatibility, batch, analysis, and viewer APIs.
- [`viewer`](viewer) is an independent presentation client with no simulation-engine ownership.

Start with the following design documents:

- [Native backend policy](docs/architecture/0001-native-backends.md)
- [Contact and mechanics design](docs/architecture/0002-contact-mechanics.md)
- [Species-rate representation](docs/architecture/0003-species-rates.md)
- [Checkpoint and resume contract](docs/architecture/0004-checkpoints.md)
- [Batch execution](docs/architecture/0005-batch-execution.md)
- [Grid signaling](docs/architecture/0006-grid-signaling.md)
- [Native controllers](docs/architecture/0014-native-controllers.md)
- [Numerical contract](docs/architecture/numerical-contract.md)
- [Validation workflow](docs/development/validation.md)

## Provenance and license

CellModeller2 is a clean-slate successor inspired by CellModeller, created by Tim Rudge, PJ Steiner, Jim Haseloff, and contributors at the University of Cambridge. No legacy source code has been copied into this implementation. The project owner has not yet selected a license for new CellModeller2 code; see [NOTICE.md](NOTICE.md) for the full provenance statement.

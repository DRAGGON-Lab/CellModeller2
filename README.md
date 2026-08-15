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
- CPU capsule contacts and native Metal/CUDA contact implementations;
- typed CPU and native Metal plane and inside/outside sphere constraints;
- matrix-free CPU and native Metal rod-mechanics solvers with diagnostics;
- a C++ CPU reference backend;
- a small Python API backed by nanobind;
- C++ and Python conformance tests;
- native Metal growth and CUDA growth implementations;
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
broad phase is under construction. Plane and sphere constraints are conformant
on CPU and native Metal, where their rows participate in the same matrix-free
mechanics solve and relaxation path as cell-cell contacts. The native CUDA
constraint implementation is the next compatibility slice.

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

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
- a C++ CPU reference backend;
- a small Python API backed by nanobind;
- C++ and Python conformance tests;
- native Metal growth and CUDA growth implementations;
- the cross-backend architecture and feature ledger.

Metal growth is validated on Apple GPU hardware. CUDA growth is implemented
with the native CUDA Runtime API and CUDA C++, but remains unadvertised until
the conformance suite passes on NVIDIA hardware. Deterministic equal division
is backend-neutral and conformed on CPU and Metal; contact mechanics remain
unimplemented. Metal contact geometry passes the exhaustive reference suite;
CUDA contact geometry compiles and links but still needs NVIDIA hardware
validation. Both GPU paths use an exhaustive correctness stage while the
scalable broad phase is under construction.

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

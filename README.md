# CellModeller2

CellModeller2 is an accelerator-native successor to [CellModeller](https://github.com/cellmodeller/CellModeller) for individual-based multicellular modeling. It combines a Python modeling interface with a C++23 engine and independent CPU, Apple Metal, and NVIDIA CUDA implementations.

Models can combine rod-shaped growth and division, lineage, contact mechanics and constraints, intracellular dynamics, and cell-grid signaling. Versioned checkpoints, batch manifests, data-only scenes, and Parquet/Zarr exports support reproducible research workflows.

## Backend status

| Backend | Status | Role |
| --- | --- | --- |
| CPU | Feature complete | Portable execution and numerical reference |
| Apple Metal | Feature complete | Native Apple GPU execution |
| NVIDIA CUDA | Under active development | Native NVIDIA GPU execution |

CPU and Metal implement the complete current modeling workflow. CUDA is developed independently with the CUDA Runtime API and CUDA C++; it does not translate Metal kernels or use a cross-platform GPU abstraction. Backend support requires native execution on corresponding hardware without CPU fallback; the [validation policy](docs/development/validation.md) defines the acceptance criteria.

## Quick start

CellModeller2 requires Python 3.12, CMake 3.25 or newer, Ninja, a C++23 compiler, and [uv](https://docs.astral.sh/uv/).

```console
uv sync --group dev
uv run cm devices
uv run cm run \
  --model examples/batch_model.py \
  --backend cpu \
  --seed 42 \
  --parameter growth_rate=0.25 \
  --steps 100 \
  --dt 0.05 \
  --output results/colony.json
```

Continue with the [tutorial suite](docs/tutorials/README.md), or inspect [`examples/native_controller.py`](examples/native_controller.py) for a complete restartable model.

## Documentation

| Topic | Entry point |
| --- | --- |
| Tutorials | [Modeling tutorials](docs/tutorials/README.md) |
| Architecture and numerics | [Design documents](docs/README.md#architecture-and-numerics) |
| HPC environments | [CPU, Metal, and CUDA setup](docs/README.md#execution-environments) |
| Analysis and visualization | [Research output workflows](docs/README.md#analysis-and-visualization) |
| CellModeller compatibility | [Scope and evidence](docs/README.md#cellmodeller-compatibility) |
| Development | [Testing and validation](docs/development/validation.md) |

The complete documentation index is available at [`docs/README.md`](docs/README.md).

## License

CellModeller2 is available under the [MIT License](LICENSE).

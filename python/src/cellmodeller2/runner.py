"""Deterministic batch construction and execution primitives."""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import cast

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    Simulation,
    backend_available,
)
from .checkpoint import JSONValue, save_checkpoint
from .legacy import LegacyModelAdapter

_UINT64_MAX = (1 << 64) - 1


class BatchError(RuntimeError):
    """Raised when a batch run cannot be configured or completed safely."""


def _empty_parameters() -> dict[str, JSONValue]:
    return {}


@dataclass(slots=True)
class ModelContext:
    """Inputs supplied to a batch model's ``build(context)`` function."""

    backend: BackendKind
    device_index: int
    seed: int
    parameters: Mapping[str, JSONValue] = field(default_factory=_empty_parameters)
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise BatchError("device index must be non-negative")
        if self.seed < 0 or self.seed > _UINT64_MAX:
            raise BatchError("seed must be an unsigned 64-bit integer")
        if not backend_available(self.backend, self.device_index):
            raise BatchError(
                f"backend {self.backend.name.lower()} device {self.device_index} is unavailable"
            )
        self.parameters = MappingProxyType(dict(self.parameters))
        self.rng = random.Random(self.seed)

    def simulation(self, *, reserved_capacity: int = 0, species_count: int = 0) -> Simulation:
        """Construct a simulation on the backend selected by the runner."""

        return Simulation(
            self.backend,
            reserved_capacity=reserved_capacity,
            species_count=species_count,
            device_index=self.device_index,
        )


@dataclass(frozen=True, slots=True)
class RunProgress:
    completed_steps: int
    requested_steps: int
    time: float
    cell_count: int


@dataclass(frozen=True, slots=True)
class RunSummary:
    completed_steps: int
    time: float
    cell_count: int
    output: Path
    periodic_checkpoints: tuple[Path, ...]


type ProgressCallback = Callable[[RunProgress], None]
type RunnableModel = Simulation | LegacyModelAdapter


def native_simulation(model: RunnableModel) -> Simulation:
    """Return the native state owner behind any supported runnable model."""

    return model.simulation if isinstance(model, LegacyModelAdapter) else model


def controller_state(model: RunnableModel) -> JSONValue:
    """Capture optional data-only controller state for a runnable model."""

    return model.controller_state() if isinstance(model, LegacyModelAdapter) else None


def _periodic_path(output: Path, step: int) -> Path:
    suffix = ".cm2.json"
    name = output.name
    stem = name[: -len(suffix)] if name.endswith(suffix) else name
    return output.with_name(f"{stem}.step-{step:08d}{suffix}")


def _run_provenance(
    base: Mapping[str, JSONValue], *, status: str, completed_steps: int, steps: int, dt: float
) -> dict[str, JSONValue]:
    result = dict(base)
    result["run"] = {
        "status": status,
        "completed_steps": completed_steps,
        "requested_steps": steps,
        "dt": dt,
    }
    return result


def run_simulation(
    simulation: RunnableModel,
    *,
    steps: int,
    dt: float,
    output: str | Path,
    checkpoint_every: int = 0,
    overwrite: bool = False,
    provenance: Mapping[str, JSONValue] | None = None,
    progress: ProgressCallback | None = None,
) -> RunSummary:
    """Advance a simulation and write periodic and final atomic checkpoints."""

    if steps < 0:
        raise BatchError("steps must be non-negative")
    if not math.isfinite(dt) or dt < 0.0:
        raise BatchError("time step must be finite and non-negative")
    if checkpoint_every < 0:
        raise BatchError("checkpoint interval must be non-negative")
    native = native_simulation(simulation)
    native.validate()

    destination = Path(output)
    periodic_steps = (
        tuple(range(checkpoint_every, steps + 1, checkpoint_every))
        if checkpoint_every > 0
        else ()
    )
    periodic_paths = tuple(_periodic_path(destination, step) for step in periodic_steps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        collisions = [path for path in (*periodic_paths, destination) if path.exists()]
        if collisions:
            raise BatchError(f"output already exists: {collisions[0]}")

    base_provenance = dict(provenance) if provenance is not None else {}
    periodic_by_step = dict(zip(periodic_steps, periodic_paths, strict=True))
    for completed_steps in range(1, steps + 1):
        simulation.step(dt)
        if progress is not None:
            progress(
                RunProgress(
                    completed_steps=completed_steps,
                    requested_steps=steps,
                    time=native.time,
                    cell_count=native.cell_count,
                )
            )
        periodic = periodic_by_step.get(completed_steps)
        if periodic is not None:
            save_checkpoint(
                native,
                periodic,
                provenance=_run_provenance(
                    base_provenance,
                    status="running" if completed_steps < steps else "complete",
                    completed_steps=completed_steps,
                    steps=steps,
                    dt=dt,
                ),
                controller=controller_state(simulation),
            )

    save_checkpoint(
        native,
        destination,
        provenance=_run_provenance(
            base_provenance,
            status="complete",
            completed_steps=steps,
            steps=steps,
            dt=dt,
        ),
        controller=controller_state(simulation),
    )
    return RunSummary(
        completed_steps=steps,
        time=native.time,
        cell_count=native.cell_count,
        output=destination,
        periodic_checkpoints=periodic_paths,
    )


def build_model(
    path: str | Path, context: ModelContext
) -> tuple[Simulation, dict[str, JSONValue]]:
    """Execute an explicit Python model file and call its ``build`` function."""

    source_path = Path(path).resolve()
    try:
        source = source_path.read_bytes()
    except OSError as error:
        raise BatchError(f"could not read model {source_path}") from error
    digest = hashlib.sha256(source).hexdigest()
    module_name = f"_cellmodeller2_model_{digest[:16]}"
    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""

    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    sys.path.insert(0, str(source_path.parent))
    try:
        code = compile(source, str(source_path), "exec")
        exec(code, module.__dict__)
        build_value = module.__dict__.get("build")
        if not callable(build_value):
            raise BatchError(f"model {source_path} must define build(context)")
        build = cast(Callable[[ModelContext], object], build_value)
        simulation_value = build(context)
    except BatchError:
        raise
    except Exception as error:
        raise BatchError(f"model {source_path} failed: {error}") from error
    finally:
        sys.path.pop(0)
        if previous_module is None:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = previous_module

    if not isinstance(simulation_value, Simulation):
        raise BatchError(f"model {source_path} build(context) did not return a Simulation")
    info = simulation_value.backend_info
    if info.kind != context.backend or info.device_index != context.device_index:
        raise BatchError("model returned a simulation on a different backend or device")
    simulation_value.validate()
    provenance: dict[str, JSONValue] = {
        "model": {
            "path": str(source_path),
            "sha256": digest,
            "seed": context.seed,
            "parameters": dict(context.parameters),
        }
    }
    return simulation_value, provenance

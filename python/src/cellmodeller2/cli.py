"""Command-line entry point for CellModeller2 batch work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, cast

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    Simulation,
    backend_available,
    backend_device_count,
)
from .checkpoint import CheckpointError, JSONValue, load_checkpoint, load_checkpoint_bundle
from .legacy_loader import build_legacy_model, resume_legacy_model
from .runner import BatchError, ModelContext, RunProgress, build_model, run_simulation

_BACKENDS = {
    "cpu": BackendKind.CPU,
    "metal": BackendKind.METAL,
    "cuda": BackendKind.CUDA,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cm2", description="CellModeller2 batch runner")
    commands = parser.add_subparsers(dest="command", required=True)

    devices = commands.add_parser("devices", help="list available native compute devices")
    devices.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    run = commands.add_parser("run", help="run a model or resume a checkpoint")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, help="Python file defining build(context)")
    source.add_argument(
        "--legacy-model", type=Path, help="CellModeller 1 growth/mechanics model"
    )
    run.add_argument("--resume", type=Path, help="CellModeller2 checkpoint to resume")
    run.add_argument("--backend", choices=tuple(_BACKENDS), default="cpu")
    run.add_argument("--device-index", type=int, default=0)
    run.add_argument("--seed", type=int, default=0, help="model-construction seed")
    run.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="model parameter; repeat for multiple values",
    )
    run.add_argument("--steps", type=int, required=True)
    run.add_argument("--dt", type=float, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint-every", type=int, default=0)
    run.add_argument("--progress-every", type=int, default=100)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--quiet", action="store_true")
    return parser


def _json_value(value: object, path: str) -> JSONValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BatchError(f"{path} must be finite JSON")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{path}[]") for item in cast(list[object], value)]
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise BatchError(f"{path} must use string object keys")
        return {
            cast(str, key): _json_value(item, f"{path}.{key}") for key, item in mapping.items()
        }
    raise BatchError(f"{path} is not JSON data")


def _parameters(values: Sequence[str]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for value in values:
        name, separator, encoded = value.partition("=")
        if not separator or not name:
            raise BatchError(f"invalid parameter {value!r}; expected NAME=JSON")
        if name in result:
            raise BatchError(f"duplicate parameter {name!r}")

        def reject_constant(constant: str, parameter_name: str = name) -> NoReturn:
            raise BatchError(f"parameter {parameter_name!r} contains {constant}")

        try:
            decoded = json.loads(encoded, parse_constant=reject_constant)
        except json.JSONDecodeError as error:
            raise BatchError(f"parameter {name!r} is not valid JSON: {error.msg}") from error
        result[name] = _json_value(cast(object, decoded), f"parameter {name!r}")
    return result


def _device_records() -> list[dict[str, JSONValue]]:
    records: list[dict[str, JSONValue]] = []
    for name, backend in _BACKENDS.items():
        count = backend_device_count(backend)
        if count == 0:
            records.append({"backend": name, "available": False, "devices": []})
            continue
        devices: list[JSONValue] = []
        for device_index in range(count):
            info = Simulation(backend, device_index=device_index).backend_info
            devices.append({"index": info.device_index, "name": info.device})
        records.append({"backend": name, "available": True, "devices": devices})
    return records


def _devices(json_output: bool) -> int:
    records = _device_records()
    if json_output:
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    for record in records:
        backend = cast(str, record["backend"])
        devices = cast(list[JSONValue], record["devices"])
        if not devices:
            print(f"{backend}: unavailable")
            continue
        for device in devices:
            device_record = cast(dict[str, JSONValue], device)
            print(f"{backend}:{device_record['index']} {device_record['name']}")
    return 0


def _resume_provenance(path: Path) -> dict[str, JSONValue]:
    source = path.resolve()
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise BatchError(f"could not read checkpoint {source}") from error
    return {"resume": {"path": str(source), "sha256": digest}}


def _progress_printer(interval: int, quiet: bool):
    if quiet:
        return None
    if interval <= 0:
        raise BatchError("progress interval must be positive unless --quiet is used")

    def report(progress: RunProgress) -> None:
        if progress.completed_steps % interval == 0 or (
            progress.completed_steps == progress.requested_steps
        ):
            print(
                f"step {progress.completed_steps}/{progress.requested_steps} "
                f"time={progress.time:.9g} cells={progress.cell_count}",
                file=sys.stderr,
            )

    return report


def _run(arguments: argparse.Namespace) -> int:
    backend = _BACKENDS[cast(str, arguments.backend)]
    device_index = cast(int, arguments.device_index)
    if not backend_available(backend, device_index):
        count = backend_device_count(backend)
        raise BatchError(
            f"backend {arguments.backend} device {device_index} is unavailable "
            f"({count} device(s) found)"
        )
    parameters = _parameters(cast(list[str], arguments.parameter))
    model_path = cast(Path | None, arguments.model)
    legacy_model_path = cast(Path | None, arguments.legacy_model)
    resume_path = cast(Path | None, arguments.resume)
    if model_path is not None:
        if resume_path is not None:
            raise BatchError("--model cannot be combined with --resume")
        context = ModelContext(
            backend=backend,
            device_index=device_index,
            seed=cast(int, arguments.seed),
            parameters=parameters,
        )
        simulation, provenance = build_model(model_path, context)
    elif legacy_model_path is not None:
        if resume_path is None:
            context = ModelContext(
                backend=backend,
                device_index=device_index,
                seed=cast(int, arguments.seed),
                parameters=parameters,
            )
            simulation, provenance = build_legacy_model(legacy_model_path, context)
        else:
            if parameters:
                raise BatchError(
                    "legacy resume uses the checkpoint parameters; do not pass --parameter"
                )
            bundle = load_checkpoint_bundle(
                resume_path,
                backend=backend,
                device_index=device_index,
            )
            model_value = bundle.provenance.get("model")
            if not isinstance(model_value, dict):
                raise BatchError("legacy checkpoint is missing model provenance")
            seed_value = model_value.get("seed")
            saved_parameters = model_value.get("parameters")
            if (
                not isinstance(seed_value, int)
                or isinstance(seed_value, bool)
                or not isinstance(saved_parameters, dict)
            ):
                raise BatchError("legacy checkpoint model provenance is invalid")
            context = ModelContext(
                backend=backend,
                device_index=device_index,
                seed=seed_value,
                parameters=cast(dict[str, JSONValue], saved_parameters),
            )
            simulation, model_provenance = resume_legacy_model(
                legacy_model_path,
                context,
                bundle,
            )
            provenance = dict(model_provenance)
            provenance.update(_resume_provenance(resume_path))
    else:
        if resume_path is None:
            raise BatchError("a model or checkpoint is required")
        if parameters:
            raise BatchError("--parameter is only valid with --model")
        simulation = load_checkpoint(
            resume_path,
            backend=backend,
            device_index=device_index,
        )
        provenance = _resume_provenance(resume_path)

    summary = run_simulation(
        simulation,
        steps=cast(int, arguments.steps),
        dt=cast(float, arguments.dt),
        output=cast(Path, arguments.output),
        checkpoint_every=cast(int, arguments.checkpoint_every),
        overwrite=cast(bool, arguments.overwrite),
        provenance=provenance,
        progress=_progress_printer(
            cast(int, arguments.progress_every), cast(bool, arguments.quiet)
        ),
    )
    print(
        f"wrote {summary.output} steps={summary.completed_steps} "
        f"time={summary.time:.9g} cells={summary.cell_count}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``cm2`` command and return its process status."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "devices":
            return _devices(cast(bool, arguments.json))
        return _run(arguments)
    except (BatchError, CheckpointError, IndexError, ValueError, RuntimeError) as error:
        print(f"cm2: {error}", file=sys.stderr)
        return 2

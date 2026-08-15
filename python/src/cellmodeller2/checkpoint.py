"""Versioned, non-executable CellModeller2 checkpoints."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NoReturn, cast

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    CellSnapshot,
    RateInstruction,
    RateOp,
    Simulation,
    SpeciesRatePlan,
    SphereRegion,
    Vec3,
    _ConstraintSetCheckpoint,
    _LineageEntry,
    _PlaneConstraint,
    _SimulationCheckpoint,
    _SphereConstraint,
    _WorldStateCheckpoint,
)

CHECKPOINT_FORMAT = "cellmodeller2-checkpoint"
CHECKPOINT_VERSION = 1
MAX_CHECKPOINT_BYTES = 1 << 30

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_FLOAT32_MAX = 3.4028234663852886e38

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be safely decoded or validated."""


_RATE_OP_NAMES = {
    RateOp.CONSTANT: "constant",
    RateOp.SPECIES: "species",
    RateOp.POSITION_X: "position_x",
    RateOp.POSITION_Y: "position_y",
    RateOp.POSITION_Z: "position_z",
    RateOp.CELL_LENGTH: "cell_length",
    RateOp.CELL_RADIUS: "cell_radius",
    RateOp.GROWTH_RATE: "growth_rate",
    RateOp.CELL_TYPE: "cell_type",
    RateOp.CELL_VOLUME: "cell_volume",
    RateOp.CELL_SURFACE_AREA: "cell_surface_area",
    RateOp.ADD: "add",
    RateOp.SUBTRACT: "subtract",
    RateOp.MULTIPLY: "multiply",
    RateOp.DIVIDE: "divide",
    RateOp.POWER: "power",
    RateOp.MINIMUM: "minimum",
    RateOp.MAXIMUM: "maximum",
    RateOp.NEGATE: "negate",
    RateOp.EXPONENTIAL: "exponential",
    RateOp.LOGARITHM: "logarithm",
    RateOp.LESS: "less",
    RateOp.LESS_EQUAL: "less_equal",
    RateOp.GREATER: "greater",
    RateOp.GREATER_EQUAL: "greater_equal",
    RateOp.EQUAL: "equal",
    RateOp.SELECT: "select",
}
_RATE_OPS = {name: operation for operation, name in _RATE_OP_NAMES.items()}
_SPHERE_REGION_NAMES = {
    SphereRegion.OUTSIDE: "outside",
    SphereRegion.INSIDE: "inside",
}
_SPHERE_REGIONS = {name: region for region, name in _SPHERE_REGION_NAMES.items()}
_BACKEND_NAMES = {
    BackendKind.CPU: "cpu",
    BackendKind.METAL: "metal",
    BackendKind.CUDA: "cuda",
}


def _installed_version() -> str:
    try:
        return version("cellmodeller2")
    except PackageNotFoundError:
        return "0+unknown"


def _vec3_to_json(value: Vec3) -> list[JSONValue]:
    return [value.x, value.y, value.z]


def _simulation_to_json(checkpoint: _SimulationCheckpoint) -> dict[str, JSONValue]:
    cells: list[JSONValue] = []
    for cell in checkpoint.world.cells:
        cells.append(
            {
                "id": cell.id,
                "slot": cell.slot,
                "position": _vec3_to_json(cell.position),
                "direction": _vec3_to_json(cell.direction),
                "length": cell.length,
                "radius": cell.radius,
                "growth_rate": cell.growth_rate,
                "cell_type": cell.cell_type,
                "species": list(cell.species),
            }
        )

    lineage: list[JSONValue] = [
        {"child": entry.child, "parent": entry.parent} for entry in checkpoint.world.lineage
    ]
    planes: list[JSONValue] = [
        {
            "id": plane.id,
            "point": _vec3_to_json(plane.point),
            "inward_normal": _vec3_to_json(plane.inward_normal),
            "coefficient": plane.coefficient,
        }
        for plane in checkpoint.constraints.planes
    ]
    spheres: list[JSONValue] = [
        {
            "id": sphere.id,
            "center": _vec3_to_json(sphere.center),
            "radius": sphere.radius,
            "coefficient": sphere.coefficient,
            "allowed_region": _SPHERE_REGION_NAMES[sphere.allowed_region],
        }
        for sphere in checkpoint.constraints.spheres
    ]
    instructions: list[JSONValue] = [
        {
            "operation": _RATE_OP_NAMES[instruction.operation],
            "first": instruction.first,
            "second": instruction.second,
            "third": instruction.third,
            "value": instruction.value,
        }
        for instruction in checkpoint.species_rate_plan.instructions
    ]
    return {
        "time": checkpoint.time,
        "world": {
            "species_count": checkpoint.world.species_count,
            "next_id": checkpoint.world.next_id,
            "cells": cells,
            "lineage": lineage,
        },
        "constraints": {
            "next_id": checkpoint.constraints.next_id,
            "planes": planes,
            "spheres": spheres,
        },
        "species_rate_plan": {
            "species_count": checkpoint.species_rate_plan.species_count,
            "instructions": instructions,
            "outputs": list(checkpoint.species_rate_plan.outputs),
        },
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def save_checkpoint(
    simulation: Simulation,
    path: str | os.PathLike[str],
    *,
    provenance: Mapping[str, JSONValue] | None = None,
) -> None:
    """Atomically save a complete simulation checkpoint as validated JSON."""

    checkpoint = simulation._checkpoint()
    checkpoint.validate()
    state = _simulation_to_json(checkpoint)
    digest = hashlib.sha256(_canonical_json(state)).hexdigest()
    backend = simulation.backend_info
    document: dict[str, JSONValue] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "producer": {"name": "cellmodeller2", "version": _installed_version()},
        "source_backend": {
            "kind": _BACKEND_NAMES[backend.kind],
            "name": backend.name,
            "device": backend.device,
            "native": backend.native,
        },
        "provenance": dict(provenance) if provenance is not None else {},
        "integrity": {"algorithm": "sha256", "simulation": digest},
        "simulation": state,
    }
    try:
        encoded = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError, RecursionError) as error:
        raise CheckpointError("checkpoint provenance is not finite JSON data") from error

    destination = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as error:
        raise CheckpointError(f"could not write checkpoint {destination}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fail(path: str, message: str) -> NoReturn:
    raise CheckpointError(f"{path}: {message}")


def _reject_constant(value: str) -> NoReturn:
    raise CheckpointError(f"checkpoint contains non-finite JSON number {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"checkpoint contains duplicate key {key!r}")
        result[key] = value
    return result


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        _fail(path, "expected string object keys")
    return cast(dict[str, object], mapping)


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return cast(list[object], value)


def _keys(
    value: dict[str, object], path: str, required: set[str], optional: set[str] | None = None
) -> None:
    allowed = required | (optional or set())
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing:
        _fail(path, f"missing keys {sorted(missing)}")
    if unknown:
        _fail(path, f"unknown keys {sorted(unknown)}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    return value


def _integer(value: object, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected an integer")
    if value < minimum or value > maximum:
        _fail(path, f"integer is outside [{minimum}, {maximum}]")
    return value


def _number(value: object, path: str, *, float32: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(path, "expected a number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail(path, "number is outside the finite float64 range")
    if not math.isfinite(result):
        _fail(path, "number must be finite")
    if float32 and abs(result) > _FLOAT32_MAX:
        _fail(path, "number is outside the finite float32 range")
    return result


def _vec3(value: object, path: str) -> Vec3:
    values = _array(value, path)
    if len(values) != 3:
        _fail(path, "expected exactly three coordinates")
    return Vec3(
        _number(values[0], f"{path}[0]", float32=True),
        _number(values[1], f"{path}[1]", float32=True),
        _number(values[2], f"{path}[2]", float32=True),
    )


def _cell(value: object, path: str) -> CellSnapshot:
    data = _object(value, path)
    _keys(
        data,
        path,
        {
            "id",
            "slot",
            "position",
            "direction",
            "length",
            "radius",
            "growth_rate",
            "cell_type",
            "species",
        },
    )
    cell = CellSnapshot()
    cell.id = _integer(data["id"], f"{path}.id", 1, _UINT64_MAX)
    cell.slot = _integer(data["slot"], f"{path}.slot", 0, _UINT32_MAX)
    cell.position = _vec3(data["position"], f"{path}.position")
    cell.direction = _vec3(data["direction"], f"{path}.direction")
    cell.length = _number(data["length"], f"{path}.length", float32=True)
    cell.radius = _number(data["radius"], f"{path}.radius", float32=True)
    cell.growth_rate = _number(data["growth_rate"], f"{path}.growth_rate", float32=True)
    cell.cell_type = _integer(data["cell_type"], f"{path}.cell_type", _INT32_MIN, _INT32_MAX)
    cell.species = [
        _number(item, f"{path}.species[{index}]", float32=True)
        for index, item in enumerate(_array(data["species"], f"{path}.species"))
    ]
    return cell


def _lineage_entry(value: object, path: str) -> _LineageEntry:
    data = _object(value, path)
    _keys(data, path, {"child", "parent"})
    entry = _LineageEntry()
    entry.child = _integer(data["child"], f"{path}.child", 1, _UINT64_MAX)
    entry.parent = _integer(data["parent"], f"{path}.parent", 1, _UINT64_MAX)
    return entry


def _plane(value: object, path: str) -> _PlaneConstraint:
    data = _object(value, path)
    _keys(data, path, {"id", "point", "inward_normal", "coefficient"})
    plane = _PlaneConstraint()
    plane.id = _integer(data["id"], f"{path}.id", 1, _UINT64_MAX)
    plane.point = _vec3(data["point"], f"{path}.point")
    plane.inward_normal = _vec3(data["inward_normal"], f"{path}.inward_normal")
    plane.coefficient = _number(data["coefficient"], f"{path}.coefficient", float32=True)
    return plane


def _sphere(value: object, path: str) -> _SphereConstraint:
    data = _object(value, path)
    _keys(data, path, {"id", "center", "radius", "coefficient", "allowed_region"})
    sphere = _SphereConstraint()
    sphere.id = _integer(data["id"], f"{path}.id", 1, _UINT64_MAX)
    sphere.center = _vec3(data["center"], f"{path}.center")
    sphere.radius = _number(data["radius"], f"{path}.radius", float32=True)
    sphere.coefficient = _number(data["coefficient"], f"{path}.coefficient", float32=True)
    region_name = _string(data["allowed_region"], f"{path}.allowed_region")
    try:
        sphere.allowed_region = _SPHERE_REGIONS[region_name]
    except KeyError:
        _fail(f"{path}.allowed_region", f"unknown sphere region {region_name!r}")
    return sphere


def _instruction(value: object, path: str) -> RateInstruction:
    data = _object(value, path)
    _keys(data, path, {"operation", "first", "second", "third", "value"})
    operation_name = _string(data["operation"], f"{path}.operation")
    instruction = RateInstruction()
    try:
        instruction.operation = _RATE_OPS[operation_name]
    except KeyError:
        _fail(f"{path}.operation", f"unknown rate operation {operation_name!r}")
    instruction.first = _integer(data["first"], f"{path}.first", 0, _UINT32_MAX)
    instruction.second = _integer(data["second"], f"{path}.second", 0, _UINT32_MAX)
    instruction.third = _integer(data["third"], f"{path}.third", 0, _UINT32_MAX)
    instruction.value = _number(data["value"], f"{path}.value", float32=True)
    return instruction


def _native_checkpoint(value: object, schema_version: int) -> _SimulationCheckpoint:
    data = _object(value, "$.simulation")
    _keys(data, "$.simulation", {"time", "world", "constraints", "species_rate_plan"})

    world_data = _object(data["world"], "$.simulation.world")
    _keys(world_data, "$.simulation.world", {"species_count", "next_id", "cells", "lineage"})
    world = _WorldStateCheckpoint()
    world.species_count = _integer(
        world_data["species_count"], "$.simulation.world.species_count", 0, _UINT64_MAX
    )
    world.next_id = _integer(
        world_data["next_id"], "$.simulation.world.next_id", 1, _UINT64_MAX
    )
    world.cells = [
        _cell(item, f"$.simulation.world.cells[{index}]")
        for index, item in enumerate(
            _array(world_data["cells"], "$.simulation.world.cells")
        )
    ]
    world.lineage = [
        _lineage_entry(item, f"$.simulation.world.lineage[{index}]")
        for index, item in enumerate(
            _array(world_data["lineage"], "$.simulation.world.lineage")
        )
    ]

    constraint_data = _object(data["constraints"], "$.simulation.constraints")
    _keys(constraint_data, "$.simulation.constraints", {"next_id", "planes", "spheres"})
    constraints = _ConstraintSetCheckpoint()
    constraints.next_id = _integer(
        constraint_data["next_id"], "$.simulation.constraints.next_id", 1, _UINT64_MAX
    )
    constraints.planes = [
        _plane(item, f"$.simulation.constraints.planes[{index}]")
        for index, item in enumerate(
            _array(constraint_data["planes"], "$.simulation.constraints.planes")
        )
    ]
    constraints.spheres = [
        _sphere(item, f"$.simulation.constraints.spheres[{index}]")
        for index, item in enumerate(
            _array(constraint_data["spheres"], "$.simulation.constraints.spheres")
        )
    ]

    plan_data = _object(data["species_rate_plan"], "$.simulation.species_rate_plan")
    _keys(
        plan_data,
        "$.simulation.species_rate_plan",
        {"species_count", "instructions", "outputs"},
    )
    plan_species_count = _integer(
        plan_data["species_count"],
        "$.simulation.species_rate_plan.species_count",
        0,
        _UINT64_MAX,
    )
    instructions = [
        _instruction(item, f"$.simulation.species_rate_plan.instructions[{index}]")
        for index, item in enumerate(
            _array(plan_data["instructions"], "$.simulation.species_rate_plan.instructions")
        )
    ]
    outputs = [
        _integer(item, f"$.simulation.species_rate_plan.outputs[{index}]", 0, _UINT32_MAX)
        for index, item in enumerate(
            _array(plan_data["outputs"], "$.simulation.species_rate_plan.outputs")
        )
    ]

    checkpoint = _SimulationCheckpoint()
    checkpoint.schema_version = schema_version
    checkpoint.time = _number(data["time"], "$.simulation.time")
    checkpoint.world = world
    checkpoint.constraints = constraints
    checkpoint.species_rate_plan = SpeciesRatePlan(plan_species_count, instructions, outputs)
    try:
        checkpoint.validate()
    except (ValueError, OverflowError) as error:
        raise CheckpointError(f"checkpoint state is invalid: {error}") from error
    return checkpoint


def load_checkpoint(
    path: str | os.PathLike[str], *, backend: BackendKind = BackendKind.CPU
) -> Simulation:
    """Load and validate a checkpoint without evaluating executable content."""

    source = Path(path)
    try:
        with source.open("rb") as stream:
            encoded = stream.read(MAX_CHECKPOINT_BYTES + 1)
        if not encoded:
            raise CheckpointError("checkpoint is empty")
        if len(encoded) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError(
                f"checkpoint exceeds the {MAX_CHECKPOINT_BYTES}-byte limit"
            )
    except OSError as error:
        raise CheckpointError(f"could not read checkpoint {source}") from error

    try:
        decoded = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except CheckpointError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise CheckpointError(f"checkpoint is not valid UTF-8 JSON: {error}") from error

    root = _object(cast(object, decoded), "$")
    _keys(
        root,
        "$",
        {
            "format",
            "version",
            "producer",
            "source_backend",
            "provenance",
            "integrity",
            "simulation",
        },
    )
    if _string(root["format"], "$.format") != CHECKPOINT_FORMAT:
        _fail("$.format", "not a CellModeller2 checkpoint")
    schema_version = _integer(root["version"], "$.version", 0, _UINT32_MAX)
    if schema_version != CHECKPOINT_VERSION:
        _fail("$.version", f"unsupported checkpoint version {schema_version}")
    _object(root["producer"], "$.producer")
    _object(root["source_backend"], "$.source_backend")
    _object(root["provenance"], "$.provenance")

    integrity = _object(root["integrity"], "$.integrity")
    _keys(integrity, "$.integrity", {"algorithm", "simulation"})
    if _string(integrity["algorithm"], "$.integrity.algorithm") != "sha256":
        _fail("$.integrity.algorithm", "unsupported integrity algorithm")
    expected_digest = _string(integrity["simulation"], "$.integrity.simulation")
    actual_digest = hashlib.sha256(_canonical_json(root["simulation"])).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        _fail("$.integrity.simulation", "state digest does not match")

    checkpoint = _native_checkpoint(root["simulation"], schema_version)
    return Simulation(backend, checkpoint)

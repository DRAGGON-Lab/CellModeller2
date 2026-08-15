"""Deterministic, immutable analysis datasets built from checkpoints."""

# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from zarr.codecs import ZstdCodec

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    ConstraintContactParameters,
    ContactParameters,
    ExternalConstraintKind,
    RodEndpoint,
)
from .checkpoint import CheckpointBundle, JSONValue, load_checkpoint_bundle
from .scene import SceneFrame, SceneGridBoundary, SceneSignalGrid, capture_scene

ANALYSIS_FORMAT = "cellmodeller2-analysis"
ANALYSIS_VERSION = 1

_BACKEND_NAMES = {
    BackendKind.CPU: "cpu",
    BackendKind.METAL: "metal",
    BackendKind.CUDA: "cuda",
}
_CONSTRAINT_NAMES = {
    ExternalConstraintKind.PLANE: "plane",
    ExternalConstraintKind.SPHERE: "sphere",
}
_ENDPOINT_NAMES = {
    RodEndpoint.NEGATIVE: "negative",
    RodEndpoint.POSITIVE: "positive",
}


class AnalysisError(ValueError):
    """Raised when an analysis dataset cannot be constructed safely."""


@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    """Summary of a published analysis dataset."""

    output: Path
    dataset_id: str
    frame_count: int
    cell_rows: int
    species_rows: int
    contact_rows: int
    external_contact_rows: int
    signal_epochs: int


@dataclass(frozen=True, slots=True)
class _SourceFrame:
    index: int
    label: str
    digest: str
    bundle: CheckpointBundle
    scene: SceneFrame


@dataclass(slots=True)
class _SignalEpoch:
    signature: tuple[object, ...]
    frames: list[_SourceFrame]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
    except OSError as error:
        raise AnalysisError(f"could not read {path}") from error
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _schema_record(schema: Any) -> list[dict[str, JSONValue]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]


def _write_table(path: Path, rows: list[dict[str, object]], schema: Any) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        version="2.6",
        compression="zstd",
        compression_level=7,
        write_statistics=True,
        row_group_size=131_072,
    )


def _boundary_signature(boundary: SceneGridBoundary) -> tuple[object, ...]:
    return (boundary.kind, *boundary.values)


def _grid_signature(grid: SceneSignalGrid) -> tuple[object, ...]:
    return (
        grid.signal_count,
        *grid.shape,
        *grid.origin,
        *grid.spacing,
        _boundary_signature(grid.x_lower),
        _boundary_signature(grid.x_upper),
        _boundary_signature(grid.y_lower),
        _boundary_signature(grid.y_upper),
        _boundary_signature(grid.z_lower),
        _boundary_signature(grid.z_upper),
    )


def _boundary_record(boundary: SceneGridBoundary) -> dict[str, JSONValue]:
    return {"kind": boundary.kind, "values": list(boundary.values)}


def _signal_epochs(frames: Sequence[_SourceFrame]) -> list[_SignalEpoch]:
    epochs: list[_SignalEpoch] = []
    previous_had_grid = False
    for frame in frames:
        grid = frame.scene.signal_grid
        if grid is None:
            previous_had_grid = False
            continue
        signature = _grid_signature(grid)
        if not previous_had_grid or not epochs or epochs[-1].signature != signature:
            epochs.append(_SignalEpoch(signature, []))
        epochs[-1].frames.append(frame)
        previous_had_grid = True
    return epochs


def _write_signals(path: Path, epochs: Sequence[_SignalEpoch]) -> list[dict[str, JSONValue]]:
    root = zarr.open_group(
        path,
        mode="w",
        zarr_format=3,
        attributes={"format": ANALYSIS_FORMAT, "version": ANALYSIS_VERSION},
    )
    records: list[dict[str, JSONValue]] = []
    compressor = ZstdCodec(level=7, checksum=True)
    for epoch_index, epoch in enumerate(epochs):
        first_grid = epoch.frames[0].scene.signal_grid
        if first_grid is None:  # pragma: no cover - guaranteed by _signal_epochs
            raise AssertionError("signal epoch has no grid")
        shape = first_grid.shape
        array_shape = (len(epoch.frames), first_grid.signal_count, *shape)
        chunks = (1, 1, min(shape[0], 64), min(shape[1], 64), min(shape[2], 16))
        levels = np.empty(array_shape, dtype=np.float32)
        for local_index, frame in enumerate(epoch.frames):
            grid = frame.scene.signal_grid
            if grid is None:  # pragma: no cover - guaranteed by _signal_epochs
                raise AssertionError("signal epoch contains an empty grid")
            levels[local_index] = np.asarray(grid.levels, dtype=np.float32).reshape(
                (grid.signal_count, *grid.shape)
            )

        name = f"epoch-{epoch_index:04d}"
        group = root.create_group(name)
        boundaries: dict[str, JSONValue] = {
            "x_lower": _boundary_record(first_grid.x_lower),
            "x_upper": _boundary_record(first_grid.x_upper),
            "y_lower": _boundary_record(first_grid.y_lower),
            "y_upper": _boundary_record(first_grid.y_upper),
            "z_lower": _boundary_record(first_grid.z_lower),
            "z_upper": _boundary_record(first_grid.z_upper),
        }
        group.attrs.update(
            {
                "origin": list(first_grid.origin),
                "spacing": list(first_grid.spacing),
                "boundaries": boundaries,
            }
        )
        group.create_array(
            "levels",
            data=levels,
            chunks=chunks,
            compressors=[compressor],
            dimension_names=("frame", "channel", "x", "y", "z"),
        )
        group.create_array(
            "frame_index",
            data=np.asarray([frame.index for frame in epoch.frames], dtype=np.uint32),
            chunks=(min(len(epoch.frames), 4096),),
            compressors=[compressor],
            dimension_names=("frame",),
        )
        group.create_array(
            "time",
            data=np.asarray([frame.scene.time for frame in epoch.frames], dtype=np.float64),
            chunks=(min(len(epoch.frames), 4096),),
            compressors=[compressor],
            dimension_names=("frame",),
        )
        records.append(
            {
                "name": name,
                "frame_indices": [frame.index for frame in epoch.frames],
                "signal_count": first_grid.signal_count,
                "shape": list(first_grid.shape),
                "origin": list(first_grid.origin),
                "spacing": list(first_grid.spacing),
                "boundaries": boundaries,
                "dimension_order": ["frame", "channel", "x", "y", "z"],
                "chunks": list(chunks),
                "dtype": "float32",
                "compression": {"codec": "zstd", "level": 7, "checksum": True},
            }
        )
    return records


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(child)))
    return digest.hexdigest()


def _load_sources(
    checkpoints: Sequence[str | os.PathLike[str]],
    backend: BackendKind,
    device_index: int,
    path_provenance: bool,
) -> list[_SourceFrame]:
    if not checkpoints:
        raise AnalysisError("at least one checkpoint is required")
    frames: list[_SourceFrame] = []
    previous_time: float | None = None
    for index, value in enumerate(checkpoints):
        path = Path(value)
        bundle = load_checkpoint_bundle(path, backend=backend, device_index=device_index)
        scene = capture_scene(bundle.simulation)
        if previous_time is not None and scene.time < previous_time:
            raise AnalysisError(
                f"checkpoint {path} has time {scene.time:.9g}, before prior time "
                f"{previous_time:.9g}"
            )
        previous_time = scene.time
        frames.append(
            _SourceFrame(
                index=index,
                label=str(path.resolve()) if path_provenance else path.name,
                digest=_sha256(path),
                bundle=bundle,
                scene=scene,
            )
        )
    return frames


_FRAMES_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.uint32(), nullable=False),
        pa.field("time", pa.float64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("checkpoint_version", pa.uint32(), nullable=False),
        pa.field("source_backend_kind", pa.string(), nullable=False),
        pa.field("source_backend_name", pa.string(), nullable=False),
        pa.field("source_backend_device", pa.string(), nullable=False),
        pa.field("source_backend_device_index", pa.uint32(), nullable=False),
        pa.field("source_backend_native", pa.bool_(), nullable=False),
        pa.field("reconstruction_backend_kind", pa.string(), nullable=False),
        pa.field("reconstruction_backend_name", pa.string(), nullable=False),
        pa.field("reconstruction_backend_device", pa.string(), nullable=False),
        pa.field("reconstruction_backend_device_index", pa.uint32(), nullable=False),
        pa.field("reconstruction_backend_native", pa.bool_(), nullable=False),
        pa.field("cell_count", pa.uint64(), nullable=False),
        pa.field("species_count", pa.uint32(), nullable=False),
        pa.field("signal_count", pa.uint32(), nullable=False),
        pa.field("contact_count", pa.uint64(), nullable=False),
        pa.field("external_contact_count", pa.uint64(), nullable=False),
    ]
)

_CELLS_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.uint32(), nullable=False),
        pa.field("id", pa.uint64(), nullable=False),
        pa.field("parent_id", pa.uint64(), nullable=True),
        pa.field("slot", pa.uint32(), nullable=False),
        pa.field("position_x", pa.float32(), nullable=False),
        pa.field("position_y", pa.float32(), nullable=False),
        pa.field("position_z", pa.float32(), nullable=False),
        pa.field("direction_x", pa.float32(), nullable=False),
        pa.field("direction_y", pa.float32(), nullable=False),
        pa.field("direction_z", pa.float32(), nullable=False),
        pa.field("cylinder_length", pa.float32(), nullable=False),
        pa.field("radius", pa.float32(), nullable=False),
        pa.field("capsule_length", pa.float32(), nullable=False),
        pa.field("growth_rate", pa.float32(), nullable=False),
        pa.field("cell_type", pa.int32(), nullable=False),
        pa.field("fixed", pa.bool_(), nullable=False),
    ]
)

_SPECIES_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.uint32(), nullable=False),
        pa.field("cell_id", pa.uint64(), nullable=False),
        pa.field("channel", pa.uint32(), nullable=False),
        pa.field("level", pa.float32(), nullable=False),
    ]
)

_CONTACTS_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.uint32(), nullable=False),
        pa.field("first_id", pa.uint64(), nullable=False),
        pa.field("second_id", pa.uint64(), nullable=False),
        pa.field("first_slot", pa.uint32(), nullable=False),
        pa.field("second_slot", pa.uint32(), nullable=False),
        pa.field("ordinal", pa.uint8(), nullable=False),
        pa.field("point_x", pa.float32(), nullable=False),
        pa.field("point_y", pa.float32(), nullable=False),
        pa.field("point_z", pa.float32(), nullable=False),
        pa.field("normal_x", pa.float32(), nullable=False),
        pa.field("normal_y", pa.float32(), nullable=False),
        pa.field("normal_z", pa.float32(), nullable=False),
        pa.field("signed_separation", pa.float32(), nullable=False),
        pa.field("overlap", pa.float32(), nullable=False),
        pa.field("weight", pa.float32(), nullable=False),
    ]
)

_EXTERNAL_CONTACTS_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.uint32(), nullable=False),
        pa.field("cell_id", pa.uint64(), nullable=False),
        pa.field("cell_slot", pa.uint32(), nullable=False),
        pa.field("constraint_id", pa.uint64(), nullable=False),
        pa.field("constraint_kind", pa.string(), nullable=False),
        pa.field("endpoint", pa.string(), nullable=False),
        pa.field("point_x", pa.float32(), nullable=False),
        pa.field("point_y", pa.float32(), nullable=False),
        pa.field("point_z", pa.float32(), nullable=False),
        pa.field("normal_x", pa.float32(), nullable=False),
        pa.field("normal_y", pa.float32(), nullable=False),
        pa.field("normal_z", pa.float32(), nullable=False),
        pa.field("signed_separation", pa.float32(), nullable=False),
        pa.field("overlap", pa.float32(), nullable=False),
        pa.field("weight", pa.float32(), nullable=False),
    ]
)


def _parameter_record(parameters: object, names: Sequence[str]) -> dict[str, JSONValue]:
    return {name: cast(float, getattr(parameters, name)) for name in names}


def _publish(temporary: Path, destination: Path, *, replace: bool) -> None:
    if destination.exists() and not replace:
        raise AnalysisError(f"output already exists: {destination}")
    backup: Path | None = None
    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent)
        )
        backup.rmdir()
        destination.rename(backup)
    try:
        temporary.rename(destination)
    except OSError:
        if backup is not None:
            backup.rename(destination)
        raise
    if backup is not None:
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def export_dataset(
    checkpoints: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    *,
    backend: BackendKind = BackendKind.CPU,
    device_index: int = 0,
    include_contacts: bool = False,
    include_external_contacts: bool = False,
    contact_parameters: ContactParameters | None = None,
    constraint_parameters: ConstraintContactParameters | None = None,
    path_provenance: bool = False,
    replace: bool = False,
) -> AnalysisSummary:
    """Export ordered checkpoints to a versioned Parquet/Zarr dataset directory."""

    if device_index < 0:
        raise AnalysisError("device index must be non-negative")
    if backend == BackendKind.CUDA and (include_contacts or include_external_contacts):
        raise AnalysisError(
            "CUDA contact export is unavailable until NVIDIA hardware conformance passes"
        )
    destination = Path(output)
    if destination.exists() and not replace:
        raise AnalysisError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = _load_sources(checkpoints, backend, device_index, path_provenance)
    contact_values = contact_parameters or ContactParameters()
    constraint_values = constraint_parameters or ConstraintContactParameters()

    frame_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    species_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    external_rows: list[dict[str, object]] = []
    for source in frames:
        scene = source.scene
        for cell in scene.cells:
            cell_rows.append(
                {
                    "frame_index": source.index,
                    "id": cell.id,
                    "parent_id": cell.parent_id,
                    "slot": cell.slot,
                    "position_x": cell.position[0],
                    "position_y": cell.position[1],
                    "position_z": cell.position[2],
                    "direction_x": cell.direction[0],
                    "direction_y": cell.direction[1],
                    "direction_z": cell.direction[2],
                    "cylinder_length": cell.length,
                    "radius": cell.radius,
                    "capsule_length": cell.length + (2.0 * cell.radius),
                    "growth_rate": cell.growth_rate,
                    "cell_type": cell.cell_type,
                    "fixed": cell.fixed,
                }
            )
            for channel, level in enumerate(cell.species):
                species_rows.append(
                    {
                        "frame_index": source.index,
                        "cell_id": cell.id,
                        "channel": channel,
                        "level": level,
                    }
                )

        frame_contact_count = 0
        if include_contacts:
            graph = source.bundle.simulation.find_cell_contacts(contact_values)
            for contact in graph.contacts:
                contact_rows.append(
                    {
                        "frame_index": source.index,
                        "first_id": contact.first_id,
                        "second_id": contact.second_id,
                        "first_slot": contact.first_slot,
                        "second_slot": contact.second_slot,
                        "ordinal": contact.ordinal,
                        "point_x": contact.point_on_first.x,
                        "point_y": contact.point_on_first.y,
                        "point_z": contact.point_on_first.z,
                        "normal_x": contact.normal.x,
                        "normal_y": contact.normal.y,
                        "normal_z": contact.normal.z,
                        "signed_separation": contact.signed_separation,
                        "overlap": max(0.0, -contact.signed_separation),
                        "weight": contact.weight,
                    }
                )
            frame_contact_count = len(graph)

        frame_external_count = 0
        if include_external_contacts:
            external_graph = source.bundle.simulation.find_external_contacts(constraint_values)
            for contact in external_graph.contacts:
                external_rows.append(
                    {
                        "frame_index": source.index,
                        "cell_id": contact.cell_id,
                        "cell_slot": contact.cell_slot,
                        "constraint_id": contact.constraint_id,
                        "constraint_kind": _CONSTRAINT_NAMES[contact.constraint_kind],
                        "endpoint": _ENDPOINT_NAMES[contact.endpoint],
                        "point_x": contact.point_on_cell.x,
                        "point_y": contact.point_on_cell.y,
                        "point_z": contact.point_on_cell.z,
                        "normal_x": contact.normal.x,
                        "normal_y": contact.normal.y,
                        "normal_z": contact.normal.z,
                        "signed_separation": contact.signed_separation,
                        "overlap": max(0.0, -contact.signed_separation),
                        "weight": contact.weight,
                    }
                )
            frame_external_count = len(external_graph)

        source_backend = source.bundle.source_backend
        reconstructed = scene.backend
        frame_rows.append(
            {
                "frame_index": source.index,
                "time": scene.time,
                "source": source.label,
                "source_sha256": source.digest,
                "checkpoint_version": source.bundle.schema_version,
                "source_backend_kind": source_backend.kind,
                "source_backend_name": source_backend.name,
                "source_backend_device": source_backend.device,
                "source_backend_device_index": source_backend.device_index,
                "source_backend_native": source_backend.native,
                "reconstruction_backend_kind": reconstructed.kind,
                "reconstruction_backend_name": reconstructed.name,
                "reconstruction_backend_device": reconstructed.device,
                "reconstruction_backend_device_index": reconstructed.device_index,
                "reconstruction_backend_native": reconstructed.native,
                "cell_count": len(scene.cells),
                "species_count": scene.species_count,
                "signal_count": scene.signal_grid.signal_count if scene.signal_grid else 0,
                "contact_count": frame_contact_count,
                "external_contact_count": frame_external_count,
            }
        )

    epoch_values = _signal_epochs(frames)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        tables: list[tuple[str, list[dict[str, object]], Any]] = [
            ("frames.parquet", frame_rows, _FRAMES_SCHEMA),
            ("cells.parquet", cell_rows, _CELLS_SCHEMA),
            ("species.parquet", species_rows, _SPECIES_SCHEMA),
        ]
        if include_contacts:
            tables.append(("contacts.parquet", contact_rows, _CONTACTS_SCHEMA))
        if include_external_contacts:
            tables.append(("external_contacts.parquet", external_rows, _EXTERNAL_CONTACTS_SCHEMA))
        table_manifest: dict[str, JSONValue] = {}
        for filename, rows, schema in tables:
            path = temporary / filename
            _write_table(path, rows, schema)
            table_manifest[filename] = cast(
                JSONValue,
                {
                    "rows": len(rows),
                    "schema": _schema_record(schema),
                    "sha256": _sha256(path),
                },
            )

        signal_manifest: dict[str, JSONValue] | None = None
        if epoch_values:
            signals_path = temporary / "signals.zarr"
            epoch_records = _write_signals(signals_path, epoch_values)
            signal_manifest = cast(
                dict[str, JSONValue],
                {
                    "path": "signals.zarr",
                    "sha256_tree": _directory_digest(signals_path),
                    "epochs": epoch_records,
                },
            )

        options: dict[str, JSONValue] = {
            "backend": _BACKEND_NAMES[backend],
            "device_index": device_index,
            "include_contacts": include_contacts,
            "include_external_contacts": include_external_contacts,
            "path_provenance": path_provenance,
            "contact_parameters": (
                _parameter_record(
                    contact_values,
                    ("activation_margin", "parallel_sine_threshold", "degeneracy_epsilon"),
                )
                if include_contacts
                else None
            ),
            "constraint_contact_parameters": (
                _parameter_record(constraint_values, ("activation_margin", "degeneracy_epsilon"))
                if include_external_contacts
                else None
            ),
            "contact_conformance": (
                "cpu_reference" if backend == BackendKind.CPU else "hardware_conformant"
            )
            if include_contacts or include_external_contacts
            else None,
        }
        source_manifest: list[JSONValue] = [
            {
                "frame_index": source.index,
                "path": source.label,
                "sha256": source.digest,
                "checkpoint_version": source.bundle.schema_version,
                "provenance": source.bundle.provenance,
            }
            for source in frames
        ]
        dataset_id = hashlib.sha256(
            _canonical_json(
                {
                    "format": ANALYSIS_FORMAT,
                    "version": ANALYSIS_VERSION,
                    "sources": source_manifest,
                    "options": options,
                }
            )
        ).hexdigest()
        manifest: dict[str, JSONValue] = {
            "format": ANALYSIS_FORMAT,
            "version": ANALYSIS_VERSION,
            "dataset_id": dataset_id,
            "sources": source_manifest,
            "options": options,
            "tables": table_manifest,
            "signals": signal_manifest,
        }
        (temporary / "manifest.json").write_bytes(
            json.dumps(
                manifest,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        _publish(temporary, destination, replace=replace)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return AnalysisSummary(
        output=destination,
        dataset_id=dataset_id,
        frame_count=len(frames),
        cell_rows=len(cell_rows),
        species_rows=len(species_rows),
        contact_rows=len(contact_rows),
        external_contact_rows=len(external_rows),
        signal_epochs=len(epoch_values),
    )

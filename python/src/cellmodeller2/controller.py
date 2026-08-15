"""Restartable Python controller contract for native simulations."""

from __future__ import annotations

import math
import random
from typing import Protocol, cast, runtime_checkable

from ._core import Simulation  # pyright: ignore[reportMissingModuleSource]
from .checkpoint import JSONValue

_RANDOM_STATE_KIND = "python-random-mt19937"
_RANDOM_STATE_VERSION = 1
_MT_STATE_WORDS = 624
_UINT32_MAX = (1 << 32) - 1


class ControllerStateError(ValueError):
    """Raised when persisted controller state is malformed or unsupported."""


@runtime_checkable
class SimulationController(Protocol):
    """Structural contract for Python orchestration around a native simulation.

    The controller owns runtime policy and must return all state needed by its
    model module's ``resume(context, checkpoint)`` function as finite JSON.
    """

    @property
    def simulation(self) -> Simulation:
        """Return the native simulation that owns checkpointed engine state."""

        ...

    def step(self, dt: float) -> None:
        """Advance exactly one biological step."""

        ...

    def controller_state(self) -> JSONValue:
        """Return complete non-null data-only state needed for exact resume."""

        ...


def capture_random_state(stream: random.Random) -> dict[str, JSONValue]:
    """Encode a dedicated Python random stream as closed-schema JSON data."""

    state_version, internal_state, gaussian = stream.getstate()
    if state_version != 3 or len(internal_state) != _MT_STATE_WORDS + 1:
        raise ControllerStateError("Python random stream uses an unsupported state format")
    if gaussian is not None and not math.isfinite(gaussian):
        raise ControllerStateError("Python random stream has a non-finite Gaussian cache")
    return {
        "kind": _RANDOM_STATE_KIND,
        "version": _RANDOM_STATE_VERSION,
        "state_version": state_version,
        "state": list(internal_state),
        "gauss_next": gaussian,
    }


def restore_random_state(value: JSONValue) -> random.Random:
    """Restore a random stream produced by :func:`capture_random_state`."""

    if not isinstance(value, dict):
        raise ControllerStateError("random state must be an object")
    if set(value) != {"kind", "version", "state_version", "state", "gauss_next"}:
        raise ControllerStateError("random state has unexpected fields")
    if value["kind"] != _RANDOM_STATE_KIND or value["version"] != _RANDOM_STATE_VERSION:
        raise ControllerStateError("random state kind or version is unsupported")
    if value["state_version"] != 3:
        raise ControllerStateError("Python random state version is unsupported")
    words = value["state"]
    if not isinstance(words, list) or len(words) != _MT_STATE_WORDS + 1:
        raise ControllerStateError("random state vector is invalid")
    for index, word in enumerate(words):
        upper = _MT_STATE_WORDS if index == _MT_STATE_WORDS else _UINT32_MAX
        if not isinstance(word, int) or isinstance(word, bool) or word < 0 or word > upper:
            raise ControllerStateError("random state vector is invalid")
    gaussian = value["gauss_next"]
    if gaussian is not None and (
        not isinstance(gaussian, int | float)
        or isinstance(gaussian, bool)
        or not math.isfinite(gaussian)
    ):
        raise ControllerStateError("random state Gaussian cache is invalid")

    stream = random.Random()
    try:
        stream.setstate(
            (
                3,
                tuple(cast(list[int], words)),
                float(gaussian) if gaussian is not None else None,
            )
        )
    except (TypeError, ValueError) as error:
        raise ControllerStateError("random state is invalid") from error
    return stream

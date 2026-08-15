from __future__ import annotations

import copy
import random
from collections.abc import Callable
from typing import Any, cast

import pytest
from cellmodeller2 import ControllerStateError, capture_random_state, restore_random_state


def test_random_stream_round_trip_preserves_uniform_and_gaussian_draws() -> None:
    stream = random.Random(1729)
    stream.random()
    stream.gauss(0.0, 1.0)
    state = capture_random_state(stream)
    restored = restore_random_state(state)

    assert [restored.random() for _ in range(8)] == [stream.random() for _ in range(8)]
    assert [restored.gauss(0.0, 1.0) for _ in range(8)] == [
        stream.gauss(0.0, 1.0) for _ in range(8)
    ]


def test_random_stream_rejects_malformed_state() -> None:
    def change_version(value: dict[str, Any]) -> None:
        value["version"] = 2

    def add_field(value: dict[str, Any]) -> None:
        value["extra"] = None

    def shorten_vector(value: dict[str, Any]) -> None:
        cast(list[Any], value["state"]).pop()

    def invalidate_word(value: dict[str, Any]) -> None:
        cast(list[Any], value["state"])[0] = -1

    def invalidate_gaussian(value: dict[str, Any]) -> None:
        value["gauss_next"] = float("nan")

    mutations: list[Callable[[dict[str, Any]], None]] = [
        change_version,
        add_field,
        shorten_vector,
        invalidate_word,
        invalidate_gaussian,
    ]
    for mutation in mutations:
        value = cast(dict[str, Any], copy.deepcopy(capture_random_state(random.Random(1))))
        mutation(value)
        with pytest.raises(ControllerStateError, match="random state"):
            restore_random_state(value)

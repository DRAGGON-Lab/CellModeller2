from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer
from cellmodeller2 import BackendKind, CellInit, Simulation, load_checkpoint
from cellmodeller2.checkpoint import JSONValue
from cellmodeller2.viewer_server import (
    LiveSession,
    LiveViewerError,
    create_live_app,
    parse_command,
)


def _factory() -> tuple[Simulation, dict[str, JSONValue]]:
    simulation = Simulation(BackendKind.CPU)
    cell = CellInit()
    cell.length = 2.0
    cell.growth_rate = 0.5
    simulation.add_cell(cell)
    return simulation, {"model": {"name": "viewer-test"}}


def _dist(path: Path) -> Path:
    dist = path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")
    (assets / "app.js").write_text("", encoding="utf-8")
    return dist


def test_live_session_steps_resets_and_writes_only_configured_checkpoint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "live.cm2.json"
    session = LiveSession(_factory, dt=0.25, checkpoint_output=output)
    initial = session.frame_message(playing=False)
    assert initial["revision"] == 0
    assert cast(dict[str, Any], initial["scene"])["frame"]["time"] == 0.0

    session.step(2)
    stepped = session.frame_message(playing=False)
    assert stepped["completed_steps"] == 2
    assert cast(dict[str, Any], stepped["scene"])["frame"]["time"] == 0.5
    assert session.checkpoint() == output.resolve()
    assert load_checkpoint(output).time == 0.5

    session.reset()
    reset = session.frame_message(playing=False)
    assert reset["revision"] == 2
    assert reset["completed_steps"] == 0
    assert cast(dict[str, Any], reset["scene"])["frame"]["time"] == 0.0

    disabled = LiveSession(_factory, dt=0.25)
    with pytest.raises(LiveViewerError, match="not configured"):
        disabled.checkpoint()


@pytest.mark.parametrize(
    ("encoded", "message"),
    [
        ('{"type":"step","steps":0}', "step count"),
        ('{"type":"step","path":"elsewhere"}', "unknown fields"),
        ('{"type":"reset","extra":true}', "unknown fields"),
        ('{"type":"eval"}', "unknown command"),
        ('["step"]', "JSON object"),
    ],
)
def test_command_protocol_is_closed(encoded: str, message: str) -> None:
    with pytest.raises(LiveViewerError, match=message):
        parse_command(encoded)


def test_live_websocket_requires_same_origin_token_and_controls_session(tmp_path: Path) -> None:
    async def exercise() -> None:
        checkpoint = tmp_path / "socket.cm2.json"
        application, token = create_live_app(
            LiveSession(_factory, dt=0.1, checkpoint_output=checkpoint),
            _dist(tmp_path),
            token="a" * 32,
            fps=60.0,
        )
        client = TestClient(TestServer(application))
        await client.start_server()
        origin = str(client.make_url("/")).rstrip("/")
        try:
            with pytest.raises(WSServerHandshakeError) as wrong_token:
                await client.ws_connect(
                    "/api/v1/session?token=wrong",
                    headers={"Origin": origin},
                )
            assert wrong_token.value.status == 403
            with pytest.raises(WSServerHandshakeError) as wrong_origin:
                await client.ws_connect(
                    f"/api/v1/session?token={token}",
                    headers={"Origin": "https://attacker.invalid"},
                )
            assert wrong_origin.value.status == 403

            socket = await client.ws_connect(
                f"/api/v1/session?token={token}",
                headers={"Origin": origin},
            )
            initial = cast(dict[str, Any], await socket.receive_json())
            assert initial["type"] == "frame"
            assert initial["playing"] is False

            await socket.send_str(json.dumps({"type": "step", "steps": 2}))
            stepped = cast(dict[str, Any], await socket.receive_json())
            assert stepped["completed_steps"] == 2
            assert abs(cast(float, stepped["scene"]["frame"]["time"]) - 0.2) < 1.0e-7

            await socket.send_json({"type": "checkpoint"})
            saved = cast(dict[str, Any], await socket.receive_json())
            assert saved == {"type": "checkpoint", "path": str(checkpoint.resolve())}
            assert checkpoint.exists()

            await socket.send_json({"type": "reset"})
            reset = cast(dict[str, Any], await socket.receive_json())
            assert reset["completed_steps"] == 0
            assert reset["scene"]["frame"]["time"] == 0.0

            await socket.send_json({"type": "eval", "source": "malicious()"})
            error = cast(dict[str, Any], await socket.receive_json())
            assert error["type"] == "error"
            assert "unknown command" in error["message"]
            await socket.close()
        finally:
            await client.close()

    asyncio.run(exercise())

"""Read-only smoke check for an explicitly configured local robot service."""

from __future__ import annotations

import asyncio
import os

import pytest

from tests.support.optional_robot import validated_loopback_host, validated_secret
from transport import RobotClient, TransportConfig


@pytest.mark.external_service
def test_configured_local_robot_service_answers_read_only_check() -> None:
    """Use a read-only connectivity request; never speak, move, or infer."""
    host = os.environ.get("DESKBUDDY_TEST_ROBOT_HOST")
    port_value = os.environ.get("DESKBUDDY_TEST_ROBOT_PORT")
    secret = os.environ.get("DESKBUDDY_TEST_ROBOT_AUTH_SECRET")
    if not host or not port_value or not secret:
        pytest.skip(
            "set DESKBUDDY_TEST_ROBOT_HOST, DESKBUDDY_TEST_ROBOT_PORT, and "
            "DESKBUDDY_TEST_ROBOT_AUTH_SECRET to check a configured local robot service"
        )
    secret = validated_secret(secret)
    host = validated_loopback_host(host)

    try:
        port = int(port_value)
    except ValueError as exc:
        pytest.fail(f"invalid local robot host or port configuration: {exc}")
    if not 1 <= port <= 65535:
        pytest.fail("DESKBUDDY_TEST_ROBOT_PORT must be between 1 and 65535")

    config = TransportConfig(host=host, port=port, secret=secret)

    async def check_connection():
        async with RobotClient(config) as client:
            return await client.request("simple_test", {}, side_effects=False)

    try:
        response = asyncio.run(check_connection())
    except Exception as exc:
        pytest.fail(f"local robot connectivity check failed: {type(exc).__name__}: {exc}")
    if not response.ok:
        error = response.error or {}
        code = error.get("code", "unknown")
        message = error.get("message", "robot service rejected connectivity check")
        pytest.fail(f"local robot connectivity check failed: [{code}] {message}")

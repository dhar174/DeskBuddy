"""Read-only smoke check for an explicitly configured local robot service."""

from __future__ import annotations

import asyncio
import os
from ipaddress import ip_address

import pytest

from transport import RobotClient, TransportConfig


@pytest.mark.external_service
def test_configured_local_robot_service_answers_ping() -> None:
    """Ping only; the test never requests speech, movement, or inference."""
    host = os.environ.get("DESKBUDDY_TEST_ROBOT_HOST")
    port_value = os.environ.get("DESKBUDDY_TEST_ROBOT_PORT")
    secret = os.environ.get("DESKBUDDY_TEST_ROBOT_AUTH_SECRET")
    if not host or not port_value or not secret:
        pytest.skip(
            "set DESKBUDDY_TEST_ROBOT_HOST, DESKBUDDY_TEST_ROBOT_PORT, and "
            "DESKBUDDY_TEST_ROBOT_AUTH_SECRET to ping a configured local robot service"
        )

    try:
        is_loopback = ip_address(host).is_loopback
        port = int(port_value)
    except ValueError as exc:
        pytest.fail(f"invalid local robot host or port configuration: {exc}")
    if not is_loopback:
        pytest.fail("the optional smoke test only permits a loopback robot host")
    if not 1 <= port <= 65535:
        pytest.fail("DESKBUDDY_TEST_ROBOT_PORT must be between 1 and 65535")

    config = TransportConfig(host=host, port=port, secret=secret)

    async def ping() -> bool:
        async with RobotClient(config) as client:
            return await client.ping()

    assert asyncio.run(ping()), "the configured local robot service did not answer ping"

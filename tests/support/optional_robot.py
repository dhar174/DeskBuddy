"""Validation helpers for opt-in local robot service tests."""

from __future__ import annotations

from ipaddress import ip_address

import pytest


def validated_secret(secret: str) -> str:
    """Reject blank credentials while preserving the configured secret bytes."""
    if not secret.strip():
        pytest.fail("DESKBUDDY_TEST_ROBOT_AUTH_SECRET must contain a non-whitespace value")
    return secret


def validated_loopback_host(host: str) -> str:
    """Accept localhost without DNS or a literal loopback IP."""
    if host.lower() in {"localhost", "localhost."}:
        return "127.0.0.1"
    try:
        is_loopback = ip_address(host).is_loopback
    except ValueError as exc:
        pytest.fail(f"local robot host must be localhost or a loopback IP literal: {exc}")
    if not is_loopback:
        pytest.fail("the optional smoke test only permits a loopback robot host")
    return host

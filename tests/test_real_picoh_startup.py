"""Regression test verifying real-Picoh hardware startup does not raise NameError(MockPicoh)."""

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest


def test_real_picoh_module_startup_no_name_error(monkeypatch: pytest.MonkeyPatch):
    """Simulate real picoh hardware package (where MockPicoh is NOT defined)

    and verify buddy modules import and initialize without NameError.
    """
    # Create fake real picoh module
    fake_real_picoh = types.ModuleType("picoh")
    fake_picoh_instance = MagicMock()
    fake_picoh_instance.HEADNOD = 0
    fake_picoh_instance.HEADTURN = 1
    fake_picoh_instance.EYETURN = 2
    fake_picoh_instance.LIDBLINK = 3
    fake_picoh_instance.TOPLIP = 4
    fake_picoh_instance.BOTTOMLIP = 5
    fake_picoh_instance.EYETILT = 6
    fake_real_picoh.picoh = fake_picoh_instance

    # Crucially: fake_real_picoh does NOT have MockPicoh attribute,
    # and MockPicoh is NOT in builtins or globals.

    monkeypatch.setitem(sys.modules, "picoh", fake_real_picoh)

    import buddy1
    from windows import buddy1 as win_buddy1
    from windows import buddy2 as win_buddy2

    buddy1 = importlib.reload(buddy1)
    win_buddy1 = importlib.reload(win_buddy1)
    win_buddy2 = importlib.reload(win_buddy2)

    try:
        assert buddy1._USING_MOCK_PICOH is False
        assert win_buddy1._USING_MOCK_PICOH is False
        assert win_buddy2._USING_MOCK_PICOH is False

        # Calling init_hardware must not raise NameError
        buddy1.init_hardware()
        assert fake_picoh_instance.reset.called

        # Initializing robot server path across all three entrypoints must not raise NameError
        from transport.config import TransportConfig

        dummy_cfg = TransportConfig(host="127.0.0.1", port=0, secret="real-picoh-test-secret")
        srv_buddy = buddy1.create_robot_server(dummy_cfg)
        assert srv_buddy is not None
        assert "move" in srv_buddy.registry.registered_commands

        srv_win1 = win_buddy1.create_robot_server(dummy_cfg)
        assert srv_win1 is not None
        assert "move" in srv_win1.registry.registered_commands

        srv_win2 = win_buddy2.create_robot_server(dummy_cfg)
        assert srv_win2 is not None
        assert "move" in srv_win2.registry.registered_commands
    finally:
        # Restore mock picoh for subsequent tests
        monkeypatch.undo()
        importlib.reload(buddy1)
        importlib.reload(win_buddy1)
        importlib.reload(win_buddy2)

"""Unit tests for dynamic command timeout calculation and wait/speech scaling."""

from transport.config import compute_command_timeout


def test_normal_command_uses_base_timeout():
    assert compute_command_timeout("ping", {}, 5.0) == 5.0
    assert compute_command_timeout("move", {"motor": 1, "position": 5.0}, 3.0) == 3.0
    assert compute_command_timeout("check_for_response", {}, 2.5) == 2.5
    assert compute_command_timeout("simple_test", None, 4.0) == 4.0


def test_wait_command_scales_with_seconds_and_margin():
    # Long wait scales beyond base timeout
    assert compute_command_timeout("wait", {"seconds": 12.0}, base_timeout=5.0, margin=10.0) == 22.0
    assert compute_command_timeout("picoh_wait", {"seconds": 20.0}, base_timeout=5.0, margin=5.0) == 25.0

    # Short wait preserves base timeout when base > wait + margin
    assert compute_command_timeout("wait", {"seconds": 1.0}, base_timeout=20.0, margin=5.0) == 20.0

    # Wait with string float representation
    assert compute_command_timeout("wait", {"seconds": "7.5"}, base_timeout=5.0, margin=10.0) == 17.5


def test_wait_command_invalid_or_missing_seconds():
    # Non-numeric seconds falls back to base timeout safely
    assert compute_command_timeout("wait", {"seconds": "not-a-number"}, base_timeout=5.0) == 5.0
    assert compute_command_timeout("wait", {}, base_timeout=5.0) == 5.0
    assert compute_command_timeout("wait", None, base_timeout=5.0) == 5.0


def test_speech_command_scales_with_character_count():
    # 200 characters = 20.0s speech margin + 10.0 margin = 30.0s
    long_speech = "a" * 200
    assert compute_command_timeout("reply", {"message": long_speech}, base_timeout=5.0, margin=10.0) == 30.0
    assert compute_command_timeout("say", {"text": long_speech}, base_timeout=5.0, margin=10.0) == 30.0
    assert compute_command_timeout("picoh_say", {"text": long_speech}, base_timeout=5.0, margin=10.0) == 30.0

    # Short text preserves margin ceiling
    short_text = "Hi"
    assert compute_command_timeout("reply", {"message": short_text}, base_timeout=5.0, margin=10.0) == 10.2


def test_speech_command_invalid_or_missing_text():
    assert compute_command_timeout("reply", {"message": 123}, base_timeout=5.0) == 5.0
    assert compute_command_timeout("reply", {}, base_timeout=5.0) == 5.0
    assert compute_command_timeout("say", None, base_timeout=5.0) == 5.0

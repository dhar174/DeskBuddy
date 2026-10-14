"""Unit tests for HMAC-SHA256 authentication, clock-skew expiration, and replay prevention."""

import time

from transport.auth import (
    HMACAuthenticator,
    ReplayCache,
    canonical_arguments_string,
    sign_request,
    sign_response,
    verify_response,
)
from transport.protocol import (
    RobotRequest,
    RobotResponse,
    TransportErrorCode,
)


def test_canonical_arguments_deterministic():
    # Key order must not affect canonical string
    dict1 = {"b": 2, "a": 1, "nested": {"z": 10, "y": 20}}
    dict2 = {"a": 1, "nested": {"y": 20, "z": 10}, "b": 2}
    assert canonical_arguments_string(dict1) == canonical_arguments_string(dict2)
    assert canonical_arguments_string(dict1) == '{"a":1,"b":2,"nested":{"y":20,"z":10}}'


def test_valid_authentication_success(test_secret: str):
    authenticator = HMACAuthenticator(secret=test_secret, max_clock_skew=10.0)
    req = RobotRequest.create("ping", {"message": "hello"})
    sign_request(test_secret, req)

    valid, err_code, err_msg = authenticator.authenticate(req)
    assert valid is True
    assert err_code is None
    assert err_msg is None


def test_incorrect_secret_rejected(test_secret: str):
    authenticator = HMACAuthenticator(secret=test_secret, max_clock_skew=10.0)
    req = RobotRequest.create("ping", {"message": "hello"})
    sign_request("wrong-secret-value-12345", req)

    valid, err_code, err_msg = authenticator.authenticate(req)
    assert valid is False
    assert err_code == TransportErrorCode.UNAUTHORIZED
    assert "verification failed" in err_msg


def test_tampered_payload_after_signing_rejected(test_secret: str):
    authenticator = HMACAuthenticator(secret=test_secret, max_clock_skew=10.0)
    req = RobotRequest.create("move", {"motor": 1, "position": 5.0})
    sign_request(test_secret, req)

    # Attacker alters motor position after signing
    req.arguments["position"] = 10.0

    valid, err_code, err_msg = authenticator.authenticate(req)
    assert valid is False
    assert err_code == TransportErrorCode.UNAUTHORIZED


def test_tampered_command_after_signing_rejected(test_secret: str):
    authenticator = HMACAuthenticator(secret=test_secret, max_clock_skew=10.0)
    req = RobotRequest.create("ping")
    sign_request(test_secret, req)

    # Attacker modifies command
    req.command = "reply"

    valid, err_code, err_msg = authenticator.authenticate(req)
    assert valid is False
    assert err_code == TransportErrorCode.UNAUTHORIZED


def test_expired_timestamp_rejected(test_secret: str):
    authenticator = HMACAuthenticator(secret=test_secret, max_clock_skew=5.0)
    # Request timestamp is 20 seconds in the past
    req = RobotRequest.create("ping", timestamp=time.time() - 20.0)
    sign_request(test_secret, req)

    valid, err_code, err_msg = authenticator.authenticate(req)
    assert valid is False
    assert err_code == TransportErrorCode.EXPIRED_REQUEST
    assert "expired" in err_msg


def test_future_timestamp_beyond_skew_rejected(test_secret: str):
    authenticator = HMACAuthenticator(secret=test_secret, max_clock_skew=5.0)
    # Request timestamp is 20 seconds into the future
    req = RobotRequest.create("ping", timestamp=time.time() + 20.0)
    sign_request(test_secret, req)

    valid, err_code, err_msg = authenticator.authenticate(req)
    assert valid is False
    assert err_code == TransportErrorCode.EXPIRED_REQUEST


def test_replay_protection_rejects_duplicate_request_id(test_secret: str):
    authenticator = HMACAuthenticator(secret=test_secret, max_clock_skew=10.0)
    now = time.time()
    req = RobotRequest.create("ping", timestamp=now)
    sign_request(test_secret, req)

    # First presentation succeeds
    valid1, err_code1, _ = authenticator.authenticate(req, current_time=now)
    assert valid1 is True

    # Replay of exact same request within skew window fails
    valid2, err_code2, err_msg2 = authenticator.authenticate(req, current_time=now + 1.0)
    assert valid2 is False
    assert err_code2 == TransportErrorCode.REPLAYED_REQUEST
    assert "replay detected" in err_msg2


def test_missing_or_unsupported_auth_scheme(test_secret: str):
    authenticator = HMACAuthenticator(secret=test_secret)

    # Missing auth block
    req_no_auth = RobotRequest.create("ping")
    valid, err_code, _ = authenticator.authenticate(req_no_auth)
    assert valid is False
    assert err_code == TransportErrorCode.UNAUTHORIZED

    # Bad scheme
    req_bad_scheme = RobotRequest.create("ping")
    req_bad_scheme.auth = {"scheme": "plaintext", "signature": "secret"}
    valid, err_code, err_msg = authenticator.authenticate(req_bad_scheme)
    assert valid is False
    assert err_code == TransportErrorCode.UNAUTHORIZED
    assert "Unsupported authentication scheme" in err_msg


def test_replay_cache_prunes_old_entries():
    cache = ReplayCache(max_entries=5)
    now = 1000.0

    # Insert 5 items
    for i in range(5):
        assert cache.check_and_record(f"req-{i}", now, now, max_clock_skew=10.0) is True

    assert len(cache) == 5

    # Insert a 6th item; cache should stay bounded <= 5
    assert cache.check_and_record("req-5", now, now, max_clock_skew=10.0) is True
    assert len(cache) <= 5
    # The oldest (req-0) was evicted
    assert "req-0" not in cache._seen


def test_invalid_signature_does_not_poison_replay_cache(test_secret: str):
    authenticator = HMACAuthenticator(secret=test_secret, max_clock_skew=10.0)
    now = time.time()
    req = RobotRequest.create("ping", request_id="poison-target-1", timestamp=now)

    # 1. Attacker sends with bad signature
    sign_request("attacker-fake-secret", req)
    valid, err_code, _ = authenticator.authenticate(req, current_time=now)
    assert valid is False
    assert err_code == TransportErrorCode.UNAUTHORIZED

    # Request ID must NOT be recorded in replay cache after failure
    assert not authenticator.replay_cache.is_seen("poison-target-1", now, 10.0)

    # 2. Legitimate client presents request with valid signature
    sign_request(test_secret, req)
    valid_legit, err_legit, _ = authenticator.authenticate(req, current_time=now)
    assert valid_legit is True
    assert err_legit is None

    # 3. Subsequent replay of the valid request is properly blocked
    valid_replay, err_replay, _ = authenticator.authenticate(req, current_time=now + 1.0)
    assert valid_replay is False
    assert err_replay == TransportErrorCode.REPLAYED_REQUEST


def test_response_signing_and_verification_success(test_secret: str):
    req_id = "req-12345"
    resp = RobotResponse.success(request_id=req_id, result={"status": "ok"})
    sign_response(test_secret, resp)

    valid, err_code, err_msg = verify_response(
        test_secret,
        resp,
        expected_request_id=req_id,
        max_clock_skew=10.0,
    )
    assert valid is True
    assert err_code is None
    assert err_msg is None


def test_domain_separation_cross_protocol_signature_rejection(test_secret: str):
    # Ensure request signature cannot authenticate a response and vice versa
    req = RobotRequest.create("ping", request_id="shared-id-1")
    sign_request(test_secret, req)

    # Fake response with identical request_id and request's signature
    resp = RobotResponse.success(request_id="shared-id-1", result=None, timestamp=req.timestamp)
    resp.auth = dict(req.auth)

    valid, err_code, _ = verify_response(test_secret, resp, expected_request_id="shared-id-1")
    assert valid is False
    assert err_code == TransportErrorCode.UNAUTHORIZED


def test_response_tampered_result_rejected(test_secret: str):
    req_id = "req-tamper-result"
    resp = RobotResponse.success(request_id=req_id, result={"val": 1})
    sign_response(test_secret, resp)

    # Tamper with result
    resp.result = {"val": 2}

    valid, err_code, _ = verify_response(test_secret, resp, expected_request_id=req_id)
    assert valid is False
    assert err_code == TransportErrorCode.UNAUTHORIZED


def test_response_tampered_error_rejected(test_secret: str):
    req_id = "req-tamper-error"
    resp = RobotResponse.fail(request_id=req_id, code=TransportErrorCode.INVALID_ARGUMENTS, message="bad input")
    sign_response(test_secret, resp)

    # Tamper with error message
    resp.error["message"] = "different message"

    valid, err_code, _ = verify_response(test_secret, resp, expected_request_id=req_id)
    assert valid is False
    assert err_code == TransportErrorCode.UNAUTHORIZED


def test_response_mismatched_request_id_rejected(test_secret: str):
    resp = RobotResponse.success(request_id="server-req-id", result="hello")
    sign_response(test_secret, resp)

    # Client expects a different request_id
    valid, err_code, err_msg = verify_response(
        test_secret,
        resp,
        expected_request_id="client-expected-different-id",
    )
    assert valid is False
    assert err_code == TransportErrorCode.MALFORMED_REQUEST
    assert "Correlation mismatch" in err_msg


def test_response_wrong_secret_rejected(test_secret: str):
    req_id = "req-wrong-secret"
    resp = RobotResponse.success(request_id=req_id, result="ok")
    sign_response("wrong-server-secret-xyz", resp)

    valid, err_code, _ = verify_response(test_secret, resp, expected_request_id=req_id)
    assert valid is False
    assert err_code == TransportErrorCode.UNAUTHORIZED


def test_response_expired_timestamp_rejected(test_secret: str):
    req_id = "req-expired-resp"
    past_ts = time.time() - 60.0
    resp = RobotResponse.success(request_id=req_id, result="ok", timestamp=past_ts)
    sign_response(test_secret, resp)

    valid, err_code, err_msg = verify_response(
        test_secret,
        resp,
        expected_request_id=req_id,
        max_clock_skew=10.0,
    )
    assert valid is False
    assert err_code == TransportErrorCode.EXPIRED_REQUEST
    assert "expired" in err_msg


def test_response_future_timestamp_rejected(test_secret: str):
    req_id = "req-future-resp"
    future_ts = time.time() + 60.0
    resp = RobotResponse.success(request_id=req_id, result="ok", timestamp=future_ts)
    sign_response(test_secret, resp)

    valid, err_code, err_msg = verify_response(
        test_secret,
        resp,
        expected_request_id=req_id,
        max_clock_skew=10.0,
    )
    assert valid is False
    assert err_code == TransportErrorCode.EXPIRED_REQUEST
    assert "skew:" in err_msg


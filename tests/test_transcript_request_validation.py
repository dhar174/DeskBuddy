"""Check correlation for invalid transcript request identifiers."""

import pytest

from transport.auth import sign_request
from transport.config import TransportConfig
from transport.protocol import RobotRequest
from transport.server import RobotServer
from transport.transcripts import TranscriptQueue


@pytest.mark.parametrize("invalid_id", ["x" * 37, "café", "line\nbreak"])
async def test_invalid_transcript_request_omits_unsafe_correlation_id(
    ephemeral_server_config: TransportConfig, invalid_id: str
) -> None:
    server = RobotServer(ephemeral_server_config, transcript_queue=TranscriptQueue())
    request = RobotRequest.create(command="poll_transcript", request_id=invalid_id)
    sign_request(ephemeral_server_config.secret, request)

    response = await server._process_request_payload(request.to_json().encode("utf-8"))

    assert response.request_id is None
    assert not response.ok
    assert response.error is not None
    assert response.error["code"] == "invalid_arguments"

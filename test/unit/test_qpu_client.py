##############################################################################
# Copyright 2016-2021 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
import time
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest import mock

import pytest
import rpcq
from dateutil.tz import tzutc
from pytest import raises
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials

from pyquil.api._qpu_client import (
    QPUClient,
    BufferResponse,
    GetBuffersResponse,
    GetBuffersRequest,
    RunProgramRequest,
    RunProgramResponse,
)
from test.unit.utils import run_rpcq_server


def test_init__sets_processor_and_timeout(mock_engagement_manager: mock.MagicMock):
    qpu_client = QPUClient(
        quantum_processor_id="some-processor",
        engagement_manager=mock_engagement_manager,
        request_timeout=3.14,
    )

    assert qpu_client.quantum_processor_id == "some-processor"
    assert qpu_client.timeout == 3.14


def test_run_program__returns_job_info(
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
    rpcq_server_with_auth: rpcq.Server,
    port: int,
):
    qpu_client = QPUClient(quantum_processor_id="some-processor", engagement_manager=mock_engagement_manager)

    mock_engagement_manager.get_engagement.return_value = engagement(
        quantum_processor_id="some-processor",
        seconds_left=10,
        credentials=engagement_credentials,
        port=port,
    )

    @rpcq_server_with_auth.rpc_handler
    def execute_qpu_request(request: rpcq.messages.QPURequest, user: str, priority: int) -> str:
        assert request == rpcq.messages.QPURequest(
            id="some-qpu-request",
            program="encrypted-program",
            patch_values={"foo": [42]},
        )
        # NOTE: user no longer needs to be passed to server by client-under-test, but we still need to ensure the mock
        # handler signature matches that of the server
        assert user is None
        assert priority == 1
        return "some-job"

    with run_rpcq_server(rpcq_server_with_auth, port):
        request = RunProgramRequest(
            id="some-qpu-request",
            priority=1,
            program="encrypted-program",
            patch_values={"foo": [42]},
        )
        assert qpu_client.run_program(request) == RunProgramResponse(job_id="some-job")


def test_get_buffers__returns_buffers_for_job(
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
    rpcq_server_with_auth: rpcq.Server,
    port: int,
):
    qpu_client = QPUClient(quantum_processor_id="some-processor", engagement_manager=mock_engagement_manager)

    mock_engagement_manager.get_engagement.return_value = engagement(
        quantum_processor_id="some-processor",
        seconds_left=10,
        credentials=engagement_credentials,
        port=port,
    )

    @rpcq_server_with_auth.rpc_handler
    def get_buffers(job_id: str, wait: bool) -> Dict[str, Any]:
        assert job_id == "some-job"
        assert wait is True
        return {
            "ro": {
                "shape": (1000, 2),
                "dtype": "float64",
                "data": b"buffer-data",
            },
        }

    with run_rpcq_server(rpcq_server_with_auth, port):
        request = GetBuffersRequest(
            job_id="some-job",
            wait=True,
        )
        assert qpu_client.get_buffers(request) == GetBuffersResponse(
            buffers={
                "ro": BufferResponse(
                    shape=(1000, 2),
                    dtype="float64",
                    data=b"buffer-data",
                )
            }
        )


def test_fetches_engagement_for_quantum_processor_on_request(
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
    rpcq_server_with_auth: rpcq.Server,
    port: int,
):
    qpu_client = QPUClient(
        quantum_processor_id="some-processor",
        engagement_manager=mock_engagement_manager,
        request_timeout=3.14,
    )

    def mock_get_engagement(quantum_processor_id: str, request_timeout: float) -> EngagementWithCredentials:
        assert quantum_processor_id == "some-processor"
        assert request_timeout == qpu_client.timeout
        return engagement(
            quantum_processor_id="some-processor",
            seconds_left=9999,
            credentials=engagement_credentials,
            port=port,
        )

    mock_engagement_manager.get_engagement.side_effect = mock_get_engagement

    @rpcq_server_with_auth.rpc_handler
    def execute_qpu_request(request: rpcq.messages.QPURequest, user: str, priority: int):
        return ""

    with run_rpcq_server(rpcq_server_with_auth, port):
        request = RunProgramRequest(
            id="",
            priority=0,
            program="",
            patch_values={},
        )
        qpu_client.run_program(request)


def test_sets_timeout_on_requests__engagement_expires_later(
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
    rpcq_server_with_auth: rpcq.Server,
    port: int,
):
    """
    Tests that the original request timeout is honored when time until engagement expiration is longer than timeout.
    """
    qpu_client = QPUClient(
        quantum_processor_id="some-processor",
        engagement_manager=mock_engagement_manager,
        request_timeout=0.1,
    )

    mock_engagement_manager.get_engagement.return_value = engagement(
        quantum_processor_id="some-processor",
        seconds_left=qpu_client.timeout * 10,
        credentials=engagement_credentials,
        port=port,
    )

    @rpcq_server_with_auth.rpc_handler
    def execute_qpu_request(request: rpcq.messages.QPURequest, user: str, priority: int):
        time.sleep(qpu_client.timeout * 2)

    with run_rpcq_server(rpcq_server_with_auth, port):
        with raises(TimeoutError, match=f"Timeout on client tcp://localhost:{port}, method name execute_qpu_request"):
            request = RunProgramRequest(
                id="",
                priority=0,
                program="",
                patch_values={},
            )
            qpu_client.run_program(request)


def test_sets_timeout_on_requests__engagement_expires_sooner(
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
    rpcq_server_with_auth: rpcq.Server,
    port: int,
):
    """
    Tests that the original request timeout is truncated when time until engagement expiration is sooner than timeout.
    """
    qpu_client = QPUClient(
        quantum_processor_id="some-processor",
        engagement_manager=mock_engagement_manager,
        request_timeout=1.0,
    )

    engagement_seconds_left = qpu_client.timeout / 10
    mock_engagement_manager.get_engagement.return_value = engagement(
        quantum_processor_id="some-processor",
        seconds_left=engagement_seconds_left,
        credentials=engagement_credentials,
        port=port,
    )

    @rpcq_server_with_auth.rpc_handler
    def execute_qpu_request(request: rpcq.messages.QPURequest, user: str, priority: int):
        time.sleep(engagement_seconds_left * 2)

    with run_rpcq_server(rpcq_server_with_auth, port):
        with raises(TimeoutError, match=f"Timeout on client tcp://localhost:{port}, method name execute_qpu_request"):
            request = RunProgramRequest(
                id="",
                priority=0,
                program="",
                patch_values={},
            )
            qpu_client.run_program(request)


def test_handles_contiguous_engagements(
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
    rpcq_server_with_auth: rpcq.Server,
    port: int,
):
    """Test that a request crossing the boundary between two contiguous engagements can successfully complete."""

    qpu_client = QPUClient(
        quantum_processor_id="some-processor",
        engagement_manager=mock_engagement_manager,
        request_timeout=1.0,
    )

    first_engagement_seconds_left = qpu_client.timeout / 10
    mock_engagement_manager.get_engagement.side_effect = [
        engagement(
            quantum_processor_id="some-processor",
            seconds_left=first_engagement_seconds_left,
            credentials=engagement_credentials,
            port=port,
        ),
        engagement(
            quantum_processor_id="some-processor",
            seconds_left=9999,
            credentials=engagement_credentials,
            port=port,
        ),
    ]

    @rpcq_server_with_auth.rpc_handler
    def execute_qpu_request(request: rpcq.messages.QPURequest, user: str, priority: int):
        time.sleep(first_engagement_seconds_left * 2)
        return "some-job"

    with run_rpcq_server(rpcq_server_with_auth, port):
        request = RunProgramRequest(
            id="",
            priority=0,
            program="",
            patch_values={},
        )
        assert qpu_client.run_program(request) == RunProgramResponse(job_id="some-job")


def engagement(
    *, quantum_processor_id: str, seconds_left: float, credentials: EngagementCredentials, port: int
) -> EngagementWithCredentials:
    return EngagementWithCredentials(
        address=f"tcp://localhost:{port}",
        credentials=credentials,
        endpoint_id="some-endpoint",
        expires_at=str(datetime.now(tzutc()) + timedelta(seconds=seconds_left)),
        quantum_processor_id=quantum_processor_id,
        user_id="some-user",
    )


@pytest.fixture()
@mock.patch("pyquil.api.EngagementManager", autospec=True)
def mock_engagement_manager(mock_engagement_manager_class: mock.MagicMock) -> mock.MagicMock:
    return mock_engagement_manager_class.return_value

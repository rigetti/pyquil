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
from datetime import datetime, timedelta
from unittest import mock

import pytest
import rpcq
from dateutil.tz import tzutc
from pytest_mock import MockerFixture
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials
from rpcq.messages import QPURequest

from pyquil.api._qpu_client import (
    QPUClient,
    BufferResponse,
    GetBuffersResponse,
    GetBuffersRequest,
    RunProgramRequest,
    RunProgramResponse,
)
from test.unit.utils import patch_rpc_client


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
    mocker: MockerFixture,
):
    qpu_client = QPUClient(quantum_processor_id="some-processor", engagement_manager=mock_engagement_manager)

    mock_engagement_manager.get_engagement.return_value = engagement(
        quantum_processor_id="some-processor",
        seconds_left=10,
        credentials=engagement_credentials,
        port=1234,
    )

    client = patch_rpc_client(mocker=mocker, return_value="some-job")
    request = RunProgramRequest(
        id="some-qpu-request",
        priority=1,
        program="encrypted-program",
        patch_values={"foo": [42]},
    )
    assert qpu_client.run_program(request) == RunProgramResponse(job_id="some-job")
    client.call.assert_called_once_with(
        "execute_qpu_request",
        request=rpcq.messages.QPURequest(
            id="some-qpu-request",
            program="encrypted-program",
            patch_values={"foo": [42]},
        ),
        priority=request.priority,
        user=None,
    )


def test_get_buffers__returns_buffers_for_job(
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
    mocker: MockerFixture,
):
    qpu_client = QPUClient(quantum_processor_id="some-processor", engagement_manager=mock_engagement_manager)
    mock_engagement_manager.get_engagement.return_value = engagement(
        quantum_processor_id="some-processor",
        seconds_left=10,
        credentials=engagement_credentials,
        port=1234,
    )
    client = patch_rpc_client(mocker=mocker, return_value={
        "ro": {
            "shape": (1000, 2),
            "dtype": "float64",
            "data": b"buffer-data",
        },
    })
    job_id = "some-job"
    wait = True
    request = GetBuffersRequest(
        job_id=job_id,
        wait=wait,
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
    client.call.assert_called_once_with(
        "get_buffers",
        job_id=job_id,
        wait=wait,
    )


def test_fetches_engagement_for_quantum_processor_on_request(
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
    mocker: MockerFixture,
):
    processor_id = "some-processor"
    qpu_client = QPUClient(
        quantum_processor_id=processor_id,
        engagement_manager=mock_engagement_manager,
        request_timeout=3.14,
    )
    mock_engagement_manager.get_engagement.return_value = engagement(
            quantum_processor_id=processor_id,
            seconds_left=9999,
            credentials=engagement_credentials,
            port=1234,
        )

    patch_rpc_client(mocker=mocker, return_value="")

    request = RunProgramRequest(
        id="",
        priority=0,
        program="",
        patch_values={},
    )
    qpu_client.run_program(request)
    mock_engagement_manager.get_engagement.assert_called_once_with(
        quantum_processor_id=processor_id,
        request_timeout=qpu_client.timeout,
    )


def test__calculate_timeout_engagement_is_longer_than_timeout(
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
):
    """
    Tests that the original request timeout is honored when time until engagement expiration is longer than timeout.
    """
    client_timeout = 0.1
    qpu_client = QPUClient(
        quantum_processor_id="some-processor",
        engagement_manager=mock_engagement_manager,
        request_timeout=client_timeout,
    )
    _engagement = engagement(
        quantum_processor_id="some-processor",
        seconds_left=qpu_client.timeout * 10,
        credentials=engagement_credentials,
        port=1234,
    )
    mock_engagement_manager.get_engagement.return_value = _engagement

    assert qpu_client._calculate_timeout(engagement=_engagement) == client_timeout


def test__calculate_timeout_engagement_is_shorter_than_timeout(
    freezer,  # This freezes time so that timeout can be compared with ==
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
):
    """
    Tests that the original request timeout is truncated when time until engagement expiration is sooner than timeout.
    """
    mock_engagement_manager.get_engagement.return_value = {}
    client_timeout = 1.0
    seconds_left = 0.5
    qpu_client = QPUClient(
        quantum_processor_id="some-processor",
        engagement_manager=mock_engagement_manager,
        request_timeout=client_timeout,
    )
    _engagement = engagement(
        quantum_processor_id="some-processor",
        seconds_left=seconds_left,
        credentials=engagement_credentials,
        port=1234,
    )
    mock_engagement_manager.get_engagement.return_value = _engagement

    assert qpu_client._calculate_timeout(engagement=_engagement) == seconds_left


def test_run_program__retries_on_timeout(
    mock_engagement_manager: mock.MagicMock,
    engagement_credentials: EngagementCredentials,
    mocker: MockerFixture,
):
    """Test that if a program times out, it will be retried.

    A real-world example is a request crossing the boundary between two contiguous engagements.
    """

    # SETUP
    processor_id = "some-processor"
    job_id = "some-job"
    qpu_client = QPUClient(
        quantum_processor_id=processor_id,
        engagement_manager=mock_engagement_manager,
        request_timeout=1.0,
    )
    mock_engagement_manager.get_engagement.return_value = engagement(
            quantum_processor_id=processor_id,
            seconds_left=0,
            credentials=engagement_credentials,
            port=1234,
        )
    client = patch_rpc_client(mocker=mocker, return_value=None)
    client.call.side_effect = [
        TimeoutError,  # First request must look like it timed out so we can verify retry
        job_id,
    ]
    request_kwargs = {"id": "TestingContiguous", "patch_values": {}, "program": ""}
    request = RunProgramRequest(  # Thing we give to QPUClient
        priority=0,
        **request_kwargs,
    )
    qpu_request = QPURequest(**request_kwargs)  # Thing QPUClient gives to rpcq.Client

    # ACT
    assert qpu_client.run_program(request) == RunProgramResponse(job_id=job_id)

    # ASSERT
    # Engagement should be fetched twice, once per RPC call
    mock_engagement_manager.get_engagement.assert_has_calls([
        mocker.call(quantum_processor_id='some-processor', request_timeout=1.0),
        mocker.call(quantum_processor_id='some-processor', request_timeout=1.0),
    ])
    # RPC call should happen twice since the first one times out
    client.call.assert_has_calls([
        mocker.call("execute_qpu_request", request=qpu_request, priority=0, user=None),
        mocker.call("execute_qpu_request", request=qpu_request, priority=0, user=None),
    ])


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
def mock_engagement_manager(mocker: MockerFixture) -> mock.MagicMock:
    mock_engagement_manager_class = mocker.patch("pyquil.api.EngagementManager", autospec=True)
    return mock_engagement_manager_class.return_value

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
from typing import Any, Dict

import pytest
import rpcq
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials

from pyquil.api._qpu_client import (
    QPUClient,
    BufferResponse,
    GetBuffersResponse,
    GetBuffersRequest,
    RunProgramRequest,
    RunProgramResponse,
)
from pyquil.tests.utils import run_rpcq_server


def test_init__sets_engagement_and_timeout(engagement: EngagementWithCredentials):
    qpu_client = QPUClient(engagement=engagement, request_timeout=3.14)

    assert qpu_client.engagement == engagement
    assert qpu_client.timeout == 3.14


def test_run_program__returns_job_info(engagement: EngagementWithCredentials, rpcq_server_with_auth: rpcq.Server):
    qpu_client = QPUClient(engagement=engagement)

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

    with run_rpcq_server(rpcq_server_with_auth, 5557):
        request = RunProgramRequest(
            id="some-qpu-request",
            priority=1,
            program="encrypted-program",
            patch_values={"foo": [42]},
        )
        assert qpu_client.run_program(request) == RunProgramResponse(job_id="some-job")


def test_get_buffers__returns_buffers_for_job(
    engagement: EngagementWithCredentials, rpcq_server_with_auth: rpcq.Server
):
    qpu_client = QPUClient(engagement=engagement)

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

    with run_rpcq_server(rpcq_server_with_auth, 5557):
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


@pytest.fixture
def engagement(engagement_credentials: EngagementCredentials) -> EngagementWithCredentials:
    return EngagementWithCredentials(
        address="tcp://localhost:5557",
        credentials=engagement_credentials,
        endpoint_id="some-endpoint",
        expires_at="9999-01-01T00:00:00Z",
        quantum_processor_id="some-processor",
        user_id="some-user",
    )

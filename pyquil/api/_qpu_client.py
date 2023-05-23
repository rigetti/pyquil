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
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, cast, Tuple, Union, List, Any
from attr import field

import rpcq
from dateutil.parser import parse as parsedate
from dateutil.tz import tzutc
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from pyquil.api import EngagementManager
from pyquil._version import DOCS_URL


@dataclass
class RunProgramRequest:
    """
    Request to run a program.
    """

    id: str
    """Identifier for request."""

    priority: int
    """Priority for request."""

    program: str
    """Encrypted program to run."""

    patch_values: Dict[str, List[Union[int, float]]]
    """Map of data names to data values for patching the program."""


@dataclass
class RunProgramResponse:
    """
    Run program response.
    """

    job_id: str
    """Identifier for created run job."""


@dataclass
class GetBuffersRequest:
    """
    Request for getting job buffers.
    """

    job_id: str
    """Job for which to get buffers."""

    wait: bool
    """Whether or not to wait until buffers become available."""


@dataclass
class BufferResponse:
    """
    Job buffer.
    """

    shape: Tuple[int, int]
    """Shape of the buffer: (<trials>, <slots>)."""

    dtype: str
    """Buffer data type."""

    data: bytes
    """Raw buffer data (C order)."""


@dataclass
class GetBuffersResponse:
    """
    Job buffers response.
    """

    buffers: Dict[str, BufferResponse]
    """Job buffers, by buffer name."""

    execution_duration_microseconds: Optional[int] = field(default=None)
    "Duration job held exclusive hardware access."


class QPUClient:
    """
    Client for making requests to a QPU.
    """

    def __init__(
        self,
        *,
        quantum_processor_id: str,
        engagement_manager: EngagementManager,
        endpoint_id: Optional[str] = None,
        request_timeout: float = 10.0,
    ) -> None:
        """
        Instantiate a new QPU client, authenticated using the given engagement.

        :param quantum_processor_id: ID of quantum processor to target.
        :param engagement_manager: Manager for QPU engagements.
        :param request_timeout: Timeout for requests, in seconds.
        """
        self.quantum_processor_id = quantum_processor_id
        self._endpoint_id = endpoint_id
        self._engagement_manager = engagement_manager
        self.timeout = request_timeout

    def run_program(self, request: RunProgramRequest) -> RunProgramResponse:
        """
        Run a program on a QPU.
        """
        rpcq_request = rpcq.messages.QPURequest(
            id=request.id,
            program=request.program,
            patch_values=request.patch_values,
        )
        job_id = self._rpcq_request(
            "execute_qpu_request",
            request=rpcq_request,
            priority=request.priority,
            user=None,
        )
        return RunProgramResponse(job_id=job_id)

    def get_buffers(self, request: GetBuffersRequest) -> GetBuffersResponse:
        """
        Get job buffers.
        """
        buffs = self._rpcq_request(
            "get_buffers",
            job_id=request.job_id,
            wait=request.wait,
        )
        return GetBuffersResponse(
            buffers={
                name: BufferResponse(
                    shape=cast(Tuple[int, int], tuple(val["shape"])),
                    dtype=val["dtype"],
                    data=val["data"],
                )
                for name, val in buffs.items()
            }
        )

    def get_execution_results(self, request: GetBuffersRequest) -> GetBuffersResponse:
        """
        Get job buffers and execution metadata.
        """
        result = self._rpcq_request(
            "get_execution_results",
            job_id=request.job_id,
            wait=request.wait,
        )
        return GetBuffersResponse(
            buffers={
                name: BufferResponse(
                    shape=cast(Tuple[int, int], tuple(val["shape"])),
                    dtype=val["dtype"],
                    data=val["data"],
                )
                for name, val in result.buffers.items()
            },
            execution_duration_microseconds=result.execution_duration_microseconds,
        )

    @retry(retry=retry_if_exception_type(TimeoutError), stop=stop_after_attempt(2), reraise=True)
    def _rpcq_request(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        engagement = self._engagement_manager.get_engagement(
            endpoint_id=self._endpoint_id,
            quantum_processor_id=self.quantum_processor_id,
            request_timeout=self.timeout,
        )
        client = rpcq.Client(
            endpoint=engagement.address,
            timeout=self._calculate_timeout(engagement),
            auth_config=self._auth_config(engagement.credentials),
        )
        try:
            return client.call(method_name, *args, **kwargs)
        except TimeoutError as e:
            raise TimeoutError(
                f"Request to QPU at {engagement.address} timed out. "
                f"See the Troubleshooting Guide: {DOCS_URL}/troubleshooting.html"
            ) from e
        finally:
            client.close()  # type: ignore

    def _calculate_timeout(self, engagement: EngagementWithCredentials) -> float:
        engagement_time_left = parsedate(engagement.expires_at) - datetime.now(tzutc())
        return min(self.timeout, engagement_time_left.total_seconds())

    @staticmethod
    def _auth_config(credentials: EngagementCredentials) -> rpcq.ClientAuthConfig:
        return rpcq.ClientAuthConfig(
            client_secret_key=credentials.client_secret.encode(),
            client_public_key=credentials.client_public.encode(),
            server_public_key=credentials.server_public.encode(),
        )

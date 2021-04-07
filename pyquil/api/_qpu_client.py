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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, cast, Tuple, Union, List

import rpcq
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials


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


class QPUClient:
    """
    Client for making requests to a QPU.
    """

    def __init__(self, *, engagement: EngagementWithCredentials, request_timeout: float = 5.0) -> None:
        """
        Instantiate a new QPU client, authenticated using the given engagement.

        :param engagement: Engagement for target quantum processor.
        :param request_timeout: Timeout for requests, in seconds.
        """
        self.engagement = engagement
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
        with self._rpcq_client() as rpcq_client:  # type: rpcq.Client
            job_id = rpcq_client.call(
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
        with self._rpcq_client() as rpcq_client:  # type: rpcq.Client
            buffs = rpcq_client.call(
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

    @contextmanager
    def _rpcq_client(self) -> Iterator[rpcq.Client]:
        client = rpcq.Client(
            endpoint=self.engagement.address,
            timeout=self.timeout,
            auth_config=self._auth_config(self.engagement.credentials),
        )
        try:
            yield client
        finally:
            client.close()  # type: ignore

    @staticmethod
    def _auth_config(credentials: EngagementCredentials) -> rpcq.ClientAuthConfig:
        return rpcq.ClientAuthConfig(
            client_secret_key=credentials.client_secret.encode(),
            client_public_key=credentials.client_public.encode(),
            server_public_key=credentials.server_public.encode(),
        )

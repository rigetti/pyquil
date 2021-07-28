from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

import httpx
from pyquil.quilatom import ExpressionDesignator
from sys import executable
from typing import Dict, Iterator, Optional, cast

import numpy as np
from rpcq.messages import ParameterAref

from pyquil.api import EngagementManager
from pyquil.api._qpu import QPU
from pyquil.experimental.api._compiler import ExperimentalExecutable
from qcs_api_client.models.endpoint import Endpoint
from qcs_api_client.operations.asyncio import get_default_endpoint
from qcs_api_client.client._configuration.configuration import QCSClientConfiguration
from qcs_api_client.client.client import build_async_client
from qcs_api_client.grpc.services.controller import ControllerStub
from qcs_api_client.grpc.models.controller import ControllerJobExecutionResultStatus, DataValue, ReadoutValues


@dataclass
class ExperimentalQPUExecuteResponse:
    job_id: str
    """The ID of the job sent for execution."""

    executable: ExperimentalExecutable
    """The executable sent for execution."""


@dataclass
class ExperimentalQPUExecutionResult:
    memory_values: Dict[str, np.ndarray]
    """
    Data values stored in named memory locations at the completion of the program.
    """

    readout_values: Dict[str, np.ndarray]
    """
    Data captured from the control hardware during capture operations on reference frames.
    """

    success: bool
    """
    Whether or not execution succeeded.
    """


class ExperimentalQPU:
    _client_configuration: QCSClientConfiguration
    _service_stub_cache: Optional[ControllerStub]
    _quantum_processor_id: str
    _timeout: Optional[int]

    def __init__(
        self,
        *,
        quantum_processor_id: str,
        timeout: Optional[int] = None,
        client_configuration: Optional[QCSClientConfiguration] = None,
    ) -> None:
        self._quantum_processor_id = quantum_processor_id
        self._client_configuration = client_configuration or QCSClientConfiguration.load()
        self._timeout = timeout
        self._service_stub_cache = None

    async def execute(self, executable: ExperimentalExecutable) -> ExperimentalQPUExecuteResponse:
        assert isinstance(
            executable, ExperimentalExecutable
        ), "ExperimentalQPU#execute requires an ExperimentalExecutable. Create one with ExperimentalQuantumComputer#compile"

        response = await self._get_service_stub().execute_encrypted_controller_job(job=executable.job)

        return ExperimentalQPUExecuteResponse(job_id=response.job_id, executable=executable)

    async def get_result(self, execute_response: ExperimentalQPUExecuteResponse) -> ExperimentalQPUExecutionResult:
        """
        Retrieve results from execution on the QPU.
        """
        response = await self._get_service_stub().get_controller_job_results(job_id=execute_response.job_id)

        return ExperimentalQPUExecutionResult(
            memory_values=_transform_memory_values(
                response.result.memory_values, recalculation_table=execute_response.executable.recalculation_table
            ),
            readout_values=_transform_readout_values(response.result.readout_values),
            success=response.result.status == ControllerJobExecutionResultStatus.SUCCESS,
        )

    async def run(self, executable: ExperimentalExecutable) -> ExperimentalQPUExecutionResult:
        return await self.get_result(execute_response=self.execute(executable=executable))

    async def _get_execution_url(self) -> str:
        """
        Return the URL to which to send requests for execution.

        Return a cached result if stored; otherwise, query the QCS API and store the result.
        """
        async with self._qcs_client() as client:
            response = await get_default_endpoint(client, quantum_processor_id=self._quantum_processor_id)
            response.raise_for_status()
            body = cast(Endpoint, response.parsed)
            url = body.addresses.grpc
            assert url is not None, f"no gRPC address found for quantum processor {self._quantum_processor_id}"


    @asynccontextmanager
    async def _qcs_client(self) -> Iterator[httpx.Client]:
        async with build_async_client(configuration=self._client_configuration) as client:  # type: httpx.Client
            yield client

    def _get_service_stub(self) -> Iterator[ControllerStub]:
        if self._service_stub_cache is None:
            url = self._get_execution_url()
            self._service_stub_cache = ControllerStub(url=url)
        return self._service_stub_cache


def _transform_memory_values(
    memory_values: Dict[str, "DataValue"], recalculation_table: Dict[ParameterAref, ExpressionDesignator]
) -> Dict[str, np.ndarray]:
    """
    Transform the memory values returned from the experimental backend into more ergonomic values
    for use in Python: numpy arrays.

    TODO: Use recalculation table to map values to the user-provided names. See ``pyquil.api._qpu.QPU``.
    """

    result = {}

    for key, value in memory_values.items():
        if value.binary is not None:
            bytes = np.frombuffer(value.binary.data, dtype=np.uint8)

            # TODO: only `unpackbits` if the memory datatype is `BIT`; return bytes if `OCTET`
            # TODO: ensure bit order (endianness) is correct; set `bitorder` if not
            # TODO: use `count` kwarg to handle bit arrays that are not a multiple of 8 in length
            result[key] = np.unpackbits(bytes)
        elif value.integer is not None:
            result[key] = np.asarray(value.integer.data, dtype=np.int64)
        elif value.real is not None:
            result[key] = np.asarray(value.real.data, dtype=np.float64)
        else:
            RuntimeError(f"memory values for {key} were unset; expected binary, integer, or real")

    return result


def _transform_readout_values(readout_values: Dict[str, "ReadoutValues"]) -> Dict[str, np.ndarray]:
    """
    Transform the readout values returned from the experimental backend into more ergonomic values
    for use in Python: numpy arrays.
    """
    result = {}

    for key, value in readout_values.items():
        if value.integer_values is not None:
            result[key] = np.asarray(value.integer_values.values, dtype=np.int32)
        elif value.complex_values is not None:
            result[key] = np.fromiter(
                (v.real + v.imaginary * 1j for v in value.complex_values.values), dtype=np.complex64
            )
        else:
            RuntimeError(f"readout values for {key} were unset; expected integer or complex data")

    return result

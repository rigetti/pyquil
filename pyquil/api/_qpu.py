##############################################################################
# Copyright 2016-2018 Rigetti Computing
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
import asyncio
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import ParameterSpec

from pyquil.api import QuantumExecutable, EncryptedProgram, EngagementManager

from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qpu_client import QPUClient, BufferResponse
from pyquil.quilatom import (
    MemoryReference,
)
import qcs_sdk


def decode_buffer(buffer: "qcs_sdk.ExecutionResult") -> np.ndarray:
    """
    Translate a DataBuffer into a numpy array.

    :param buffer: Dictionary with 'data' byte array, 'dtype', and 'shape' fields
    :return: NumPy array of decoded data
    """
    if buffer["dtype"] == "complex":
        buffer["data"] = [complex(re, im) for re, im in buffer["data"]]  # type: ignore
        buffer["dtype"] = np.complex64  # type: ignore
    elif buffer["dtype"] == "integer":
        buffer["dtype"] = np.int32  # type: ignore
    return np.array(buffer["data"], dtype=buffer["dtype"])


def _extract_memory_regions(
    memory_descriptors: Dict[str, ParameterSpec],
    ro_sources: Dict[MemoryReference, str],
    buffers: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:

    # hack to extract num_shots indirectly from the shape of the returned data
    first, *rest = buffers.values()
    num_shots = first.shape[0]

    def alloc(spec: ParameterSpec) -> np.ndarray:
        dtype = {
            "BIT": np.int64,
            "INTEGER": np.int64,
            "REAL": np.float64,
            "FLOAT": np.float64,
        }
        try:
            return np.ndarray((num_shots, spec.length), dtype=dtype[spec.type])
        except KeyError:
            raise ValueError(f"Unexpected memory type {spec.type}.")

    regions: Dict[str, np.ndarray] = {}

    for mref, key in ro_sources.items():
        # Translation sometimes introduces ro_sources that the user didn't ask for.
        # That's fine, we just ignore them.
        if mref.name not in memory_descriptors:
            continue
        elif mref.name not in regions:
            regions[mref.name] = alloc(memory_descriptors[mref.name])

        buf = buffers[key]
        if buf.ndim == 1:
            buf = buf.reshape((num_shots, 1))

        if np.iscomplexobj(buf):  # type: ignore
            buf = np.column_stack((buf.real, buf.imag))  # type: ignore
        _, width = buf.shape

        end = mref.offset + width
        region_width = memory_descriptors[mref.name].length
        if end > region_width:
            raise ValueError(
                f"Attempted to fill {mref.name}[{mref.offset}, {end})"
                f"but the declared region has width {region_width}."
            )

        regions[mref.name][:, mref.offset : end] = buf

    return regions


@dataclass
class QPUExecuteResponse:
    job_id: str
    _executable: EncryptedProgram


class QPU(QAM[QPUExecuteResponse]):
    def __init__(
        self,
        *,
        quantum_processor_id: str,
        priority: int = 1,
        timeout: float = 10.0,
        client_configuration: Optional[QCSClientConfiguration] = None,
        engagement_manager: Optional[EngagementManager] = None,
        endpoint_id: Optional[str] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        use_gateway: bool = True,
    ) -> None:
        """
        A connection to the QPU.

        :param quantum_processor_id: Processor to run against.
        :param priority: The priority with which to insert jobs into the QPU queue. Lower integers
            correspond to higher priority.
        :param timeout: Time limit for requests, in seconds.
        :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        :param endpoint_id: Optional endpoint ID to be used for engagement.
        :param engagement_manager: Optional engagement manager. If none is provided, a default one will be created.
        :param use_gateway: Disable to skip the Gateway server and perform direct execution.
        """
        super().__init__()

        self.priority = priority

        client_configuration = client_configuration or QCSClientConfiguration.load()
        engagement_manager = engagement_manager or EngagementManager(client_configuration=client_configuration)
        self._qpu_client = QPUClient(
            quantum_processor_id=quantum_processor_id,
            endpoint_id=endpoint_id,
            engagement_manager=engagement_manager,
            request_timeout=timeout,
        )
        self._last_results: Dict[str, np.ndarray] = {}
        self._memory_results: Dict[str, Optional[np.ndarray]] = defaultdict(lambda: None)

        if event_loop is None:
            event_loop = asyncio.get_event_loop()
        self._event_loop = event_loop
        self._use_gateway = use_gateway

    @property
    def quantum_processor_id(self) -> str:
        """ID of quantum processor targeted."""
        return self._qpu_client.quantum_processor_id

    def execute(self, executable: QuantumExecutable) -> QPUExecuteResponse:
        """
        Enqueue a job for execution on the QPU. Returns a ``QPUExecuteResponse``, a
        job descriptor which should be passed directly to ``QPU.get_result`` to retrieve
        results.
        """
        executable = executable.copy()

        assert isinstance(
            executable, EncryptedProgram
        ), "QPU#execute requires an rpcq.EncryptedProgram. Create one with QuantumComputer#compile"

        assert (
            executable.ro_sources is not None
        ), "To run on a QPU, a program must include ``MEASURE``, ``CAPTURE``, and/or ``RAW-CAPTURE`` instructions"

        # executable._memory.values is a dict of ParameterARef -> numbers, where ParameterARef is data class w/ name and index
        # ParamterARef == Parameter on the Rust side
        mem_values = defaultdict(list)
        for k, v in executable._memory.values.items():
            mem_values[k.name].append(v)
        patch_values = qcs_sdk.build_patch_values(executable.recalculation_table, mem_values)

        async def _submit(*args) -> str:  # type: ignore
            return await qcs_sdk.submit(*args)

        job_id = self._event_loop.run_until_complete(
            _submit(executable.program, patch_values, self.quantum_processor_id, self._use_gateway)
        )

        return QPUExecuteResponse(_executable=executable, job_id=job_id)

    def get_result(self, execute_response: QPUExecuteResponse) -> QAMExecutionResult:
        """
        Retrieve results from execution on the QPU.
        """

        async def _get_result(*args) -> qcs_sdk.ExecutionResults:  # type: ignore
            return await qcs_sdk.retrieve_results(*args)

        results = self._event_loop.run_until_complete(
            _get_result(execute_response.job_id, self.quantum_processor_id, self._use_gateway)
        )

        ro_sources = execute_response._executable.ro_sources
        decoded_buffers = {k: decode_buffer(v) for k, v in results["buffers"].items()}

        result_memory = {}
        if len(decoded_buffers) != 0:
            extracted = _extract_memory_regions(
                execute_response._executable.memory_descriptors, ro_sources, decoded_buffers
            )
            for name, array in extracted.items():
                result_memory[name] = array
        elif not ro_sources:
            result_memory["ro"] = np.zeros((0, 0), dtype=np.int64)

        return QAMExecutionResult(
            executable=execute_response._executable,
            readout_data=result_memory,
            execution_duration_microseconds=results["execution_duration_microseconds"],
        )

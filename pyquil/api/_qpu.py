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
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Union, cast

import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import ParameterAref, ParameterSpec

from pyquil.api import QuantumExecutable, EncryptedProgram, EngagementManager

from pyquil._memory import Memory
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qpu_client import GetBuffersRequest, QPUClient, BufferResponse, RunProgramRequest
from pyquil.quilatom import (
    MemoryReference,
    BinaryExp,
    Function,
    Parameter,
    ExpressionDesignator,
)


def decode_buffer(buffer: BufferResponse) -> np.ndarray:
    """
    Translate a DataBuffer into a numpy array.

    :param buffer: Dictionary with 'data' byte array, 'dtype', and 'shape' fields
    :return: NumPy array of decoded data
    """
    buf = np.frombuffer(buffer.data, dtype=buffer.dtype)  # type: ignore
    return buf.reshape(buffer.shape)  # type: ignore


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

        request = RunProgramRequest(
            id=str(uuid.uuid4()),
            priority=self.priority,
            program=executable.program,
            patch_values=self._build_patch_values(executable),
        )
        job_id = self._qpu_client.run_program(request).job_id
        return QPUExecuteResponse(_executable=executable, job_id=job_id)

    def get_result(self, execute_response: QPUExecuteResponse) -> QAMExecutionResult:
        """
        Retrieve results from execution on the QPU.
        """
        request = GetBuffersRequest(job_id=execute_response.job_id, wait=True)
        results = self._qpu_client.get_execution_results(request)

        ro_sources = execute_response._executable.ro_sources
        decoded_buffers = {k: decode_buffer(v) for k, v in results.buffers.items()}

        result_memory = {}
        if decoded_buffers is not None:
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
            execution_duration_microseconds=results.execution_duration_microseconds,
        )

    @classmethod
    def _build_patch_values(cls, program: EncryptedProgram) -> Dict[str, List[Union[int, float]]]:
        """
        Construct the patch values from the program to be used in execution.
        """
        patch_values = {}

        # Now that we are about to run, we have to resolve any gate parameter arithmetic that was
        # saved in the executable's recalculation table, and add those values to the variables shim
        cls._update_memory_with_recalculation_table(program=program)

        # Initialize our patch table
        assert isinstance(program, EncryptedProgram)
        recalculation_table = program.recalculation_table
        memory_ref_names = list(set(mr.name for mr in recalculation_table.keys()))
        if memory_ref_names:
            assert len(memory_ref_names) == 1, (
                "We expected only one declared memory region for "
                "the gate parameter arithmetic replacement references."
            )
            memory_reference_name = memory_ref_names[0]
            patch_values[memory_reference_name] = [0.0] * len(recalculation_table)

        for name, spec in program.memory_descriptors.items():
            # NOTE: right now we fake reading out measurement values into classical memory
            # hence we omit them here from the patch table.
            if any(name == mref.name for mref in program.ro_sources):
                continue
            initial_value = 0.0 if spec.type == "REAL" else 0
            patch_values[name] = [initial_value] * spec.length

        # Fill in our patch table
        for k, v in program._memory.values.items():
            patch_values[k.name][k.index] = v

        return patch_values

    @classmethod
    def _update_memory_with_recalculation_table(cls, program: EncryptedProgram) -> None:
        """
        Update the program's memory with the final values to be patched into the gate parameters,
        according to the arithmetic expressions in the original program.

        For example::

            DECLARE theta REAL
            DECLARE beta REAL
            RZ(3 * theta) 0
            RZ(beta+theta) 0

        gets translated to::

            DECLARE theta REAL
            DECLARE __P REAL[2]
            RZ(__P[0]) 0
            RZ(__P[1]) 0

        and the recalculation table will contain::

            {
                ParameterAref('__P', 0): Mul(3.0, <MemoryReference theta[0]>),
                ParameterAref('__P', 1): Add(<MemoryReference beta[0]>, <MemoryReference theta[0]>)
            }

        Let's say we've made the following two function calls:

        .. code-block:: python

            compiled_program.write_memory(region_name='theta', value=0.5)
            compiled_program.write_memory(region_name='beta', value=0.1)

        After executing this function, our self.variables_shim in the above example would contain
        the following:

        .. code-block:: python

            {
                ParameterAref('theta', 0): 0.5,
                ParameterAref('beta', 0): 0.1,
                ParameterAref('__P', 0): 1.5,       # (3.0) * theta[0]
                ParameterAref('__P', 1): 0.6        # beta[0] + theta[0]
            }

        Once the _variables_shim is filled, execution continues as with regular binary patching.
        """
        assert isinstance(program, EncryptedProgram)
        for mref, expression in program.recalculation_table.items():
            # Replace the user-declared memory references with any values the user has written,
            # coerced to a float because that is how we declared it.
            program._memory.values[mref] = float(cls._resolve_memory_references(expression, memory=program._memory))

    @classmethod
    def _resolve_memory_references(cls, expression: ExpressionDesignator, memory: Memory) -> Union[float, int]:
        """
        Traverse the given Expression, and replace any Memory References with whatever values
        have been so far provided by the user for those memory spaces. Declared memory defaults
        to zero.

        :param expression: an Expression
        """
        if isinstance(expression, BinaryExp):
            left = cls._resolve_memory_references(expression.op1, memory=memory)
            right = cls._resolve_memory_references(expression.op2, memory=memory)
            return cast(Union[float, int], expression.fn(left, right))
        elif isinstance(expression, Function):
            return cast(
                Union[float, int],
                expression.fn(cls._resolve_memory_references(expression.expression, memory=memory)),
            )
        elif isinstance(expression, Parameter):
            raise ValueError(f"Unexpected Parameter in gate expression: {expression}")
        elif isinstance(expression, (float, int)):
            return expression
        elif isinstance(expression, MemoryReference):
            return memory.values.get(ParameterAref(name=expression.name, index=expression.offset), 0)
        else:
            raise ValueError(f"Unexpected expression in gate parameter: {expression}")

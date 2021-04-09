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
import uuid
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Union, cast

import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import ParameterAref, ParameterSpec

from pyquil.api import QuantumExecutable, EncryptedProgram, EngagementManager
from pyquil.api._error_reporting import _record_call
from pyquil.api._qam import QAM
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
    buf = np.frombuffer(buffer.data, dtype=buffer.dtype)
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

        if np.iscomplexobj(buf):
            buf = np.column_stack((buf.real, buf.imag))
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


class QPU(QAM):
    @_record_call
    def __init__(
        self,
        *,
        quantum_processor_id: str,
        priority: int = 1,
        timeout: float = 5.0,
        client_configuration: Optional[QCSClientConfiguration] = None,
        engagement_manager: Optional[EngagementManager] = None,
    ) -> None:
        """
        A connection to the QPU.

        :param quantum_processor_id: Processor to run against.
        :param priority: The priority with which to insert jobs into the QPU queue. Lower
                         integers correspond to higher priority.
        :param timeout: Time limit for requests, in seconds.
        :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        :param engagement_manager: Optional engagement manager. If none is provided, a default one will be created.
        """
        super().__init__()

        self.priority = priority

        client_configuration = client_configuration or QCSClientConfiguration.load()
        engagement_manager = engagement_manager or EngagementManager(client_configuration=client_configuration)
        self._qpu_client = QPUClient(
            quantum_processor_id=quantum_processor_id,
            engagement_manager=engagement_manager,
            request_timeout=timeout,
        )
        self._last_results: Dict[str, np.ndarray] = {}
        self._memory_results: Dict[str, Optional[np.ndarray]] = defaultdict(lambda: None)

    @property
    def quantum_processor_id(self) -> str:
        """ID of quantum processor targeted."""
        return self._qpu_client.quantum_processor_id

    @_record_call
    def load(self, executable: QuantumExecutable) -> "QPU":
        """
        Initialize a QAM into a fresh state. Load the executable and parse the expressions
        in the recalculation table (if any) into pyQuil Expression objects.

        :param executable: Load a compiled executable onto the QAM.
        """
        if not isinstance(executable, EncryptedProgram):
            raise TypeError(
                "`executable` argument must be an `EncryptedProgram`. Make "
                "sure you have explicitly compiled your program via `qc.compile` "
                "or `qc.compiler.native_quil_to_executable(...)` for more "
                "fine-grained control."
            )

        super().load(executable)
        return self

    @_record_call
    def run(self, run_priority: Optional[int] = None) -> "QPU":
        """
        Run a pyquil program on the QPU.

        This formats the classified data from the QPU server by stacking measured bits into
        an array of shape (trials, classical_addresses). The mapping of qubit to
        classical address is backed out from MEASURE instructions in the program, so
        only do measurements where there is a 1-to-1 mapping between qubits and classical
        addresses. If no MEASURE instructions are present in the program, a 0-by-0 array is
        returned.

        :param run_priority: The priority with which to insert jobs into the QPU queue. Lower
                             integers correspond to higher priority. If not specified, the QPU
                             object's default priority is used.
        :return: The QPU object itself.
        """
        super().run()
        assert isinstance(self.executable, EncryptedProgram)

        request = RunProgramRequest(
            id=str(uuid.uuid4()),
            priority=run_priority if run_priority is not None else self.priority,
            program=self.executable.program,
            patch_values=self._build_patch_values(),
        )
        job_id = self._qpu_client.run_program(request).job_id
        results = self._get_buffers(job_id)
        ro_sources = self.executable.ro_sources

        self._memory_results = defaultdict(lambda: None)
        if results:
            extracted = _extract_memory_regions(self.executable.memory_descriptors, ro_sources, results)
            for name, array in extracted.items():
                self._memory_results[name] = array
        elif not ro_sources:
            warnings.warn(
                "You are running a QPU program with no MEASURE instructions. "
                "The result of this program will always be an empty array. Are "
                "you sure you didn't mean to measure some of your qubits?"
            )
            self._memory_results["ro"] = np.zeros((0, 0), dtype=np.int64)

        self._last_results = results

        return self

    def _get_buffers(self, job_id: str) -> Dict[str, np.ndarray]:
        """
        Return the decoded result buffers for particular job_id.

        :param job_id: Unique identifier for the job in question
        :return: Decoded buffers or throw an error
        """
        request = GetBuffersRequest(job_id=job_id, wait=True)
        buffers = self._qpu_client.get_buffers(request).buffers
        return {k: decode_buffer(v) for k, v in buffers.items()}

    def _build_patch_values(self) -> Dict[str, List[Union[int, float]]]:
        patch_values = {}

        # Now that we are about to run, we have to resolve any gate parameter arithmetic that was
        # saved in the executable's recalculation table, and add those values to the variables shim
        self._update_variables_shim_with_recalculation_table()

        # Initialize our patch table
        assert isinstance(self.executable, EncryptedProgram)
        recalculation_table = self.executable.recalculation_table
        memory_ref_names = list(set(mr.name for mr in recalculation_table.keys()))
        if memory_ref_names:
            assert len(memory_ref_names) == 1, (
                "We expected only one declared memory region for "
                "the gate parameter arithmetic replacement references."
            )
            memory_reference_name = memory_ref_names[0]
            patch_values[memory_reference_name] = [0.0] * len(recalculation_table)

        for name, spec in self.executable.memory_descriptors.items():
            # NOTE: right now we fake reading out measurement values into classical memory
            # hence we omit them here from the patch table.
            if any(name == mref.name for mref in self.executable.ro_sources):
                continue
            initial_value = 0.0 if spec.type == "REAL" else 0
            patch_values[name] = [initial_value] * spec.length

        # Fill in our patch table
        for k, v in self._variables_shim.items():
            patch_values[k.name][k.index] = v

        return patch_values

    def _update_variables_shim_with_recalculation_table(self) -> None:
        """
        Update self._variables_shim with the final values to be patched into the gate parameters,
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

            qpu.write_memory(region_name='theta', value=0.5)
            qpu.write_memory(region_name='beta', value=0.1)

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
        assert isinstance(self.executable, EncryptedProgram)
        for mref, expression in self.executable.recalculation_table.items():
            # Replace the user-declared memory references with any values the user has written,
            # coerced to a float because that is how we declared it.
            self._variables_shim[mref] = float(self._resolve_memory_references(expression))

    def _resolve_memory_references(self, expression: ExpressionDesignator) -> Union[float, int]:
        """
        Traverse the given Expression, and replace any Memory References with whatever values
        have been so far provided by the user for those memory spaces. Declared memory defaults
        to zero.

        :param expression: an Expression
        """
        if isinstance(expression, BinaryExp):
            left = self._resolve_memory_references(expression.op1)
            right = self._resolve_memory_references(expression.op2)
            return cast(Union[float, int], expression.fn(left, right))
        elif isinstance(expression, Function):
            return cast(
                Union[float, int],
                expression.fn(self._resolve_memory_references(expression.expression)),
            )
        elif isinstance(expression, Parameter):
            raise ValueError(f"Unexpected Parameter in gate expression: {expression}")
        elif isinstance(expression, (float, int)):
            return expression
        elif isinstance(expression, MemoryReference):
            return self._variables_shim.get(ParameterAref(name=expression.name, index=expression.offset), 0)
        else:
            raise ValueError(f"Unexpected expression in gate parameter: {expression}")

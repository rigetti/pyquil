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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rpcq import Client
from rpcq.messages import QPURequest, ParameterAref

from pyquil import parser
from pyquil.api._qam import QAM
from pyquil.api._error_reporting import _record_call
from pyquil.quilatom import MemoryReference, BinaryExp, Function, Parameter, Expression


def decode_buffer(buffer: dict) -> np.ndarray:
    """
    Translate a DataBuffer into a numpy array.

    :param buffer: Dictionary with 'data' byte array, 'dtype', and 'shape' fields
    :return: NumPy array of decoded data
    """
    buf = np.frombuffer(buffer['data'], dtype=buffer['dtype'])
    return buf.reshape(buffer['shape'])


def _extract_bitstrings(ro_sources: List[Optional[Tuple[int, int]]],
                        buffers: Dict[str, np.ndarray]
                        ) -> np.ndarray:
    """
    De-mux qubit readout results and assemble them into the ro-bitstrings in the correct order.

    :param ro_sources: Specification of the ro_sources, cf
        :py:func:`pyquil.api._compiler._collect_classical_memory_write_locations`.
        It is a list whose value ``(q, m)`` at index ``addr`` records that the ``m``-th measurement
        of qubit ``q`` was measured into ``ro`` address ``addr``. A value of `None` means nothing
        was measured into ``ro`` address ``addr``.
    :param buffers: A dictionary of readout results returned from the qpu.
    :return: A numpy array of shape ``(num_shots, len(ro_sources))`` with the readout bits.
    """
    # hack to extract num_shots indirectly from the shape of the returned data
    first, *rest = buffers.values()
    num_shots = first.shape[0]
    bitstrings = np.zeros((num_shots, len(ro_sources)), dtype=np.int64)
    for col_idx, src in enumerate(ro_sources):
        if src:
            qubit, meas_idx = src
            buf = buffers[f"q{qubit}"]
            if buf.ndim == 1:
                buf = buf.reshape((num_shots, 1))
            bitstrings[:, col_idx] = buf[:, meas_idx]
    return bitstrings


class QPU(QAM):
    @_record_call
    def __init__(self, endpoint: str, user: str = "pyquil-user") -> None:
        """
        A connection to the QPU.

        :param endpoint: Address to connect to the QPU server.
        :param user: A string identifying who's running jobs.
        """
        super().__init__()
        self.client = Client(endpoint)
        self.user = user
        self._last_results: Dict[str, np.ndarray] = {}

    def get_version_info(self) -> dict:
        """
        Return version information for this QPU's execution engine and its dependencies.

        :return: Dictionary of version information.
        """
        return self.client.call('get_version_info')

    @_record_call
    def run(self):
        """
        Run a pyquil program on the QPU.

        This formats the classified data from the QPU server by stacking measured bits into
        an array of shape (trials, classical_addresses). The mapping of qubit to
        classical address is backed out from MEASURE instructions in the program, so
        only do measurements where there is a 1-to-1 mapping between qubits and classical
        addresses.

        :return: The QPU object itself.
        """
        super().run()

        request = QPURequest(program=self._executable.program,
                             patch_values=self._build_patch_values(),
                             id=str(uuid.uuid4()))

        job_id = self.client.call('execute_qpu_request', request=request, user=self.user)
        results = self._get_buffers(job_id)
        ro_sources = self._executable.ro_sources

        if results:
            bitstrings = _extract_bitstrings(ro_sources, results)
        elif not ro_sources:
            warnings.warn("You are running a QPU program with no MEASURE instructions. "
                          "The result of this program will always be an empty array. Are "
                          "you sure you didn't mean to measure some of your qubits?")
            bitstrings = np.zeros((0, 0), dtype=np.int64)
        else:
            bitstrings = None

        self._bitstrings = bitstrings
        self._last_results = results
        return self

    def _get_buffers(self, job_id: str) -> Dict[str, np.ndarray]:
        """
        Return the decoded result buffers for particular job_id.

        :param job_id: Unique identifier for the job in question
        :return: Decoded buffers or throw an error
        """
        buffers = self.client.call('get_buffers', job_id, wait=True)
        return {k: decode_buffer(v) for k, v in buffers.items()}

    def _build_patch_values(self) -> dict:
        patch_table = {}

        for name, spec in self._executable.memory_descriptors.items():
            # NOTE: right now we fake reading out measurement values into classical memory
            if name == "ro":
                continue
            patch_table[name] = [0] * spec.length

        mref_name = self._build_arithmetic_patch_values()
        if mref_name:
            patch_table[mref_name] = [0] * len(self._executable.recalculation_table)

        for k, v in self._variables_shim.items():
            # NOTE: right now we fake reading out measurement values into classical memory
            if k.name == "ro":
                continue

            # floats stored in tsunami memory are expected to be in revolutions rather than radians.
            if isinstance(v, float):
                v /= 2 * np.pi

            patch_table[k.name][k.index] = v

        return patch_table

    def _build_arithmetic_patch_values(self) -> str:
        """
        Update the variables shim with the final values to be patched into the gate parameters,
        according to the arithmetic expressions in the original program.

        For example:

            DECLARE theta REAL
            RZ(3 * theta) 0

        gets translated to:

            DECLARE theta REAL
            DECLARE __P REAL[0]
            RZ(__P[0]) 0

        and the recalculation table gets the entry: __P[0] = '3 * theta'. In our example, we
        look up the value of 'theta' in the variables shim, multiply that by 3, and then add
        an entry for __P[0] to the variables shim.

        :return: Name of the region used to patch the parameters.
        """
        try:
            recalculation_table = self._executable.recalculation_table
        except AttributeError:
            # No recalculation table, no work to be done here.
            return ""

        # We only declare one region to hold these values, and we need to add that name to
        memory_reference_name = None
        for memory_reference, recalc_rule in recalculation_table.items():
            # Parse the arithmetic expression
            expression = parser.parse(f"RZ({recalc_rule}) 0")[0].params[0]
            # Replace the memory references with any values the user has written
            value = self._resolve_memory_references(expression)
            self._variables_shim[memory_reference] = value
            memory_reference_name = memory_reference.name
        return memory_reference_name

    def _resolve_memory_references(self, expression: Expression) -> Union[float, int]:
        """
        Traverse the given Expression, and replace any Memory References with whatever values
        have been so far provided by the user for those memory spaces. Declared memory defaults
        to zero.

        :param expression: an Expression
        """
        if isinstance(expression, BinaryExp):
            left = self._resolve_memory_references(expression.op1)
            right = self._resolve_memory_references(expression.op2)
            return expression.fn(left, right)
        elif isinstance(expression, Function):
            return expression.fn(_resolve_memory_references(expression.expression))
        elif isinstance(expression, Parameter):
            raise ValueError(f"Unexpected Parameter in gate expression: {expression}")
        elif isinstance(expression, float) or isinstance(expression, int):
            return expression
        elif isinstance(expression, MemoryReference):
            return self._variables_shim.get(ParameterAref(name=expression.name, index=expression.offset), 0)
        else:
            raise ValueError(f"Unexpected expression in gate parameter: {expression}")

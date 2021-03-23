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
import logging
from typing import Dict, Optional, cast, List

from qcs_api_client.models import TranslateNativeQuilToEncryptedBinaryResponse
from qcs_api_client.models.get_quilt_calibrations_response import GetQuiltCalibrationsResponse
from qcs_api_client.models.translate_native_quil_to_encrypted_binary_request import (
    TranslateNativeQuilToEncryptedBinaryRequest,
)
from qcs_api_client.operations.sync import (
    translate_native_quil_to_encrypted_binary,
    get_quilt_calibrations,
)
from qcs_api_client.types import UNSET
from rpcq.messages import ParameterSpec

from pyquil.api import Client
from pyquil.api._error_reporting import _record_call
from pyquil.api._abstract_compiler import AbstractCompiler, QuantumExecutable, EncryptedProgram
from pyquil.api._rewrite_arithmetic import rewrite_arithmetic
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.parser import parse_program, parse
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, ExpressionDesignator
from pyquil.quilbase import Declare, Gate

_log = logging.getLogger(__name__)

PYQUIL_PROGRAM_PROPERTIES = ["native_quil_metadata", "num_shots"]


class QPUCompilerNotRunning(Exception):
    pass


def parse_mref(val: str) -> MemoryReference:
    """ Parse a memory reference from its string representation. """
    val = val.strip()
    try:
        if val[-1] == "]":
            name, offset = val.split("[")
            return MemoryReference(name, int(offset[:-1]))
        else:
            return MemoryReference(val)
    except Exception:
        raise ValueError(f"Unable to parse memory reference {val}.")


def _collect_memory_descriptors(program: Program) -> Dict[str, ParameterSpec]:
    """Collect Declare instructions that are important for building the patch table.

    :return: A dictionary of variable names to specs about the declared region.
    """

    return {
        instr.name: ParameterSpec(type=instr.memory_type, length=instr.memory_size)
        for instr in program
        if isinstance(instr, Declare)
    }


class QPUCompiler(AbstractCompiler):
    """
    Client to communicate with the compiler and translation service.
    """

    quantum_processor_id: str
    _calibration_program: Optional[Program] = None

    @_record_call
    def __init__(
        self,
        *,
        quantum_processor_id: str,
        quantum_processor: AbstractQuantumProcessor,
        client: Optional[Client] = None,
        timeout: float = 10,
    ) -> None:
        """
        Instantiate a new QPU compiler client.

        :param quantum_processor_id: Processor to target.
        :param quantum_processor: Quantum processor to use as compilation target.
        :param client: Optional QCS client. If none is provided, a default client will be created.
        :param timeout: Number of seconds to wait for a response from the client.
        """
        super().__init__(quantum_processor=quantum_processor, client=client, timeout=timeout)
        self.quantum_processor_id = quantum_processor_id

    @_record_call
    def native_quil_to_executable(self, nq_program: Program) -> QuantumExecutable:
        arithmetic_response = rewrite_arithmetic(nq_program)
        request = TranslateNativeQuilToEncryptedBinaryRequest(
            quil=arithmetic_response.quil, num_shots=nq_program.num_shots
        )

        # TODO(andrew): timeout?
        response = cast(
            TranslateNativeQuilToEncryptedBinaryResponse,
            self._client.qcs_request(
                translate_native_quil_to_encrypted_binary,
                quantum_processor_id=self.quantum_processor_id,
                json_body=request,
            ),
        )

        ro_sources = cast(List[List[str]], [] if response.ro_sources == UNSET else response.ro_sources)

        def to_expression(rule: str) -> ExpressionDesignator:
            # We can only parse complete lines of Quil, so we wrap the arithmetic expression
            # in a valid Quil instruction to parse it.
            # TODO: This hack should be replaced after #687
            return cast(ExpressionDesignator, cast(Gate, parse(f"RZ({rule}) 0")[0]).params[0])

        return EncryptedProgram(
            program=response.program,
            memory_descriptors=_collect_memory_descriptors(nq_program),
            ro_sources={parse_mref(mref): source for mref, source in ro_sources},
            recalculation_table={
                mref: to_expression(rule) for mref, rule in arithmetic_response.recalculation_table.items()
            },
        )

    def _get_calibration_program(self) -> Program:
        response = cast(
            GetQuiltCalibrationsResponse,
            self._client.qcs_request(get_quilt_calibrations, quantum_processor_id=self.quantum_processor_id),
        )
        return parse_program(response.quilt)

    @property  # type: ignore
    @_record_call
    def calibration_program(self) -> Program:
        """
        Get the Quil-T calibration program associated with the underlying QPU.

        A calibration program contains a number of DEFCAL, DEFWAVEFORM, and
        DEFFRAME instructions. In sum, those instructions describe how a Quil-T
        program should be translated into analog instructions for execution on
        hardware.

        :returns: A Program object containing the calibration definitions."""
        if self._calibration_program is None:
            try:
                self._calibration_program = self._get_calibration_program()
            except Exception as ex:
                raise RuntimeError("Could not refresh calibrations") from ex

        return self._calibration_program

    @_record_call
    def reset(self) -> None:
        """
        Reset the state of the QPUCompiler.
        """
        super().reset()
        self._calibration_program = None


class QVMCompiler(AbstractCompiler):
    """
    Client to communicate with the compiler.
    """

    @_record_call
    def __init__(
        self, *, quantum_processor: AbstractQuantumProcessor, client: Optional[Client] = None, timeout: float = 10
    ) -> None:
        """
        Client to communicate with compiler.

        :param quantum_processor: Quantum processor to use as compilation target.
        :param client: Optional QCS client. If none is provided, a default client will be created.
        :param timeout: Number of seconds to wait for a response from the client.
        """
        super().__init__(quantum_processor=quantum_processor, client=client, timeout=timeout)

    @_record_call
    def native_quil_to_executable(self, nq_program: Program) -> QuantumExecutable:
        return nq_program

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
from typing import Any, Dict, Optional

from qcs_sdk import QCSClient
from qcs_sdk.qpu.rewrite_arithmetic import rewrite_arithmetic
from qcs_sdk.qpu.translation import (
    get_quilt_calibrations,
    translate,
    TranslationOptions as QPUCompilerAPIOptions,
)
from qcs_sdk.compiler.quilc import QuilcClient
from rpcq.messages import ParameterSpec

from pyquil.api._abstract_compiler import AbstractCompiler, EncryptedProgram, QuantumExecutable
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare


class QPUCompilerNotRunning(Exception):
    pass


def parse_mref(val: str) -> MemoryReference:
    """Parse a memory reference from its string representation."""
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

    api_options: Optional[QPUCompilerAPIOptions]

    def __init__(
        self,
        *,
        quantum_processor_id: str,
        quantum_processor: AbstractQuantumProcessor,
        timeout: float = 10.0,
        client_configuration: Optional[QCSClient] = None,
        api_options: Optional[QPUCompilerAPIOptions] = None,
        quilc_client: Optional[QuilcClient] = None,
    ) -> None:
        """
        Instantiate a new QPU compiler client.

        :param quantum_processor_id: Processor to target.
        :param quantum_processor: Quantum processor to use as compilation target.
        :param timeout: Time limit for requests, in seconds.
        :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        :param api_options: Options to pass to the QPU compiler API. See ``qcs-sdk-python`` for details.
        """
        super().__init__(
            quantum_processor=quantum_processor,
            timeout=timeout,
            client_configuration=client_configuration,
        )

        self.api_options = api_options
        self.quantum_processor_id = quantum_processor_id
        self._calibration_program: Optional[Program] = None

    def native_quil_to_executable(
        self, nq_program: Program, *, api_options: Optional[QPUCompilerAPIOptions] = None, **kwargs: Any
    ) -> QuantumExecutable:
        """
        Convert a native Quil program into an executable binary which can be executed by a QPU.

        If `api_options` is provided, it overrides the options set on `self`.
        """
        rewrite_response = rewrite_arithmetic(nq_program.out())

        translated_program = translate(
            native_quil=rewrite_response.program,
            num_shots=nq_program.num_shots,
            quantum_processor_id=self.quantum_processor_id,
            translation_options=api_options or self.api_options,
        )

        ro_sources = translated_program.ro_sources or {}

        return EncryptedProgram(
            program=translated_program.program,
            memory_descriptors=_collect_memory_descriptors(nq_program),
            ro_sources={parse_mref(mref): source for mref, source in ro_sources.items() or []},
            recalculation_table=list(rewrite_response.recalculation_table),
        )

    def _fetch_calibration_program(self) -> Program:
        response = get_quilt_calibrations(
            quantum_processor_id=self.quantum_processor_id,
        )
        return Program(response.quilt)

    def get_calibration_program(self, force_refresh: bool = False) -> Program:
        """
        Get the Quil-T calibration program associated with the underlying QPU.

        This will return a cached copy of the calibration program if present.
        Otherwise (or if forcing a refresh), it will fetch and store the
        calibration program from the QCS API.

        A calibration program contains a number of DEFCAL, DEFWAVEFORM, and
        DEFFRAME instructions. In sum, those instructions describe how a Quil-T
        program should be translated into analog instructions for execution on
        hardware.

        :param force_refresh: Whether or not to fetch a new calibration program before returning.
        :returns: A Program object containing the calibration definitions."""
        if force_refresh or self._calibration_program is None:
            try:
                self._calibration_program = self._fetch_calibration_program()
            except Exception as ex:
                raise RuntimeError("Could not fetch calibrations") from ex

        assert self._calibration_program is not None
        return self._calibration_program

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

    def __init__(
        self,
        *,
        quantum_processor: AbstractQuantumProcessor,
        timeout: float = 10.0,
        client_configuration: Optional[QCSClient] = None,
        quilc_client: Optional[QuilcClient] = None,
    ) -> None:
        """
        Client to communicate with compiler.

        :param quantum_processor: Quantum processor to use as compilation target.
        :param timeout: Number of seconds to wait for a response from the client.
        :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        """
        super().__init__(
            quantum_processor=quantum_processor,
            timeout=timeout,
            client_configuration=client_configuration,
            quilc_client=quilc_client,
        )

    def native_quil_to_executable(self, nq_program: Program, **kwargs: Any) -> QuantumExecutable:
        return nq_program

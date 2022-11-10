##############################################################################
# Copyright 2018 Rigetti Computing
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
from abc import ABC, abstractmethod
from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Union
import asyncio
import json

import qcs_sdk

from pyquil._memory import Memory
from pyquil._version import pyquil_version
from pyquil.api._compiler_client import CompilerClient, CompileToNativeQuilRequest
from pyquil.external.rpcq import compiler_isa_to_target_quantum_processor
from pyquil.parser import parse_program
from pyquil.paulis import PauliTerm
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quil import Program
from pyquil.quilatom import ExpressionDesignator, MemoryReference
from pyquil.quilbase import Gate
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import NativeQuilMetadata, ParameterAref, ParameterSpec


class QuilcVersionMismatch(Exception):
    pass


class QuilcNotRunning(Exception):
    pass


@dataclass
class EncryptedProgram:
    """
    Encrypted binary, executable on a QPU.
    """

    program: str
    """String representation of an encrypted Quil program."""

    memory_descriptors: Dict[str, ParameterSpec]
    """Descriptors for memory executable's regions, mapped by name."""

    ro_sources: Dict[MemoryReference, str]
    """Readout sources, mapped by memory reference."""

    recalculation_table: List[str]
    """A mapping from memory references to the original gate arithmetic."""

    _memory: Memory
    """Memory values (parameters) to be sent with the program."""

    def copy(self) -> "EncryptedProgram":
        """
        Return a deep copy of this EncryptedProgram.
        """
        return dataclasses.replace(self, _memory=self._memory.copy())

    def write_memory(
        self,
        *,
        region_name: str,
        value: Union[int, float, Sequence[int], Sequence[float]],
        offset: Optional[int] = None,
    ) -> "EncryptedProgram":
        self._memory._write_value(parameter=ParameterAref(name=region_name, index=(offset or 0)), value=value)
        return self


QuantumExecutable = Union[EncryptedProgram, Program]


class AbstractCompiler(ABC):
    """The abstract interface for a compiler."""

    _event_loop: asyncio.AbstractEventLoop

    def __init__(
        self,
        *,
        quantum_processor: AbstractQuantumProcessor,
        timeout: float,
        client_configuration: Optional[QCSClientConfiguration] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.quantum_processor = quantum_processor
        self._timeout = timeout

        self._client_configuration = client_configuration or QCSClientConfiguration.load()

        if event_loop is None:
            event_loop = asyncio.get_event_loop()
        self._event_loop = event_loop

        self._compiler_client = CompilerClient(
            client_configuration=self._client_configuration, request_timeout=timeout, event_loop=self._event_loop
        )

        self._connect()

    def get_version_info(self) -> Dict[str, Any]:
        """
        Return version information for this compiler and its dependencies.

        :return: Dictionary of version information.
        """
        return {"quilc": self._compiler_client.get_version()}

    def quil_to_native_quil(self, program: Program, *, protoquil: Optional[bool] = None) -> Program:
        """
        Convert a Quil program into native Quil, which is supported for execution on a QPU.
        """

        # This is a work-around needed because calling `qcs_sdk.compile` happens _before_
        # the event loop is available. Wrapping it in a Python async function ensures that
        # the event loop is available. This is a limitation of pyo3:
        # https://pyo3.rs/v0.17.1/ecosystem/async-await.html#a-note-about-asynciorun
        async def _compile(*args, **kwargs) -> str:  # type: ignore
            return await qcs_sdk.compile(*args, **kwargs)

        # TODO This ISA isn't always going to be available. Specifically, if the quantum processor is
        # a QVM-type processor, then `quantum_processor` will have a CompilerISA, not a QCSISA.
        # This will have to be addressed as part of this issue: https://github.com/rigetti/pyquil/issues/1496
        target_device = compiler_isa_to_target_quantum_processor(self.quantum_processor.to_compiler_isa())
        native_quil = self._event_loop.run_until_complete(
            _compile(
                program.out(calibrations=False),
                json.dumps(target_device.asdict(), indent=2),  # type: ignore
                timeout=self._compiler_client.timeout,
            )
        )

        native_program = Program(native_quil)
        native_program.num_shots = program.num_shots
        native_program._calibrations = program._calibrations
        native_program._waveforms = program._waveforms
        native_program._memory = program._memory.copy()

        return native_program

    def _connect(self) -> None:
        try:
            _check_quilc_version(self._compiler_client.get_version())
        except TimeoutError:
            raise QuilcNotRunning(
                f"Request to quilc at {self._compiler_client.base_url} timed out. "
                "This could mean that quilc is not running, is not reachable, or is "
                "responding slowly."
            )

    @abstractmethod
    def native_quil_to_executable(self, nq_program: Program) -> QuantumExecutable:
        """
        Compile a native quil program to a binary executable.

        :param nq_program: Native quil to compile
        :return: An (opaque) binary executable
        """

    def reset(self) -> None:
        """
        Reset the state of the this compiler.
        """
        pass


def _check_quilc_version(version: str) -> None:
    """
    Verify that there is no mismatch between pyquil and quilc versions.

    :param version: quilc version.
    """
    major, minor, _ = map(int, version.split("."))
    if major == 1 and minor < 8:
        raise QuilcVersionMismatch(
            "Must use quilc >= 1.8.0 with pyquil >= 2.8.0, but you " f"have quilc {version} and pyquil {pyquil_version}"
        )


class AbstractBenchmarker(ABC):
    @abstractmethod
    def apply_clifford_to_pauli(self, clifford: Program, pauli_in: PauliTerm) -> PauliTerm:
        r"""
        Given a circuit that consists only of elements of the Clifford group,
        return its action on a PauliTerm.

        In particular, for Clifford C, and Pauli P, this returns the PauliTerm
        representing PCP^{\dagger}.

        :param clifford: A Program that consists only of Clifford operations.
        :param pauli_in: A PauliTerm to be acted on by clifford via conjugation.
        :return: A PauliTerm corresponding to pauli_in * clifford * pauli_in^{\dagger}
        """

    @abstractmethod
    def generate_rb_sequence(
        self,
        depth: int,
        gateset: Sequence[Gate],
        seed: Optional[int] = None,
        interleaver: Optional[Program] = None,
    ) -> List[Program]:
        """
        Construct a randomized benchmarking experiment on the given qubits, decomposing into
        gateset. If interleaver is not provided, the returned sequence will have the form

            C_1 C_2 ... C_(depth-1) C_inv ,

        where each C is a Clifford element drawn from gateset, C_{< depth} are randomly selected,
        and C_inv is selected so that the entire sequence composes to the identity.  If an
        interleaver G (which must be a Clifford, and which will be decomposed into the native
        gateset) is provided, then the sequence instead takes the form

            C_1 G C_2 G ... C_(depth-1) G C_inv .

        The JSON response is a list of lists of indices, or Nones. In the former case, they are the
        index of the gate in the gateset.

        :param int depth: The number of Clifford gates to include in the randomized benchmarking
         experiment. This is different than the number of gates in the resulting experiment.
        :param list gateset: A list of pyquil gates to decompose the Clifford elements into. These
         must generate the clifford group on the qubits of interest. e.g. for one qubit
         [RZ(np.pi/2), RX(np.pi/2)].
        :param seed: A positive integer used to seed the PRNG.
        :param interleaver: A Program object that encodes a Clifford element.
        :return: A list of pyquil programs. Each pyquil program is a circuit that represents an
         element of the Clifford group. When these programs are composed, the resulting Program
         will be the randomized benchmarking experiment of the desired depth. e.g. if the return
         programs are called cliffords then `sum(cliffords, Program())` will give the randomized
         benchmarking experiment, which will compose to the identity program.
        """

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
import dataclasses
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Union

from deprecated.sphinx import deprecated
from qcs_sdk import QCSClient
from qcs_sdk.compiler.quilc import CompilationResult, CompilerOpts, QuilcClient, TargetDevice, compile_program
from rpcq.messages import ParameterSpec

from pyquil._version import pyquil_version
from pyquil.api._compiler_client import CompilerClient
from pyquil.external.rpcq import compiler_isa_to_target_quantum_processor
from pyquil.paulis import PauliTerm
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Gate


class QuilcVersionMismatch(Exception):
    pass


class QuilcNotRunning(Exception):
    pass


@dataclass
class EncryptedProgram:
    """Encrypted binary, executable on a QPU."""

    program: str
    """String representation of an encrypted Quil program."""

    memory_descriptors: dict[str, ParameterSpec]
    """Descriptors for memory executable's regions, mapped by name."""

    ro_sources: dict[MemoryReference, str]
    """Readout sources, mapped by memory reference."""

    _recalculation_table: list[str] = field(default_factory=list, repr=False, init=False)
    """A mapping from memory references to the original gate arithmetic."""

    def copy(self) -> "EncryptedProgram":
        """Return a deep copy of this EncryptedProgram."""
        return dataclasses.replace(self)

    def __post_init__(self) -> None:
        if any(f.name == "recalculation_table" for f in fields(self)):
            warnings.warn(
                "The recalculation_table field is no longer used. It will be removed in future versions.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    @deprecated(
        version="4.12.0",
        reason="The recalculation_table field is no longer used. It will be removed in future versions.",
    )
    def recalculation_table(self) -> list[str]:
        return self._recalculation_table

    @recalculation_table.setter
    @deprecated(
        version="4.12.0",
        reason="The recalculation_table field is no longer used. Recalculation is now performed by the execution service. This field will be removed in future versions.",
    )
    def recalculation_table(self, value: list[str]) -> None:
        self._recalculation_table = value


QuantumExecutable = Union[EncryptedProgram, Program]


class AbstractCompiler(ABC):
    """The abstract interface for a compiler."""

    def __init__(
        self,
        *,
        quantum_processor: AbstractQuantumProcessor,
        timeout: float,
        client_configuration: Optional[QCSClient] = None,
        quilc_client: Optional[QuilcClient] = None,
    ) -> None:
        self.quantum_processor = quantum_processor
        self._timeout = timeout

        self._client_configuration = client_configuration or QCSClient.load()

        self._compiler_client = CompilerClient(
            client_configuration=self._client_configuration,
            request_timeout=timeout,
            quilc_client=quilc_client,
        )

    def get_version_info(self) -> dict[str, Any]:
        """Return version information for this compiler and its dependencies.

        :return: Dictionary of version information.
        """
        return {"quilc": self._compiler_client.get_version()}

    def quil_to_native_quil(self, program: Program, *, protoquil: Optional[bool] = None) -> Program:
        """Convert a Quil program into native Quil, which is supported for execution on a QPU."""
        result = self._compile_with_quilc(
            program.out(calibrations=False),
            options=CompilerOpts(protoquil=protoquil, timeout=self._compiler_client.timeout),
        )

        native_program = program.copy_everything_except_instructions()
        native_program.native_quil_metadata = result.native_quil_metadata
        native_program.inst(result.program)

        return native_program

    def _compile_with_quilc(self, input: str, options: Optional[CompilerOpts] = None) -> CompilationResult:
        self._connect()

        # convert the pyquil ``TargetDevice`` to the qcs_sdk ``TargetDevice``
        compiler_isa = self.quantum_processor.to_compiler_isa()
        target_device_json = json.dumps(compiler_isa_to_target_quantum_processor(compiler_isa).asdict())  # type: ignore
        target_device = TargetDevice.from_json(target_device_json)

        return compile_program(
            quil=input,
            target=target_device,
            client=self._compiler_client.quilc_client,
            options=options,
        )

    def _connect(self) -> None:
        try:
            _check_quilc_version(self._compiler_client.get_version())
        except TimeoutError as e:
            raise QuilcNotRunning(
                f"Request to quilc at {self._compiler_client.base_url} timed out. "
                "This could mean that quilc is not running, is not reachable, or is "
                "responding slowly. See the Troubleshooting Guide: "
                "{DOCS_URL}/troubleshooting.html"
            ) from e

    def transpile_qasm_2(self, qasm: str) -> Program:
        """Transpile a QASM 2.0 program string to Quil, returning the result as a :py:class:~`pyquil.quil.Program`."""
        result = self._compile_with_quilc(qasm, options=CompilerOpts(timeout=self._compiler_client.timeout))
        return Program(result.program)

    @abstractmethod
    def native_quil_to_executable(self, nq_program: Program, **kwargs: Any) -> QuantumExecutable:
        """Compile a native quil program to a binary executable.

        :param nq_program: Native quil to compile
        :return: An (opaque) binary executable
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the state of the this compiler."""


def _check_quilc_version(version: str) -> None:
    """Verify that there is no mismatch between pyquil and quilc versions.

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
        r"""Given a circuit that consists only of elements of the Clifford group, return its action on a PauliTerm.

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
    ) -> list[Program]:
        r"""Construct a randomized benchmarking experiment on the given qubits, decomposing into gateset.

        If interleaver is not provided, the returned sequence will have the form

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

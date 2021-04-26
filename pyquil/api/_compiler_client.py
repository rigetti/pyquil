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
from typing import Iterator, Optional, List

import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor


@dataclass
class CompileToNativeQuilRequest:
    """
    Request to compile to native Quil.
    """

    program: str
    """Program to compile."""

    target_quantum_processor: TargetQuantumProcessor
    """Quantum processor to target."""

    protoquil: Optional[bool]
    """Whether or not to restrict to protoquil. Overrides server default when provided."""


@dataclass
class NativeQuilMetadataResponse:
    """
    Metadata for a native Quil program.
    """

    final_rewiring: List[int]
    """Output qubit index relabeling due to SWAP insertion."""

    gate_depth: Optional[int]
    """Maximum number of successive gates in the native Quil program."""

    gate_volume: Optional[int]
    """Total number of gates in the native Quil program."""

    multiqubit_gate_depth: Optional[int]
    """Maximum number of successive two-qubit gates in the native Quil program."""

    program_duration: Optional[float]
    """Rough estimate of native Quil program length in nanoseconds."""

    program_fidelity: Optional[float]
    """Rough estimate of the fidelity of the full native Quil program."""

    topological_swaps: Optional[int]
    """Total number of SWAPs in the native Quil program."""

    qpu_runtime_estimation: Optional[float]
    """
    The estimated runtime (milliseconds) on a Rigetti QPU (protoquil program). Available only for protoquil-compliant
    programs.
    """


@dataclass
class CompileToNativeQuilResponse:
    """
    Compile to native Quil response.
    """

    native_program: str
    """Native Quil program."""

    metadata: Optional[NativeQuilMetadataResponse]
    """Metadata for the returned Native Quil."""


@dataclass
class ConjugatePauliByCliffordRequest:
    """
    Request to conjugate a Pauli element by a Clifford element.
    """

    pauli_indices: List[int]
    """Qubit indices onto which the factors of the Pauli term are applied."""

    pauli_symbols: List[str]
    """Ordered factors of the Pauli term."""

    clifford: str
    """Clifford element."""


@dataclass
class ConjugatePauliByCliffordResponse:
    """
    Conjugate Pauli by Clifford response.
    """

    phase_factor: int
    """Encoded global phase factor on the emitted Pauli."""

    pauli: str
    """Description of the encoded Pauli."""


@dataclass
class GenerateRandomizedBenchmarkingSequenceRequest:
    """
    Request to generate a randomized benchmarking sequence.
    """

    depth: int
    """Depth of the benchmarking sequence."""

    num_qubits: int
    """Number of qubits involved in the benchmarking sequence."""

    gateset: List[str]
    """List of Quil programs, each describing a Clifford."""

    seed: Optional[int]
    """PRNG seed. Set this to guarantee repeatable results."""

    interleaver: Optional[str]
    """Fixed Clifford, specified as a Quil string, to interleave through an RB sequence."""


@dataclass
class GenerateRandomizedBenchmarkingSequenceResponse:
    """
    Randomly generated benchmarking sequence response.
    """

    sequence: List[List[int]]
    """List of Cliffords, each expressed as a list of generator indices."""


class CompilerClient:
    """
    Client for making requests to a Quil compiler.
    """

    def __init__(self, *, client_configuration: QCSClientConfiguration, request_timeout: float = 10.0) -> None:
        """
        Instantiate a new compiler client.

        :param client_configuration: Configuration for client.
        :param request_timeout: Timeout for requests, in seconds.
        """
        base_url = client_configuration.profile.applications.pyquil.quilc_url
        if not base_url.startswith("tcp://"):
            raise ValueError(f"Expected compiler URL '{base_url}' to start with 'tcp://'")

        self.base_url = base_url
        self.timeout = request_timeout

    def get_version(self) -> str:
        """
        Get version info for compiler server.
        """
        with self._rpcq_client() as rpcq_client:  # type: rpcq.Client
            version: Optional[str] = rpcq_client.call("get_version_info").get("quilc")
            if version is None:
                raise ValueError("Expected compiler version info to contain a 'quilc' field.")

            return version

    def compile_to_native_quil(self, request: CompileToNativeQuilRequest) -> CompileToNativeQuilResponse:
        """
        Compile Quil program to native Quil.
        """
        rpcq_request = rpcq.messages.NativeQuilRequest(
            quil=request.program,
            target_device=request.target_quantum_processor,
        )
        with self._rpcq_client() as rpcq_client:  # type: rpcq.Client
            response: rpcq.messages.NativeQuilResponse = rpcq_client.call(
                "quil_to_native_quil",
                rpcq_request,
                protoquil=request.protoquil,
            )
            metadata: Optional[NativeQuilMetadataResponse] = None
            if response.metadata is not None:
                metadata = NativeQuilMetadataResponse(
                    final_rewiring=response.metadata.final_rewiring,
                    gate_depth=response.metadata.gate_depth,
                    gate_volume=response.metadata.gate_volume,
                    multiqubit_gate_depth=response.metadata.multiqubit_gate_depth,
                    program_duration=response.metadata.program_duration,
                    program_fidelity=response.metadata.program_fidelity,
                    topological_swaps=response.metadata.topological_swaps,
                    qpu_runtime_estimation=response.metadata.qpu_runtime_estimation,
                )
            return CompileToNativeQuilResponse(native_program=response.quil, metadata=metadata)

    def conjugate_pauli_by_clifford(self, request: ConjugatePauliByCliffordRequest) -> ConjugatePauliByCliffordResponse:
        """
        Conjugate a Pauli element by a Clifford element.
        """
        rpcq_request = rpcq.messages.ConjugateByCliffordRequest(
            pauli=rpcq.messages.PauliTerm(indices=request.pauli_indices, symbols=request.pauli_symbols),
            clifford=request.clifford,
        )
        with self._rpcq_client() as rpcq_client:  # type: rpcq.Client
            response: rpcq.messages.ConjugateByCliffordResponse = rpcq_client.call(
                "conjugate_pauli_by_clifford",
                rpcq_request,
            )
            return ConjugatePauliByCliffordResponse(phase_factor=response.phase, pauli=response.pauli)

    def generate_randomized_benchmarking_sequence(
        self, request: GenerateRandomizedBenchmarkingSequenceRequest
    ) -> GenerateRandomizedBenchmarkingSequenceResponse:
        """
        Generate a randomized benchmarking sequence.
        """
        rpcq_request = rpcq.messages.RandomizedBenchmarkingRequest(
            depth=request.depth,
            qubits=request.num_qubits,
            gateset=request.gateset,
            seed=request.seed,
            interleaver=request.interleaver,
        )
        with self._rpcq_client() as rpcq_client:  # type: rpcq.Client
            response: rpcq.messages.RandomizedBenchmarkingResponse = rpcq_client.call(
                "generate_rb_sequence",
                rpcq_request,
            )
            return GenerateRandomizedBenchmarkingSequenceResponse(sequence=response.sequence)

    @contextmanager
    def _rpcq_client(self) -> Iterator[rpcq.Client]:
        client = rpcq.Client(
            endpoint=self.base_url,
            timeout=self.timeout,
        )
        try:
            yield client
        finally:
            client.close()  # type: ignore

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
import json
from dataclasses import dataclass
from typing import Optional

from qcs_sdk import QCSClient
from qcs_sdk.compiler.quilc import (
    CompilerOpts,
    ConjugateByCliffordRequest,
    ConjugatePauliByCliffordResponse,
    GenerateRandomizedBenchmarkingSequenceResponse,
    NativeQuilMetadata,
    QuilcClient,
    RandomizedBenchmarkingRequest,
    TargetDevice,
    compile_program,
    conjugate_pauli_by_clifford,
    generate_randomized_benchmarking_sequence,
    get_version_info,
)
from rpcq.messages import TargetDevice as TargetQuantumProcessor


@dataclass
class CompileToNativeQuilRequest:
    """Request to compile to native Quil."""

    program: str
    """Program to compile."""

    target_quantum_processor: TargetQuantumProcessor
    """Quantum processor to target."""

    protoquil: Optional[bool]
    """Whether or not to restrict to protoquil. Overrides server default when provided."""


@dataclass
class NativeQuilMetadataResponse:
    """Metadata for a native Quil program."""

    final_rewiring: list[int]
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
    """Compile to native Quil response."""

    native_program: str
    """Native Quil program."""

    metadata: Optional[NativeQuilMetadata]
    """Metadata for the returned Native Quil."""


class CompilerClient:
    """Client for making requests to a Quil compiler."""

    def __init__(
        self,
        *,
        client_configuration: QCSClient,
        request_timeout: float = 10.0,
        quilc_client: Optional[QuilcClient] = None,
    ) -> None:
        """Instantiate a new compiler client.

        :param client_configuration: Configuration for client.
        :param request_timeout: Timeout for requests, in seconds.
        """
        self._client_configuration = client_configuration
        base_url = client_configuration.quilc_url
        if not base_url.startswith("tcp://"):
            raise ValueError(f"Expected compiler URL '{base_url}' to start with 'tcp://'")

        if quilc_client is None:
            self.quilc_client = QuilcClient.new_rpcq(base_url)
        elif isinstance(quilc_client, QuilcClient):
            self.quilc_client = quilc_client
        else:
            raise TypeError(f"Unsupported type for Quilc client: {quilc_client}")

        self.base_url = base_url
        self.timeout = request_timeout

    def get_version(self) -> str:
        """Get version info for compiler server."""
        return get_version_info(client=self.quilc_client)

    def compile_to_native_quil(self, request: CompileToNativeQuilRequest) -> CompileToNativeQuilResponse:
        """Compile Quil program to native Quil."""
        target_device_json = json.dumps(request.target_quantum_processor.asdict())  # type: ignore
        target_device = TargetDevice.from_json(target_device_json)

        result = compile_program(
            quil=request.program,
            target=target_device,
            client=self.quilc_client,
            options=CompilerOpts(protoquil=request.protoquil, timeout=self.timeout),
        )
        return CompileToNativeQuilResponse(native_program=result.program, metadata=result.native_quil_metadata)

    def conjugate_pauli_by_clifford(self, request: ConjugateByCliffordRequest) -> ConjugatePauliByCliffordResponse:
        """Conjugate a Pauli element by a Clifford element."""
        return conjugate_pauli_by_clifford(request=request, client=self.quilc_client)

    def generate_randomized_benchmarking_sequence(
        self, request: RandomizedBenchmarkingRequest
    ) -> GenerateRandomizedBenchmarkingSequenceResponse:
        """Generate a randomized benchmarking sequence."""
        return generate_randomized_benchmarking_sequence(request=request, client=self.quilc_client)

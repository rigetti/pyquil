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
import time
from functools import partial
from typing import Dict

import rpcq
from _pytest.monkeypatch import MonkeyPatch
from pytest import raises
from qcs_api_client.client import QCSClientConfiguration

from pyquil.api._compiler_client import (
    CompilerClient,
    GenerateRandomizedBenchmarkingSequenceResponse,
    GenerateRandomizedBenchmarkingSequenceRequest,
    ConjugatePauliByCliffordResponse,
    ConjugatePauliByCliffordRequest,
    NativeQuilMetadataResponse,
    CompileToNativeQuilResponse,
    CompileToNativeQuilRequest,
)
from pyquil.external.rpcq import CompilerISA, compiler_isa_to_target_quantum_processor
from test.unit.utils import run_rpcq_server


def test_init__sets_base_url_and_timeout(monkeypatch: MonkeyPatch, port: int):
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL", f"tcp://localhost:{port}")
    client_configuration = QCSClientConfiguration.load()

    compiler_client = CompilerClient(client_configuration=client_configuration, request_timeout=3.14)

    assert compiler_client.base_url == f"tcp://localhost:{port}"
    assert compiler_client.timeout == 3.14


def test_init__validates_compiler_url(monkeypatch: MonkeyPatch, port: int):
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL", "not-http-or-tcp://example.com")
    client_configuration = QCSClientConfiguration.load()

    with raises(
        ValueError,
        match="Expected compiler URL 'not-http-or-tcp://example.com' to start with 'tcp://'",
    ):
        CompilerClient(client_configuration=client_configuration)


def get_version_info_timeout(timeout: int) -> None:
    time.sleep(timeout * 2)


def test_sets_timeout_on_requests(rpcq_server: rpcq.Server, monkeypatch: MonkeyPatch, port: int):
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL", f"tcp://localhost:{port}")
    client_configuration = QCSClientConfiguration.load()
    compiler_client = CompilerClient(client_configuration=client_configuration, request_timeout=0.1)

    get_version_info = partial(get_version_info_timeout, compiler_client.timeout)
    get_version_info.__name__ = "get_version_info"
    rpcq_server.rpc_handler(get_version_info)

    with run_rpcq_server(rpcq_server, port):
        with raises(TimeoutError, match=f"Timeout on client tcp://localhost:{port}, method name get_version_info"):
            compiler_client.get_version()


def get_version_info() -> Dict[str, str]:
    return {"quilc": "1.2.3"}


def test_get_version__returns_version(rpcq_server: rpcq.Server, monkeypatch: MonkeyPatch, port: int):
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL", f"tcp://localhost:{port}")
    client_configuration = QCSClientConfiguration.load()
    compiler_client = CompilerClient(client_configuration=client_configuration)

    rpcq_server.rpc_handler(get_version_info)

    with run_rpcq_server(rpcq_server, port):
        assert compiler_client.get_version() == "1.2.3"


def quil_to_native_quil(
    aspen8_compiler_isa: CompilerISA, request: rpcq.messages.NativeQuilRequest, protoquil: bool,
) -> rpcq.messages.NativeQuilResponse:
    assert request == rpcq.messages.NativeQuilRequest(
        quil="some-program",
        target_device=compiler_isa_to_target_quantum_processor(aspen8_compiler_isa),
    )
    assert protoquil is True
    return rpcq.messages.NativeQuilResponse(
        quil="native-program",
        metadata=rpcq.messages.NativeQuilMetadata(
            final_rewiring=[0, 1, 2],
            gate_depth=10,
            gate_volume=42,
            multiqubit_gate_depth=5,
            program_duration=3.14,
            program_fidelity=0.99,
            topological_swaps=3,
            qpu_runtime_estimation=0.1618,
        ),
    )


def test_compile_to_native_quil__returns_native_quil(
    rpcq_server: rpcq.Server,
    aspen8_compiler_isa: CompilerISA,
    monkeypatch: MonkeyPatch,
    port: int,
):
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL", f"tcp://localhost:{port}")
    client_configuration = QCSClientConfiguration.load()
    compiler_client = CompilerClient(client_configuration=client_configuration)

    _quil_to_native_quil = partial(quil_to_native_quil, aspen8_compiler_isa)
    _quil_to_native_quil.__name__ = "quil_to_native_quil"
    rpcq_server.rpc_handler(_quil_to_native_quil)

    with run_rpcq_server(rpcq_server, port):
        request = CompileToNativeQuilRequest(
            program="some-program",
            target_quantum_processor=compiler_isa_to_target_quantum_processor(aspen8_compiler_isa),
            protoquil=True,
        )
        assert compiler_client.compile_to_native_quil(request) == CompileToNativeQuilResponse(
            native_program="native-program",
            metadata=NativeQuilMetadataResponse(
                final_rewiring=[0, 1, 2],
                gate_depth=10,
                gate_volume=42,
                multiqubit_gate_depth=5,
                program_duration=3.14,
                program_fidelity=0.99,
                topological_swaps=3,
                qpu_runtime_estimation=0.1618,
            ),
        )


def conjugate_pauli_by_clifford(
    request: rpcq.messages.ConjugateByCliffordRequest,
) -> rpcq.messages.ConjugateByCliffordResponse:
    assert request == rpcq.messages.ConjugateByCliffordRequest(
        pauli=rpcq.messages.PauliTerm(indices=[0, 1, 2], symbols=["x", "y", "z"]),
        clifford="cliff",
    )
    return rpcq.messages.ConjugateByCliffordResponse(phase=42, pauli="pauli")


def test_conjugate_pauli_by_clifford__returns_conjugation_result(
    rpcq_server: rpcq.Server, monkeypatch: MonkeyPatch, port: int
):
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL", f"tcp://localhost:{port}")
    client_configuration = QCSClientConfiguration.load()
    compiler_client = CompilerClient(client_configuration=client_configuration)

    rpcq_server.rpc_handler(conjugate_pauli_by_clifford)

    with run_rpcq_server(rpcq_server, port):
        request = ConjugatePauliByCliffordRequest(
            pauli_indices=[0, 1, 2],
            pauli_symbols=["x", "y", "z"],
            clifford="cliff",
        )
        assert compiler_client.conjugate_pauli_by_clifford(request) == ConjugatePauliByCliffordResponse(
            phase_factor=42,
            pauli="pauli",
        )


def generate_rb_sequence(
    request: rpcq.messages.RandomizedBenchmarkingRequest,
) -> rpcq.messages.RandomizedBenchmarkingResponse:
    assert request == rpcq.messages.RandomizedBenchmarkingRequest(
        depth=42,
        qubits=3,
        gateset=["some", "gate", "set"],
        seed=314,
        interleaver="some-interleaver",
    )
    return rpcq.messages.RandomizedBenchmarkingResponse(sequence=[[3, 1, 4], [1, 6, 1]])


def test_generate_randomized_benchmarking_sequence__returns_benchmarking_sequence(
    rpcq_server: rpcq.Server,
    monkeypatch: MonkeyPatch,
    port: int,
):
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL", f"tcp://localhost:{port}")
    client_configuration = QCSClientConfiguration.load()
    compiler_client = CompilerClient(client_configuration=client_configuration)

    rpcq_server.rpc_handler(generate_rb_sequence)

    with run_rpcq_server(rpcq_server, port):
        request = GenerateRandomizedBenchmarkingSequenceRequest(
            depth=42,
            num_qubits=3,
            gateset=["some", "gate", "set"],
            seed=314,
            interleaver="some-interleaver",
        )
        assert compiler_client.generate_randomized_benchmarking_sequence(
            request
        ) == GenerateRandomizedBenchmarkingSequenceResponse(sequence=[[3, 1, 4], [1, 6, 1]])

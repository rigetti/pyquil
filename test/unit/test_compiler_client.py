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

import rpcq
from _pytest.monkeypatch import MonkeyPatch
from pytest import raises
from pytest_mock import MockerFixture
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
from test.unit.utils import patch_rpcq_client


def test_init__sets_base_url_and_timeout(monkeypatch: MonkeyPatch):
    host = "tcp://localhost:1234"
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL", host)
    client_configuration = QCSClientConfiguration.load()

    compiler_client = CompilerClient(client_configuration=client_configuration, request_timeout=3.14)

    assert compiler_client.base_url == host
    assert compiler_client.timeout == 3.14


def test_init__validates_compiler_url(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL", "not-http-or-tcp://example.com")
    client_configuration = QCSClientConfiguration.load()

    with raises(
        ValueError,
        match="Expected compiler URL 'not-http-or-tcp://example.com' to start with 'tcp://'",
    ):
        CompilerClient(client_configuration=client_configuration)


def test_sets_timeout_on_requests(mocker: MockerFixture):
    client_configuration = QCSClientConfiguration.load()
    compiler_client = CompilerClient(client_configuration=client_configuration, request_timeout=0.1)

    patch_rpcq_client(mocker=mocker, return_value={})

    with compiler_client._rpcq_client() as client:
        assert client.timeout == compiler_client.timeout


def test_get_version__returns_version(mocker: MockerFixture):
    client_configuration = QCSClientConfiguration.load()
    compiler_client = CompilerClient(client_configuration=client_configuration)

    rpcq_client = patch_rpcq_client(mocker=mocker, return_value={"quilc": "1.2.3"})

    assert compiler_client.get_version() == "1.2.3"
    rpcq_client.call.assert_called_once_with(
        "get_version_info"
    )


def test_compile_to_native_quil__returns_native_quil(
    aspen8_compiler_isa: CompilerISA,
    mocker: MockerFixture,
):
    client_configuration = QCSClientConfiguration.load()
    compiler_client = CompilerClient(client_configuration=client_configuration)

    rpcq_client = patch_rpcq_client(
        mocker=mocker,
        return_value=rpcq.messages.NativeQuilResponse(
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
    )
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
    rpcq_client.call.assert_called_once_with(
        "quil_to_native_quil",
        rpcq.messages.NativeQuilRequest(
            quil="some-program",
            target_device=compiler_isa_to_target_quantum_processor(aspen8_compiler_isa),
        ),
        protoquil=True,
    )


def test_conjugate_pauli_by_clifford__returns_conjugation_result(
    mocker: MockerFixture
):
    client_configuration = QCSClientConfiguration.load()
    compiler_client = CompilerClient(client_configuration=client_configuration)
    rpcq_client = patch_rpcq_client(mocker=mocker, return_value=rpcq.messages.ConjugateByCliffordResponse(phase=42, pauli="pauli"))

    request = ConjugatePauliByCliffordRequest(
        pauli_indices=[0, 1, 2],
        pauli_symbols=["x", "y", "z"],
        clifford="cliff",
    )
    assert compiler_client.conjugate_pauli_by_clifford(request) == ConjugatePauliByCliffordResponse(
        phase_factor=42,
        pauli="pauli",
    )
    rpcq_client.call.assert_called_once_with(
        "conjugate_pauli_by_clifford",
        rpcq.messages.ConjugateByCliffordRequest(
            pauli=rpcq.messages.PauliTerm(indices=[0, 1, 2], symbols=["x", "y", "z"]),
            clifford="cliff",
        )
    )


def test_generate_randomized_benchmarking_sequence__returns_benchmarking_sequence(
    mocker: MockerFixture,
):
    client_configuration = QCSClientConfiguration.load()
    compiler_client = CompilerClient(client_configuration=client_configuration)

    rpcq_client = patch_rpcq_client(mocker=mocker, return_value=rpcq.messages.RandomizedBenchmarkingResponse(sequence=[[3, 1, 4], [1, 6, 1]]))

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
    rpcq_client.call.assert_called_once_with(
        "generate_rb_sequence",
        rpcq.messages.RandomizedBenchmarkingRequest(
            depth=42,
            qubits=3,
            gateset=["some", "gate", "set"],
            seed=314,
            interleaver="some-interleaver",
        )
    )

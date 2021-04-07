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

from pytest_httpx import HTTPXMock
from qcs_api_client.client import QCSClientConfiguration

from pyquil.api._qvm_client import (
    QVMClient,
    GetWavefunctionResponse,
    GetWavefunctionRequest,
    MeasureExpectationResponse,
    MeasureExpectationRequest,
    RunAndMeasureProgramResponse,
    RunAndMeasureProgramRequest,
    RunProgramRequest,
    RunProgramResponse,
)


def test_init__sets_base_url_and_timeout(client_configuration: QCSClientConfiguration):
    qvm_client = QVMClient(client_configuration=client_configuration, request_timeout=3.14)

    assert qvm_client.base_url == client_configuration.profile.applications.pyquil.qvm_url
    assert qvm_client.timeout == 3.14


def test_get_version__returns_version(client_configuration: QCSClientConfiguration, httpx_mock: HTTPXMock):
    qvm_client = QVMClient(client_configuration=client_configuration)

    httpx_mock.add_response(
        url=client_configuration.profile.applications.pyquil.qvm_url,
        match_content=json.dumps({"type": "version"}).encode(),
        data="1.2.3 [abc123]",
    )

    assert qvm_client.get_version() == "1.2.3"


def test_run_program__returns_results(client_configuration: QCSClientConfiguration, httpx_mock: HTTPXMock):
    qvm_client = QVMClient(client_configuration=client_configuration)

    httpx_mock.add_response(
        url=client_configuration.profile.applications.pyquil.qvm_url,
        match_content=json.dumps(
            {
                "type": "multishot",
                "compiled-quil": "some-program",
                "addresses": {"ro": True},
                "trials": 1,
                "measurement-noise": (3.14, 1.61, 6.28),
                "gate-noise": (1.0, 2.0, 3.0),
                "rng-seed": 314,
            },
        ).encode(),
        json={"ro": [[1, 0, 1]]},
    )

    request = RunProgramRequest(
        program="some-program",
        addresses={"ro": True},
        trials=1,
        measurement_noise=(3.14, 1.61, 6.28),
        gate_noise=(1.0, 2.0, 3.0),
        seed=314,
    )
    assert qvm_client.run_program(request) == RunProgramResponse(results={"ro": [[1, 0, 1]]})


def test_run_and_measure_program__returns_results(client_configuration: QCSClientConfiguration, httpx_mock: HTTPXMock):
    qvm_client = QVMClient(client_configuration=client_configuration)

    httpx_mock.add_response(
        url=client_configuration.profile.applications.pyquil.qvm_url,
        match_content=json.dumps(
            {
                "type": "multishot-measure",
                "compiled-quil": "some-program",
                "qubits": [0, 1, 2],
                "trials": 1,
                "measurement-noise": (3.14, 1.61, 6.28),
                "gate-noise": (1.0, 2.0, 3.0),
                "rng-seed": 314,
            },
        ).encode(),
        json=[[1, 0, 1]],
    )

    request = RunAndMeasureProgramRequest(
        program="some-program",
        qubits=[0, 1, 2],
        trials=1,
        measurement_noise=(3.14, 1.61, 6.28),
        gate_noise=(1.0, 2.0, 3.0),
        seed=314,
    )
    assert qvm_client.run_and_measure_program(request) == RunAndMeasureProgramResponse(results=[[1, 0, 1]])


def test_measure_expectation__returns_expectation(client_configuration: QCSClientConfiguration, httpx_mock: HTTPXMock):
    qvm_client = QVMClient(client_configuration=client_configuration)

    httpx_mock.add_response(
        url=client_configuration.profile.applications.pyquil.qvm_url,
        match_content=json.dumps(
            {
                "type": "expectation",
                "state-preparation": "some-program",
                "operators": ["some-op-program"],
                "rng-seed": 314,
            },
        ).encode(),
        json=[0.161],
    )

    request = MeasureExpectationRequest(
        prep_program="some-program",
        pauli_operators=["some-op-program"],
        seed=314,
    )
    assert qvm_client.measure_expectation(request) == MeasureExpectationResponse(expectations=[0.161])


def test_get_wavefunction__returns_wavefunction(client_configuration: QCSClientConfiguration, httpx_mock: HTTPXMock):
    qvm_client = QVMClient(client_configuration=client_configuration)

    httpx_mock.add_response(
        url=client_configuration.profile.applications.pyquil.qvm_url,
        match_content=json.dumps(
            {
                "type": "wavefunction",
                "compiled-quil": "some-program",
                "measurement-noise": (3.14, 1.61, 6.28),
                "gate-noise": (1.0, 2.0, 3.0),
                "rng-seed": 314,
            },
        ).encode(),
        data=b"some-wavefunction",
    )

    request = GetWavefunctionRequest(
        program="some-program",
        measurement_noise=(3.14, 1.61, 6.28),
        gate_noise=(1.0, 2.0, 3.0),
        seed=314,
    )
    assert qvm_client.get_wavefunction(request) == GetWavefunctionResponse(wavefunction=b"some-wavefunction")

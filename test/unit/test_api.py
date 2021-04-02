#!/usr/bin/python
##############################################################################
# Copyright 2016-2017 Rigetti Computing
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
from math import pi
from unittest import mock

import numpy as np
import pytest
from qcs_api_client.client import QCSClientConfiguration

from pyquil.api import QVMConnection
from pyquil.api._qvm_client import (
    MeasureExpectationResponse,
    MeasureExpectationRequest,
    RunAndMeasureProgramResponse,
    RunAndMeasureProgramRequest,
    RunProgramRequest,
    RunProgramResponse,
)
from pyquil.external.rpcq import _compiler_isa_from_dict
from pyquil.gates import CNOT, H, MEASURE, PHASE, Z, RZ, RX, CZ
from pyquil.paulis import PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Halt, Declare
from pyquil.simulation.tools import program_unitary

EMPTY_PROGRAM = Program()
BELL_STATE = Program(H(0), CNOT(0, 1))
BELL_STATE_MEASURE = Program(
    Declare("ro", "BIT", 2),
    H(0),
    CNOT(0, 1),
    MEASURE(0, MemoryReference("ro", 0)),
    MEASURE(1, MemoryReference("ro", 1)),
)
COMPILED_BELL_STATE = Program(
    [
        RZ(pi / 2, 0),
        RX(pi / 2, 0),
        RZ(-pi / 2, 1),
        RX(pi / 2, 1),
        CZ(1, 0),
        RZ(-pi / 2, 0),
        RX(-pi / 2, 1),
        RZ(pi / 2, 1),
        Halt(),
    ]
)
DUMMY_ISA_DICT = {"1Q": {"0": {}, "1": {}}, "2Q": {"0-1": {}}}
DUMMY_ISA = _compiler_isa_from_dict(DUMMY_ISA_DICT)

COMPILED_BYTES_ARRAY = b"SUPER SECRET PACKAGE"
RB_ENCODED_REPLY = [[0, 0], [1, 1]]
RB_REPLY = [Program("H 0\nH 0\n"), Program("PHASE(pi/2) 0\nPHASE(pi/2) 0\n")]


def test_sync_run_mock(qvm_connection_with_mock_client: QVMConnection):
    mock_qvm_client = qvm_connection_with_mock_client._qvm_client

    def mock_run_program(request: RunProgramRequest) -> RunProgramResponse:
        assert request == RunProgramRequest(
            program=BELL_STATE_MEASURE.out(),
            addresses={"ro": [0, 1]},
            trials=2,
            measurement_noise=(0.2, 0.3, 0.5),
            gate_noise=(0.2, 0.3, 0.5),
            seed=52,
        )
        return RunProgramResponse(
            results={
                "ro": [[0, 0], [1, 1]],
            },
        )

    mock_qvm_client.run_program.side_effect = mock_run_program

    assert qvm_connection_with_mock_client.run(BELL_STATE_MEASURE, [0, 1], trials=2) == [[0, 0], [1, 1]]

    # Test no classical addresses
    assert qvm_connection_with_mock_client.run(BELL_STATE_MEASURE, trials=2) == [[0, 0], [1, 1]]

    with pytest.raises(ValueError):
        qvm_connection_with_mock_client.run(EMPTY_PROGRAM)


def test_sync_run(qvm_connection: QVMConnection):
    assert qvm_connection.run(BELL_STATE_MEASURE, [0, 1], trials=2) == [[0, 0], [1, 1]]

    # Test range as well
    assert qvm_connection.run(BELL_STATE_MEASURE, range(2), trials=2) == [[0, 0], [1, 1]]

    # Test numpy ints
    assert qvm_connection.run(BELL_STATE_MEASURE, np.arange(2), trials=2) == [[0, 0], [1, 1]]

    # Test no classical addresses
    assert qvm_connection.run(BELL_STATE_MEASURE, trials=2) == [[0, 0], [1, 1]]

    with pytest.raises(ValueError):
        qvm_connection.run(EMPTY_PROGRAM)


def test_sync_run_and_measure_mock(qvm_connection_with_mock_client: QVMConnection):
    mock_qvm_client = qvm_connection_with_mock_client._qvm_client

    def mock_run_and_measure_program(request: RunAndMeasureProgramRequest) -> RunAndMeasureProgramResponse:
        assert request == RunAndMeasureProgramRequest(
            program=BELL_STATE.out(),
            qubits=[0, 1],
            trials=2,
            measurement_noise=(0.2, 0.3, 0.5),
            gate_noise=(0.2, 0.3, 0.5),
            seed=52,
        )
        return RunAndMeasureProgramResponse(
            results=[[0, 0], [1, 1]],
        )

    mock_qvm_client.run_and_measure_program.side_effect = mock_run_and_measure_program

    assert qvm_connection_with_mock_client.run_and_measure(BELL_STATE, [0, 1], trials=2) == [[0, 0], [1, 1]]

    with pytest.raises(ValueError):
        qvm_connection_with_mock_client.run_and_measure(EMPTY_PROGRAM, [0])


def test_sync_run_and_measure(qvm_connection: QVMConnection):
    assert qvm_connection.run_and_measure(BELL_STATE, [0, 1], trials=2) == [[1, 1], [0, 0]]
    assert qvm_connection.run_and_measure(BELL_STATE, [0, 1]) == [[1, 1]]

    with pytest.raises(ValueError):
        qvm_connection.run_and_measure(EMPTY_PROGRAM, [0])


WAVEFUNCTION_PROGRAM = Program(Declare("ro", "BIT"), H(0), CNOT(0, 1), MEASURE(0, MemoryReference("ro")), H(0))


def test_sync_expectation_mock(qvm_connection_with_mock_client: QVMConnection):
    mock_qvm_client = qvm_connection_with_mock_client._qvm_client

    def mock_measure_expectation(request: MeasureExpectationRequest) -> MeasureExpectationResponse:
        assert request == MeasureExpectationRequest(
            prep_program=BELL_STATE.out(),
            pauli_operators=["Z 0\n", "Z 1\n", "Z 0\nZ 1\n"],
            seed=52,
        )
        return MeasureExpectationResponse(
            expectations=[0.0, 0.0, 1.0],
        )

    mock_qvm_client.measure_expectation.side_effect = mock_measure_expectation

    result = qvm_connection_with_mock_client.expectation(
        BELL_STATE, [Program(Z(0)), Program(Z(1)), Program(Z(0), Z(1))]
    )
    exp_expected = [0.0, 0.0, 1.0]
    np.testing.assert_allclose(exp_expected, result)

    z0 = PauliTerm("Z", 0)
    z1 = PauliTerm("Z", 1)
    z01 = z0 * z1
    result = qvm_connection_with_mock_client.pauli_expectation(BELL_STATE, [z0, z1, z01])
    exp_expected = [0.0, 0.0, 1.0]
    np.testing.assert_allclose(exp_expected, result)


def test_sync_expectation(qvm_connection: QVMConnection):
    result = qvm_connection.expectation(BELL_STATE, [Program(Z(0)), Program(Z(1)), Program(Z(0), Z(1))])
    exp_expected = [0.0, 0.0, 1.0]
    np.testing.assert_allclose(exp_expected, result)


def test_sync_pauli_expectation_mock(qvm_connection_with_mock_client: QVMConnection):
    mock_qvm_client = qvm_connection_with_mock_client._qvm_client

    def mock_measure_expectation(request: MeasureExpectationRequest) -> MeasureExpectationResponse:
        assert request == MeasureExpectationRequest(
            prep_program=BELL_STATE.out(),
            pauli_operators=["Z 0\nZ 1\n", "Z 0\n", "Z 1\n"],
            seed=52,
        )
        return MeasureExpectationResponse(
            expectations=[1.0, 0.0, 0.0],
        )

    mock_qvm_client.measure_expectation.side_effect = mock_measure_expectation

    z0 = PauliTerm("Z", 0)
    z1 = PauliTerm("Z", 1)
    z01 = z0 * z1
    result = qvm_connection_with_mock_client.pauli_expectation(BELL_STATE, 1j * z01 + z0 + z1)
    exp_expected = 1j
    np.testing.assert_allclose(exp_expected, result)


def test_sync_pauli_expectation(qvm_connection: QVMConnection):
    z0 = PauliTerm("Z", 0)
    z1 = PauliTerm("Z", 1)
    z01 = z0 * z1
    result = qvm_connection.pauli_expectation(BELL_STATE, [z0, z1, z01])
    exp_expected = [0.0, 0.0, 1.0]
    np.testing.assert_allclose(exp_expected, result)


def test_sync_wavefunction(qvm_connection: QVMConnection):
    qvm_connection.random_seed = 0  # this test uses a stochastic program and assumes we measure 0
    result = qvm_connection.wavefunction(WAVEFUNCTION_PROGRAM)
    wf_expected = np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.70710678 + 0.0j, -0.70710678 + 0.0j])
    np.testing.assert_allclose(result.amplitudes, wf_expected)


def test_quil_to_native_quil(compiler):
    response = compiler.quil_to_native_quil(BELL_STATE)
    p_unitary = program_unitary(response, n_qubits=2)
    compiled_p_unitary = program_unitary(COMPILED_BELL_STATE, n_qubits=2)
    from pyquil.simulation.tools import scale_out_phase

    assert np.allclose(p_unitary, scale_out_phase(compiled_p_unitary, p_unitary))


def test_local_rb_sequence(benchmarker):
    response = benchmarker.generate_rb_sequence(2, [PHASE(np.pi / 2, 0), H(0)], seed=52)
    assert [prog.out() for prog in response] == [
        "H 0\nPHASE(pi/2) 0\nH 0\nPHASE(pi/2) 0\nPHASE(pi/2) 0\n",
        "H 0\nPHASE(pi/2) 0\nH 0\nPHASE(pi/2) 0\nPHASE(pi/2) 0\n",
    ]


def test_local_conjugate_request(benchmarker):
    response = benchmarker.apply_clifford_to_pauli(Program("H 0"), PauliTerm("X", 0, 1.0))
    assert isinstance(response, PauliTerm)
    assert str(response) == "(1+0j)*Z0"


def test_apply_clifford_to_pauli(benchmarker):
    response = benchmarker.apply_clifford_to_pauli(Program("H 0"), PauliTerm("I", 0, 0.34))
    assert response == PauliTerm("I", 0, 0.34)


@pytest.fixture()
@mock.patch("pyquil.api._qvm.QVMClient", autospec=True)
def qvm_connection_with_mock_client(
    mock_qvm_client_class: mock.MagicMock, client_configuration: QCSClientConfiguration
):
    mock_qvm_client = mock_qvm_client_class.return_value
    mock_qvm_client.get_version.return_value = "1.8.0"

    return QVMConnection(
        client_configuration=client_configuration,
        measurement_noise=(0.2, 0.3, 0.5),
        gate_noise=(0.2, 0.3, 0.5),
        random_seed=52,
    )

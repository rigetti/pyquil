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
import json
from math import pi

import numpy as np
import pytest
from pytest_httpx import HTTPXMock

from pyquil.api import QVMConnection
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


def test_sync_run_mock(qvm: QVMConnection, httpx_mock: HTTPXMock):
    mock_qvm = qvm
    mock_endpoint = mock_qvm.client.qvm_url
    httpx_mock.add_response(
        method="POST",
        url=mock_endpoint,
        match_content=json.dumps(
            {
                "type": "multishot",
                "addresses": {"ro": [0, 1]},
                "trials": 2,
                "compiled-quil": "DECLARE ro BIT[2]\nH 0\nCNOT 0 1\nMEASURE 0 ro[0]" + "\nMEASURE 1 ro[1]\n",
                "rng-seed": 52,
            }
        ).encode(),
        json={"ro": [[0, 0], [1, 1]]},
    )

    assert mock_qvm.run(BELL_STATE_MEASURE, [0, 1], trials=2) == [[0, 0], [1, 1]]

    # Test no classical addresses
    assert mock_qvm.run(BELL_STATE_MEASURE, trials=2) == [[0, 0], [1, 1]]

    with pytest.raises(ValueError):
        mock_qvm.run(EMPTY_PROGRAM)


def test_sync_run(qvm: QVMConnection):
    assert qvm.run(BELL_STATE_MEASURE, [0, 1], trials=2) == [[0, 0], [1, 1]]

    # Test range as well
    assert qvm.run(BELL_STATE_MEASURE, range(2), trials=2) == [[0, 0], [1, 1]]

    # Test numpy ints
    assert qvm.run(BELL_STATE_MEASURE, np.arange(2), trials=2) == [[0, 0], [1, 1]]

    # Test no classical addresses
    assert qvm.run(BELL_STATE_MEASURE, trials=2) == [[0, 0], [1, 1]]

    with pytest.raises(ValueError):
        qvm.run(EMPTY_PROGRAM)


def test_sync_run_and_measure_mock(qvm: QVMConnection, httpx_mock: HTTPXMock):
    mock_qvm = qvm
    mock_endpoint = mock_qvm.client.qvm_url
    httpx_mock.add_response(
        method="POST",
        url=mock_endpoint,
        match_content=json.dumps(
            {
                "type": "multishot-measure",
                "qubits": [0, 1],
                "trials": 2,
                "compiled-quil": "H 0\nCNOT 0 1\n",
                "rng-seed": 52,
            }
        ).encode(),
        json=[[0, 0], [1, 1]],
    )

    assert mock_qvm.run_and_measure(BELL_STATE, [0, 1], trials=2) == [[0, 0], [1, 1]]

    with pytest.raises(ValueError):
        mock_qvm.run_and_measure(EMPTY_PROGRAM, [0])


def test_sync_run_and_measure(qvm):
    assert qvm.run_and_measure(BELL_STATE, [0, 1], trials=2) == [[1, 1], [0, 0]]
    assert qvm.run_and_measure(BELL_STATE, [0, 1]) == [[1, 1]]

    with pytest.raises(ValueError):
        qvm.run_and_measure(EMPTY_PROGRAM, [0])


WAVEFUNCTION_PROGRAM = Program(Declare("ro", "BIT"), H(0), CNOT(0, 1), MEASURE(0, MemoryReference("ro")), H(0))


def test_sync_expectation_mock(qvm: QVMConnection, httpx_mock: HTTPXMock):
    mock_qvm = qvm
    mock_endpoint = mock_qvm.client.qvm_url
    httpx_mock.add_response(
        method="POST",
        url=mock_endpoint,
        match_content=json.dumps(
            {
                "type": "expectation",
                "state-preparation": BELL_STATE.out(),
                "operators": ["Z 0\n", "Z 1\n", "Z 0\nZ 1\n"],
                "rng-seed": 52,
            }
        ).encode(),
        json=[0.0, 0.0, 1.0],
    )

    result = mock_qvm.expectation(BELL_STATE, [Program(Z(0)), Program(Z(1)), Program(Z(0), Z(1))])
    exp_expected = [0.0, 0.0, 1.0]
    np.testing.assert_allclose(exp_expected, result)

    z0 = PauliTerm("Z", 0)
    z1 = PauliTerm("Z", 1)
    z01 = z0 * z1
    result = mock_qvm.pauli_expectation(BELL_STATE, [z0, z1, z01])
    exp_expected = [0.0, 0.0, 1.0]
    np.testing.assert_allclose(exp_expected, result)


def test_sync_expectation(qvm):
    result = qvm.expectation(BELL_STATE, [Program(Z(0)), Program(Z(1)), Program(Z(0), Z(1))])
    exp_expected = [0.0, 0.0, 1.0]
    np.testing.assert_allclose(exp_expected, result)


def test_sync_expectation_2(qvm):
    z0 = PauliTerm("Z", 0)
    z1 = PauliTerm("Z", 1)
    z01 = z0 * z1
    result = qvm.pauli_expectation(BELL_STATE, [z0, z1, z01])
    exp_expected = [0.0, 0.0, 1.0]
    np.testing.assert_allclose(exp_expected, result)


def test_sync_paulisum_expectation(qvm: QVMConnection, httpx_mock: HTTPXMock):
    mock_qvm = qvm
    mock_endpoint = mock_qvm.client.qvm_url
    httpx_mock.add_response(
        method="POST",
        url=mock_endpoint,
        match_content=json.dumps(
            {
                "type": "expectation",
                "state-preparation": BELL_STATE.out(),
                "operators": ["Z 0\nZ 1\n", "Z 0\n", "Z 1\n"],
                "rng-seed": 52,
            }
        ).encode(),
        json=[1.0, 0.0, 0.0],
    )

    z0 = PauliTerm("Z", 0)
    z1 = PauliTerm("Z", 1)
    z01 = z0 * z1
    result = mock_qvm.pauli_expectation(BELL_STATE, 1j * z01 + z0 + z1)
    exp_expected = 1j
    np.testing.assert_allclose(exp_expected, result)


def test_sync_wavefunction(qvm):
    qvm.random_seed = 0  # this test uses a stochastic program and assumes we measure 0
    result = qvm.wavefunction(WAVEFUNCTION_PROGRAM)
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

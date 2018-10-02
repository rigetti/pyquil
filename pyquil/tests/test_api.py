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
import asyncio
import json
import os
import signal
import time
from math import pi
from multiprocessing import Process
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest
import requests_mock
from pidgin.core_messages import (BinaryExecutableRequest, BinaryExecutableResponse,
                                  NativeQuilRequest, NativeQuilResponse, NativeQuilMetadata,
                                  ConjugateByCliffordRequest, ConjugateByCliffordResponse,
                                  RandomizedBenchmarkingRequest, RandomizedBenchmarkingResponse)
from pidgin.json_rpc.server import Server

from pyquil.api import (QVMConnection, QPUCompiler, BenchmarkConnection,
                        get_qc, LocalQVMCompiler, QVMCompiler, LocalBenchmarkConnection)
from pyquil.api._base_connection import validate_noise_probabilities, validate_qubit_list, \
    prepare_register_list
from pyquil.api._config import PyquilConfig
from pyquil.device import ISA, NxDevice
from pyquil.gates import CNOT, H, MEASURE, PHASE, Z, RZ, RX, CZ
from pyquil.paulis import PauliTerm
from pyquil.quil import Program
from pyquil.quilbase import Pragma, Declare

EMPTY_PROGRAM = Program()
BELL_STATE = Program(H(0), CNOT(0, 1))
BELL_STATE_MEASURE = Program(H(0), CNOT(0, 1), MEASURE(0, 0), MEASURE(1, 1))
COMPILED_BELL_STATE = Program([
    Declare("ro", "BIT", 2),
    Pragma("EXPECTED_REWIRING", ('"#(0 1 2 3)"',)),
    RZ(pi / 2, 0),
    RX(pi / 2, 0),
    RZ(-pi / 2, 1),
    RX(pi / 2, 1),
    CZ(1, 0),
    RZ(-pi / 2, 0),
    RX(-pi / 2, 1),
    RZ(pi / 2, 1),
    Pragma("CURRENT_REWIRING", ('"#(0 1 2 3)"',)),
    Pragma("EXPECTED_REWIRING", ('"#(0 1 2 3)"',)),
    Pragma("CURRENT_REWIRING", ('"#(0 1 2 3)"',)),
])
DUMMY_ISA_DICT = {"1Q": {"0": {}, "1": {}}, "2Q": {"0-1": {}}}
DUMMY_ISA = ISA.from_dict(DUMMY_ISA_DICT)

COMPILED_BYTES_ARRAY = b'SUPER SECRET PACKAGE'
RB_ENCODED_REPLY = [[0, 0], [1, 1]]
RB_REPLY = [Program("H 0\nH 0\n"), Program("PHASE(pi/2) 0\nPHASE(pi/2) 0\n")]

mock_qvm = QVMConnection()


def test_sync_run_mock():
    def mock_response(request, context):
        assert json.loads(request.text) == {
            "type": "multishot",
            "addresses": {'ro': [0, 1]},
            "trials": 2,
            "compiled-quil": "DECLARE ro BIT[2]\nH 0\nCNOT 0 1\nMEASURE 0 ro[0]\nMEASURE 1 ro[1]\n"
        }
        return '{"ro": [[0,0],[1,1]]}'

    with requests_mock.Mocker() as m:
        m.post('http://localhost:5000/qvm', text=mock_response)
        assert mock_qvm.run(BELL_STATE_MEASURE,
                            [0, 1],
                            trials=2) == [[0, 0], [1, 1]]

        # Test no classical addresses
        m.post('http://localhost:5000/qvm', text=mock_response)
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


def test_sync_run_and_measure_mock():
    def mock_response(request, context):
        assert json.loads(request.text) == {
            "type": "multishot-measure",
            "qubits": [0, 1],
            "trials": 2,
            "compiled-quil": "H 0\nCNOT 0 1\n"
        }
        return '[[0,0],[1,1]]'

    with requests_mock.Mocker() as m:
        m.post('http://localhost:5000/qvm', text=mock_response)
        assert mock_qvm.run_and_measure(BELL_STATE, [0, 1], trials=2) == [[0, 0], [1, 1]]

    with pytest.raises(ValueError):
        mock_qvm.run_and_measure(EMPTY_PROGRAM, [0])


def test_sync_run_and_measure(qvm):
    assert qvm.run_and_measure(BELL_STATE, [0, 1], trials=2) == [[1, 1], [0, 0]]
    assert qvm.run_and_measure(BELL_STATE, [0, 1]) == [[1, 1]]

    with pytest.raises(ValueError):
        qvm.run_and_measure(EMPTY_PROGRAM, [0])


WAVEFUNCTION_BINARY = (b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00?\xe6\xa0\x9ef'
                       b'\x7f;\xcc\x00\x00\x00\x00\x00\x00\x00\x00\xbf\xe6\xa0\x9ef\x7f;\xcc\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00')
WAVEFUNCTION_PROGRAM = Program(H(0), CNOT(0, 1), MEASURE(0, 0), H(0))


def test_sync_expectation_mock():
    def mock_response(request, context):
        assert json.loads(request.text) == {
            "type": "expectation",
            "state-preparation": BELL_STATE.out(),
            "operators": ["Z 0\n", "Z 1\n", "Z 0\nZ 1\n"]
        }
        return b'[0.0, 0.0, 1.0]'

    with requests_mock.Mocker() as m:
        m.post('http://localhost:5000/qvm', content=mock_response)
        result = mock_qvm.expectation(BELL_STATE, [Program(Z(0)), Program(Z(1)), Program(Z(0), Z(1))])
        exp_expected = [0.0, 0.0, 1.0]
        np.testing.assert_allclose(exp_expected, result)

    with requests_mock.Mocker() as m:
        m.post('http://localhost:5000/qvm', content=mock_response)
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
    result = mock_qvm.pauli_expectation(BELL_STATE, [z0, z1, z01])
    exp_expected = [0.0, 0.0, 1.0]
    np.testing.assert_allclose(exp_expected, result)


def test_sync_paulisum_expectation():
    def mock_response(request, context):
        assert json.loads(request.text) == {
            "type": "expectation",
            "state-preparation": BELL_STATE.out(),
            "operators": ["Z 0\nZ 1\n", "Z 0\n", "Z 1\n"]
        }
        return b'[1.0, 0.0, 0.0]'

    with requests_mock.Mocker() as m:
        m.post('http://localhost:5000/qvm', content=mock_response)
        z0 = PauliTerm("Z", 0)
        z1 = PauliTerm("Z", 1)
        z01 = z0 * z1
        result = mock_qvm.pauli_expectation(BELL_STATE, 1j * z01 + z0 + z1)
        exp_expected = 1j
        np.testing.assert_allclose(exp_expected, result)


def test_sync_wavefunction(qvm):
    qvm.random_seed = 0  # this test uses a stochastic program and assumes we measure 0
    result = qvm.wavefunction(WAVEFUNCTION_PROGRAM)
    wf_expected = np.array([0. + 0.j, 0. + 0.j, 0.70710678 + 0.j, -0.70710678 + 0.j])
    np.testing.assert_allclose(result.amplitudes, wf_expected)


def test_seeded_qvm(test_device):
    def mock_response(request, context):
        assert json.loads(request.text) == {
            "type": "multishot-measure",
            "qubits": [0, 1],
            "trials": 2,
            "compiled-quil": "H 0\nCNOT 0 1\n"
        }
        return '[[0,0],[1,1]]'

    with patch.object(LocalQVMCompiler, "quil_to_native_quil") as m_compile,\
            patch('pyquil.api._qvm.apply_noise_model') as m_anm,\
            requests_mock.Mocker() as m:
        m.post('http://localhost:5000/qvm', text=mock_response)
        m_compile.side_effect = [BELL_STATE]
        m_anm.side_effect = [BELL_STATE]

        qvm = QVMConnection(test_device)
        assert qvm.noise_model == test_device.noise_model
        qvm.run_and_measure(BELL_STATE, qubits=[0, 1], trials=2)
        assert m_compile.call_count == 1
        assert m_anm.call_count == 1

        test_device.noise_model = None
        qvm = QVMConnection(test_device)
        assert qvm.noise_model is None
        qvm.run_and_measure(BELL_STATE, qubits=[0, 1], trials=2)
        assert m_compile.call_count == 1
        assert m_anm.call_count == 1

        qvm = QVMConnection()
        assert qvm.noise_model is None
        qvm.run_and_measure(BELL_STATE, qubits=[0, 1], trials=2)
        assert m_compile.call_count == 1
        assert m_anm.call_count == 1


def test_validate_noise_probabilities():
    with pytest.raises(TypeError):
        validate_noise_probabilities(1)
    with pytest.raises(TypeError):
        validate_noise_probabilities(['a', 'b', 'c'])
    with pytest.raises(ValueError):
        validate_noise_probabilities([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        validate_noise_probabilities([0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        validate_noise_probabilities([-0.5, -0.5, -0.5])


def test_validate_qubit_list():
    with pytest.raises(TypeError):
        validate_qubit_list([-1, 1])
    with pytest.raises(TypeError):
        validate_qubit_list(['a', 0], 1)


def test_prepare_register_list():
    with pytest.raises(TypeError):
        prepare_register_list({'ro': [-1, 1]})


def test_config_parsing():
    with patch.dict('os.environ', {"FOREST_CONFIG": os.path.join(os.path.dirname(__file__),
                                                                 "data/forest_config.test"),
                                   "QCS_CONFIG": os.path.join(os.path.dirname(__file__),
                                                              "data/qcs_config.test")}):
        pq_config = PyquilConfig()
        assert pq_config.forest_url == "http://dummy_forest_url"
        assert pq_config.api_key == "pyquil_user_key"
        assert pq_config.user_id == "pyquil_user_token"
        assert pq_config.engage_cmd == "dummy_command"
        assert pq_config.qpu_url == "tcp://dummy_qpu_url"
        assert pq_config.qvm_url == "http://dummy_qvm_url"
        assert pq_config.compiler_url == "tcp://dummy_compiler_server_url"


# ---------------------
# compiler-server tests
# ---------------------


def test_get_qc_returns_local_qvm_compiler():
    with patch.dict('os.environ', {"COMPILER_URL": "http://localhost:7000"}):
        qc = get_qc("9q-generic-qvm")
        assert isinstance(qc.compiler, LocalQVMCompiler)


def test_get_qc_returns_remote_qvm_compiler():
    with patch.dict('os.environ', {"COMPILER_URL": "tcp://192.168.0.0:5555"}):
        qc = get_qc("9q-generic-qvm")
        assert isinstance(qc.compiler, QVMCompiler)


mock_compiler_server = Server()


@mock_compiler_server.rpc_handler
def quil_to_native_quil(payload: NativeQuilRequest) -> NativeQuilResponse:
    assert payload.quil == BELL_STATE.out()
    time.sleep(0.1)
    return NativeQuilResponse(quil=COMPILED_BELL_STATE.out(),
                              metadata=NativeQuilMetadata(final_rewiring=[],
                                                          gate_depth=0,
                                                          gate_volume=0,
                                                          multiqubit_gate_depth=0,
                                                          program_duration=0.0,
                                                          program_fidelity=0.0,
                                                          topological_swaps=0))


@mock_compiler_server.rpc_handler
def native_quil_to_binary(payload: BinaryExecutableRequest) -> BinaryExecutableResponse:
    assert payload.quil == COMPILED_BELL_STATE.out()
    time.sleep(0.1)
    return BinaryExecutableResponse(program=COMPILED_BYTES_ARRAY)


@mock_compiler_server.rpc_handler
def generate_rb_sequence(payload: RandomizedBenchmarkingRequest) -> RandomizedBenchmarkingResponse:
    assert payload.depth == 2
    time.sleep(0.1)
    return RandomizedBenchmarkingResponse(sequence=RB_ENCODED_REPLY)


@mock_compiler_server.rpc_handler
def conjugate_pauli_by_clifford(payload: ConjugateByCliffordRequest) -> ConjugateByCliffordResponse:
    time.sleep(0.1)
    return ConjugateByCliffordResponse(phase=0, pauli="Z")


@pytest.fixture
def m_endpoints():
    return "tcp://localhost:5555", "tcp://*:5555"


def run_mock(_, endpoint):
    # Need a new event loop for a new process
    mock_compiler_server.run(endpoint, loop=asyncio.new_event_loop())


@pytest.fixture
def server(request, m_endpoints):
    proc = Process(target=run_mock, args=m_endpoints)
    proc.start()
    yield proc
    os.kill(proc.pid, signal.SIGINT)


@pytest.fixture
def mock_compiler(request, m_endpoints):
    return QPUCompiler(endpoint=m_endpoints[0],
                             device=NxDevice(nx.Graph([(0, 1)])))


@pytest.fixture
def mock_rb_cxn(request, m_endpoints):
    return BenchmarkConnection(endpoint=m_endpoints[0])


def test_quil_to_native_quil(server, mock_compiler):
    response = mock_compiler.quil_to_native_quil(BELL_STATE)
    assert response.out() == COMPILED_BELL_STATE.out()


def test_native_quil_to_binary(server, mock_compiler):
    p = COMPILED_BELL_STATE.copy()
    p.wrap_in_numshots_loop(10)
    response = mock_compiler.native_quil_to_executable(p)
    assert response.program == COMPILED_BYTES_ARRAY


def test_rb_sequence(server, mock_rb_cxn):
    response = mock_rb_cxn.generate_rb_sequence(2, [PHASE(np.pi/2, 0), H(0)])
    assert [prog.out() for prog in response] == [prog.out() for prog in RB_REPLY]


def test_conjugate_request(server, mock_rb_cxn):
    response = mock_rb_cxn.apply_clifford_to_pauli(Program("H 0"), PauliTerm("X", 0, 1.0))
    assert isinstance(response, PauliTerm)
    assert str(response) == "(1+0j)*Z0"


def test_local_rb_sequence(benchmarker):
    config = PyquilConfig()
    if config.compiler_url is not None:
        cxn = LocalBenchmarkConnection(endpoint=config.compiler_url)
        response = cxn.generate_rb_sequence(2, [PHASE(np.pi / 2, 0), H(0)], seed=52)
        assert [prog.out() for prog in response] == \
               ["H 0\nPHASE(pi/2) 0\nH 0\nPHASE(pi/2) 0\nPHASE(pi/2) 0\n",
                "H 0\nPHASE(pi/2) 0\nH 0\nPHASE(pi/2) 0\nPHASE(pi/2) 0\n"]


def test_local_conjugate_request(benchmarker):
    config = PyquilConfig()
    if config.compiler_url is not None:
        cxn = LocalBenchmarkConnection(endpoint=config.compiler_url)
        response = cxn.apply_clifford_to_pauli(Program("H 0"), PauliTerm("X", 0, 1.0))
        assert isinstance(response, PauliTerm)
        assert str(response) == "(1+0j)*Z0"

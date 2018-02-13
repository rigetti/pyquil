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
import base64

import requests_mock
import json
import numpy as np
import pytest

from pyquil.api import QVMConnection, QPUConnection, CompilerConnection
from pyquil.api._base_connection import validate_noise_probabilities, validate_run_items
from pyquil.api.qpu import append_measures_to_program
from pyquil.quil import Program
from pyquil.gates import CNOT, H, MEASURE
from pyquil.device import ISA

BELL_STATE = Program(H(0), CNOT(0, 1))
BELL_STATE_MEASURE = Program(H(0), CNOT(0, 1), MEASURE(0, 0), MEASURE(1, 1))
DUMMY_ISA_DICT = {"1Q": {"0": {}, "1": {}}, "2Q": {"0-1": {}}}
DUMMY_ISA = ISA.from_dict(DUMMY_ISA_DICT)

qvm = QVMConnection(api_key='api_key', user_id='user_id')
async_qvm = QVMConnection(api_key='api_key', user_id='user_id', use_queue=True)
compiler = CompilerConnection(isa_source=DUMMY_ISA, api_key='api_key', user_id='user_id')
async_compiler = CompilerConnection(isa_source=DUMMY_ISA, api_key='api_key', user_id='user_id', use_queue=True)


def test_sync_run():
    def mock_response(request, context):
        assert json.loads(request.text) == {
            "type": "multishot",
            "addresses": [0, 1],
            "trials": 2,
            "compiled-quil": "H 0\nCNOT 0 1\n"
        }
        return '[[0,0],[1,1]]'

    with requests_mock.Mocker() as m:
        m.post('https://api.rigetti.com/qvm', text=mock_response)
        assert qvm.run(BELL_STATE, [0, 1], trials=2) == [[0, 0], [1, 1]]

        # Test range as well
        m.post('https://api.rigetti.com/qvm', text=mock_response)
        assert qvm.run(BELL_STATE, range(2), trials=2) == [[0, 0], [1, 1]]


def test_sync_run_and_measure():
    def mock_response(request, context):
        assert json.loads(request.text) == {
            "type": "multishot-measure",
            "qubits": [0, 1],
            "trials": 2,
            "compiled-quil": "H 0\nCNOT 0 1\n"
        }
        return '[[0,0],[1,1]]'

    with requests_mock.Mocker() as m:
        m.post('https://api.rigetti.com/qvm', text=mock_response)
        assert qvm.run_and_measure(BELL_STATE, [0, 1], trials=2) == [[0, 0], [1, 1]]


WAVEFUNCTION_BINARY = (b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00?\xe6'
                       b'\xa0\x9ef\x7f;\xcc\x00\x00\x00\x00\x00\x00\x00\x00\xbf\xe6\xa0\x9ef'
                       b'\x7f;\xcc\x00\x00\x00\x00\x00\x00\x00\x00')
WAVEFUNCTION_PROGRAM = Program(H(0), CNOT(0, 1), MEASURE(0, 0), H(0))


def test_sync_wavefunction():
    def mock_response(request, context):
        assert json.loads(request.text) == {
            "type": "wavefunction",
            "compiled-quil": "H 0\nCNOT 0 1\nMEASURE 0 [0]\nH 0\n",
            "addresses": [0, 1]
        }
        return WAVEFUNCTION_BINARY

    with requests_mock.Mocker() as m:
        m.post('https://api.rigetti.com/qvm', content=mock_response)
        result = qvm.wavefunction(WAVEFUNCTION_PROGRAM, [0, 1])
        wf_expected = np.array([0. + 0.j, 0. + 0.j, 0.70710678 + 0.j, -0.70710678 + 0.j])
        assert np.all(np.isclose(result.amplitudes, wf_expected))
        assert result.classical_memory == [1, 0]


JOB_ID = 'abc'


def test_job_run():
    program = {
        "type": "multishot",
        "addresses": [0, 1],
        "trials": 2,
        "compiled-quil": "H 0\nCNOT 0 1\n"
    }

    def mock_queued_response(request, context):
        assert json.loads(request.text) == {
            "machine": "QVM",
            "program": program
        }
        return json.dumps({"jobId": JOB_ID, "status": "QUEUED"})

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response)
        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, [
            {'text': json.dumps({"jobId": JOB_ID, "status": "RUNNING"})},
            {'text': json.dumps({"jobId": JOB_ID, "status": "FINISHED",
                                 "result": [[0, 0], [1, 1]], "program": program})}
        ])

        result = async_qvm.run(BELL_STATE, [0, 1], trials=2)
        assert result == [[0, 0], [1, 1]]


def test_async_wavefunction():
    program = {
        "type": "wavefunction",
        "compiled-quil": "H 0\nCNOT 0 1\nMEASURE 0 [0]\nH 0\n",
        "addresses": [0, 1]
    }

    def mock_queued_response(request, context):
        assert json.loads(request.text) == {
            "machine": "QVM",
            "program": program
        }
        return json.dumps({"jobId": JOB_ID, "status": "QUEUED"})

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response)
        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, text=json.dumps({
            "jobId": JOB_ID,
            "status": "FINISHED",
            "result": base64.b64encode(WAVEFUNCTION_BINARY).decode(),
            "program": program
        }))
        result = async_qvm.wavefunction(WAVEFUNCTION_PROGRAM, [0, 1])

        wf_expected = np.array([0. + 0.j, 0. + 0.j, 0.70710678 + 0.j, -0.70710678 + 0.j])
        assert np.all(np.isclose(result.amplitudes, wf_expected))
        assert result.classical_memory == [1, 0]


def test_qpu_connection(test_device):
    qpu = QPUConnection(device=test_device)

    run_program = {
        "type": "multishot",
        "addresses": [0, 1],
        "trials": 2,
        "uncompiled-quil": "H 0\nCNOT 0 1\nMEASURE 0 [0]\nMEASURE 1 [1]\n"
    }

    run_and_measure_program = {
        "type": "multishot-measure",
        "qubits": [0, 1],
        "trials": 2,
        "uncompiled-quil": "H 0\nCNOT 0 1\nMEASURE 0 [0]\nMEASURE 1 [1]\n"
    }

    reply_program = {
        "type": "multishot-measure",
        "qubits": [0, 1],
        "trials": 2,
        "uncompiled-quil": "H 0\nCNOT 0 1\nMEASURE 0 [0]\nMEASURE 1 [1]\n",
        "compiled-quil": "H 0\nCNOT 0 1\nMEASURE 0 [0]\nMEASURE 1 [1]\n"
    }

    def mock_queued_response_run(request, context):
        assert json.loads(request.text) == {
            "machine": "QPU",
            "program": run_program,
            "device": "test_device"
        }
        return json.dumps({"jobId": JOB_ID, "status": "QUEUED"})

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response_run)
        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, [
            {'text': json.dumps({"jobId": JOB_ID, "status": "RUNNING"})},
            {'text': json.dumps({"jobId": JOB_ID, "status": "FINISHED",
                                 "result": [[0, 0], [1, 1]], "program": reply_program})}
        ])

        result = qpu.run(BELL_STATE_MEASURE, [0, 1], trials=2)
        assert result == [[0, 0], [1, 1]]

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response_run)
        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, [
            {'text': json.dumps({"jobId": JOB_ID, "status": "RUNNING"})},
            {'text': json.dumps({"jobId": JOB_ID, "status": "FINISHED",
                                 "result": [[0, 0], [1, 1]], "program": reply_program,
                                 "metadata": {
                                     "compiled_quil": "H 0\nCNOT 0 1\nMEASURE 0 [0]\nMEASURE 1 [1]\n",
                                     "topological_swaps": 0,
                                     "gate_depth": 2
                                 }})}
        ])

        job = qpu.wait_for_job(qpu.run_async(BELL_STATE_MEASURE, [0, 1], trials=2))
        assert job.result() == [[0, 0], [1, 1]]
        assert job.compiled_quil() == Program(H(0), CNOT(0, 1), MEASURE(0, 0), MEASURE(1, 1))
        assert job.topological_swaps() == 0
        assert job.gate_depth() == 2

    def mock_queued_response_run_and_measure(request, context):
        assert json.loads(request.text) == {
            "machine": "QPU",
            "program": run_and_measure_program,
            "device": "test_device"
        }
        return json.dumps({"jobId": JOB_ID, "status": "QUEUED"})

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response_run_and_measure)
        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, [
            {'text': json.dumps({"jobId": JOB_ID, "status": "RUNNING"})},
            {'text': json.dumps({"jobId": JOB_ID, "status": "FINISHED",
                                 "result": [[0, 0], [1, 1]], "program": reply_program})}
        ])

        result = qpu.run_and_measure(BELL_STATE, [0, 1], trials=2)
        assert result == [[0, 0], [1, 1]]

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response_run_and_measure)
        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, [
            {'text': json.dumps({"jobId": JOB_ID, "status": "RUNNING"})},
            {'text': json.dumps({"jobId": JOB_ID, "status": "FINISHED",
                                 "result": [[0, 0], [1, 1]],
                                 "program": reply_program,
                                 "metadata": {
                                     "topological_swaps": 0,
                                     "gate_depth": 2
                                 }})}
        ])

        job = qpu.wait_for_job(qpu.run_and_measure_async(BELL_STATE, [0, 1], trials=2))
        assert job.result() == [[0, 0], [1, 1]]
        assert job.compiled_quil() == Program(H(0), CNOT(0, 1), MEASURE(0, 0), MEASURE(1, 1))
        assert job.topological_swaps() == 0
        assert job.gate_depth() == 2


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


def test_validate_run_items():
    with pytest.raises(TypeError):
        validate_run_items(-1, 1)
    with pytest.raises(TypeError):
        validate_run_items(['a', 0], 1)


def test_append_measures_to_program():
    gate_program = Program()
    meas_program = Program(MEASURE(0, 0), MEASURE(1, 1))
    assert gate_program + meas_program == append_measures_to_program(gate_program, [0, 1])


def test_sync_compile():
    def mock_response(request, context):
        assert json.loads(request.text) == {
            "type": "multishot",
            "qubits": [],
            "uncompiled-quil": "H 0\nCNOT 0 1\n",
            "target-device": {"isa": DUMMY_ISA_DICT}
        }
        return json.dumps({
            "type": "multishot",
            "qubits": [],
            "uncompiled-quil": "H 0\nCNOT 0 1\n",
            "compiled-quil": "H 0\nCNOT 0 1\n",
            "target-device": {"isa": DUMMY_ISA_DICT}})

    with requests_mock.Mocker() as m:
        m.post('https://api.rigetti.com/quilc', text=mock_response)
        assert compiler.compile(BELL_STATE) == BELL_STATE


def test_job_compile():
    processed_program = {
        "type": "multishot",
        "qubits": [],
        "uncompiled-quil": "H 0\nCNOT 0 1\n",
        "target-device": {"isa": DUMMY_ISA_DICT}
    }
    postprocessed_program = {
        "type": "multishot",
        "qubits": [],
        "uncompiled-quil": "H 0\nCNOT 0 1\n",
        "compiled-quil": "H 0\nCNOT 0 1\n",
        "target-device": {"isa": DUMMY_ISA_DICT}
    }

    def mock_queued_response(request, context):
        assert json.loads(request.text) == {
            "machine": "QUILC",
            "program": processed_program
        }
        return json.dumps({"jobId": JOB_ID, "status": "QUEUED"})

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response)
        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, [
            {'text': json.dumps({"jobId": JOB_ID, "status": "COMPILING"})},
            {'text': json.dumps({"jobId": JOB_ID, "status": "FINISHED",
                                 "program": postprocessed_program, "metadata": {}})}
        ])

        result = async_compiler.compile(BELL_STATE)
        assert result == BELL_STATE

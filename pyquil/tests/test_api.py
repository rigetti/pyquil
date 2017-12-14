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

from pyquil.api import QVMConnection, QPUConnection
from pyquil.api._base_connection import validate_noise_probabilities, validate_run_items
from pyquil.job_results import wait_for_job
from pyquil.quil import Program
from pyquil.gates import CNOT, H, MEASURE

BELL_STATE = Program(H(0), CNOT(0, 1))

qvm = QVMConnection(api_key='api_key', user_id='user_id')
async_qvm = QVMConnection(api_key='api_key', user_id='user_id', use_queue=True)


def test_sync_run():
    def mock_response(request, context):
        assert json.loads(request.text) == {
            "type": "multishot",
            "addresses": [0, 1],
            "trials": 2,
            "quil-instructions": "H 0\nCNOT 0 1\n"
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
            "quil-instructions": "H 0\nCNOT 0 1\n"
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
            "quil-instructions": "H 0\nCNOT 0 1\nMEASURE 0 [0]\nH 0\n",
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
        "quil-instructions": "H 0\nCNOT 0 1\n"
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
        "quil-instructions": "H 0\nCNOT 0 1\nMEASURE 0 [0]\nH 0\n",
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


def test_qpu_connection():
    qpu = QPUConnection(device_name='fake_device')

    program = {
        "type": "multishot-measure",
        "qubits": [0, 1],
        "trials": 2,
        "quil-instructions": "H 0\nCNOT 0 1\n"
    }

    def mock_queued_response(request, context):
        assert json.loads(request.text) == {
            "machine": "QPU",
            "program": program,
            "device": "fake_device"
        }
        return json.dumps({"jobId": JOB_ID, "status": "QUEUED"})

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response)
        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, [
            {'text': json.dumps({"jobId": JOB_ID, "status": "RUNNING"})},
            {'text': json.dumps({"jobId": JOB_ID, "status": "FINISHED",
                                 "result": [[0, 0], [1, 1]], "program": program})}
        ])

        result = qpu.run_and_measure(BELL_STATE, [0, 1], trials=2)
        assert result == [[0, 0], [1, 1]]

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response)
        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, [
            {'text': json.dumps({"jobId": JOB_ID, "status": "RUNNING"})},
            {'text': json.dumps({"jobId": JOB_ID, "status": "FINISHED",
                                 "result": [[0, 0], [1, 1]], "program": program,
                                 "metadata": {
                                     "compiled_quil": "H 0\nCNOT 0 1\n",
                                     "topological_swaps": 0,
                                     "gate_depth": 2
                                 }})}
        ])

        job = qpu.wait_for_job(qpu.run_and_measure_async(BELL_STATE, [0, 1], trials=2))
        assert job.result() == [[0, 0], [1, 1]]
        assert job.compiled_quil() == Program(H(0), CNOT(0, 1))
        assert job.topological_swaps() == 0
        assert job.gate_depth() == 2

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response)
        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, [
            {'text': json.dumps({"jobId": JOB_ID, "status": "RUNNING"})},
            {'text': json.dumps({"jobId": JOB_ID, "status": "FINISHED",
                                 "result": [[0, 0], [1, 1]], "program": program})}
        ])

        job = qpu.wait_for_job(qpu.run_and_measure_async(BELL_STATE, [0, 1], trials=2))
        assert job.result() == [[0, 0], [1, 1]]
        assert job.compiled_quil() is None
        assert job.topological_swaps() is None
        assert job.gate_depth() is None


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

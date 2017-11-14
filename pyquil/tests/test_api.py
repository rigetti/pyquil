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
from mock import Mock
import json
import numpy as np
import pytest

from pyquil.api import (add_noise_to_payload, add_rng_seed_to_payload, SyncConnection,
                        _validate_noise_probabilities, validate_run_items, JobConnection)
from pyquil.job_results import wait_for_job
from pyquil.quil import Program
from pyquil.gates import CNOT, H, MEASURE

BELL_STATE = Program(H(0), CNOT(0, 1))

qvm = SyncConnection(api_key='api_key', user_id='user_id')
job_qvm = JobConnection(api_key='api_key', user_id='user_id')


def test_ping():
    def mock_response(request, context):
        assert json.loads(request.text) == {"type": "ping"}
        return 'pong'

    with requests_mock.Mocker() as m:
        m.post('https://api.rigetti.com/qvm', text=mock_response)
        assert qvm.ping() == 'pong'


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
        assert np.all(np.isclose(result[0].amplitudes, wf_expected))
        assert result[1] == [1, 0]


JOB_ID = 'abc'


def test_async_run():
    def mock_queued_response(request, context):
        assert json.loads(request.text) == {
            "machine": "QVM",
            "userId": "user_id",
            "program": {
                "type": "multishot",
                "addresses": [0, 1],
                "trials": 2,
                "quil-instructions": "H 0\nCNOT 0 1\n"
            }
        }
        return json.dumps({"jobId": JOB_ID, "status": "QUEUED"})

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response)
        result = job_qvm.run(BELL_STATE, [0, 1], trials=2)
        assert result.job_id() == JOB_ID

        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, [
            {'text': json.dumps({"jobId": JOB_ID, "status": "RUNNING"})},
            {'text': json.dumps({"jobId": JOB_ID, "status": "FINISHED", "result": [[0, 0], [1, 1]]})}
        ])
        assert not result.is_done()
        assert result.is_done()
        assert result.decode() == [[0, 0], [1, 1]]


def test_async_wavefunction():
    def mock_queued_response(request, context):
        assert json.loads(request.text) == {
            "machine": "QVM",
            "userId": "user_id",
            "program": {
                "type": "wavefunction",
                "quil-instructions": "H 0\nCNOT 0 1\nMEASURE 0 [0]\nH 0\n",
                "addresses": [0, 1]
            }
        }
        return json.dumps({"jobId": JOB_ID, "status": "QUEUED"})

    with requests_mock.Mocker() as m:
        m.post('https://job.rigetti.com/beta/job', text=mock_queued_response)
        result = job_qvm.wavefunction(WAVEFUNCTION_PROGRAM, [0, 1])

        m.get('https://job.rigetti.com/beta/job/' + JOB_ID, text=json.dumps({
            "jobId": JOB_ID,
            "status": "FINISHED",
            "result": base64.b64encode(WAVEFUNCTION_BINARY).decode()
        }))
        wait_for_job(result)

        wf_expected = np.array([0. + 0.j, 0. + 0.j, 0.70710678 + 0.j, -0.70710678 + 0.j])
        assert np.all(np.isclose(result.decode()[0].amplitudes, wf_expected))
        assert result.decode()[1] == [1, 0]


########################################################################################################################


class MockPostJson(object):
    def __init__(self):
        self.return_value = Mock()

    def __call__(self, payload, route):
        json.dumps(payload)
        return self.return_value


@pytest.fixture
def cxn():
    c = SyncConnection()
    c.post_json = MockPostJson()
    c.post_json.return_value.text = json.dumps('Success')
    c.measurement_noise = 1
    return c


WAVEFUNCTION_BINARY = (b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00?\xe6'
                       b'\xa0\x9ef\x7f;\xcc\x00\x00\x00\x00\x00\x00\x00\x00\xbf\xe6\xa0\x9ef'
                       b'\x7f;\xcc\x00\x00\x00\x00\x00\x00\x00\x00')


@pytest.fixture
def cxn_wf(cxn):
    cxn.post_json.return_value.content = WAVEFUNCTION_BINARY
    return cxn


@pytest.fixture
def prog():
    p = Program()
    p.inst(H(0), CNOT(0, 1), MEASURE(0, 0), MEASURE(1, 1))
    return p


@pytest.fixture
def prog_wf():
    p = Program()
    p.inst(H(0)).inst(CNOT(0, 1)).measure(0, 0)
    p.inst(H(0))
    return p


def test_add_rng_seed_to_payload():
    payload = {}
    add_rng_seed_to_payload(payload, 1)
    assert payload['rng-seed'] == 1


def test_dont_add_rng_seed_to_payload():
    payload = {}
    add_rng_seed_to_payload(payload, None)
    assert 'rng-seed' not in payload


def test_add_noise_to_payload():
    payload = {}
    add_noise_to_payload(payload, 1, None)
    assert payload['gate-noise'] == 1
    assert 'measurement-noise' not in payload


def test_validate_noise_probabilities():
    with pytest.raises(TypeError):
        _validate_noise_probabilities(1)
    with pytest.raises(TypeError):
        _validate_noise_probabilities(['a', 'b', 'c'])
    with pytest.raises(ValueError):
        _validate_noise_probabilities([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        _validate_noise_probabilities([0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        _validate_noise_probabilities([-0.5, -0.5, -0.5])


def test_validate_run_items():
    with pytest.raises(TypeError):
        validate_run_items(-1, 1)
    with pytest.raises(TypeError):
        validate_run_items(['a', 0], 1)


def test_run(cxn, prog):
    with pytest.raises(TypeError):
        cxn.run(prog, [0, 1], 'a')
    assert cxn.run(prog, [0, 1], 1) == 'Success'


def test_run_and_measure(cxn, prog):
    with pytest.raises(TypeError):
        cxn.run_and_measure(prog, [0, 1], 'a')
    assert cxn.run_and_measure(prog, [0, 1], 1) == 'Success'


def test_expectation(cxn, prog):
    assert cxn.expectation(prog) == 'Success'


def test_wavefunction(cxn_wf, prog_wf):
    wf, mem = cxn_wf.wavefunction(prog_wf, [0, 1])
    wf_expected = np.array(
        [0.00000000 + 0.j, 0.00000000 + 0.j, 0.70710678 + 0.j, -0.70710678 + 0.j])
    mem_expected = [1, 0]
    assert np.all(np.isclose(wf.amplitudes, wf_expected))
    assert mem == mem_expected

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


import pytest
import json
from mock import Mock
import numpy as np

import pyquil.forest as qvm
import pyquil.quil as pq
from pyquil.gates import *


class MockPostJson(object):
    def __init__(self):
        self.return_value = Mock()

    def __call__(self, payload):
        json.dumps(payload)
        return self.return_value


@pytest.fixture
def cxn():
    c = qvm.LocalConnection()
    c.post_json = MockPostJson()
    c.post_json.return_value.text = json.dumps("Success")
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
    p = pq.Program()
    p.inst(H(0), CNOT(0, 1), MEASURE(0, 0), MEASURE(1, 1))
    return p


@pytest.fixture
def prog_wf():
    p = pq.Program()
    p.inst(H(0)).inst(CNOT(0, 1)).measure(0, 0)
    p.inst(H(0))
    return p


def test_add_rng_seed_to_payload():
    payload = {}
    qvm.add_rng_seed_to_payload(payload, 1)
    assert payload['rng-seed'] == 1


def test_dont_add_rng_seed_to_payload():
    payload = {}
    qvm.add_rng_seed_to_payload(payload, None)
    assert 'rng-seed' not in payload


def test_add_noise_to_payload():
    payload = {}
    qvm.add_noise_to_payload(payload, 1, None)
    assert payload["gate-noise"] == 1
    assert "measurement-noise" not in payload


def test_validate_noise_probabilities():
    with pytest.raises(TypeError):
        qvm._validate_noise_probabilities(1)
    with pytest.raises(TypeError):
        qvm._validate_noise_probabilities(['a', 'b', 'c'])
    with pytest.raises(ValueError):
        qvm._validate_noise_probabilities([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        qvm._validate_noise_probabilities([0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        qvm._validate_noise_probabilities([-0.5, -0.5, -0.5])


def test_validate_run_items():
    with pytest.raises(TypeError):
        qvm._validate_run_items(-1, 1)
    with pytest.raises(TypeError):
        qvm._validate_run_items(["a", 0], 1)


def test_run(cxn, prog):
    with pytest.raises(TypeError):
        cxn.run(prog, [0, 1], "a")
    assert cxn.run(prog, [0, 1], 1) == "Success"


def test_run_and_measure(cxn, prog):
    with pytest.raises(TypeError):
        cxn.run_and_measure(prog, [0, 1], "a")
    assert cxn.run_and_measure(prog, [0, 1], 1) == "Success"


def test_expectation(cxn, prog):
    assert cxn.expectation(prog) == "Success"


def test_wavefunction(cxn_wf, prog_wf):
    wf, mem = cxn_wf.wavefunction(prog_wf, [0, 1])
    wf_expected = np.array(
        [0.00000000 + 0.j, 0.00000000 + 0.j, 0.70710678 + 0.j, -0.70710678 + 0.j])
    mem_expected = [1, 0]
    assert np.all(np.isclose(wf.amplitudes, wf_expected))
    assert mem == mem_expected

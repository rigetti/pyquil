import numpy as np
import pytest
from pyquil.api import QVMConnection, CompilerConnection, ForestConnection
from pyquil.api.errors import UnknownApiError
from pyquil.device import Device, ISA
from pyquil.gates import I
from pyquil.quil import Program
from requests import RequestException


@pytest.fixture
def isa_dict():
    return {
        "1Q": {
            "0": {
                'type': 'Xhalves'
            },
            "1": {},
            "2": {},
            "3": {
                "dead": True
            }
        },
        "2Q": {
            "0-1": {},
            "1-2": {
                "type": "ISWAP"
            },
            "0-2": {
                "type": "CPHASE"
            },
            "0-3": {
                "dead": True
            }
        }
    }


@pytest.fixture
def noise_model_dict():
    return {'gates': [{'gate': 'I',
                       'params': (5.0,),
                       'targets': (0, 1),
                       'kraus_ops': [[[[1.]], [[1.0]]]],
                       'fidelity': 1.0},
                      {'gate': 'RX',
                       'params': (np.pi / 2.,),
                       'targets': (0,),
                       'kraus_ops': [[[[1.]], [[1.0]]]],
                       'fidelity': 1.0}],
            'assignment_probs': {'1': [[1.0, 0.0], [0.0, 1.0]],
                                 '0': [[1.0, 0.0], [0.0, 1.0]]},
            }


@pytest.fixture
def device_raw(isa_dict, noise_model_dict):
    return {'isa': isa_dict,
            'noise_model': noise_model_dict,
            'is_online': True,
            'is_retuning': False}


@pytest.fixture
def test_device(device_raw):
    return Device('test_device', device_raw)


@pytest.fixture(scope='session')
def qvm():
    try:
        qvm = QVMConnection(random_seed=52)
        qvm.run(Program(I(0)), [0])
        return qvm
    except (RequestException, UnknownApiError) as e:
        return pytest.skip("This test requires QVM connection: {}".format(e))


@pytest.fixture(scope='session')
def compiler():
    try:
        compiler = CompilerConnection(isa_source=ISA.from_dict(isa_dict()))
        compiler.compile(Program(I(0)))
        return compiler
    except (RequestException, UnknownApiError) as e:
        return pytest.skip("This test requires compiler connection: {}".format(e))


@pytest.fixture(scope='session')
def forest():
    try:
        connection = ForestConnection()
        connection._wavefunction(Program(I(0)), [], 52)
        return connection
    except (RequestException, UnknownApiError) as e:
        return pytest.skip("This test requires a Forest connection: {}".format(e))

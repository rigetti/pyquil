import numpy as np
import pytest

from pyquil.device import Device


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
            "2-0": {
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

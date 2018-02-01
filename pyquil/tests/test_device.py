from collections import OrderedDict

import numpy as np
import pytest

from pyquil.device import Device, KrausModel, ISA, Qubit, Edge, NoiseModel

DEVICE_FIXTURE_NAME = 'mixed_architecture_chip'


@pytest.fixture
def isa_dict():
    return {
        'id': {
            'name': DEVICE_FIXTURE_NAME,
            'version': '0.0',
            'timestamp': '20180104103600'
        },
        'logical-hardware': [
            [
                {
                    'qubit-id': 0,
                    'type': 'Xhalves'
                },
                {
                    'qubit-id': 1
                },
                {
                    'qubit-id': 2
                }
            ],
            [
                {
                    'action-qubits': [0, 1],
                    'type': 'CZ'
                },
                {
                    'action-qubits': [1, 2],
                    'type': 'ISWAP'
                },
                {
                    'action-qubits': [2, 0],
                    'type': 'CPHASE'
                }
            ]
        ]
    }


@pytest.fixture
def kraus_model_I_dict():
    return {'gate': 'I',
            'fidelity': 1.0,
            'kraus_ops': [[[[1.]], [[1.0]]]],
            'targets': (0, 1),
            'params': (5.0,)}


@pytest.fixture
def kraus_model_RX90_dict():
    return {'gate': 'RX',
            'fidelity': 1.0,
            'kraus_ops': [[[[1.]], [[1.0]]]],
            'targets': (0,),
            'params': (np.pi / 2.,)}


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
            'isa_name': DEVICE_FIXTURE_NAME}


@pytest.fixture
def device_raw(isa_dict, noise_model_dict):
    return {'isa': isa_dict,
            'noise_model': noise_model_dict,
            'is_online': True,
            'is_retuning': False}


def test_isa(isa_dict):
    isa = ISA.from_dict(isa_dict)
    assert isa == ISA(
        name=isa_dict['id']['name'],
        version=isa_dict['id']['version'],
        timestamp=isa_dict['id']['timestamp'],
        qubits=[
            Qubit(id=0, type='Xhalves', dead=None),
            Qubit(id=1, type=None, dead=None),
            Qubit(id=2, type=None, dead=None),
        ],
        edges=[
            Edge(targets=[0, 1], type='CZ', dead=None),
            Edge(targets=[1, 2], type='ISWAP', dead=None),
            Edge(targets=[2, 0], type='CPHASE', dead=None),
        ])
    d = isa.to_dict()
    assert d == isa_dict


def test_kraus_model(kraus_model_I_dict):
    km = KrausModel.from_dict(kraus_model_I_dict)
    assert km == KrausModel(
        gate=kraus_model_I_dict['gate'],
        params=kraus_model_I_dict['params'],
        targets=kraus_model_I_dict['targets'],
        kraus_ops=[KrausModel.unpack_kraus_matrix(kraus_op)
                   for kraus_op in kraus_model_I_dict['kraus_ops']],
        fidelity=kraus_model_I_dict['fidelity'])
    d = km.to_dict()
    assert d == OrderedDict([
        ('gate', km.gate),
        ('params', km.params),
        ('targets', (0, 1)),
        ('kraus_ops', [[[[1.]], [[1.0]]]]),
        ('fidelity', 1.0)
    ])


def test_noise_model(kraus_model_I_dict, kraus_model_RX90_dict):
    noise_model_dict = {'gates': [kraus_model_I_dict,
                                  kraus_model_RX90_dict],
                        'assignment_probs': {'1': [[1.0, 0.0], [0.0, 1.0]],
                                             '0': [[1.0, 0.0], [0.0, 1.0]]},
                        'isa_name': DEVICE_FIXTURE_NAME}

    nm = NoiseModel.from_dict(noise_model_dict)
    km1 = KrausModel.from_dict(kraus_model_I_dict)
    km2 = KrausModel.from_dict(kraus_model_RX90_dict)
    assert nm == NoiseModel(isa_name=DEVICE_FIXTURE_NAME,
                            gates=[km1, km2],
                            assignment_probs={0: np.eye(2), 1: np.eye(2)})
    assert nm.gates_by_name('I') == [km1]
    assert nm.gates_by_name('RX') == [km2]
    assert nm.to_dict() == noise_model_dict


def test_device(isa_dict, noise_model_dict):
    device_raw = {'isa': isa_dict,
                  'noise_model': noise_model_dict,
                  'is_online': True,
                  'is_retuning': False}

    device = Device(DEVICE_FIXTURE_NAME, device_raw)
    assert device.name == DEVICE_FIXTURE_NAME
    assert device.is_online()
    assert not device.is_retuning()

    isa = ISA.from_dict(isa_dict)
    noise_model = NoiseModel.from_dict(noise_model_dict)
    assert isinstance(device.isa, ISA)
    assert device.isa == isa
    assert isinstance(device.noise_model, NoiseModel)
    assert device.noise_model == noise_model

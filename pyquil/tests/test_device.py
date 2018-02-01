import numpy as np
import pytest

from pyquil.device import Device, ISA, Qubit, Edge, THETA, gates_in_isa
from pyquil.noise import NoiseModel, KrausModel
from pyquil.gates import RZ, RX, I, CZ, ISWAP, CPHASE
from collections import OrderedDict


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
                },
                {
                    'qubit-id': 3,
                    'dead': True
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
                },
                {
                    'action-qubits': [0, 3],
                    'type': 'CZ',
                    'dead': True
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
            Qubit(id=0, type='Xhalves', dead=False),
            Qubit(id=1, type='Xhalves', dead=False),
            Qubit(id=2, type='Xhalves', dead=False),
            Qubit(id=3, type='Xhalves', dead=True),
        ],
        edges=[
            Edge(targets=[0, 1], type='CZ', dead=False),
            Edge(targets=[1, 2], type='ISWAP', dead=False),
            Edge(targets=[2, 0], type='CPHASE', dead=False),
            Edge(targets=[0, 3], type='CZ', dead=True),
        ])
    assert isa == ISA.from_dict(isa.to_dict())


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


def test_gates_in_isa(isa_dict):
    isa = ISA.from_dict(isa_dict)
    gates = gates_in_isa(isa)
    for q in [0, 1, 2]:
        for g in [I, RX(np.pi / 2), RX(-np.pi / 2), RZ(THETA)]:
            assert g(q) in gates

    assert CZ(0, 1) in gates
    assert CZ(1, 0) in gates
    assert ISWAP(1, 2) in gates
    assert ISWAP(2, 1) in gates
    assert CPHASE(THETA)(2, 0) in gates
    assert CPHASE(THETA)(0, 2) in gates

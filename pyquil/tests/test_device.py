import numpy as np
import pytest

from pyquil.device import (Device, ISA, Qubit, Edge, Specs, QubitSpecs,
                           EdgeSpecs, THETA, gates_in_isa)
from pyquil.noise import NoiseModel, KrausModel
from pyquil.gates import RZ, RX, I, CZ, ISWAP, CPHASE
from collections import OrderedDict


DEVICE_FIXTURE_NAME = 'mixed_architecture_chip'


@pytest.fixture
def specs_dict():
    return {
        '1Q': {
            "0": {
                "f1QRB": 0.99,
                "fRO": 0.93,
                "T1": 20e-6,
                "T2": 15e-6
            },
            "1": {
                "f1QRB": 0.989,
                "fRO": 0.92,
                "T1": 19e-6,
                "T2": 12e-6
            },
            "2": {
                "f1QRB": 0.983,
                "fRO": 0.95,
                "T1": 21e-6,
                "T2": 16e-6
            },
            "3": {
                "f1QRB": 0.988,
                "fRO": 0.94,
                "T1": 18e-6,
                "T2": 11e-6
            }
        },
        '2Q': {
            "0-1": {
                "fBellState": 0.90,
                "fCZ": 0.89,
                "fCPHASE": 0.88
            },
            "1-2": {
                "fBellState": 0.91,
                "fCZ": 0.90,
                "fCPHASE": 0.89
            },
            "2-0": {
                "fBellState": 0.92,
                "fCZ": 0.91,
                "fCPHASE": 0.90
            },
            "0-3": {
                "fBellState": 0.89,
                "fCZ": 0.88,
                "fCPHASE": 0.87
            }
        }
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


def test_isa(isa_dict):
    isa = ISA.from_dict(isa_dict)
    assert isa == ISA(
        qubits=[
            Qubit(id=0, type='Xhalves', dead=False),
            Qubit(id=1, type='Xhalves', dead=False),
            Qubit(id=2, type='Xhalves', dead=False),
            Qubit(id=3, type='Xhalves', dead=True),
        ],
        edges=[
            Edge(targets=[0, 1], type='CZ', dead=False),
            Edge(targets=[0, 3], type='CZ', dead=True),
            Edge(targets=[1, 2], type='ISWAP', dead=False),
            Edge(targets=[2, 0], type='CPHASE', dead=False),
        ])
    assert isa == ISA.from_dict(isa.to_dict())


def test_specs(specs_dict):
    specs = Specs.from_dict(specs_dict)
    assert specs == Specs(
        qubits_specs=[
            QubitSpecs(id=0, f1QRB=0.99, fRO=0.93, T1=20e-6, T2=15e-6),
            QubitSpecs(id=1, f1QRB=0.989, fRO=0.92, T1=19e-6, T2=12e-6),
            QubitSpecs(id=2, f1QRB=0.983, fRO=0.95, T1=21e-6, T2=16e-6),
            QubitSpecs(id=3, f1QRB=0.988, fRO=0.94, T1=18e-6, T2=11e-6)
        ],
        edges_specs=[
            EdgeSpecs(targets=[0, 1], fBellState=0.90, fCZ=0.89, fCPHASE=0.88),
            EdgeSpecs(targets=[0, 3], fBellState=0.89, fCZ=0.88, fCPHASE=0.87),
            EdgeSpecs(targets=[1, 2], fBellState=0.91, fCZ=0.90, fCPHASE=0.89),
            EdgeSpecs(targets=[2, 0], fBellState=0.92, fCZ=0.91, fCPHASE=0.90)
        ])

    assert specs == Specs.from_dict(specs.to_dict())

    assert specs.f1QRBs() == {0: 0.99, 1: 0.989, 2: 0.983, 3: 0.988}
    assert specs.fROs() == {0: 0.93, 1: 0.92, 2: 0.95, 3: 0.94}
    assert specs.T1s() == {0: 20e-6, 1: 19e-6, 2: 21e-6, 3: 18e-6}
    assert specs.T2s() == {0: 15e-6, 1: 12e-6, 2: 16e-6, 3: 11e-6}

    assert specs.fBellStates() == {(0, 1): 0.90, (0, 3): 0.89, (1, 2): 0.91, (2, 0): 0.92}
    assert specs.fCZs() == {(0, 1): 0.89, (0, 3): 0.88, (1, 2): 0.90, (2, 0): 0.91}
    assert specs.fCPHASEs() == {(0, 1): 0.88, (0, 3): 0.87, (1, 2): 0.89, (2, 0): 0.90}


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
                        }

    nm = NoiseModel.from_dict(noise_model_dict)
    km1 = KrausModel.from_dict(kraus_model_I_dict)
    km2 = KrausModel.from_dict(kraus_model_RX90_dict)
    assert nm == NoiseModel(gates=[km1, km2],
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

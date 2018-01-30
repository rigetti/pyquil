from collections import OrderedDict

import numpy as np

from pyquil.device_info import KrausModel, ISA, Qubit, Edge, NoiseModel

ARCH_QPU = {
    "id": {
        "name": "Mixed architecture chip",
        "version": "0.0",
        "timestamp": "20180104103600"
    },
    "logical-hardware": [
        [
            {
                "qubit-id": 0,
                "type": "Xhalves"
            },
            {
                "qubit-id": 1
            },
            {
                "qubit-id": 2
            }
        ],
        [
            {
                "action-qubits": [0, 1],
                "type": "CZ"
            },
            {
                "action-qubits": [1, 2],
                "type": "ISWAP"
            },
            {
                "action-qubits": [2, 0],
                "type": "CPHASE"
            }
        ]
    ]
}


def test_isa():
    isa = ISA.from_dict(ARCH_QPU)
    assert isa == ISA(
        name=ARCH_QPU["id"]["name"],
        version=ARCH_QPU["id"]["version"],
        timestamp=ARCH_QPU["id"]["timestamp"],
        qubits=[
            Qubit(id=0, type="Xhalves", dead=None),
            Qubit(id=1, type=None, dead=None),
            Qubit(id=2, type=None, dead=None),
        ],
        edges=[
            Edge(targets=[0, 1], type="CZ", dead=None),
            Edge(targets=[1, 2], type="ISWAP", dead=None),
            Edge(targets=[2, 0], type="CPHASE", dead=None),
        ])
    d = isa.to_dict()
    assert d == ARCH_QPU


def test_kraus_model():
    km = KrausModel('I', (5.,), (0, 1), [np.array([[1 + 1j]])], 1.0)
    d = km.to_dict()
    assert d == OrderedDict([
        ('gate', km.gate),
        ('params', km.params),
        ('targets', (0, 1)),
        ('kraus_ops', [[[[1.]], [[1.0]]]]),
        ('fidelity', 1.0)
    ])
    assert KrausModel.from_dict(d) == km


def test_noise_model():
    km1 = KrausModel('I', (5.,), (0, 1), [np.array([[1 + 1j]])], 1.0)
    km2 = KrausModel('RX', (np.pi / 2,), (0,), [np.array([[1 + 1j]])], 1.0)
    nm = NoiseModel("my_qpu", [km1, km2], {0: np.eye(2), 1: np.eye(2)})

    assert nm == NoiseModel.from_dict(nm.to_dict())
    assert nm.gates_by_name("I") == [km1]
    assert nm.gates_by_name("RX") == [km2]

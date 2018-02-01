import numpy as np

from pyquil.device import ISA, Qubit, Edge, gates_in_isa, THETA
from pyquil.gates import RZ, RX, I, CZ, ISWAP, CPHASE

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
            },
            {
                "qubit-id": 3,
                "dead": True
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
            },
            {
                "action-qubits": [0, 3],
                "type": "CZ",
                "dead": True
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
            Qubit(id=3, type=None, dead=True),
        ],
        edges=[
            Edge(targets=[0, 1], type="CZ", dead=None),
            Edge(targets=[1, 2], type="ISWAP", dead=None),
            Edge(targets=[2, 0], type="CPHASE", dead=None),
            Edge(targets=[0, 3], type="CZ", dead=True),
        ])
    d = isa.to_dict()
    assert d == ARCH_QPU


def test_gates_in_isa():
    isa = ISA.from_dict(ARCH_QPU)
    gates = gates_in_isa(isa)
    for q in [0, 1, 2]:
        for g in [I, RX(np.pi/2), RX(-np.pi/2), RZ(THETA)]:
            assert g(q) in gates

    assert CZ(0, 1) in gates
    assert CZ(1, 0) in gates
    assert ISWAP(1, 2) in gates
    assert ISWAP(2, 1) in gates
    assert CPHASE(THETA)(2, 0) in gates
    assert CPHASE(THETA)(0, 2) in gates

from builtins import range
from pyquil.resource_manager import *
import pyquil.quil as pq
from pyquil.gates import *
import pytest


@pytest.fixture
def five_qubit_prog():
    p = pq.Program()
    qubits = [p.alloc() for _ in range(5)]
    p.inst([H(q) for q in qubits])
    return p, qubits


def test_alloc():
    p, qubits = five_qubit_prog()
    for qubit in qubits:
        assert qubit in p.resource_manager.live_qubits
        assert not instantiated(qubit)
        check_live_qubit(qubit)
    # Give the qubits labels
    for qubit in qubits:
        p.resource_manager.instantiate(qubit)
        assert instantiated(qubit)
    for qubit in qubits:
        p.free(qubit)
        assert qubit in p.resource_manager.dead_qubits


def test_add_resource_managers():
    p, p_qubits = five_qubit_prog()
    q, q_qubits = five_qubit_prog()
    summed_program = p + q
    assert (set(summed_program.resource_manager.live_qubits)
            == set.union(set(p_qubits), set(q_qubits)))


def test_direct_qubit_comparison():
    q0 = DirectQubit(0)
    q1 = DirectQubit(0)

    assert q0 == q1
    if q0 != q1:
        # This is a weird behavior in python2. x!=y and x<>y call __ne__() instead of negating
        # __eq__(). In python 3 __ne__ defaults to the inversion of__eq__
        raise AssertionError("Should never have gotten here! Python2 bites again")

    q2 = DirectQubit(2)

    assert q0 != q2

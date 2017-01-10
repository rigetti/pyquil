from pyquil.resource_manager import *
import pyquil.quil as pq
from pyquil.gates import *
import pytest
import json
from mock import Mock
import numpy as np


@pytest.fixture
def five_qubit_prog():
    p = pq.Program()
    qubits = [p.alloc() for _ in xrange(5)]
    p.inst(map(H, qubits))
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


def test_add_qubits():
    p = pq.Program()
    q = pq.Program()
    p_qubits = [p.alloc(), p.alloc()]
    q_qubits = [q.alloc(), q.alloc()]
    p.inst(H(p_qubits[0]), CNOT(*p_qubits))
    q.inst(H(q_qubits[0]), CNOT(*q_qubits))
    two_bell = p + q
    assert p.resource_manager.live_qubits + q.resource_manager.live_qubits == \
           two_bell.resource_manager.live_qubits
    assert p.resource_manager.dead_qubits + q.resource_manager.dead_qubits == \
           two_bell.resource_manager.dead_qubits


def test_synthesize():
    p, qubits = five_qubit_prog()
    p_resource_manager = p.resource_manager
    for qubit in qubits:
        assert qubit.resource_manager == p_resource_manager
        assert qubit.live
    p.synthesize()
    for qubit in qubits:
        assert qubit.resource_manager is None
        assert p.resource_manager == p_resource_manager
        assert not qubit.live


def test_index_handling():
    progs = []
    qubit_lists = []
    for i in xrange(2):
        prog, qubits = five_qubit_prog()
        progs.append(prog)
        qubit_lists.append(qubits)
    # We synthesize the second program to fix the indices of the qubits
    total_prog = reduce(lambda x, y: x + y, progs)
    total_prog.synthesize()
    for index, qubit in progs[0].resource_manager.in_use.iteritems():
        assert total_prog.resource_manager.in_use[index] == qubit
    for index, qubit in progs[1].resource_manager.in_use.iteritems():
        assert total_prog.resource_manager.in_use[len(qubit_lists[0]) + index] == qubit

    total_prog = reduce(lambda x, y: x + y, reversed(progs))
    total_prog.synthesize()
    for index, qubit in progs[1].resource_manager.in_use.iteritems():
        assert total_prog.resource_manager.in_use[len(qubit_lists[0]) + index] == qubit
    for index, qubit in progs[0].resource_manager.in_use.iteritems():
        assert total_prog.resource_manager.in_use[index] == qubit

    progs[0].synthesize()
    total_prog = reduce(lambda x, y: x+y, progs)
    for index, qubit in progs[0].resource_manager.in_use.iteritems():
        assert total_prog.resource_manager.in_use[index] == qubit

    progs[1].synthesize()
    total_prog = reduce(lambda x, y: x + y, progs)
    for index, qubit in progs[1].resource_manager.in_use.iteritems():
        assert total_prog.resource_manager.in_use[index] == qubit

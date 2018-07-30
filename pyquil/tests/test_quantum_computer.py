import networkx as nx
import numpy as np
import pytest

from pyquil import Program
from pyquil.api import QVM
from pyquil.api.quantum_computer import _get_flipped_protoquil_program, QuantumComputer, \
    list_quantum_computers, _parse_name, get_qc
from pyquil.device import NxDevice
from pyquil.gates import *
from pyquil.noise import decoherance_noise_with_asymettric_ro


def test_get_flipped_program():
    program = Program([
        I(0),
        RX(2.3, 1),
        CNOT(0, 1),
        MEASURE(0, 0),
        MEASURE(1, 1),
    ])

    flipped_program = _get_flipped_protoquil_program(program)
    assert flipped_program.out().splitlines()[-6::] == [
        'PRAGMA PRESERVE_BLOCK',
        'RX(pi) 0',
        'RX(pi) 1',
        'PRAGMA END_PRESERVE_BLOCK',
        'MEASURE 0 [0]',
        'MEASURE 1 [1]',
    ]


def test_get_flipped_program_only_measure():
    program = Program([
        MEASURE(0, 0),
        MEASURE(1, 1),
    ])

    flipped_program = _get_flipped_protoquil_program(program)
    assert flipped_program.out().splitlines() == [
        'PRAGMA PRESERVE_BLOCK',
        'RX(pi) 0',
        'RX(pi) 1',
        'PRAGMA END_PRESERVE_BLOCK',
        'MEASURE 0 [0]',
        'MEASURE 1 [1]',
    ]


def test_device_stuff():
    topo = nx.from_edgelist([(0, 4), (0, 99)])
    qc = QuantumComputer(
        name='testy!',
        qam=None,  # not necessary for this test
        device=NxDevice(topo),
    )
    assert nx.is_isomorphic(qc.qubit_topology(), topo)

    isa = qc.get_isa(twoq_type='CPHASE')
    assert sorted(isa.edges)[0].type == 'CPHASE'
    assert sorted(isa.edges)[0].targets == [0, 4]


def test_run(forest):
    qc = QuantumComputer(
        name='testy!',
        qam=QVM(connection=forest, gate_noise=[0.01] * 3),
        device=NxDevice(nx.complete_graph(3)),
    )
    bitstrings = qc.run(Program(
        H(0),
        CNOT(0, 1),
        CNOT(1, 2),
        MEASURE(0, 0),
        MEASURE(1, 1),
        MEASURE(2, 2),
    ), classical_addresses=[0, 1, 2], trials=1000)

    assert bitstrings.shape == (1000, 3)
    parity = np.sum(bitstrings, axis=1) % 3
    assert 0 < np.mean(parity) < 0.15


def test_readout_symmetrization(forest):
    device = NxDevice(nx.complete_graph(3))
    noise_model = decoherance_noise_with_asymettric_ro(device.get_isa())
    qc = QuantumComputer(
        name='testy!',
        qam=QVM(connection=forest, noise_model=noise_model),
        device=device
    )

    prog = Program(I(0), X(1),
                   MEASURE(0, 0),
                   MEASURE(1, 1))

    bs1 = qc.run(prog, [0, 1], 1000)
    avg0_us = np.mean(bs1[:, 0])
    avg1_us = 1 - np.mean(bs1[:, 1])
    diff_us = avg1_us - avg0_us
    print(avg0_us, avg1_us, diff_us)
    assert diff_us > 0.03

    bs2 = qc.run_symmetrized_readout(prog, [0, 1], 1000)
    avg0_s = np.mean(bs2[:, 0])
    avg1_s = 1 - np.mean(bs2[:, 1])
    diff_s = avg1_s - avg0_s
    print(avg0_s, avg1_s, diff_s)
    assert diff_s < 0.05


def test_list_qc():
    qc_names = list_quantum_computers()
    assert qc_names == ['8Q-Agave', '8Q-Agave-qvm', '8Q-Agave-noisy-qvm',
                        '19Q-Acorn', '19Q-Acorn-qvm', '19Q-Acorn-noisy-qvm',
                        '9q-generic-qvm', '9q-generic-noisy-qvm']


def test_parse_qc_name():
    name, as_qvm, noisy = _parse_name('9q-generic', None, None)
    assert name == '9q-generic'
    assert not as_qvm
    assert not noisy

    name, as_qvm, noisy = _parse_name('9q-generic-qvm', None, None)
    assert name == '9q-generic'
    assert as_qvm
    assert not noisy

    name, as_qvm, noisy = _parse_name('9q-generic-noisy-qvm', None, None)
    assert name == '9q-generic'
    assert as_qvm
    assert noisy


def test_parse_qc_flags():
    name, as_qvm, noisy = _parse_name('9q-generic', False, False)
    assert name == '9q-generic'
    assert not as_qvm
    assert not noisy

    name, as_qvm, noisy = _parse_name('9q-generic', True, None)
    assert name == '9q-generic'
    assert as_qvm
    assert not noisy

    name, as_qvm, noisy = _parse_name('9q-generic', True, True)
    assert name == '9q-generic'
    assert as_qvm
    assert noisy


def test_parse_qc_redundant():
    name, as_qvm, noisy = _parse_name('9q-generic', False, False)
    assert name == '9q-generic'
    assert not as_qvm
    assert not noisy

    name, as_qvm, noisy = _parse_name('9q-generic-qvm', True, False)
    assert name == '9q-generic'
    assert as_qvm
    assert not noisy

    name, as_qvm, noisy = _parse_name('9q-generic-noisy-qvm', True, True)
    assert name == '9q-generic'
    assert as_qvm
    assert noisy


def test_parse_qc_conflicting():
    with pytest.raises(ValueError) as e:
        name, as_qvm, noisy = _parse_name('9q-generic-qvm', False, False)

    assert e.match(r'.*but you have specified `as_qvm=False`')

    with pytest.raises(ValueError) as e:
        name, as_qvm, noisy = _parse_name('9q-generic-noisy-qvm', True, False)
    assert e.match(r'.*but you have specified `noisy=False`')


def test_qc(forest):
    qc = get_qc('9q-generic-noisy-qvm', connection=forest)
    assert isinstance(qc, QuantumComputer)
    assert isinstance(qc.qam, QVM)
    assert qc.qam.noise_model is not None
    assert qc.qubit_topology().number_of_nodes() == 9
    assert qc.qubit_topology().degree[0] == 2
    assert qc.qubit_topology().degree[4] == 4

    # TODO: have `qubits` default to all device qubits?
    bs = qc.run_and_measure(Program(X(0)), qubits=[0], trials=1)
    np.testing.assert_array_equal(bs, [[1]])

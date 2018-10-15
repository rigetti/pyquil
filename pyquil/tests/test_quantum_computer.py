import itertools

import networkx as nx
import numpy as np
import pytest

from pyquil import Program, get_qc, list_quantum_computers
from pyquil.api import QVM, QuantumComputer
from pyquil.api._qac import AbstractCompiler
from pyquil.api._quantum_computer import _get_flipped_protoquil_program, _parse_name
from pyquil.device import NxDevice, gates_in_isa
from pyquil.gates import *
from pyquil.noise import decoherence_noise_with_asymmetric_ro


class DummyCompiler(AbstractCompiler):
    def quil_to_native_quil(self, program: Program):
        return program

    def native_quil_to_executable(self, nq_program: Program):
        return nq_program


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
        'MEASURE 0 ro[0]',
        'MEASURE 1 ro[1]',
    ]


def test_get_flipped_program_only_measure():
    program = Program([
        MEASURE(0, 0),
        MEASURE(1, 1),
    ])

    flipped_program = _get_flipped_protoquil_program(program)
    assert flipped_program.out().splitlines() == [
        'DECLARE ro BIT[2]',
        'PRAGMA PRESERVE_BLOCK',
        'RX(pi) 0',
        'RX(pi) 1',
        'PRAGMA END_PRESERVE_BLOCK',
        'MEASURE 0 ro[0]',
        'MEASURE 1 ro[1]',
    ]


def test_device_stuff():
    topo = nx.from_edgelist([(0, 4), (0, 99)])
    qc = QuantumComputer(
        name='testy!',
        qam=None,  # not necessary for this test
        device=NxDevice(topo),
        compiler=DummyCompiler()
    )
    assert nx.is_isomorphic(qc.qubit_topology(), topo)

    isa = qc.get_isa(twoq_type='CPHASE')
    assert sorted(isa.edges)[0].type == 'CPHASE'
    assert sorted(isa.edges)[0].targets == [0, 4]


def test_run(forest):
    device = NxDevice(nx.complete_graph(3))
    qc = QuantumComputer(
        name='testy!',
        qam=QVM(connection=forest, gate_noise=[0.01] * 3),
        device=device,
        compiler=DummyCompiler()
    )
    bitstrings = qc.run(
        Program(
            H(0),
            CNOT(0, 1),
            CNOT(1, 2),
            MEASURE(0, 0),
            MEASURE(1, 1),
            MEASURE(2, 2)).wrap_in_numshots_loop(1000)
    )

    assert bitstrings.shape == (1000, 3)
    parity = np.sum(bitstrings, axis=1) % 3
    assert 0 < np.mean(parity) < 0.15


def test_readout_symmetrization(forest):
    device = NxDevice(nx.complete_graph(3))
    noise_model = decoherence_noise_with_asymmetric_ro(gates=gates_in_isa(device.get_isa()))
    qc = QuantumComputer(
        name='testy!',
        qam=QVM(connection=forest, noise_model=noise_model),
        device=device,
        compiler=DummyCompiler()
    )

    prog = Program(I(0), X(1),
                   MEASURE(0, 0),
                   MEASURE(1, 1))
    prog.wrap_in_numshots_loop(1000)

    bs1 = qc.run(prog)
    avg0_us = np.mean(bs1[:, 0])
    avg1_us = 1 - np.mean(bs1[:, 1])
    diff_us = avg1_us - avg0_us
    assert diff_us > 0.03

    bs2 = qc.run_symmetrized_readout(prog, 1000)
    avg0_s = np.mean(bs2[:, 0])
    avg1_s = 1 - np.mean(bs2[:, 1])
    diff_s = avg1_s - avg0_s
    assert diff_s < 0.05


def test_list_qc():
    qc_names = list_quantum_computers()
    # TODO: update with deployed qpus
    assert qc_names == ['9q-generic-qvm', '9q-generic-noisy-qvm']


def test_parse_qc_name():
    name, qvm_type, noisy = _parse_name('9q-generic', None, None)
    assert name == '9q-generic'
    assert qvm_type is None
    assert not noisy

    name, qvm_type, noisy = _parse_name('9q-generic-qvm', None, None)
    assert name == '9q-generic'
    assert qvm_type == 'qvm'
    assert not noisy

    name, qvm_type, noisy = _parse_name('9q-generic-noisy-qvm', None, None)
    assert name == '9q-generic'
    assert qvm_type == 'qvm'
    assert noisy


def test_parse_qc_flags():
    name, qvm_type, noisy = _parse_name('9q-generic', False, False)
    assert name == '9q-generic'
    assert qvm_type is None
    assert not noisy

    name, qvm_type, noisy = _parse_name('9q-generic', True, None)
    assert name == '9q-generic'
    assert qvm_type == 'qvm'
    assert not noisy

    name, qvm_type, noisy = _parse_name('9q-generic', True, True)
    assert name == '9q-generic'
    assert qvm_type == 'qvm'
    assert noisy


def test_parse_qc_redundant():
    name, qvm_type, noisy = _parse_name('9q-generic', False, False)
    assert name == '9q-generic'
    assert qvm_type is None
    assert not noisy

    name, qvm_type, noisy = _parse_name('9q-generic-qvm', True, False)
    assert name == '9q-generic'
    assert qvm_type == 'qvm'
    assert not noisy

    name, qvm_type, noisy = _parse_name('9q-generic-noisy-qvm', True, True)
    assert name == '9q-generic'
    assert qvm_type == 'qvm'
    assert noisy


def test_parse_qc_conflicting():
    with pytest.raises(ValueError) as e:
        name, qvm_type, noisy = _parse_name('9q-generic-qvm', False, False)

    assert e.match(r'.*but you have specified `as_qvm=False`')

    with pytest.raises(ValueError) as e:
        name, qvm_type, noisy = _parse_name('9q-generic-noisy-qvm', True, False)
    assert e.match(r'.*but you have specified `noisy=False`')


def test_parse_qc_strip():
    # Originally used `str.strip` to remove the suffixes. This is not correct!
    name, _, _ = _parse_name("mvq-qvm", None, None)
    assert name == 'mvq'

    name, _, _ = _parse_name("mvq-noisy-qvm", None, None)
    assert name == 'mvq'


def test_parse_qc_no_prefix():
    prefix, qvm_type, noisy = _parse_name('qvm', None, None)
    assert qvm_type == 'qvm'
    assert not noisy
    assert prefix == ''

    prefix, qvm_type, noisy = _parse_name('', True, None)
    assert qvm_type == 'qvm'
    assert not noisy
    assert prefix == ''


def test_parse_qc_no_prefix_2():
    prefix, qvm_type, noisy = _parse_name('noisy-qvm', None, None)
    assert qvm_type == 'qvm'
    assert noisy
    assert prefix == ''

    prefix, qvm_type, noisy = _parse_name('', True, True)
    assert qvm_type == 'qvm'
    assert noisy
    assert prefix == ''


def test_parse_qc_pyqvm():
    prefix, qvm_type, noisy = _parse_name('9q-generic-pyqvm', None, None)
    assert prefix == '9q-generic'
    assert qvm_type == 'pyqvm'
    assert not noisy


def test_qc_name():
    qc = get_qc("qvm")
    assert qc.name == 'qvm'


def test_qc_name_2():
    qc = get_qc("noisy-qvm")
    assert qc.name == 'noisy-qvm'


def test_qc_name_3():
    qc = get_qc("9q-generic-noisy-qvm")
    assert qc.name == '9q-generic-noisy-qvm'


def test_qc(qvm, compiler):
    qc = get_qc('9q-generic-noisy-qvm')
    assert isinstance(qc, QuantumComputer)
    assert isinstance(qc.qam, QVM)
    assert qc.qam.noise_model is not None
    assert qc.qubit_topology().number_of_nodes() == 9
    assert qc.qubit_topology().degree[0] == 2
    assert qc.qubit_topology().degree[4] == 4

    bs = qc.run_and_measure(Program(X(0)), trials=3)
    assert bs.shape == (3, 9)


def test_fully_connected_qvm_qc():
    qc = get_qc('qvm')
    for q1, q2 in itertools.permutations(range(34), r=2):
        assert (q1, q2) in qc.qubit_topology().edges

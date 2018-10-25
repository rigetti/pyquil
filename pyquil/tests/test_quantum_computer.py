import itertools

import networkx as nx
import numpy as np
import pytest

from pyquil import Program, get_qc, list_quantum_computers
from pyquil.api import QVM, QuantumComputer, local_qvm
from pyquil.api._qac import AbstractCompiler
from pyquil.api._quantum_computer import _get_flipped_protoquil_program, _parse_name
from pyquil.device import NxDevice, gates_in_isa
from pyquil.gates import *
from pyquil.noise import decoherence_noise_with_asymmetric_ro
from rpcq.messages import PyQuilExecutableResponse


class DummyCompiler(AbstractCompiler):
    def get_version_info(self):
        return {}

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
    qc_names = list_quantum_computers(qpus=False)
    # TODO: update with deployed qpus
    assert qc_names == ['9q-square-qvm', '9q-square-noisy-qvm']


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


def test_parse_qc_strip():
    # Originally used `str.strip` to remove the suffixes. This is not correct!
    name, _, _ = _parse_name("mvq-qvm", None, None)
    assert name == 'mvq'

    name, _, _ = _parse_name("mvq-noisy-qvm", None, None)
    assert name == 'mvq'


def test_parse_qc_no_prefix():
    prefix, as_qvm, noisy = _parse_name('qvm', None, None)
    assert as_qvm
    assert not noisy
    assert prefix == ''

    prefix, as_qvm, noisy = _parse_name('', True, None)
    assert as_qvm
    assert not noisy
    assert prefix == ''


def test_parse_qc_no_prefix_2():
    prefix, as_qvm, noisy = _parse_name('noisy-qvm', None, None)
    assert as_qvm
    assert noisy
    assert prefix == ''

    prefix, as_qvm, noisy = _parse_name('', True, True)
    assert as_qvm
    assert noisy
    assert prefix == ''


def test_qc():
    qc = get_qc('9q-square-noisy-qvm')
    assert isinstance(qc, QuantumComputer)
    assert isinstance(qc.qam, QVM)
    assert qc.qam.noise_model is not None
    assert qc.qubit_topology().number_of_nodes() == 9
    assert qc.qubit_topology().degree[0] == 2
    assert qc.qubit_topology().degree[4] == 4
    assert str(qc) == "9q-square-noisy-qvm"


def test_qc_run(qvm, compiler):
    qc = get_qc('9q-square-noisy-qvm')
    bs = qc.run_and_measure(Program(X(0)), trials=3)
    assert len(bs) == 9
    for q, bits in bs.items():
        assert bits.shape == (3,)


def test_nq_qvm_qc():
    for n_qubits in [2, 4, 7, 19]:
        qc = get_qc(f'{n_qubits}q-qvm')
        for q1, q2 in itertools.permutations(range(n_qubits), r=2):
            assert (q1, q2) in qc.qubit_topology().edges
        assert qc.name == f'{n_qubits}q-qvm'


def test_qc_noisy():
    qc = get_qc('5q', as_qvm=True, noisy=True)
    assert isinstance(qc, QuantumComputer)


def test_qc_compile():
    qc = get_qc('5q', as_qvm=True, noisy=True)
    qc.compiler = DummyCompiler()
    prog = Program()
    prog += H(0)
    prog1 = qc.compile(prog)
    assert prog1 == prog


def test_qc_error():
    # QVM is not a QPU
    with pytest.raises(ValueError):
        get_qc('9q-square-noisy-qvm', as_qvm=False)

    with pytest.raises(ValueError):
        get_qc('5q', as_qvm=False)


def test_run_and_measure(local_qvm_quilc):
    qc = get_qc("9q-generic-qvm")
    prog = Program(I(8))
    trials = 11
    with local_qvm():   # Redundant with test fixture.
        bitstrings = qc.run_and_measure(prog, trials)
    bitstring_array = np.vstack(bitstrings[q] for q in sorted(qc.qubits())).T
    assert bitstring_array.shape == (trials, len(qc.qubits()))


def test_run_symmetrized_readout_error(local_qvm_quilc):
    qc = get_qc("9q-generic-qvm")
    trials = 11
    prog = Program(I(8))

    # Trials not even
    with pytest.raises(ValueError):
        bitstrings = qc.run_symmetrized_readout(prog, trials)


def test_qvm_compile_pickiness(forest):
    p = Program(X(0), MEASURE(0, 0))
    p.wrap_in_numshots_loop(1000)
    nq = PyQuilExecutableResponse(program=p.out(), attributes={'num_shots': 1000})

    # Ok, non-realistic
    qc = get_qc('9q-qvm')
    qc.run(p)

    # Also ok
    qc.run(nq)

    # Not ok
    qc = get_qc('9q-square-qvm')
    with pytest.raises(TypeError):
        qc.run(p)

    # Yot ok
    qc.run(nq)

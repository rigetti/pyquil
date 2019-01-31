import itertools

import networkx as nx
import numpy as np
import pytest

from pyquil import Program, get_qc, list_quantum_computers
from pyquil.api import QVM, QuantumComputer, local_qvm
from pyquil.api._qac import AbstractCompiler
from pyquil.api._quantum_computer import _get_flipped_protoquil_program, _parse_name, \
    _get_qvm_with_topology
from pyquil.device import NxDevice, gates_in_isa
from pyquil.gates import *
from pyquil.quilbase import Declare, MemoryReference
from pyquil.noise import decoherence_noise_with_asymmetric_ro
from pyquil.pyqvm import PyQVM
from rpcq.messages import ParameterAref, PyQuilExecutableResponse


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


def test_run_pyqvm_noiseless():
    device = NxDevice(nx.complete_graph(3))
    qc = QuantumComputer(
        name='testy!',
        qam=PyQVM(n_qubits=3),
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
    assert np.mean(parity) == 0


def test_run_pyqvm_noisy():
    device = NxDevice(nx.complete_graph(3))
    qc = QuantumComputer(
        name='testy!',
        qam=PyQVM(n_qubits=3, post_gate_noise_probabilities={'relaxation': 0.01}),
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


def test_qc(qvm, compiler):
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
    # note to devs: this is included as an example in the run_and_measure docstrings
    # so if you change it here ... change it there!
    with local_qvm():  # Redundant with test fixture.
        bitstrings = qc.run_and_measure(prog, trials)
    bitstring_array = np.vstack(bitstrings[q] for q in qc.qubits()).T
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


def test_run_with_parameters(forest):
    device = NxDevice(nx.complete_graph(3))
    qc = QuantumComputer(
        name='testy!',
        qam=QVM(connection=forest),
        device=device,
        compiler=DummyCompiler()
    )
    bitstrings = qc.run(
        executable=Program(
            Declare(name='theta', memory_type='REAL'),
            Declare(name='ro', memory_type='BIT'),
            RX(MemoryReference('theta'), 0),
            MEASURE(0, MemoryReference('ro'))
        ).wrap_in_numshots_loop(1000),
        memory_map={'theta': [np.pi]}
    )

    assert bitstrings.shape == (1000, 1)
    assert all([bit == 1 for bit in bitstrings])


def test_reset(forest):
    device = NxDevice(nx.complete_graph(3))
    qc = QuantumComputer(
        name='testy!',
        qam=QVM(connection=forest),
        device=device,
        compiler=DummyCompiler()
    )
    p = Program(
        Declare(name='theta', memory_type='REAL'),
        Declare(name='ro', memory_type='BIT'),
        RX(MemoryReference('theta'), 0),
        MEASURE(0, MemoryReference('ro'))
    ).wrap_in_numshots_loop(1000)
    qc.run(
        executable=p,
        memory_map={'theta': [np.pi]}
    )

    aref = ParameterAref(name='theta', index=0)
    assert qc.qam._variables_shim[aref] == np.pi
    assert qc.qam._executable == p
    assert qc.qam._bitstrings.shape == (1000, 1)
    assert all([bit == 1 for bit in qc.qam._bitstrings])
    assert qc.qam.status == 'done'

    qc.reset()

    assert qc.qam._variables_shim == {}
    assert qc.qam._executable is None
    assert qc.qam._bitstrings is None
    assert qc.qam.status == 'connected'


def test_get_qvm_with_topology():
    topo = nx.from_edgelist([
        (5, 6),
        (6, 7),
        (10, 11),
    ])
    # Note to developers: perhaps make `get_qvm_with_topology` public in the future
    qc = _get_qvm_with_topology(name='test-qvm', topology=topo)
    assert len(qc.qubits()) == 5
    assert min(qc.qubits()) == 5


def test_get_qvm_with_topology_2(forest):
    topo = nx.from_edgelist([
        (5, 6),
        (6, 7),
    ])
    qc = _get_qvm_with_topology(name='test-qvm', topology=topo)
    results = qc.run_and_measure(Program(X(5)), trials=5)
    assert sorted(results.keys()) == [5, 6, 7]
    assert all(x == 1 for x in results[5])


def test_parse_mix_qvm_and_noisy_flag():
    # https://github.com/rigetti/pyquil/issues/764
    name, qvm_type, noisy = _parse_name('1q-qvm', as_qvm=None, noisy=True)
    assert noisy


def test_noisy(forest):
    # https://github.com/rigetti/pyquil/issues/764
    p = Program(X(0))
    qc = get_qc('1q-qvm', noisy=True)
    result = qc.run_and_measure(p, trials=10000)
    assert result[0].mean() < 1.0

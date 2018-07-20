import networkx as nx
import numpy as np

from pyquil import Program
from pyquil.api import QVM
from pyquil.device import NxDevice, ISA, gates_in_isa
from pyquil.gates import *

from pyquil.api.quantum_computer import _get_flipped_protoquil_program, QuantumComputer
from pyquil.noise import _decoherence_noise_model, NoiseModel


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


def decoherance_noise_with_asymettric_ro(isa: ISA):
    """Reimplementation of `add_decoherance_noise` with asymmetric readout.

    For simplicity, we use the default values for T1, T2, gate times, et al. and hard-code
    readout fidelities here.
    """
    gates = gates_in_isa(isa)
    noise_model = _decoherence_noise_model(gates)
    p00 = 0.975
    p11 = 0.911
    aprobs = np.array([[p00, 1 - p00],
                       [1 - p11, p11]])
    aprobs = {q: aprobs for q in noise_model.assignment_probs.keys()}
    return NoiseModel(noise_model.gates, aprobs)


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

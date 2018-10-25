#!/usr/bin/env python

import numpy as np

from pyquil import Program, get_qc
from pyquil.gates import H, CNOT, MEASURE


def run_bell_high_level(n_shots=1000):
    # Step 1. Get a device. Either a QVM or a QPU
    qc = get_qc('9q-generic-qvm')
    q = [4, 5]  # qubits

    # Step 2. Construct your program
    program = Program(
        H(q[0]),
        CNOT(q[0], q[1])
    )

    # Step 3. Run
    res = qc.run_and_measure(program, trials=n_shots)
    bitstrings = np.zeros(shape=(n_shots, 2), dtype=int)
    bitstrings[:, 0] = np.asarray(res[q[0]])
    bitstrings[:, 1] = np.asarray(res[q[1]])

    # Bincount bitstrings
    basis = np.array([2 ** i for i in range(len(q))])
    ints = np.sum(bitstrings * basis, axis=1)
    print('bincounts', np.bincount(ints))

    # Check parity
    parities = np.sum(bitstrings, axis=1) % 2
    print('avg parity', np.mean(parities))


def run_bell_medium_level(n_shots=1000):
    # Step 1. Get a device. Either a QVM or a QPU
    qc = get_qc('9q-generic-qvm')
    q = [4, 5]  # qubits

    # Step 2. Construct your program
    program = Program()
    program += H(q[0])
    program += CNOT(q[0], q[1])

    # Step 2.1. Manage read-out memory
    ro = program.declare('ro', memory_type='BIT', memory_size='2')
    program += MEASURE(q[0], ro[0])
    program += MEASURE(q[1], ro[1])

    # Step 2.2. Run the program in a loop
    program = program.wrap_in_numshots_loop(n_shots)

    # Step 3. Compile and run
    executable = qc.compile(program)
    bitstrings = qc.run(executable)

    # Bincount bitstrings
    basis = np.array([2 ** i for i in range(len(q))])
    ints = np.sum(bitstrings * basis, axis=1)
    print('bincounts', np.bincount(ints))

    # Check parity
    parities = np.sum(bitstrings, axis=1) % 2
    print('avg parity', np.mean(parities))


def run_bell_low_level(n_shots=1000):
    # Step 1. Get some device components
    qc = get_qc('9q-generic-qvm')
    compiler = qc.compiler
    qam = qc.qam
    del qc

    q = [4, 5]  # qubits

    # Step 2. Construct your program
    program = Program()
    program += H(q[0])
    program += CNOT(q[0], q[1])

    # Step 2.1. Manage read-out memory
    ro = program.declare('ro', memory_type='BIT', memory_size='2')
    program += MEASURE(q[0], ro[0])
    program += MEASURE(q[1], ro[1])

    # Step 2.2. Run the program in a loop
    program = program.wrap_in_numshots_loop(n_shots)

    # Step 3. Compile and run
    nq_program = compiler.quil_to_native_quil(program)
    executable = compiler.native_quil_to_executable(nq_program)
    bitstrings = qam.load(executable) \
        .run() \
        .wait() \
        .read_from_memory_region(region_name="ro")

    # Bincount bitstrings
    basis = np.array([2 ** i for i in range(len(q))])
    ints = np.sum(bitstrings * basis, axis=1)
    print('bincounts', np.bincount(ints))

    # Check parity
    parities = np.sum(bitstrings, axis=1) % 2
    print('avg parity', np.mean(parities))


if __name__ == '__main__':
    run_bell_high_level()
    run_bell_medium_level()
    run_bell_low_level()

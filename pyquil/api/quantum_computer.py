from math import pi
from typing import List

import numpy as np

from pyquil.api._qam import QAM
from pyquil.device import AbstractDevice
from pyquil.gates import MEASURE, RX
from pyquil.quil import Program, get_classical_addresses_from_program
from pyquil.quilbase import Measurement, Pragma


def _get_flipped_protoquil_program(program: Program):
    """For symmetrization, generate a program where X gates are added before measurement.

    Forest 1.3 is really picky about where the measure instructions happen. It has to be
    at the end!
    """
    program = Program(program.instructions)  # Copy
    to_measure = []
    while True:
        inst = program.instructions[-1]
        if isinstance(inst, Measurement):
            program.pop()
            to_measure.append((inst.qubit, inst.classical_reg))
        else:
            break

    program += Pragma('PRESERVE_BLOCK')
    for qu, addr in to_measure[::-1]:
        program += RX(pi, qu)
    program += Pragma('END_PRESERVE_BLOCK')

    for qu, addr in to_measure[::-1]:
        program += Measurement(qubit=qu, classical_reg=addr)

    return program


class QuantumComputer:
    def __init__(self, *, name: str, qam: QAM, device: AbstractDevice, symmetrize=False):
        """
        A quantum computer for running quantum programs.

        A quantum computer has various characteristics like supported gates, qubits, qubit
        topologies, gate fidelities, and more. A quantum computer also has the ability to
        run quantum programs.

        A quantum computer can be a real Rigetti QPU that uses superconducting transmon
        qubits to run quantum programs, or it can be an emulator like the Rigetti QVM with
        noise models and mimicked topologies.

        :param name: A string identifying this particular quantum computer.
        :param qam: A quantum abstract machine which handles executing quantum programs. This
            dispatches to a QVM or QPU.
        :param device: A collection of connected qubits and associated specs and topology.
        :param symmetrize: Whether to apply readout error symmetrization. See
            :py:func:`run_symmetrized` for a complete description.
        """
        self.name = name
        self.qam = qam
        self.device = device

        self.symmetrize = symmetrize

    def qubit_topology(self):
        return self.device.qubit_topology()

    def get_isa(self, oneq_type='Xhalves', twoq_type='CZ'):
        return self.device.get_isa(oneq_type=oneq_type, twoq_type=twoq_type)

    def run(self, program, classical_addresses, trials, symmetrize=None) -> np.ndarray:
        """
        Run a quil program.

        :param program: The program to run. You probably want to put MEASURE instructions
            in your program somewhere (like at the end) because qubits are not automatically
            measured
        :param classical_addresses: The addresses of the classical bits to return. These don't
            necessarily correspond to qubit indices; rather they are the second argument to
            any MEASURE instructions you've added to your program
        :param trials: The number of times to run the program.
        :param symmetrize: Whether to apply readout error symmetrization. If not specified,
            the class attribute ``symmetrize`` will be used. See :py:func:`run_symmetrized` for
            a complete description.
        :return: A numpy array of shape (trials, len(classical_addresses)) that contains 0s and 1s
        """
        if symmetrize is None:
            symmetrize = self.symmetrize

        if not classical_addresses:
            classical_addresses = get_classical_addresses_from_program(program)

        if symmetrize:
            return self.run_symmetrized(program, classical_addresses, trials)

        return self.qam.run(program, classical_addresses, trials)

    def run_async(self, program, classical_addresses, trials, symmetrize=None) -> str:
        """
        Queue a quil program for running, but return immediately with a job id.

        Use :py:func:`QuantumComputer.wait_for_job` to get the actual job results, probably
        after queueing up a whole batch of jobs.

        See :py:func:`run` for this function's parameter descriptions.

        :returns: a job id
        """
        if symmetrize is None:
            symmetrize = self.symmetrize

        if not classical_addresses:
            classical_addresses = get_classical_addresses_from_program(program)

        if symmetrize:
            raise NotImplementedError("Async symmetrized measurement isn't supported")

        return self.qam.run_async(program, classical_addresses, trials)

    def run_symmetrized(self, program, classical_addresses, trials):
        """
        Run a quil program in such a way that the readout error is made collectively symmetric

        This means the probability of a bitstring ``b`` being mistaken for a bitstring ``c`` is
        the same as the probability of ``not(b)`` being mistaken for ``not(c)``

        A more general symmetrization would guarantee that the probability of ``b`` being
        mistaken for ``c`` depends only on which bit of ``c`` are different from ``b``. This
        would require choosing random subsets of bits to flip.

        In a noisy device, the probability of accurately reading the 0 state might be higher
        than that of the 1 state. This makes correcting for readout more difficult. This
        function runs the program normally ``(trials//2)`` times. The other half of the time,
        it will insert an ``X`` gate prior to any ``MEASURE`` instruction and then flip the
        measured classical bit back.

        See :py:func:`run` for this function's parameter descriptions.
        """
        flipped_program = _get_flipped_protoquil_program(program)
        if trials % 2 != 0:
            raise ValueError("Using symmetrized measurement functionality requires that you "
                             "take an even number of trials.")
        half_trials = trials // 2

        samples = self.run(program, classical_addresses, half_trials, symmetrize=False)
        flipped_samples = self.run(flipped_program, classical_addresses, half_trials,
                                   symmetrize=False)
        double_flipped_samples = np.logical_not(flipped_samples).astype(int)
        results = np.concatenate((samples, double_flipped_samples), axis=0)
        np.random.shuffle(results)
        return results

    def run_and_measure(self, program: Program, qubits: List[int], trials: int, symmetrize=None):
        """
        Automatically apply MEASURE instructions to each qubit.

        I think we should discourage use of this function. ``run`` is much more general.

        Unlike the old ``run_and_measure`` function, this one works with noise models.

        :param program: The program to run. We'll put MEASURE instructions at the end.
        :param qubits: Qubit indices to measure.
        :param trials: The number of times to run the program.
        :param symmetrize: Whether to apply readout error symmetrization. If not specified,
            the class attribute ``symmetrize`` will be used. See :py:func:`run_symmetrized` for
            a complete description.
        :return: A numpy array of shape (trials, len(qubits)) that contains 0s and 1s
        """
        new_prog = Program().inst(program)  # make a copy?

        classical_addrs = list(range(len(qubits)))
        for q, i in zip(qubits, classical_addrs):
            new_prog += MEASURE(q, i)

        return self.run(program=new_prog, classical_addresses=classical_addrs,
                        trials=trials, symmetrize=symmetrize)

    def wait_for_job(self, job_id):
        return self.qam.wait_for_job(job_id)

    def __str__(self):
        return self.name

from typing import Iterable, List, Union

import numpy as np
from six import integer_types

from pyquil.api import Job
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.wavefunction import Wavefunction
from ._base_connection import ForestConnection


class WavefunctionSimulator:
    def __init__(self, connection: ForestConnection = None, random_seed=None):
        """
        A simulator that propagates a wavefunction representation of a quantum state.

        :param connection: A connection to the Forest web API.
        :param random_seed: A seed for the simulator's random number generators. Either None (for
            an automatically generated seed) or a non-negative integer.
        """
        if connection is None:
            connection = ForestConnection()

        self.connection = connection

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, integer_types) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int")

    def wavefunction(self, quil_program: Program,
                     classical_addresses: Iterable[int] = None) -> Wavefunction:
        """
        Simulate a Quil program and return the wavefunction.

        .. note:: If your program contains measurements or noisy gates, this method may not do what
            you want. If the execution of ``quil_program`` is **non-deterministic** then the
            final wavefunction only represents a stochastically generated sample and the
            wavefunctions returned by *different* ``wavefunction`` calls *will generally be
            different*.

        :param quil_program: A Quil program.
        :param classical_addresses: An optional list of classical addresses to return in addition
            of the quantum wavefunction.
        :return: A Wavefunction object representing the state of the QVM.
        """
        if classical_addresses is None:
            classical_addresses = []

        return self.connection._wavefunction(quil_program=quil_program,
                                             classical_addresses=classical_addresses,
                                             random_seed=self.random_seed)

    def wavefunction_async(self, quil_program, classical_addresses=None):
        """
        Similar to wavefunction except that it returns a job id and doesn't wait for the program
        to be executed. See https://go.rigetti.com/connections for reasons to use this method.
        """
        if classical_addresses is None:
            classical_addresses = []

        return self.connection._wavefunction_async(quil_program=quil_program,
                                                   classical_addresses=classical_addresses,
                                                   random_seed=self.random_seed)

    def expectation(self, prep_prog: Program,
                    pauli_terms: Union[PauliSum, List[PauliTerm]]) -> Union[float, List[float]]:
        """
        Calculate the expectation value of Pauli operators given a state prepared by prep_program.

        If ``pauli_terms`` is a ``PauliSum`` then the returned value is a single ``float``,
        otherwise the returned value is a list of ``float``s, one for each ``PauliTerm`` in the
        list.

        .. note:: If your program contains measurements or noisy gates, this method may not do what
            you want. If the execution of ``quil_program`` is **non-deterministic** then the
            final wavefunction from which the expectation value is calculated only represents
            a stochastically generated sample and the wavefunctions returned by *different*
            ``wavefunction`` calls *will generally be different*.

        :param prep_prog: A program that prepares the state on which we measure the expectation.
        :param pauli_terms: A Pauli representation of a quantum operator.
        :return: Either a float or list of floats depending on ``pauli_terms``.
        """

        is_pauli_sum = False
        if isinstance(pauli_terms, PauliSum):
            progs, coeffs = pauli_terms.get_programs()
            is_pauli_sum = True
        else:
            coeffs = [pt.coefficient for pt in pauli_terms]
            progs = [pt.program for pt in pauli_terms]

        bare_results = self.connection._expectation(prep_prog, progs, random_seed=self.random_seed)
        results = [c * r for c, r in zip(coeffs, bare_results)]
        if is_pauli_sum:
            return sum(results)
        return results

    def run_and_measure(self, quil_program: Program, qubits: List[int] = None, trials: int = 1):
        """
        Run a Quil program once to determine the final wavefunction, and measure multiple times.

        Alternatively, consider using ``wavefunction`` and calling ``sample_bitstrings`` on the
        resulting object.

        For a large wavefunction and a low-medium number of trials, use this function.
        On the other hand, if you're sampling a small system many times you might want to
        use ``Wavefunction.sample_bitstrings``.

        .. note:: If your program contains measurements or noisy gates, this method may not do what
            you want. If the execution of ``quil_program`` is **non-deterministic** then the
            final wavefunction from which the returned bitstrings are sampled itself only
            represents a stochastically generated sample and the outcomes sampled from
            *different* ``run_and_measure`` calls *generally sample different bitstring
            distributions*.

        :param quil_program: The program to run and measure
        :param qubits: An optional list of qubits to measure. The order of this list is
            respected in the returned bitstrings. If not provided, all qubits used in
            the program will be measured and returned in their sorted order.
        :param int trials: Number of times to sample from the prepared wavefunction.
        :return: An array of measurement results (0 or 1) of shape (trials, len(qubits))
        """
        if qubits is None:
            qubits = sorted(quil_program.get_qubits(indices=True))

        return np.asarray(self.connection._run_and_measure(quil_program=quil_program, qubits=qubits,
                                                           trials=trials,
                                                           random_seed=self.random_seed))

    def run_and_measure_async(self, quil_program, qubits=None, trials=1):
        """
        Similar to run_and_measure except that it returns a job id and doesn't wait for the
        program to be executed.

        See https://go.rigetti.com/connections for reasons to use this method.
        """
        if qubits is None:
            qubits = sorted(quil_program.get_qubits(indices=True))

        return self.connection._run_and_measure_async(quil_program=quil_program, qubits=qubits,
                                                      trials=trials, random_seed=self.random_seed)

    def wait_for_job(self, job_id, ping_time=None, status_time=None) -> Job:
        """
        For async functions, wait for the specified job to be done and return the completed job.

        :param job_id: The id of the job returned by ``_async`` methods.
        :param ping_time: An optional time in seconds to poll for job completion.
        :param status_time: An optional time in seconds to print the status of a job.
        :return: The completed job.
        """
        return self.connection._wait_for_job(job_id=job_id, ping_time=ping_time,
                                             status_time=status_time, machine='QVM')

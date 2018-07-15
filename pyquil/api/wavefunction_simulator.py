import warnings

from six import integer_types

from pyquil.paulis import PauliSum
from pyquil.quil import Program
from pyquil.wavefunction import Wavefunction
from ._base_connection import ForestConnection


class WavefunctionSimulator:
    """
    Represents a connection to the QVM.
    """

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

    def run_and_measure(self, quil_program, qubits, trials=1):
        """
        Run a Quil program once to determine the final wavefunction, and measure multiple times.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the outcomes sampled from *different* ``run_and_measure`` calls *generally sample
            different bitstring distributions*.

        :param Program quil_program: A Quil program.
        :param list|range qubits: A list of qubits.
        :param int trials: Number of shots to collect.
        :return: A list of a list of bits.
        :rtype: list
        """
        return self.connection._run_and_measure(quil_program=quil_program, qubits=qubits,
                                                trials=trials, random_seed=self.random_seed)

    def run_and_measure_async(self, quil_program, qubits, trials=1):
        """
        Similar to run_and_measure except that it returns a job id and doesn't wait for the
        program to be executed.

        See https://go.rigetti.com/connections for reasons to use this method.
        """
        return self.connection._run_and_measure_async(quil_program=quil_program, qubits=qubits,
                                                      trials=trials, random_seed=self.random_seed)

    def wavefunction(self, quil_program, classical_addresses=None):
        """
        Simulate a Quil program and get the wavefunction back.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the wavefunctions returned by *different* ``wavefunction`` calls *will generally be
            different*.

        :param Program quil_program: A Quil program.
        :param list|range classical_addresses: An optional list of classical addresses.
        :param needs_compilation: If True, preprocesses the job with the compiler.
        :param isa: If set, compiles to this target ISA.
        :return: A Wavefunction object representing the state of the QVM.
        :rtype: Wavefunction
        """
        if classical_addresses is None:
            classical_addresses = []

        return self.connection._wavefunction(quil_program=quil_program,
                                             classical_addresses=classical_addresses,
                                             random_seed=self.random_seed)

    def wavefunction_async(self, quil_program, classical_addresses=None):
        """
        Similar to wavefunction except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        if classical_addresses is None:
            classical_addresses = []

        return self.connection._wavefunction_async(quil_program=quil_program,
                                                   classical_addresses=classical_addresses,
                                                   random_seed=self.random_seed)

    def expectation(self, prep_prog, pauli_terms):
        """
        Calculate the expectation value of Pauli operators given a state prepared by prep_program.

        If ``pauli_terms`` is a ``PauliSum`` then the returned value is a single ``float``,
        otherwise the returned value is a list of ``float``s, one for each ``PauliTerm`` in the
        list.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            expectation values are computed itself only represents a stochastically generated
            sample. The expectations returned from *different* ``expectation`` calls *will then
            generally be different*.

        :param Program prep_prog: Quil program for state preparation.
        :param Sequence[PauliTerm]|PauliSum pauli_terms: A list of PauliTerms or a PauliSum.
        :return: If ``pauli_terms`` is a PauliSum return its expectation value. Otherwise return
          a list of expectation values.
        :rtype: float|List[float]
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

    def expectation_async(self, prep_prog, operator_programs=None):
        """
        Similar to expectation except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        warnings.warn("`expectation_async()` is deprecated. Use `pauli_expectation`.",
                      DeprecationWarning)
        return self.connection._expectation_async(prep_prog=prep_prog,
                                                  operator_programs=operator_programs,
                                                  random_seed=self.random_seed)

import warnings

from six import integer_types

from pyquil.api.job import Job
from pyquil.paulis import PauliSum
from pyquil.quil import Program
from pyquil.wavefunction import Wavefunction
from ._base_connection import validate_run_items, TYPE_MULTISHOT_MEASURE, TYPE_WAVEFUNCTION, \
    TYPE_EXPECTATION, get_job_id, get_session, wait_for_job, post_json, get_json, SYNC_ENDPOINT, \
    ASYNC_ENDPOINT


def _run_and_measure_payload(quil_program, qubits, trials, random_seed):
    """REST payload for :py:func:`run_and_measure`"""
    if not quil_program:
        raise ValueError("You have attempted to run an empty program."
                         " Please provide gates or measure instructions to your program.")

    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")
    validate_run_items(qubits)
    if not isinstance(trials, integer_types):
        raise TypeError("trials must be an integer")

    payload = {"type": TYPE_MULTISHOT_MEASURE,
               "qubits": list(qubits),
               "trials": trials,
               'compiled-quil': quil_program.out()}

    if random_seed is not None:
        payload['rng-seed'] = random_seed

    return payload


def _wavefunction_payload(quil_program, classical_addresses, random_seed):
    """REST payload for :py:func:`wavefunction`"""
    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")
    validate_run_items(classical_addresses)

    payload = {'type': TYPE_WAVEFUNCTION,
               'addresses': list(classical_addresses),
               'compiled-quil': quil_program.out()}

    if random_seed is not None:
        payload['rng-seed'] = random_seed

    return payload


def _expectation_payload(prep_prog, operator_programs, random_seed):
    """REST payload for :py:func:`expectation`"""
    if operator_programs is None:
        operator_programs = [Program()]

    if not isinstance(prep_prog, Program):
        raise TypeError("prep_prog variable must be a Quil program object")

    payload = {'type': TYPE_EXPECTATION,
               'state-preparation': prep_prog.out(),
               'operators': [x.out() for x in operator_programs]}

    if random_seed is not None:
        payload['rng-seed'] = random_seed

    return payload


class ForestConnection:
    """
    Represents a connection to Forest
    """

    def __init__(self, sync_endpoint=SYNC_ENDPOINT,
                 async_endpoint=ASYNC_ENDPOINT, api_key=None, user_id=None,
                 use_queue=False, ping_time=0.1, status_time=2):
        """
        Constructor for wavefunction simulation. Sets up any necessary security.

        :param sync_endpoint: The endpoint of the server for running small jobs
        :param async_endpoint: The endpoint of the server for running large jobs
        :param api_key: The key to the Forest API Gateway (default behavior is to read from
            config file)
        :param user_id: Your userid for Forest (default behavior is to read from config file)
        :param bool use_queue: Disabling this parameter may improve performance for small,
            quick programs. To support larger programs, set it to True. (default: False)
            *_async methods will always use the queue; See https://go.rigetti.com/connections
            for more information.
        :param int ping_time: Time in seconds for how long to wait between polling the server
            for updated status information on a job. Note that this parameter doesn't matter if
            use_queue is False.
        :param int status_time: Time in seconds for how long to wait between printing status
            information. To disable printing of status entirely then set status_time to False.
            Note that this parameter doesn't matter if use_queue is False.
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
            automatically generated seed) or a non-negative integer.
        """
        self.async_endpoint = async_endpoint
        self.sync_endpoint = sync_endpoint
        self.session = get_session(api_key, user_id)

        self.use_queue = use_queue
        self.ping_time = ping_time
        self.status_time = status_time

    def run_and_measure(self, quil_program, qubits, trials, random_seed):
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
        payload = _run_and_measure_payload(quil_program, qubits, trials, random_seed)
        if self.use_queue:
            response = post_json(self.session, self.async_endpoint + "/job",
                                 {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
            return response.json()

    def run_and_measure_async(self, quil_program, qubits, trials, random_seed):
        """
        Similar to run_and_measure except that it returns a job id and doesn't wait for the
        program to be executed.

        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = _run_and_measure_payload(quil_program, qubits, trials, random_seed)
        response = post_json(self.session, self.async_endpoint + "/job",
                             {"machine": "QVM", "program": payload})
        return get_job_id(response)

    def wavefunction(self, quil_program, classical_addresses, random_seed):
        """
        Simulate a Quil program and get the wavefunction back.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the wavefunctions returned by *different* ``wavefunction`` calls *will generally be
            different*.

        :param Program quil_program: A Quil program.
        :param list|range classical_addresses: A list of classical addresses.
        :param needs_compilation: If True, preprocesses the job with the compiler.
        :param isa: If set, compiles to this target ISA.
        :return: A Wavefunction object representing the state of the QVM.
        :rtype: Wavefunction
        """
        if self.use_queue:
            payload = _wavefunction_payload(quil_program, classical_addresses, random_seed)
            response = post_json(self.session, self.async_endpoint + "/job",
                                 {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = _wavefunction_payload(quil_program, classical_addresses, random_seed)
            response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
            return Wavefunction.from_bit_packed_string(response.content, classical_addresses)

    def wavefunction_async(self, quil_program, classical_addresses, random_seed):
        """
        Similar to wavefunction except that it returns a job id and doesn't wait for the program
        to be executed.

        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = _wavefunction_payload(quil_program, classical_addresses, random_seed)
        response = post_json(self.session, self.async_endpoint + "/job",
                             {"machine": "QVM", "program": payload})
        return get_job_id(response)

    def expectation(self, prep_prog, operator_programs, random_seed):
        """
        Calculate the expectation value of operators given a state prepared by
        prep_program.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            expectation values are computed itself only represents a stochastically generated
            sample. The expectations returned from *different* ``expectation`` calls *will then
            generally be different*.

        To measure the expectation of a PauliSum, you probably want to
        do something like this::

                progs, coefs = hamiltonian.get_programs()
                expect_coeffs = np.array(cxn.expectation(prep_program, operator_programs=progs))
                return np.real_if_close(np.dot(coefs, expect_coeffs))

        :param Program prep_prog: Quil program for state preparation.
        :param list operator_programs: A list of Programs, each specifying an operator whose expectation
            to compute.
        :return: Expectation values of the operators.
        :rtype: List[float]
        """
        if isinstance(operator_programs, Program):
            warnings.warn(
                "You have provided a Program rather than a list of Programs. The results from expectation "
                "will be line-wise expectation values of the operator_programs.", SyntaxWarning)
        if self.use_queue:
            payload = _expectation_payload(prep_prog, operator_programs, random_seed)
            response = post_json(self.session, self.async_endpoint + "/job",
                                 {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = _expectation_payload(prep_prog, operator_programs, random_seed)
            response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
            return response.json()

    def expectation_async(self, prep_prog, operator_programs, random_seed):
        """
        Similar to expectation except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = _expectation_payload(prep_prog, operator_programs, random_seed)
        response = post_json(self.session, self.async_endpoint + "/job",
                             {"machine": "QVM", "program": payload})
        return get_job_id(response)

    def get_job(self, job_id):
        """
        Given a job id, return information about the status of the job

        :param str job_id: job id
        :return: Job object with the status and potentially results of the job
        :rtype: Job
        """
        response = get_json(self.session, self.async_endpoint + "/job/" + job_id)
        return Job(response.json(), 'QVM')

    def wait_for_job(self, job_id, ping_time=None, status_time=None):
        """
        Wait for the results of a job and periodically print status

        :param job_id: Job id
        :param ping_time: How often to poll the server.
                          Defaults to the value specified in the constructor. (0.1 seconds)
        :param status_time: How often to print status, set to False to never print status.
                            Defaults to the value specified in the constructor (2 seconds)
        :return: Completed Job
        """

        def get_job_fn():
            return self.get_job(job_id)

        return wait_for_job(get_job_fn,
                            ping_time if ping_time else self.ping_time,
                            status_time if status_time else self.status_time)


class WavefunctionSimulator:
    """
    Represents a connection to the QVM.
    """

    def __init__(self, connection: ForestConnection, random_seed=None):
        """
        Constructor for wavefunction simulation. Sets up any necessary security.

        :param sync_endpoint: The endpoint of the server for running small jobs
        :param async_endpoint: The endpoint of the server for running large jobs
        :param api_key: The key to the Forest API Gateway (default behavior is to read from
            config file)
        :param user_id: Your userid for Forest (default behavior is to read from config file)
        :param bool use_queue: Disabling this parameter may improve performance for small,
            quick programs. To support larger programs, set it to True. (default: False)
            *_async methods will always use the queue; See https://go.rigetti.com/connections
            for more information.
        :param int ping_time: Time in seconds for how long to wait between polling the server
            for updated status information on a job. Note that this parameter doesn't matter if
            use_queue is False.
        :param int status_time: Time in seconds for how long to wait between printing status
            information. To disable printing of status entirely then set status_time to False.
            Note that this parameter doesn't matter if use_queue is False.
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
            automatically generated seed) or a non-negative integer.
        """
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
        return self.connection.run_and_measure(quil_program=quil_program, qubits=qubits,
                                               trials=trials, random_seed=self.random_seed)

    def run_and_measure_async(self, quil_program, qubits, trials=1):
        """
        Similar to run_and_measure except that it returns a job id and doesn't wait for the
        program to be executed.

        See https://go.rigetti.com/connections for reasons to use this method.
        """
        return self.connection.run_and_measure_async(quil_program=quil_program, qubits=qubits,
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

        return self.connection.wavefunction(quil_program=quil_program,
                                            classical_addresses=classical_addresses,
                                            random_seed=self.random_seed)

    def wavefunction_async(self, quil_program, classical_addresses=None):
        """
        Similar to wavefunction except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        if classical_addresses is None:
            classical_addresses = []

        return self.connection.wavefunction_async(quil_program=quil_program,
                                                  classical_addresses=classical_addresses,
                                                  random_seed=self.random_seed)

    def expectation(self, prep_prog, operator_programs=None):
        """
        Calculate the expectation value of operators given a state prepared by
        prep_program.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            expectation values are computed itself only represents a stochastically generated
            sample. The expectations returned from *different* ``expectation`` calls *will then
            generally be different*.

        This function is deprecated because its API is not very helpful. In particular,
        To measure the expectation of a PauliSum, you probably want to
        do something like this::

                progs, coefs = hamiltonian.get_programs()
                expect_coeffs = np.array(cxn.expectation(prep_program, operator_programs=progs))
                return np.real_if_close(np.dot(coefs, expect_coeffs))

        Please use :py:func:`pauli_expectation` which takes PauliSums directly.

        :param Program prep_prog: Quil program for state preparation.
        :param list operator_programs: A list of Programs, each specifying an operator whose expectation to compute.
            Default is a list containing only the empty Program.
        :return: Expectation values of the operators.
        :rtype: List[float]
        """

        warnings.warn("`expectation()` is deprecated. Use `pauli_expectation`.", DeprecationWarning)
        return self.connection.expectation(prep_prog=prep_prog, operator_programs=operator_programs,
                                           random_seed=self.random_seed)

    def pauli_expectation(self, prep_prog, pauli_terms):
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

        bare_results = self.connection.expectation(prep_prog, progs, random_seed=self.random_seed)
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
        return self.connection.expectation_async(prep_prog=prep_prog,
                                                 operator_programs=operator_programs,
                                                 random_seed=self.random_seed)

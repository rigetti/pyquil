import json

import time
from six import integer_types

from pyquil.api import Job
from pyquil.quil import Program
from pyquil.wavefunction import Wavefunction
from ._base_connection import BaseConnection, validate_noise_probabilities, validate_run_items, TYPE_MULTISHOT, \
    TYPE_MULTISHOT_MEASURE, TYPE_WAVEFUNCTION, TYPE_EXPECTATION


class QVMSyncConnection(BaseConnection):
    """
    The SyncConnection makes a synchronous connection to the Forest API.
    """

    def __init__(self, endpoint='https://api.rigetti.com/qvm', api_key=None, user_id=None,
                 gate_noise=None, measurement_noise=None, random_seed=None):
        super(QVMSyncConnection, self).__init__(endpoint=endpoint, api_key=api_key, user_id=user_id)

        validate_noise_probabilities(gate_noise)
        validate_noise_probabilities(measurement_noise)
        self.gate_noise = gate_noise
        self.measurement_noise = measurement_noise

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, integer_types) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int")

    def ping(self):
        """
        Ping the QVM.
        :return: Should get "pong" back.
        :rtype: string
        """
        payload = {"type": "ping"}
        res = self.post_json(payload, route="")
        return str(res.text)

    def run(self, quil_program, classical_addresses, trials=1):
        """
        Run a Quil program multiple times, accumulating the values deposited in
        a list of classical addresses.

        :param Program quil_program: A Quil program.
        :param list classical_addresses: A list of addresses.
        :param int trials: Number of shots to collect.
        :return: A list of lists of bits. Each sublist corresponds to the values
                 in `classical_addresses`.
        :rtype: list
        """
        payload = self._run_payload(quil_program, classical_addresses, trials)
        response = self.post_json(payload, route="")
        return response.json()

    def _run_payload(self, quil_program, classical_addresses, trials):
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(classical_addresses)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT,
                   "addresses": classical_addresses,
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        self._add_noise_to_payload(payload)
        self._add_rng_seed_to_payload(payload)

        return payload

    def run_and_measure(self, quil_program, qubits, trials=1):
        """
        Run a Quil program once to determine the final wavefunction, and measure multiple times.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the outcomes sampled from *different* ``run_and_measure`` calls *generally sample
            different bitstring distributions*.

        :param Program quil_program: A Quil program.
        :param list qubits: A list of qubits.
        :param int trials: Number of shots to collect.
        :return: A list of a list of bits.
        :rtype: list
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        response = self.post_json(payload, route="")
        return response.json()

    def _run_and_measure_payload(self, quil_program, qubits, trials):
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(qubits)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT_MEASURE,
                   "qubits": qubits,
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        self._add_noise_to_payload(payload)
        self._add_rng_seed_to_payload(payload)

        return payload

    def wavefunction(self, quil_program, classical_addresses=None):
        """
        Simulate a Quil program and get the wavefunction back.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the wavefunctions returned by *different* ``wavefunction`` calls *will generally be
            different*.

        :param Program quil_program: A Quil program.
        :param list classical_addresses: An optional list of classical addresses.
        :return: A tuple whose first element is a Wavefunction object,
                 and whose second element is the list of classical bits corresponding
                 to the classical addresses.
        :rtype: Wavefunction
        """
        payload = self._wavefunction_payload(quil_program, classical_addresses)
        response = self.post_json(payload, route="")
        return Wavefunction.from_bit_packed_string(response.content, classical_addresses)

    def _wavefunction_payload(self, quil_program, classical_addresses):
        if classical_addresses is None:
            classical_addresses = []

        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(classical_addresses)

        payload = {'type': TYPE_WAVEFUNCTION,
                   'quil-instructions': quil_program.out(),
                   'addresses': list(classical_addresses)}

        self._add_noise_to_payload(payload)
        self._add_rng_seed_to_payload(payload)

        return payload

    def expectation(self, prep_prog, operator_programs=None):
        """
        Calculate the expectation value of operators given a state prepared by
        prep_program.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            expectation values are computed itself only represents a stochastically generated
            sample. The expectations returned from *different* ``expectation`` calls *will then
            generally be different*.

        :param Program prep_prog: Quil program for state preparation.
        :param list operator_programs: A list of PauliTerms. Default is Identity operator.
        :returns: Expectation value of the operators.
        :rtype: float
        """
        payload = self._expectation_payload(prep_prog, operator_programs)
        response = self.post_json(payload, route="")
        return response.json()

    def _expectation_payload(self, prep_prog, operator_programs=None):
        if operator_programs is None:
            operator_programs = [Program()]

        if not isinstance(prep_prog, Program):
            raise TypeError("prep_prog variable must be a Quil program object")

        payload = {'type': TYPE_EXPECTATION,
                   'state-preparation': prep_prog.out(),
                   'operators': [x.out() for x in operator_programs]}

        self._add_rng_seed_to_payload(payload)

    def _add_noise_to_payload(self, payload):
        """
        Set the gate noise and measurement noise of a payload.
        """
        if self.measurement_noise is not None:
            payload["measurement-noise"] = self.measurement_noise
        if self.gate_noise is not None:
            payload["gate-noise"] = self.gate_noise

    def _add_rng_seed_to_payload(self, payload):
        """
        Add a random seed to the payload.
        """
        if self.random_seed is not None:
            payload['rng-seed'] = self.random_seed


class QVMConnection(QVMSyncConnection):
    """
    Represents a connection to the QVM.
    """

    def __init__(self, endpoint='https://job.rigetti.com/beta', api_key=None, user_id=None,
                 gate_noise=None, measurement_noise=None, random_seed=None):
        """
        Constructor for QVMConnection. Sets up any necessary security, and establishes the noise
        model to use.

        :param endpoint: The endpoint of the server (default behavior is to read from config file)
        :param api_key: The key to the Forest API Gateway (default behavior is to read from config file)
        :param user_id: Your userid for Forest (default behavior is to read from config file)
        :param gate_noise: A list of three numbers [Px, Py, Pz] indicating the probability of an X,
                           Y, or Z gate getting applied to each qubit after a gate application or
                           reset. (default None)
        :param measurement_noise: A list of three numbers [Px, Py, Pz] indicating the probability of
                                  an X, Y, or Z gate getting applied before a a measurement.
                                  (default None)
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
                            automatically generated seed) or a non-negative integer.
        """
        super(QVMConnection, self).__init__(endpoint=endpoint, api_key=api_key, user_id=user_id,
                                            gate_noise=gate_noise, measurement_noise=measurement_noise,
                                            random_seed=random_seed)

    def ping(self):
        """
        Ping Forest.

        :return: Should get "ok" back.
        :rtype: string
        """
        res = self._session.get(self.endpoint + "/check")
        return str(json.loads(res.text)["rc"])

    def run(self, quil_program, classical_addresses, trials=1):
        """
        Run a Quil program multiple times, accumulating the values deposited in
        a list of classical addresses.

        :param Program quil_program: A Quil program.
        :param list classical_addresses: A list of addresses.
        :param int trials: Number of shots to collect.
        :return: A list of lists of bits. Each sublist corresponds to the values
                 in `classical_addresses`.
        :rtype: list
        """
        payload = self._run_payload(quil_program, classical_addresses, trials)
        response = self.post_json({"machine": "QVM", "program": payload}, route="/job")
        job = self._wait_for_result(self._get_job_id(response))
        return job.result()

    def run_async(self, quil_program, classical_addresses, trials=1):
        payload = self._run_payload(quil_program, classical_addresses, trials)
        response = self.post_json({"machine": "QVM", "program": payload}, route="/job")
        return self._get_job_id(response)

    def run_and_measure(self, quil_program, qubits, trials=1):
        """
        Run a Quil program once to determine the final wavefunction, and measure multiple times.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the outcomes sampled from *different* ``run_and_measure`` calls *generally sample
            different bitstring distributions*.

        :param Program quil_program: A Quil program.
        :param list qubits: A list of qubits.
        :param int trials: Number of shots to collect.
        :return: A list of a list of bits.
        :rtype: list
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        response = self.post_json({"machine": "QVM", "program": payload}, route="/job")
        job = self._wait_for_result(self._get_job_id(response))
        return job.result()

    def run_and_measure_async(self, quil_program, qubits, trials=1):
        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        response = self.post_json({"machine": "QVM", "program": payload}, route="/job")
        return self._get_job_id(response)

    def wavefunction(self, quil_program, classical_addresses=None):
        """
        Simulate a Quil program and get the wavefunction back.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the wavefunctions returned by *different* ``wavefunction`` calls *will generally be
            different*.

        :param Program quil_program: A Quil program.
        :param list classical_addresses: An optional list of classical addresses.
        :return: A tuple whose first element is a Wavefunction object,
                 and whose second element is the list of classical bits corresponding
                 to the classical addresses.
        :rtype: Wavefunction
        """
        payload = self._wavefunction_payload(quil_program, classical_addresses)
        response = self.post_json({"machine": "QVM", "program": payload}, route="/job")
        job = self._wait_for_result(self._get_job_id(response))
        return job.result()

    def wavefunction_async(self, quil_program, classical_addresses=None):
        payload = self._wavefunction_payload(quil_program, classical_addresses)
        response = self.post_json({"machine": "QVM", "program": payload}, route="/job")
        return self._get_job_id(response)

    def expectation(self, prep_prog, operator_programs=None):
        """
        Calculate the expectation value of operators given a state prepared by
        prep_program.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            expectation values are computed itself only represents a stochastically generated
            sample. The expectations returned from *different* ``expectation`` calls *will then
            generally be different*.

        :param Program prep_prog: Quil program for state preparation.
        :param list operator_programs: A list of PauliTerms. Default is Identity operator.
        :returns: Expectation value of the operators.
        :rtype: float
        """
        payload = self._expectation_payload(prep_prog, operator_programs)
        response = self.post_json({"machine": "QVM", "program": payload}, route="/job")
        job = self._wait_for_result(self._get_job_id(response))
        return job.result()

    def get_job(self, job_id):
        response = self.get_json(route="/job/" + job_id)
        return Job(response.json())

    def _get_job_id(self, response):
        return response.json()['jobId']

    def _wait_for_result(self, job_id):
        job = self.get_job(job_id)
        while not job.is_done():
            time.sleep(0.1)
            job = self.get_job(job_id)
        return job


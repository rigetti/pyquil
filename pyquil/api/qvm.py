##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

from six import integer_types

from pyquil.api import Job
from pyquil.quil import Program
from pyquil.wavefunction import Wavefunction
from ._base_connection import validate_noise_probabilities, validate_run_items, TYPE_MULTISHOT, \
    TYPE_MULTISHOT_MEASURE, TYPE_WAVEFUNCTION, TYPE_EXPECTATION, get_job_id, get_session, wait_for_job, \
    post_json, get_json


class QVMConnection(object):
    """
    Represents a connection to the QVM.
    """

    def __init__(self, sync_endpoint='https://api.rigetti.com', async_endpoint='https://job.rigetti.com/beta',
                 api_key=None, user_id=None, use_queue=False, ping_time=0.1, status_time=2,
                 gate_noise=None, measurement_noise=None, random_seed=None):
        """
        Constructor for QVMConnection. Sets up any necessary security, and establishes the noise
        model to use.

        :param sync_endpoint: The endpoint of the server for running small jobs
        :param async_endpoint: The endpoint of the server for running large jobs
        :param api_key: The key to the Forest API Gateway (default behavior is to read from config file)
        :param user_id: Your userid for Forest (default behavior is to read from config file)
        :param bool use_queue: Disabling this parameter may improve performance for small, quick programs.
                               To support larger programs, set it to True. (default: False)
                               *_async methods will always use the queue
                               See https://go.rigetti.com/connections for more information.
        :param int ping_time: Time in seconds for how long to wait between polling the server for updated status
                              information on a job. Note that this parameter doesn't matter if use_queue is False.
        :param int status_time: Time in seconds for how long to wait between printing status information.
                                To disable printing of status entirely then set status_time to False.
                                Note that this parameter doesn't matter if use_queue is False.
        :param gate_noise: A list of three numbers [Px, Py, Pz] indicating the probability of an X,
                           Y, or Z gate getting applied to each qubit after a gate application or
                           reset. (default None)
        :param measurement_noise: A list of three numbers [Px, Py, Pz] indicating the probability of
                                  an X, Y, or Z gate getting applied before a a measurement.
                                  (default None)
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
                            automatically generated seed) or a non-negative integer.
        """
        self.async_endpoint = async_endpoint
        self.sync_endpoint = sync_endpoint
        self.session = get_session(api_key, user_id)

        self.use_queue = use_queue
        self.ping_time = ping_time
        self.status_time = status_time

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
        raise DeprecationWarning("ping() function is deprecated")

    def run(self, quil_program, classical_addresses, trials=1):
        """
        Run a Quil program multiple times, accumulating the values deposited in
        a list of classical addresses.

        :param Program quil_program: A Quil program.
        :param list|range classical_addresses: A list of addresses.
        :param int trials: Number of shots to collect.
        :return: A list of lists of bits. Each sublist corresponds to the values
                 in `classical_addresses`.
        :rtype: list
        """
        payload = self._run_payload(quil_program, classical_addresses, trials)
        if self.use_queue:
            response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = self._run_payload(quil_program, classical_addresses, trials)
            response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
            return response.json()

    def run_async(self, quil_program, classical_addresses, trials=1):
        """
        Similar to run except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._run_payload(quil_program, classical_addresses, trials)
        response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
        return get_job_id(response)

    def _run_payload(self, quil_program, classical_addresses, trials):
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(classical_addresses)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT,
                   "addresses": list(classical_addresses),
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
        :param list|range qubits: A list of qubits.
        :param int trials: Number of shots to collect.
        :return: A list of a list of bits.
        :rtype: list
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        if self.use_queue:
            response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = self._run_and_measure_payload(quil_program, qubits, trials)
            response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
            return response.json()

    def run_and_measure_async(self, quil_program, qubits, trials=1):
        """
        Similar to run_and_measure except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
        return get_job_id(response)

    def _run_and_measure_payload(self, quil_program, qubits, trials):
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(qubits)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT_MEASURE,
                   "qubits": list(qubits),
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
        :param list|range classical_addresses: An optional list of classical addresses.
        :return: A tuple whose first element is a Wavefunction object,
                 and whose second element is the list of classical bits corresponding
                 to the classical addresses.
        :rtype: Wavefunction
        """
        if classical_addresses is None:
            classical_addresses = []

        if self.use_queue:
            payload = self._wavefunction_payload(quil_program, classical_addresses)
            response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = self._wavefunction_payload(quil_program, classical_addresses)
            response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
            return Wavefunction.from_bit_packed_string(response.content, classical_addresses)

    def wavefunction_async(self, quil_program, classical_addresses=None):
        """
        Similar to wavefunction except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        if classical_addresses is None:
            classical_addresses = []

        payload = self._wavefunction_payload(quil_program, classical_addresses)
        response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
        return get_job_id(response)

    def _wavefunction_payload(self, quil_program, classical_addresses):
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
        if self.use_queue:
            payload = self._expectation_payload(prep_prog, operator_programs)
            response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = self._expectation_payload(prep_prog, operator_programs)
            response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
            return response.json()

    def expectation_async(self, prep_prog, operator_programs=None):
        """
        Similar to expectation except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._expectation_payload(prep_prog, operator_programs)
        response = post_json(self.session, self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
        return get_job_id(response)

    def _expectation_payload(self, prep_prog, operator_programs):
        if operator_programs is None:
            operator_programs = [Program()]

        if not isinstance(prep_prog, Program):
            raise TypeError("prep_prog variable must be a Quil program object")

        payload = {'type': TYPE_EXPECTATION,
                   'state-preparation': prep_prog.out(),
                   'operators': [x.out() for x in operator_programs]}

        self._add_rng_seed_to_payload(payload)

        return payload

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

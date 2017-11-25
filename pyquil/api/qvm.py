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

import json

from six import integer_types

from pyquil.quil import Program
from pyquil.wavefunction import Wavefunction
from ._base_connection import BaseConnection, validate_noise_probabilities, validate_run_items, TYPE_MULTISHOT, \
    TYPE_MULTISHOT_MEASURE, TYPE_WAVEFUNCTION, TYPE_EXPECTATION, get_job_id


class QVMConnection(BaseConnection):
    """
    Represents a connection to the QVM.
    """

    def __init__(self, sync_endpoint=None, async_endpoint=None, api_key=None, user_id=None,
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
        if not sync_endpoint:
            sync_endpoint = 'https://api.rigetti.com'
        if not async_endpoint:
            async_endpoint = 'https://job.rigetti.com/beta'

        super(QVMConnection, self).__init__(async_endpoint=async_endpoint, api_key=api_key, user_id=user_id)
        self.sync_endpoint = sync_endpoint

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

    def run(self, quil_program, classical_addresses, trials=1, use_queue=False):
        """
        Run a Quil program multiple times, accumulating the values deposited in
        a list of classical addresses.

        :param Program quil_program: A Quil program.
        :param list classical_addresses: A list of addresses.
        :param int trials: Number of shots to collect.
        :param bool use_queue: Disabling this parameter may improve performance for small, quick programs.
                               To support larger programs, set it to True. (default: False)
                               See https://go.rigetti.com/connections for more information.
        :return: A list of lists of bits. Each sublist corresponds to the values
                 in `classical_addresses`.
        :rtype: list
        """
        payload = self._run_payload(quil_program, classical_addresses, trials)
        if use_queue:
            response = self._post_json(self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = self._run_payload(quil_program, classical_addresses, trials)
            response = self._post_json(self.sync_endpoint + "/qvm", payload)
            return response.json()

    def run_async(self, quil_program, classical_addresses, trials=1):
        """
        Similar to run except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._run_payload(quil_program, classical_addresses, trials)
        response = self._post_json(self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
        return get_job_id(response)

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

    def run_and_measure(self, quil_program, qubits, trials=1, use_queue=False):
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
        :param bool use_queue: Disabling this parameter may improve performance for small, quick programs.
                               To support larger programs, set it to True. (default: False)
                               See https://go.rigetti.com/connections for more information.
        :return: A list of a list of bits.
        :rtype: list
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        if use_queue:
            response = self._post_json(self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = self._run_and_measure_payload(quil_program, qubits, trials)
            response = self._post_json(self.sync_endpoint + "/qvm", payload)
            return response.json()

    def run_and_measure_async(self, quil_program, qubits, trials=1):
        """
        Similar to run_and_measure except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        response = self._post_json(self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
        return get_job_id(response)

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

    def wavefunction(self, quil_program, classical_addresses=None, use_queue=False):
        """
        Simulate a Quil program and get the wavefunction back.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the wavefunctions returned by *different* ``wavefunction`` calls *will generally be
            different*.

        :param Program quil_program: A Quil program.
        :param list classical_addresses: An optional list of classical addresses.
        :param bool use_queue: Disabling this parameter may improve performance for small, quick programs.
                               To support larger programs, set it to True. (default: False)
                               See https://go.rigetti.com/connections for more information.
        :return: A tuple whose first element is a Wavefunction object,
                 and whose second element is the list of classical bits corresponding
                 to the classical addresses.
        :rtype: Wavefunction
        """
        if classical_addresses is None:
            classical_addresses = []

        if use_queue:
            payload = self._wavefunction_payload(quil_program, classical_addresses)
            response = self._post_json(self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = self._wavefunction_payload(quil_program, classical_addresses)
            response = self._post_json(self.sync_endpoint + "/qvm", payload)
            return Wavefunction.from_bit_packed_string(response.content, classical_addresses)

    def wavefunction_async(self, quil_program, classical_addresses=None):
        """
        Similar to wavefunction except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        if classical_addresses is None:
            classical_addresses = []

        payload = self._wavefunction_payload(quil_program, classical_addresses)
        response = self._post_json(self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
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

    def expectation(self, prep_prog, operator_programs=None, use_queue=False):
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
        :param bool use_queue: Disabling this parameter may improve performance for small, quick programs.
                               To support larger programs, set it to True. (default: False)
                               See https://go.rigetti.com/connections for more information.
        :returns: Expectation value of the operators.
        :rtype: float
        """
        if use_queue:
            payload = self._expectation_payload(prep_prog, operator_programs)
            response = self._post_json(self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
            job = self.wait_for_job(get_job_id(response))
            return job.result()
        else:
            payload = self._expectation_payload(prep_prog, operator_programs)
            response = self._post_json(self.sync_endpoint + "/qvm", payload)
            return response.json()

    def expectation_async(self, prep_prog, operator_programs=None):
        """
        Similar to expectation except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._expectation_payload(prep_prog, operator_programs)
        response = self._post_json(self.async_endpoint + "/job", {"machine": "QVM", "program": payload})
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

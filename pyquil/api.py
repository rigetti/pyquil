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
"""
Module for facilitating connections to the QVM / QPU.
"""

from __future__ import print_function
from requests.adapters import HTTPAdapter
from six import integer_types
from urllib3.util.retry import Retry
import json
import requests
import sys

from pyquil.config import PyquilConfig
from pyquil.job_results import JobResult, WavefunctionResult, recover_complexes
from pyquil.quil import Program


def add_noise_to_payload(payload, gate_noise, measurement_noise):
    """
    Set the gate noise and measurement noise of a payload.

    :param dict payload: Dictionary of run information.
    :param gate_noise: Probability of a noise gate being applied at each time step.
    :param measurement_noise: Probability of a noise gate being applied before measurement.
    """
    if gate_noise is not None:
        payload["gate-noise"] = gate_noise
    if measurement_noise is not None:
        payload["measurement-noise"] = measurement_noise


def add_rng_seed_to_payload(payload, seed):
    """
    Add a random seed to the payload.

    :param payload: JSON payload.
    :param int seed: A non-negative integer.
    """
    if seed is not None:
        payload['rng-seed'] = seed


def _validate_noise_probabilities(noise_parameter):
    """
    Is noise_parameter a valid specification of noise probabilities for depolarizing noise?

    :param list noise_parameter: List of noise parameter values to be validated.
    """
    if not noise_parameter:
        return
    if not isinstance(noise_parameter, list):
        raise TypeError("noise_parameter must be a list")
    if any([not isinstance(value, float) for value in noise_parameter]):
        raise TypeError("noise_parameter values should all be floats")
    if len(noise_parameter) != 3:
        raise ValueError("noise_parameter lists must be of length 3")
    if sum(noise_parameter) > 1 or sum(noise_parameter) < 0:
        raise ValueError("sum of entries in noise_parameter must be between 0 and 1 (inclusive)")
    if any([value < 0 for value in noise_parameter]):
        raise ValueError("noise_parameter values should all be non-negative")


def validate_run_items(run_items):
    """
    Check the validity of classical addresses / qubits for the payload.

    :param list run_items: List of classical addresses or qubits to be validated.
    """
    if not isinstance(run_items, list):
        raise TypeError("run_items must be a list")
    if any([not isinstance(i, integer_types) for i in run_items]):
        raise TypeError("run_items list must contain integer values")


TYPE_EXPECTATION = "expectation"
TYPE_MULTISHOT = "multishot"
TYPE_MULTISHOT_MEASURE = "multishot-measure"
TYPE_WAVEFUNCTION = "wavefunction"


class JobConnection(object):
    """
    Represents a connection to the QVM.
    """

    #################################################################################
    # PS: WELCOME TO THE SOURCE! Do you want fast-tracked access to a Rigetti API key?
    #    Email us at batman@rigetti.com and tell us what you'd like to do.  We're looking
    #    for people like you to join our team: https://jobs.lever.co/rigetti
    #################################################################################

    def __init__(self, endpoint='https://job.rigetti.com/beta', api_key=None, user_id=None, gate_noise=None,
                 measurement_noise=None, num_retries=3, random_seed=None):
        """
        Constructor for JobConnection. Sets up any necessary security, and establishes the noise
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
        :param num_retries: The number of times to retry a request when faced with an HTTP response
                            code in [502, 503, 504, 521, 523] (default 3)
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
                            automatically generated seed) or a non-negative integer.
        """
        # Once these are set, they should not ever be cleared/changed/touched.
        # Make a new JobConnection() if you need that.
        _validate_noise_probabilities(gate_noise)
        _validate_noise_probabilities(measurement_noise)
        self.gate_noise = gate_noise
        self.measurement_noise = measurement_noise

        self.session = requests.Session()

        # We need this to get binary payload for the wavefunction call.
        self.session.headers.update({"Accept": "application/octet-stream"})

        self.num_retries = num_retries
        retry_adapter = HTTPAdapter(max_retries=Retry(total=num_retries,
                                                      method_whitelist=['POST'],
                                                      status_forcelist=[502, 503, 504, 521, 523],
                                                      backoff_factor=0.2,
                                                      raise_on_status=False))
        self.session.mount("http://", retry_adapter)
        self.session.mount("https://", retry_adapter)

        config = PyquilConfig()
        self.endpoint = endpoint
        self.api_key = api_key if api_key else config.api_key
        self.user_id = user_id if user_id else config.user_id
        self.headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'x-api-key': self.api_key,
            'x-user-id': self.user_id,
        }

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, integer_types) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int or long.")

        self.machine = 'QVM'  # default to a QVM Machine Connection

    def post_json(self, jd, headers=None, route=""):
        """
        Post JSON to the Forest endpoint.

        :param dict jd: JSON.
        :param dict headers: The headers for the post request.
        :param str route: The route to append to the endpoint, e.g. "/job"
        :return: A non-error response.
        """
        if headers is None:
            headers = self.headers
        url = self.endpoint + route
        res = self.session.post(url, json=jd, headers=headers)

        # Print some nice info for unauthorized/permission errors.
        if res.status_code == 401 or res.status_code == 403:
            print("! ERROR:\n"
                  "!   There was an issue validating your forest account.\n"
                  "!   Have you run the pyquil-config-setup command yet?\n"
                  "! The server came back with the following information:\n"
                  "%s\n%s\n%s" % ("=" * 80, res.text, "=" * 80),
                  file=sys.stderr)
            print("! If you suspect this to be a bug in pyQuil or Rigetti Forest,\n"
                  "! then please describe the problem in a GitHub issue at:\n!\n"
                  "!      https://github.com/rigetticomputing/pyquil/issues\n",
                  file=sys.stderr)

        # Print some nice info for invalid input or internal server errors.
        if res.status_code == 400 or res.status_code >= 500:
            print("! ERROR:\n"
                  "!   Server caught an error. This could be due to a bug in the server\n"
                  "!   or a bug in your code. The server came back with the following\n"
                  "!   information:\n"
                  "%s\n%s\n%s" % ("=" * 80, res.text, "=" * 80),
                  file=sys.stderr)
            print("! If you suspect this to be a bug in pyQuil or Rigetti Forest,\n"
                  "! then please describe the problem in a GitHub issue at:\n!\n"
                  "!      https://github.com/rigetticomputing/pyquil/issues\n",
                  file=sys.stderr)

        res.raise_for_status()
        return res

    def wrap_payload_into_message(self, payload):
        """
        Wraps the payload into a Forest query.

        :param dict payload:
        :return: A JSON dictionary to post as a Forest query.
        """
        message = {
            'machine': self.machine,
            'program': payload,
            'userId': self.user_id,
        }
        return message

    def post_job(self, program, headers=None):
        """
        Post a Job to the Forest endpoint.

        :param program:
        :param headers:
        :return: A non-error response.
        """
        message = self.wrap_payload_into_message(program)
        return self.post_json(message, headers, route="/job")

    def get_job(self, job_result):
        """
        :param JobResult job_result:
        :return: fills in the result in the JobResult object
        """
        url = self.endpoint + ("/job/%s" % (job_result.job_id()))
        res = requests.get(url, headers=self.headers)
        result = res.json()
        return job_result._update(res.ok, result)

    def ping(self):
        """
        Ping Forest.

        :return: Should get "ok" back.
        :rtype: string
        """
        res = self.session.get(self.endpoint + "/check")
        return str(json.loads(res.text)["rc"])

    def version(self):
        """
        Query the QVM version.

        :return: None
        """
        raise DeprecationWarning("Version checks have been deprecated.")

    def process_response(self, res):
        """
        :param res: A response object from a request
        :return: A JobResult filled in with the response data
        :rtype: JobResult
        """
        return JobResult.load_res(self, res)

    def process_wavefunction_response(self, res, payload):
        """
        Wavefunctions are processed differently as they are byte encoded.
        :param res: A response object from a request
        :param payload: The payload that was used to make that request
        :return: A WavefunctionResult with the response data filled in
        :rtype: WavefunctionResult
        """
        result = res.json()
        return WavefunctionResult(self, res.ok, result=result, payload=payload)

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
        :rtype: tuple
        """
        if classical_addresses is None:
            classical_addresses = []

        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(classical_addresses)

        payload = {'type': TYPE_WAVEFUNCTION,
                   'quil-instructions': quil_program.out(),
                   'addresses': list(classical_addresses)}
        add_noise_to_payload(payload, self.gate_noise, self.measurement_noise)
        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_job(payload, headers=self.headers)
        return self.process_wavefunction_response(res, payload)

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
        if operator_programs is None:
            operator_programs = [Program()]

        if not isinstance(prep_prog, Program):
            raise TypeError("prep_prog variable must be a Quil program object")

        payload = {'type': TYPE_EXPECTATION,
                   'state-preparation': prep_prog.out(),
                   'operators': [x.out() for x in operator_programs]}

        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_job(payload, headers=self.headers)
        return self.process_response(res)

    def bit_string_probabilities(self, quil_program):
        """
        Simulate a Quil program and get outcome probabilities back.

        :param Program quil_program: A Quil program.
        :return: A dictionary with outcomes as keys and probabilities as values.
        :rtype: dict
        """
        return DeprecationWarning, "This function is deprecated."

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
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(classical_addresses)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT,
                   "addresses": classical_addresses,
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        add_noise_to_payload(payload, self.gate_noise, self.measurement_noise)
        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_job(payload, headers=self.headers)
        return self.process_response(res)

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
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(qubits)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT_MEASURE,
                   "qubits": qubits,
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        add_noise_to_payload(payload, self.gate_noise, self.measurement_noise)
        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_job(payload, headers=self.headers)
        return self.process_response(res)


class SyncConnection(JobConnection):
    """
    The SyncConnection makes a synchronous connection to the Forest API.
    """

    def __init__(self, endpoint='https://api.rigetti.com/qvm', **kwargs):
        super(self.__class__, self).__init__(endpoint=endpoint, **kwargs)

    def ping(self):
        """
        Ping the QVM.
        :return: Should get "pong" back.
        :rtype: string
        """
        payload = {"type": "ping"}
        res = self.post_json(payload)
        return str(res.text)

    def version(self):
        """
        Query the QVM version.
        :return: The current version of the QVM.
        :rtype: string
        """
        payload = {"type": "version"}
        res = self.post_json(payload)
        return str(res.text)

    def post_job(self, program, headers=None):
        """
        Post a synchronous Job to the QVM endpoint.

        :param program:
        :param headers:
        :return: A non-error response.
        """
        return self.post_json(program, headers)

    def process_response(self, res):
        """
        :param res: A response object from a request
        :return: The json dictionary of the response
        :rtype: dict
        """
        return json.loads(res.text)

    def process_wavefunction_response(self, res, payload):
        """
        :param res: A response object from a request
        :return: The json dictionary of the response
        :rtype: dict
        """
        return recover_complexes(res.content, classical_addresses=payload['addresses'])

    def get_job(self, job_result):
        raise NotImplementedError

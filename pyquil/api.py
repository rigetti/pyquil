#!/usr/bin/python
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
from requests.packages.urllib3.util import Retry
from requests.adapters import HTTPAdapter
from copy import deepcopy
import requests
import json
import os
import os.path
import sys
from six.moves.configparser import ConfigParser, NoOptionError, NoSectionError
from six.moves import range
from six import integer_types

import pyquil.quil as pq
from pyquil.job_results import JobResult, WavefunctionResult, recover_complexes

USER_HOMEDIR = os.path.expanduser("~")

PYQUIL_CONFIG_PATH = os.getenv('PYQUIL_CONFIG', os.path.join(USER_HOMEDIR, ".pyquil_config"))
PYQUIL_CONFIG = ConfigParser()

try:
    if "~" in PYQUIL_CONFIG_PATH:
        raise RuntimeError("PYQUIL_CONFIG enviroment variable contains `~`. Use $HOME instead.")
    if len(PYQUIL_CONFIG.read(PYQUIL_CONFIG_PATH)) == 0:
        raise RuntimeError("Error locating config file")
except:
    print("! WARNING:\n"
          "!   There was an error reading your pyQuil config file.\n"
          "!   Make sure you have a .pyquil_config file either in\n"
          "!   your home directory, or you have the environment\n"
          "!   variable PYQUIL_CONFIG set to a valid path. You must\n"
          "!   have permissions to read this file.", file=sys.stderr)


def config_value(name, default=None):
    """
    Get a config file value for a particular name

    :param name: The key.
    :param default: A default if it's not found.
    :return: The value.
    """
    SECTION = "Rigetti Forest"
    try:
        return PYQUIL_CONFIG.get(SECTION, name)
    except (NoSectionError, NoOptionError, KeyError):
        return default


def env_or_config(env, name, default=None):
    """
    Get the value of the environment variable or config file value.
    The environment variable takes precedence.

    :param env: The environment variable name.
    :param name: The config file key.
    :param default: The default value to use.
    :return: The value.
    """
    env_val = os.getenv(env, None)
    if env_val is not None:
        return env_val

    config_val = config_value(name, default=None)
    if config_val is not None:
        return config_val

    return default

# Set up the configuration.
ENDPOINT = env_or_config('QVM_URL', 'url', default='https://api.rigetti.com/qvm')
API_KEY = env_or_config('QVM_API_KEY', 'key')
USER_ID = env_or_config("QVM_USER_ID", 'user_id')


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


def _validate_run_items(run_items):
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

    def __init__(self, endpoint=ENDPOINT, api_key=API_KEY, user_id=USER_ID, gate_noise=None,
                 measurement_noise=None, num_retries=3, random_seed=None):
        """
        Constructor for JobConnection. Sets up any necessary security, and establishes the noise
        model to use.

        :param endpoint: The endpoint of the server (default ENDPOINT)
        :param api_key: The key to the Forest API Gateway (default QVM_API_KEY)
        :param user_id: Your userid for Forest (default QVM_USER_ID)
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

        self.endpoint = endpoint
        self.api_key = api_key
        self.user_id = user_id
        self.json_headers = None
        self.text_headers = None
        if self.api_key or self.user_id:
            self.json_headers = {
                'Content-Type': 'application/json; charset=utf-8',
                'x-api-key': self.api_key,
                'x-user-id': self.user_id,
            }
            self.text_headers = deepcopy(self.json_headers)
            self.text_headers['Content-Type'] = 'application/text; charset=utf-8'

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, integer_types) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int or long.")

        self.machine = 'QVM' # default to a QVM Machine Connection

    def post_json(self, jd, headers=None, route=""):
        """
        Post JSON to the Forest endpoint.

        :param dict jd: JSON.
        :param dict headers: The headers for the post request.
        :param: str route: The route to append to the endpoint, e.g. "/job"
        :return: A non-error response.
        """
        if headers is None:
            headers = self.json_headers
        url = self.endpoint + route
        res = self.session.post(url, json=jd, headers=headers)

        # Print some nice info for internal server errors.
        if res.status_code == 500:
            print("! Server caught an error. This could be due to a bug in the server\n"
                  "! or a bug in your code. The server came back with the following\n"
                  "! information:\n"
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
            'jobId': '',
            'results': '',
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
        res = requests.get(url, headers=self.text_headers)
        result = json.loads(res.content.decode("utf-8"))
        return job_result._update(res.ok, result)

    def ping(self):
        """
        Ping Forest.

        :return: Should get "ok" back.
        :rtype: string
        """
        res = self.session.get(self.endpoint+"/check")
        return str(json.loads(res.text)["rc"])

    def version(self):
        """
        Query the QVM version.

        :return: The current version of the QVM.
        :rtype: string
        """
        raise DeprecationWarning, "Version checks have been deprecated."
        return None

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
        result = json.loads(res.content.decode("utf-8"))
        return WavefunctionResult(self, res.ok, result=result, payload=payload)

    def wavefunction(self, quil_program, classical_addresses=None):
        """
        Simulate a Quil program and get the wavefunction back.

        :param Program quil_program: A Quil program.
        :param list classical_addresses: An optional list of classical addresses.
        :return: A tuple whose first element is a Wavefunction object,
                 and whose second element is the list of classical bits corresponding
                 to the classical addresses.
        :rtype: tuple
        """
        if classical_addresses is None:
            classical_addresses = []

        if not isinstance(quil_program, pq.Program):
            raise TypeError("quil_program must be a Quil program object")
        _validate_run_items(classical_addresses)

        payload = {'type': TYPE_WAVEFUNCTION,
                   'quil-instructions': quil_program.out(),
                   'addresses': classical_addresses}
        add_noise_to_payload(payload, self.gate_noise, self.measurement_noise)
        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_job(payload, headers=self.json_headers)
        return self.process_wavefunction_response(res, payload)

    def expectation(self, prep_prog, operator_programs=None):
        """
        Calculate the expectation value of operators given a state prepared by
        prep_program.

        :param Program prep_prog: Quil program for state preparation.
        :param list operators: A list of PauliTerms. Default is Identity operator.
        :returns: Expectation value of the operators.
        :rtype: float
        """
        if operator_programs is None:
            operator_programs = [pq.Program()]

        if not isinstance(prep_prog, pq.Program):
            raise TypeError("prep_prog variable must be a Quil program object")

        payload = {'type': TYPE_EXPECTATION,
                   'state-preparation': prep_prog.out(),
                   'operators': [x.out() for x in operator_programs]}

        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_job(payload, headers=self.json_headers)
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
        if not isinstance(quil_program, pq.Program):
            raise TypeError("quil_program must be a Quil program object")
        _validate_run_items(classical_addresses)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT,
                   "addresses": classical_addresses,
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        add_noise_to_payload(payload, self.gate_noise, self.measurement_noise)
        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_job(payload, headers=self.json_headers)
        return self.process_response(res)

    def run_and_measure(self, quil_program, qubits, trials=1):
        """
        Run a Quil program once to determine the final wavefunction, and measure multiple times.

        :param Program quil_program: A Quil program.
        :param list qubits: A list of qubits.
        :param int trials: Number of shots to collect.
        :return: A list of a list of bits.
        :rtype: list
        """
        if not isinstance(quil_program, pq.Program):
            raise TypeError("quil_program must be a Quil program object")
        _validate_run_items(qubits)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT_MEASURE,
                   "qubits": qubits,
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        add_noise_to_payload(payload, self.gate_noise, self.measurement_noise)
        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_job(payload, headers=self.json_headers)
        return self.process_response(res)


class SyncConnection(JobConnection):
    """
    The SyncConnection makes a synchronous connection to the Forest API.
    """

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

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
import requests
import json
import os
import os.path
import numpy as np
import sys
import struct
import pyquil.quil as pq
import ConfigParser

USER_HOMEDIR = os.path.expanduser("~")

PYQUIL_CONFIG_PATH = os.getenv('PYQUIL_CONFIG', os.path.join(USER_HOMEDIR, ".pyquil_config"))
PYQUIL_CONFIG = ConfigParser.ConfigParser()

try:
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
    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError, KeyError):
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
HTTPS_CERT = env_or_config('QVM_HTTPS_CERT', 'https_cert')
HTTPS_KEY = env_or_config('QVM_HTTPS_KEY', 'https_key')


def certificate(cert=HTTPS_CERT, key=HTTPS_KEY):
    """
    Return information about the location of the client certificate. This is used for
    HTTPS authentication with the Requests library.

    :param cert: Certificate file or None (default HTTPS_CERT)
    :param key: Key file or None (default HTTPS_KEY)
    :return: Either None or a certificate file or a tuple of certificate and key files.
    """
    if cert is None:
        return None
    elif key is None:
        return cert
    else:
        return cert, key


def add_noise_to_payload(payload, gate_noise, measurement_noise):
    """
    Set the gate noise and measurement noise of a payload.
    :param payload: Dictionary of run information.
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
    :param seed: A non-negative integer.
    """
    if seed is not None:
        payload['rng-seed'] = seed


def _validate_noise_probabilities(noise_parameter):
    """
    Is noise_parameter a valid specification of noise probabilities for depolarizing noise?
    :param noise_parameter: List of noise parameter values to be validated.
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
    :param run_items: List of classical addresses or qubits to be validated.
    """
    if not isinstance(run_items, list):
        raise TypeError("run_items must be a list")
    if any([not isinstance(i, int) for i in run_items]):
        raise TypeError("run_items list must contain integer values")


def _round_to_next_multiple(n, m):
    """
    Round up the the next multiple.
    :param n: The number to round up.
    :param m: The multiple.
    :return: The rounded number
    """
    return n if n % m == 0 else n + m - n % m


def _octet_bits(o):
    """
    Get the bits of an octet.
    :param o: The octets.
    :return: The bits as a list in LSB-to-MSB order.
    """
    if not isinstance(o, (int, long)):
        raise TypeError("o should be an int or long")
    if not (0 <= o <= 255):
        raise ValueError("o should be between 0 and 255 inclusive")
    bits = [0] * 8
    for i in xrange(8):
        if 1 == o & 1:
            bits[i] = 1
        o = o >> 1
    return bits


OCTETS_PER_DOUBLE_FLOAT = 8
OCTETS_PER_COMPLEX_DOUBLE = 2 * OCTETS_PER_DOUBLE_FLOAT

TYPE_EXPECTATION = "expectation"
TYPE_MULTISHOT = "multishot"
TYPE_MULTISHOT_MEASURE = "multishot-measure"
TYPE_WAVEFUNCTION = "wavefunction"


class Connection(object):
    """
    Represents a connection to the QVM.
    """

    #################################################################################
    # PS: WELCOME TO THE SOURCE! Do you want fast-tracked access to a Rigetti API key?
    #    Email us at batman@rigetti.com and tell us what you'd like to do.  We're looking
    #    for people like you to join our team: https://jobs.lever.co/rigetti
    #################################################################################

    def __init__(self, endpoint=ENDPOINT, cert=HTTPS_CERT, key=HTTPS_KEY, api_key=API_KEY,
                 gate_noise=None, measurement_noise=None, num_retries=5, random_seed=None):
        """
        Constructor for Connection. Sets up any necessary security, and establishes the noise model
        to use.

        :param endpoint: The endpoint of the server (default ENDPOINT)
        :param cert: The certificate file or None (default HTTPS_CERT)
        :param key: The key file or None (default HTTPS_KEY)
        :param gate_noise: A list of three numbers [Px, Py, Pz] indicating the probability of an X,
                           Y, or Z gate getting applied to each qubit after a gate application or
                           reset. (default None)
        :param measurement_noise: A list of three numbers [Px, Py, Pz] indicating the probability of
                                  an X, Y, or Z gate getting applied before a a measurement.
                                  (default None)
        :param num_retries: The number of times to retry a request when faced with an HTTP response
                            code in [502, 503, 504, 521, 523] (default 5)
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
                            automatically generated seed) or a non-negative integer.
        """
        _validate_noise_probabilities(gate_noise)
        _validate_noise_probabilities(measurement_noise)
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
        self.certificate_file = cert
        self.key_file = key
        self.cached_certificate = certificate(cert, key)
        self.api_key = api_key
        if self.api_key:
            self.session.headers.update({'x-api-key': self.api_key})
        # Once these are set, they should not ever be cleared/changed/touched.
        # Make a new Connection() if you need that.
        self.gate_noise = gate_noise
        self.measurement_noise = measurement_noise

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, (int, long)) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int or long.")

    def post_json(self, j):
        """
        Post JSON to the QVM endpoint.
        :param j: JSON.
        :return: A non-error response.
        """

        if self.cached_certificate is None:
            res = self.session.post(self.endpoint, json=j)
        else:
            res = self.session.post(self.endpoint, json=j, cert=self.cached_certificate)

        # Print some nice info for internal server errors.
        if res.status_code == 500:
            print("! Server caught an error. This could be due to a bug in the server\n"
                  "! or a bug in your code. The server came back with the following\n"
                  "! information:\n"
                  "%s\n%s\n%s\n" % ("=" * 80, res.text, "=" * 80),
                  file=sys.stderr)

        res.raise_for_status()

        return res

    def ping(self):
        """
        Ping the QVM.
        :return: Should get "pong" back.
        """
        payload = {"type": "ping"}
        res = self.post_json(payload)
        return str(res.text)

    def version(self):
        """
        Query the QVM version.
        :return: The current version of the QVM.
        """
        payload = {"type": "version"}
        res = self.post_json(payload)
        return str(res.text)

    def wavefunction(self, quil_program, classical_addresses=[]):
        """
        Simulate a Quil program and get the wavefunction back.
        :param quil_program: A Quil program.
        :param classical_addresses: An optional list of classical addresses.
        :return: A tuple whose first element is a a NumPy array of amplitudes,
                 and whose second element is the list of classical bits corresponding
                 to the classical addresses.
        """

        def recover_complexes(coef_string):
            num_octets = len(coef_string)
            num_addresses = len(classical_addresses)
            num_memory_octets = _round_to_next_multiple(num_addresses, 8) / 8
            num_wavefunction_octets = num_octets - num_memory_octets

            # Parse the classical memory
            mem = []
            for i in xrange(num_memory_octets):
                octet = struct.unpack('B', coef_string[i])[0]
                mem.extend(_octet_bits(octet))

            mem = mem[0:num_addresses]

            # Parse the wavefunction
            wf = np.zeros(num_wavefunction_octets / OCTETS_PER_COMPLEX_DOUBLE, dtype=np.cfloat)
            for i, p in enumerate(xrange(num_memory_octets, num_octets, OCTETS_PER_COMPLEX_DOUBLE)):
                re_be = coef_string[p: p + OCTETS_PER_DOUBLE_FLOAT]
                im_be = coef_string[p + OCTETS_PER_DOUBLE_FLOAT: p + OCTETS_PER_COMPLEX_DOUBLE]
                re = struct.unpack('>d', re_be)[0]
                im = struct.unpack('>d', im_be)[0]
                wf[i] = complex(re, im)

            return wf, mem

        if not isinstance(quil_program, pq.Program):
            raise TypeError("quil_program must be a Quil program object")
        _validate_run_items(classical_addresses)

        payload = {'type': TYPE_WAVEFUNCTION,
                   'quil-instructions': quil_program.out(),
                   'addresses': classical_addresses}
        add_noise_to_payload(payload, self.gate_noise, self.measurement_noise)
        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_json(payload)

        return recover_complexes(res.content)

    def expectation(self, prep_prog, operator_programs=[pq.Program()]):
        """
        Calculate the expectation value of operators given a state prepared by
        prep_program.
        :params prep_prog: Quil program for state preparation.
        :params operators: (list) of PauliTerms. Default is Identity operator.
        :returns: float expectation value of the operators.
        """
        if not isinstance(prep_prog, pq.Program):
            raise TypeError("prep_prog variable must be a Quil program object")

        payload = {'type': TYPE_EXPECTATION,
                   'state-preparation': prep_prog.out(),
                   'operators': map(lambda x: x.out(), operator_programs)}

        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_json(payload)
        result_overlaps = json.loads(res.text)

        return result_overlaps

    def bit_string_probabilities(self, quil_program):
        """
        Simulate a Quil program and get outcome probabilities back.
        :param quil_program: A Quil program.
        :return: A dict with outcomes as keys and probabilities as values.
        """
        wvf, _ = self.wavefunction(quil_program)
        return get_outcome_probs(wvf)

    def run(self, quil_program, classical_addresses, trials=1):
        """
        Run a Quil program multiple times, accumulating the values deposited in
        a list of classical addresses.
        :param quil_program: A Quil program.
        :param classical_addresses: A list of addresses.
        :param trials: Number of shots to collect.
        :return: A list of lists of bits. Each sublist corresponds to the values
        in `classical_addresses`.
        """
        if not isinstance(quil_program, pq.Program):
            raise TypeError("quil_program must be a Quil program object")
        _validate_run_items(classical_addresses)
        if not isinstance(trials, int):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT,
                   "addresses": classical_addresses,
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        add_noise_to_payload(payload, self.gate_noise, self.measurement_noise)
        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_json(payload)

        return json.loads(res.text)

    def run_and_measure(self, quil_program, qubits, trials=1):
        """
        Run a Quil program once to determine the final wavefunction, and measure multiple times.
        :param quil_program: A Quil program.
        :param qubits: A list of qubits.
        :param trials: Number of shots to collect.
        :return: A list of a list of bits.
        """
        if not isinstance(quil_program, pq.Program):
            raise TypeError("quil_program must be a Quil program object")
        _validate_run_items(qubits)
        if not isinstance(trials, int):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT_MEASURE,
                   "qubits": qubits,
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        add_noise_to_payload(payload, self.gate_noise, self.measurement_noise)
        add_rng_seed_to_payload(payload, self.random_seed)

        res = self.post_json(payload)

        return json.loads(res.text)


def get_outcome_probs(wvf):
    """
    Parses a wavefunction (array of complex amplitudes) and returns a dictionary of
    outcomes and associated probabilities.
    :param wvf: A complex list of amplitudes.
    :return: A dict with outcomes as keys and probabilities as values.
    """
    outcome_dict = {}
    qubit_num = len(wvf).bit_length() - 1
    for index, amplitude in enumerate(wvf):
        outcome = bin(index)[2:].rjust(qubit_num, '0')
        outcome_dict[outcome] = abs(amplitude) ** 2
    return outcome_dict

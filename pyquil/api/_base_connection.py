##############################################################################
# Copyright 2016-2018 Rigetti Computing
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
from __future__ import print_function

import re
import warnings
from json.decoder import JSONDecodeError
from typing import Dict, Union, Sequence

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from six import integer_types
from urllib3 import Retry

from pyquil import Program
from pyquil.api._config import PyquilConfig
from pyquil.api._error_reporting import _record_call
from pyquil.api._errors import error_mapping, UnknownApiError, TooManyQubitsError
from pyquil.device import Specs, ISA
from pyquil.wavefunction import Wavefunction

TYPE_EXPECTATION = "expectation"
TYPE_MULTISHOT = "multishot"
TYPE_MULTISHOT_MEASURE = "multishot-measure"
TYPE_WAVEFUNCTION = "wavefunction"


def get_json(session, url, params: dict = None):
    """
    Get JSON from a Forest endpoint.
    """
    res = session.get(url, params=params)
    if res.status_code >= 400:
        raise parse_error(res)
    return res.json()


def post_json(session, url, json):
    """
    Post JSON to the Forest endpoint.
    """
    res = session.post(url, json=json)
    if res.status_code >= 400:
        raise parse_error(res)
    return res


def parse_error(res):
    """
    Every server error should contain a "status" field with a human readable explanation of
    what went wrong as well as a "error_type" field indicating the kind of error that can be mapped
    to a Python type.

    There's a fallback error UnknownError for other types of exceptions (network issues, api
    gateway problems, etc.)
    """
    try:
        body = res.json()
    except JSONDecodeError:
        raise UnknownApiError(res.text)

    if 'error_type' not in body:
        raise UnknownApiError(str(body))

    error_type = body['error_type']
    status = body['status']

    if re.search(r"[0-9]+ qubits were requested, but the QVM is limited to [0-9]+ qubits.", status):
        return TooManyQubitsError(status)

    error_cls = error_mapping.get(error_type, UnknownApiError)
    return error_cls(status)


def get_session():
    """
    Create a requests session to access the REST API

    :return: requests session
    :rtype: Session
    """
    config = PyquilConfig()
    session = requests.Session()
    retry_adapter = HTTPAdapter(max_retries=Retry(total=3,
                                                  method_whitelist=['POST'],
                                                  status_forcelist=[502, 503, 504, 521, 523],
                                                  backoff_factor=0.2,
                                                  raise_on_status=False))

    session.mount("http://", retry_adapter)
    session.mount("https://", retry_adapter)

    # We need this to get binary payload for the wavefunction call.
    session.headers.update({"Accept": "application/octet-stream",
                            "X-User-Id": config.user_id,
                            "X-Api-Key": config.api_key})

    session.headers.update({
        'Content-Type': 'application/json; charset=utf-8'
    })

    return session


def validate_noise_probabilities(noise_parameter):
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


def validate_qubit_list(qubit_list):
    """
    Check the validity of qubits for the payload.

    :param list|range qubit_list: List of qubits to be validated.
    """
    if not isinstance(qubit_list, (list, range)):
        raise TypeError("run_items must be a list")
    if any(not isinstance(i, integer_types) or i < 0 for i in qubit_list):
        raise TypeError("run_items list must contain positive integer values")
    return qubit_list


def prepare_register_list(register_dict: Dict[str, Union[bool, Sequence[int]]]):
    """
    Canonicalize classical addresses for the payload and ready MemoryReference instances
    for serialization.

    This function will cast keys that are iterables of int-likes to a list of Python
    ints. This is to support specifying the register offsets as ``range()`` or numpy
    arrays. This mutates ``register_dict``.

    :param register_dict: The classical memory to retrieve. Specified as a dictionary:
        the keys are the names of memory regions, and the values are either (1) a list of
        integers for reading out specific entries in that memory region, or (2) True, for
        reading out the entire memory region.
    """
    if not isinstance(register_dict, dict):
        raise TypeError("register_dict must be a dict but got " + repr(register_dict))

    for k, v in register_dict.items():
        if isinstance(v, bool):
            assert v    # If boolean v must be True
            continue

        indices = [int(x) for x in v]  # support ranges, numpy, ...

        if not all(x >= 0 for x in indices):
            raise TypeError("Negative indices into classical arrays are not allowed.")
        register_dict[k] = indices

    return register_dict


def run_and_measure_payload(quil_program, qubits, trials, random_seed):
    """REST payload for :py:func:`ForestConnection._run_and_measure`"""
    if not quil_program:
        raise ValueError("You have attempted to run an empty program."
                         " Please provide gates or measure instructions to your program.")

    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")
    qubits = validate_qubit_list(qubits)
    if not isinstance(trials, integer_types):
        raise TypeError("trials must be an integer")

    payload = {"type": TYPE_MULTISHOT_MEASURE,
               "qubits": list(qubits),
               "trials": trials,
               "compiled-quil": quil_program.out()}

    if random_seed is not None:
        payload['rng-seed'] = random_seed

    return payload


def wavefunction_payload(quil_program, random_seed):
    """REST payload for :py:func:`ForestConnection._wavefunction`"""
    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")

    payload = {'type': TYPE_WAVEFUNCTION,
               'compiled-quil': quil_program.out()}

    if random_seed is not None:
        payload['rng-seed'] = random_seed

    return payload


def expectation_payload(prep_prog, operator_programs, random_seed):
    """REST payload for :py:func:`ForestConnection._expectation`"""
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


def qvm_run_payload(quil_program, classical_addresses, trials,
                    measurement_noise, gate_noise, random_seed):
    """REST payload for :py:func:`ForestConnection._qvm_run`"""
    if not quil_program:
        raise ValueError("You have attempted to run an empty program."
                         " Please provide gates or measure instructions to your program.")
    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")
    classical_addresses = prepare_register_list(classical_addresses)
    if not isinstance(trials, integer_types):
        raise TypeError("trials must be an integer")

    payload = {"type": TYPE_MULTISHOT,
               "addresses": classical_addresses,
               "trials": trials,
               "compiled-quil": quil_program.out()}

    if measurement_noise is not None:
        payload["measurement-noise"] = measurement_noise
    if gate_noise is not None:
        payload["gate-noise"] = gate_noise
    if random_seed is not None:
        payload['rng-seed'] = random_seed

    return payload


def quilc_compile_payload(quil_program, isa, specs):
    """REST payload for :py:func:`ForestConnection._quilc_compile`"""
    if not quil_program:
        raise ValueError("You have attempted to compile an empty program."
                         " Please provide an actual program.")
    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Program object.")
    if not isinstance(isa, ISA):
        raise TypeError("isa must be an ISA object.")
    if not isinstance(specs, Specs):
        raise TypeError("specs must be a Specs object.")

    payload = {"uncompiled-quil": quil_program.out(),
               "target-device": {
                   "isa": isa.to_dict(),
                   "specs": specs.to_dict()}}

    return payload


class ForestConnection:
    @_record_call
    def __init__(self, sync_endpoint=None, compiler_endpoint=None, forest_cloud_endpoint=None):
        """
        Represents a connection to Forest containing methods to wrap all possible API endpoints.

        Users should not use methods from this class directly.

        :param sync_endpoint: The endpoint of the server for running QVM jobs
        :param compiler_endpoint: The endpoint of the server for running quilc compiler jobs
        :param forest_cloud_endpoint: The endpoint of the forest cloud server
        """
        pyquil_config = PyquilConfig()
        if sync_endpoint is None:
            sync_endpoint = pyquil_config.qvm_url
        if compiler_endpoint is None:
            compiler_endpoint = pyquil_config.compiler_url
        if forest_cloud_endpoint is None:
            forest_cloud_endpoint = pyquil_config.forest_url

        self.sync_endpoint = sync_endpoint
        self.compiler_endpoint = compiler_endpoint
        self.forest_cloud_endpoint = forest_cloud_endpoint
        self.session = get_session()

    @_record_call
    def _run_and_measure(self, quil_program, qubits, trials, random_seed) -> np.ndarray:
        """
        Run a Forest ``run_and_measure`` job.

        Users should use :py:func:`WavefunctionSimulator.run_and_measure` instead of calling
        this directly.
        """
        payload = run_and_measure_payload(quil_program, qubits, trials, random_seed)
        response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
        return np.asarray(response.json())

    @_record_call
    def _wavefunction(self, quil_program, random_seed) -> Wavefunction:
        """
        Run a Forest ``wavefunction`` job.

        Users should use :py:func:`WavefunctionSimulator.wavefunction` instead of calling
        this directly.
        """

        payload = wavefunction_payload(quil_program, random_seed)
        response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
        return Wavefunction.from_bit_packed_string(response.content)

    @_record_call
    def _expectation(self, prep_prog, operator_programs, random_seed) -> np.ndarray:
        """
        Run a Forest ``expectation`` job.

        Users should use :py:func:`WavefunctionSimulator.expectation` instead of calling
        this directly.
        """
        if isinstance(operator_programs, Program):
            warnings.warn("You have provided a Program rather than a list of Programs. The results "
                          "from expectation will be line-wise expectation values of the "
                          "operator_programs.", SyntaxWarning)

        payload = expectation_payload(prep_prog, operator_programs, random_seed)
        response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
        return np.asarray(response.json())

    @_record_call
    def _qvm_run(self, quil_program, classical_addresses, trials,
                 measurement_noise, gate_noise, random_seed) -> np.ndarray:
        """
        Run a Forest ``run`` job on a QVM.

        Users should use :py:func:`QVM.run` instead of calling this directly.
        """
        payload = qvm_run_payload(quil_program, classical_addresses, trials,
                                  measurement_noise, gate_noise, random_seed)
        response = post_json(self.session, self.sync_endpoint + "/qvm", payload)

        ram = response.json()

        for k in ram.keys():
            ram[k] = np.array(ram[k])

        return ram

    @_record_call
    def _qvm_get_version_info(self) -> dict:
        """
        Return version information for the QVM.

        :return: Dictionary with version information
        """
        response = post_json(self.session, self.sync_endpoint, {'type': 'version'})
        split_version_string = response.text.split()
        try:
            qvm_app_version = split_version_string[0]
            qvm_lib_version = split_version_string[2][:-1]
        except ValueError:
            raise TypeError(f'Malformed version string returned by the QVM: {response.text}')
        return {'qvm-app': qvm_app_version, 'qvm-lib': qvm_lib_version}

    def _quilc_compile(self, quil_program, isa, specs):
        """
        Sends a quilc job to Forest.

        Users should use :py:func:`LocalCompiler.quil_to_native_quil` instead of calling this
        directly.
        """
        payload = quilc_compile_payload(quil_program, isa, specs)
        response = post_json(self.session, self.sync_endpoint + "/quilc", payload)
        unpacked_response = response.json()
        return unpacked_response

    def _quilc_get_version_info(self) -> dict:
        """
        Return version information for quilc.

        :return: Dictionary with version information
        """
        return get_json(self.session, self.sync_endpoint + '/version')

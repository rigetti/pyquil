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

from enum import Enum
import re
import warnings
from json.decoder import JSONDecodeError
from typing import Dict, Union, Sequence

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import uuid

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

# The following RPC methods are only available in qvm-ng.
TYPE_RUN_PROGRAM = "run-program"
TYPE_CREATE_QVM = "create-qvm"
TYPE_DELETE_QVM = "delete-qvm"
TYPE_QVM_INFO = "qvm-info"


class QVMSimulationMethod(Enum):
    PURE_STATE = "pure-state"
    FULL_DENSITY_MATRIX = "full-density-matrix"


class QVMAllocationMethod(Enum):
    NATIVE = "native"
    FOREIGN = "foreign"


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
    if any(not isinstance(i, int) or i < 0 for i in qubit_list):
        raise TypeError("run_items list must contain positive integer values")
    return qubit_list


def is_valid_v4_uuid(uuid_string):
    """
    Is uuid_string a valid string representation of a v4 UUID?

    :param str uuid_string: The UUID string to check.
    """
    try:
        uid = uuid.UUID(uuid_string)
    except Exception:
        return False
    else:
        return uid.version == 4


def validate_persistent_qvm_token(qvm_token):
    """
    Check that qvm_token is a valid persistent QVM token.

    :param str qvm_token: The persistent QVM token string.
    """
    if not is_valid_v4_uuid(qvm_token):
        raise ValueError("qvm_token must be a valid v4 UUID. Got {qvm_token}.")


def validate_allocation_method(allocation_method):
    """
    Check that allocation_method is a valid QVM allocation method.

    :param QVMAllocationMethod allocation_method: The allocation method.
    """
    if not isinstance(allocation_method, QVMAllocationMethod):
        raise TypeError("allocation_method must be a QVMAllocationMethod. "
                        f"Got '{allocation_method}'.")


def validate_simulation_method(simulation_method):
    """
    Check that simulation_method is a valid QVM simulation method.

    :param QVMSimulationMethod simulation_method: The simulation method.
    """
    if not isinstance(simulation_method, QVMSimulationMethod):
        raise TypeError("simulation_method must be a QVMSimulationMethod. "
                        f"Got '{simulation_method}'.")


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
    if not isinstance(trials, int):
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
    if not isinstance(trials, int):
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


def qvm_ng_run_program_payload(quil_program, qvm_token, simulation_method, allocation_method,
                               classical_addresses, measurement_noise, gate_noise):
    """REST payload for :py:func:`ForestConnection._qvm_ng_run_program`"""
    if not quil_program:
        raise ValueError("You have attempted to run an empty program."
                         " Please provide gates or measure instructions to your program.")
    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")
    if qvm_token is not None:
        if (simulation_method is not None or allocation_method is not None
              or measurement_noise is not None or gate_noise is not None):
            raise ValueError("Cannot provide both qvm_token and any of the following: "
                             "simulation_method, allocation_method, measurement_noise, gate_noise")
        validate_persistent_qvm_token(qvm_token)
    else:
        validate_simulation_method(simulation_method)
        validate_allocation_method(allocation_method)

    classical_addresses = prepare_register_list(classical_addresses)
    payload = {"type": TYPE_RUN_PROGRAM,
               "addresses": classical_addresses,
               "compiled-quil": quil_program.out()}

    if qvm_token is not None:
        payload["qvm-token"] = qvm_token
    else:
        payload["simulation-method"] = simulation_method.value
        payload["allocation-method"] = allocation_method.value

    if measurement_noise is not None:
        payload["measurement-noise"] = measurement_noise
    if gate_noise is not None:
        payload["gate-noise"] = gate_noise

    return payload


def qvm_ng_create_qvm_payload(simulation_method, allocation_method, num_qubits, measurement_noise,
                              gate_noise):
    """REST payload for :py:func:`ForestConnection._qvm_ng_create_qvm`"""
    if not isinstance(num_qubits, integer_types) or num_qubits < 0:
        raise ValueError("num_qubits must be a positive integer.")

    validate_simulation_method(simulation_method)
    validate_allocation_method(allocation_method)

    payload = {"type": TYPE_CREATE_QVM,
               "simulation-method": simulation_method.value,
               "allocation-method": allocation_method.value,
               "num-qubits": num_qubits}

    if measurement_noise is not None:
        payload["measurement-noise"] = measurement_noise
    if gate_noise is not None:
        payload["gate-noise"] = gate_noise

    return payload


def qvm_ng_delete_qvm_payload(token):
    """REST payload for :py:func:`ForestConnection._qvm_ng_delete_qvm`"""
    validate_persistent_qvm_token(token)
    return {"type": TYPE_DELETE_QVM, "qvm-token": token}


def qvm_ng_qvm_info_payload(token):
    """REST payload for :py:func:`ForestConnection._qvm_ng_qvm_info`"""
    validate_persistent_qvm_token(token)
    return {"type": TYPE_QVM_INFO, "qvm-token": token}


class ForestConnection:
    @_record_call
    def __init__(self, sync_endpoint=None, compiler_endpoint=None, forest_cloud_endpoint=None,
                 qvm_ng_endpoint=None):
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
            compiler_endpoint = pyquil_config.quilc_url
        if forest_cloud_endpoint is None:
            forest_cloud_endpoint = pyquil_config.forest_url
        if qvm_ng_endpoint is None:
            qvm_ng_endpoint = pyquil_config.qvm_ng_url

        self.sync_endpoint = sync_endpoint
        self.compiler_endpoint = compiler_endpoint
        self.forest_cloud_endpoint = forest_cloud_endpoint
        self.qvm_ng_endpoint = qvm_ng_endpoint
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
    def _qvm_get_version_info(self) -> str:
        """
        Return version information for the QVM.

        :return: String of QVM version
        """
        response = post_json(self.session, self.sync_endpoint, {'type': 'version'})
        split_version_string = response.text.split()
        try:
            qvm_version = split_version_string[0]
        except ValueError:
            raise TypeError(f'Malformed version string returned by the QVM: {response.text}')
        return qvm_version

    @_record_call
    def _qvm_ng_run_program(self, quil_program, qvm_token, simulation_method, allocation_method,
                            classical_addresses, measurement_noise, gate_noise) -> np.ndarray:
        """
        Run a Forest ``run_program`` job on a QVM.
        """
        payload = qvm_ng_run_program_payload(quil_program, qvm_token, simulation_method,
                                             allocation_method, classical_addresses, measurement_noise,
                                             gate_noise)
        response = post_json(self.session, self.qvm_ng_endpoint + "/", payload)
        ram = response.json()

        for k in ram.keys():
            ram[k] = np.array(ram[k])

        return ram

    @_record_call
    def _qvm_ng_create_qvm(self, simulation_method, allocation_method, num_qubits, measurement_noise,
                           gate_noise) -> str:
        """
        Run a Forest ``create_qvm`` job.
        """
        payload = qvm_ng_create_qvm_payload(simulation_method, allocation_method, num_qubits,
                                            measurement_noise, gate_noise)
        response = post_json(self.session, self.qvm_ng_endpoint + "/", payload)
        json = response.json()

        if not isinstance(json, dict) or "token" not in json or not is_valid_v4_uuid(json["token"]):
            raise TypeError(f"Malformed persistent QVM token returned by the QVM: {json}")

        return json["token"]

    @_record_call
    def _qvm_ng_delete_qvm(self, token) -> bool:
        """
        Run a Forest ``delete_qvm`` job.
        """
        payload = qvm_ng_delete_qvm_payload(token)
        response = post_json(self.session, self.qvm_ng_endpoint + "/", payload)
        return response.ok

    @_record_call
    def _qvm_ng_qvm_info(self, token) -> dict:
        """
        Run a Forest ``qvm_info`` job.
        """
        payload = qvm_ng_qvm_info_payload(token)
        response = post_json(self.session, self.qvm_ng_endpoint + "/", payload)
        return response.json()

    @_record_call
    def _qvm_ng_get_version_info(self) -> dict:
        """
        Return version information for the QVM-NG.

        :return: String of QVM version
        """
        response = post_json(self.session, self.qvm_ng_endpoint, {'type': 'version'})
        split_version_string = response.text.split()
        try:
            qvm_version = split_version_string[0]
        except ValueError:
            raise TypeError(f'Malformed version string returned by the QVM: {response.text}')
        return qvm_version

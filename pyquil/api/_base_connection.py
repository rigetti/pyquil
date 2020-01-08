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
import re
import time
import warnings
from json.decoder import JSONDecodeError
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from pyquil.api._config import PyquilConfig
from pyquil.api._error_reporting import _record_call
from pyquil.api._errors import (
    error_mapping,
    ApiError,
    UserMessageError,
    UnknownApiError,
    TooManyQubitsError,
)
from pyquil.api._logger import logger
from pyquil.quil import Program
from pyquil.version import __version__
from pyquil.wavefunction import Wavefunction

TYPE_EXPECTATION = "expectation"
TYPE_MULTISHOT = "multishot"
TYPE_MULTISHOT_MEASURE = "multishot-measure"
TYPE_WAVEFUNCTION = "wavefunction"


def get_json(session: requests.Session, url: str, params: Optional[Dict[Any, Any]] = None) -> Any:
    """
    Get JSON from a Forest endpoint.
    """
    logger.debug("Sending GET request to %s. Params: %s", url, params)
    res = session.get(url, params=params)
    if res.status_code >= 400:
        raise parse_error(res)
    return res.json()


def post_json(session: requests.Session, url: str, json: Any) -> requests.models.Response:
    """
    Post JSON to the Forest endpoint.
    """
    logger.debug("Sending POST request to %s. Body: %s", url, json)
    res = session.post(url, json=json)
    if res.status_code >= 400:
        raise parse_error(res)
    return res


def parse_error(res: requests.Response) -> ApiError:
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

    if "error_type" not in body:
        raise UnknownApiError(str(body))

    error_type = body["error_type"]
    status = body["status"]

    if re.search(r"[0-9]+ qubits were requested, but the QVM is limited to [0-9]+ qubits.", status):
        return TooManyQubitsError(status)

    error_cls = error_mapping.get(error_type, UnknownApiError)
    return error_cls(status)


def get_session(*args: Any, **kwargs: Any) -> "ForestSession":
    """
    Create a requests session to access the REST API

    :return: requests session
    :rtype: Session
    """
    session = ForestSession(*args, **kwargs)
    retry_adapter = HTTPAdapter(
        max_retries=Retry(
            total=3,
            method_whitelist=["POST"],
            status_forcelist=[502, 503, 504, 521, 523],
            backoff_factor=0.2,
            raise_on_status=False,
        )
    )

    session.mount("http://", retry_adapter)
    session.mount("https://", retry_adapter)

    # We need this to get binary payload for the wavefunction call.
    session.headers.update({"Accept": "application/octet-stream"})

    session.headers.update({"Content-Type": "application/json; charset=utf-8"})

    return session


def validate_noise_probabilities(noise_parameter: Optional[List[float]]) -> None:
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


def validate_qubit_list(qubit_list: Sequence[int]) -> Sequence[int]:
    """
    Check the validity of qubits for the payload.

    :param qubit_list: List of qubits to be validated.
    """
    if not isinstance(qubit_list, Sequence):
        raise TypeError("'qubit_list' must be of type 'Sequence'")
    if any(not isinstance(i, int) or i < 0 for i in qubit_list):
        raise TypeError("'qubit_list' must contain positive integer values")
    return qubit_list


def prepare_register_list(
    register_dict: Dict[str, Union[bool, Sequence[int]]]
) -> Dict[str, Union[bool, Sequence[int]]]:
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
            assert v  # If boolean v must be True
            continue

        indices = [int(x) for x in v]  # support ranges, numpy, ...

        if not all(x >= 0 for x in indices):
            raise TypeError("Negative indices into classical arrays are not allowed.")
        register_dict[k] = indices

    return register_dict


def run_and_measure_payload(
    quil_program: Program, qubits: Sequence[int], trials: int, random_seed: int
) -> Dict[str, object]:
    """REST payload for :py:func:`ForestConnection._run_and_measure`"""
    if not quil_program:
        raise ValueError(
            "You have attempted to run an empty program."
            " Please provide gates or measure instructions to your program."
        )

    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")
    qubits = validate_qubit_list(qubits)
    if not isinstance(trials, int):
        raise TypeError("trials must be an integer")

    payload = {
        "type": TYPE_MULTISHOT_MEASURE,
        "qubits": list(qubits),
        "trials": trials,
        "compiled-quil": quil_program.out(),
    }

    if random_seed is not None:
        payload["rng-seed"] = random_seed

    return payload


def wavefunction_payload(quil_program: Program, random_seed: int) -> Dict[str, object]:
    """REST payload for :py:func:`ForestConnection._wavefunction`"""
    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")

    payload: Dict[str, object] = {"type": TYPE_WAVEFUNCTION, "compiled-quil": quil_program.out()}

    if random_seed is not None:
        payload["rng-seed"] = random_seed

    return payload


def expectation_payload(
    prep_prog: Program, operator_programs: Optional[Iterable[Program]], random_seed: int
) -> Dict[str, object]:
    """REST payload for :py:func:`ForestConnection._expectation`"""
    if operator_programs is None:
        operator_programs = [Program()]

    if not isinstance(prep_prog, Program):
        raise TypeError("prep_prog variable must be a Quil program object")

    payload: Dict[str, object] = {
        "type": TYPE_EXPECTATION,
        "state-preparation": prep_prog.out(),
        "operators": [x.out() for x in operator_programs],
    }

    if random_seed is not None:
        payload["rng-seed"] = random_seed

    return payload


def qvm_run_payload(
    quil_program: Program,
    classical_addresses: Dict[str, Union[bool, Sequence[int]]],
    trials: int,
    measurement_noise: Optional[Tuple[float, float, float]],
    gate_noise: Optional[Tuple[float, float, float]],
    random_seed: Optional[int],
) -> Dict[str, object]:
    """REST payload for :py:func:`ForestConnection._qvm_run`"""
    if not quil_program:
        raise ValueError(
            "You have attempted to run an empty program."
            " Please provide gates or measure instructions to your program."
        )
    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")
    classical_addresses = prepare_register_list(classical_addresses)
    if not isinstance(trials, int):
        raise TypeError("trials must be an integer")

    payload = {
        "type": TYPE_MULTISHOT,
        "addresses": classical_addresses,
        "trials": trials,
        "compiled-quil": quil_program.out(),
    }

    if measurement_noise is not None:
        payload["measurement-noise"] = measurement_noise
    if gate_noise is not None:
        payload["gate-noise"] = gate_noise
    if random_seed is not None:
        payload["rng-seed"] = random_seed

    return payload


class ForestSession(requests.Session):
    """
    ForestSession inherits from requests.Session. It is responsible for adding
    authentication headers to Forest server requests. Upon receiving a 401 or 403
    response, it will attempt to refresh the auth credential and update the
    PyquilConfig, which in turn writes the refreshed auth credential to file.

    Encapsulates the operations required for authorization & encryption
    with the QPU.

    Two operations are involved in authorization:

    * Requesting & storing a user authentication token, used to authenticate calls
      to Forest, Dispatch, and other Rigetti services
    * Requesting a Curve ZeroMQ keypair for connection to the QPU. The response to
      this request also comes with service endpoints: compiler server and QPU

    The authentication tokens are of the standard JWT format and are issued by Forest Server.

    The refresh token is only used to renew the access token, which is used for all transactions
    and is valid for a short period of time.

    In wrapping the PyQuilConfig object, it provides that object with a callback to
    retrieve a valid engagement when needed, because the engagement is maintained here
    but is used by the config to provide service endpoints.
    """

    def __init__(self, *, config: PyquilConfig, lattice_name: Optional[str] = None):
        super().__init__()
        self.config = config
        self.config.get_engagement = self.get_engagement
        self._engagement: Optional["Engagement"] = None
        self.headers.update(self.config.qcs_auth_headers)
        self.headers["User-Agent"] = f"PyQuil/{__version__}"
        self.lattice_name = lattice_name

    def _engage(self) -> Optional["Engagement"]:
        """
        The heart of the QPU authorization process, ``engage`` makes a request to
        the dispatch server for the information needed to communicate with the QPU.

        This is a standard GraphQL request, authenticated using the access token
        retrieved from Forest Server.

        The response includes the endpoints to the QPU and QPU Compiler Server,
        along with the set of keys necessary to connect to the QPU and the time at
        which that key set expires.
        """
        query = """
          mutation Engage($name: String!) {
            engage(input: { lattice: { name: $name }}) {
              success
              message
              engagement {
                type
                qpu {
                    endpoint
                    credentials {
                        clientPublic
                        clientSecret
                        serverPublic
                    }
                }
                compiler {
                    endpoint
                }
                expiresAt
              }
            }
          }
        """
        if not self.lattice_name:
            logger.debug("ForestSession requires lattice_name in order to engage")
            return None

        logger.debug("Requesting engagement from %s", self.config.dispatch_url)
        variables = dict(name=self.lattice_name)
        query_response = self._request_graphql_retry(
            self.config.dispatch_url, query=query, variables=variables
        )

        if query_response.get("errors"):
            errors = query_response.get("errors", [])
            error_messages = map(lambda error: error["message"], errors)  # type: ignore
            raise UserMessageError(f"Failed to engage: {','.join(error_messages)}")

        engagement_response = query_response.get("data", {}).get("engage", None)
        if engagement_response and engagement_response.get("success") is True:
            logger.debug("Engagement successful")
            engagement_data = engagement_response.get("engagement", {})
            return Engagement(
                client_secret_key=engagement_data.get("qpu", {})
                .get("credentials", {})
                .get("clientSecret", "")
                .encode("utf-8"),
                client_public_key=engagement_data.get("qpu", {})
                .get("credentials", {})
                .get("clientPublic", "")
                .encode("utf-8"),
                server_public_key=engagement_data.get("qpu", {})
                .get("credentials", {})
                .get("serverPublic", "")
                .encode("utf-8"),
                expires_at=engagement_data.get("expiresAt", {}),
                qpu_endpoint=engagement_data.get("qpu", {}).get("endpoint"),
                qpu_compiler_endpoint=engagement_data.get("compiler", {}).get("endpoint"),
            )
        else:
            raise UserMessageError(
                f"Unable to engage {self.lattice_name}: "
                f"{engagement_response.get('message', 'No message')}"
            )

    def get_engagement(self) -> Optional["Engagement"]:
        """
        Returns memoized engagement information, if still valid - or requests a new engagement
        and then stores and returns that.
        """
        if not (self._engagement and self._engagement.is_valid()):
            self._engagement = self._engage()
        return self._engagement

    def _refresh_auth_token(self) -> bool:
        self.config.assert_valid_auth_credential()
        if self.config.user_auth_token is not None:
            return self._refresh_user_auth_token()
        elif self.config.qmi_auth_token is not None:
            return self._refresh_qmi_auth_token()
        return False

    def _refresh_user_auth_token(self) -> bool:
        url = f"{self.config.forest_url}/auth/idp/oauth2/v1/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Cache-Control": "no-cache",
            "Accept": "application/json",
        }
        assert self.config.user_auth_token is not None
        data = {
            "grant_type": "refresh_token",
            "scope": self.config.user_auth_token["scope"],
            "refresh_token": self.config.user_auth_token["refresh_token"],
        }
        response = super().request("POST", url, data=data, headers=headers)
        if response.status_code == 200:
            self.config.update_user_auth_token(response.json())
            self.headers.update(self.config.qcs_auth_headers)
            return True

        logger.warning(
            f"Failed to refresh your user auth token at {self.config.user_auth_token_path}. "
            f"Server response: {response.text}"
        )
        return False

    def _refresh_qmi_auth_token(self) -> bool:
        url = f"{self.config.forest_url}/auth/qmi/refresh"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        data = self.config.qmi_auth_token
        response = super().request("POST", url, json=data, headers=headers)
        if response.status_code == 200:
            self.config.update_qmi_auth_token(response.json())
            self.headers.update(self.config.qcs_auth_headers)
            return True

        logger.warning(
            f"Failed to refresh your QMI auth token at {self.config.qmi_auth_token_path}. "
            f"Server response: {response.text}"
        )
        return False

    def request(self, *args: Any, **kwargs: Any) -> requests.models.Response:
        """
        request is a wrapper around requests.Session#request that checks for
        401 and 403 response statuses and refreshes the auth credential
        accordingly.
        """
        response = super().request(*args, **kwargs)
        if response.status_code in {401, 403}:
            if self._refresh_auth_token():
                response = super().request(*args, **kwargs)
        return response

    def _request_graphql(self, url: str, query: str, variables: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Makes a single graphql request using the session credentials, throwing an error
        if the response is not valid JSON.

        Returns the JSON parsed from the response.
        """
        response = super().post(url, json=dict(query=query, variables=variables))
        try:
            return cast(Dict[Any, Any], response.json())
        except JSONDecodeError as e:
            logger.exception(f"Unable to parse json response from endpoint {url}:", response.text)
            raise e

    def _request_graphql_retry(self, *args: Any, **kwargs: Any) -> Dict[Any, Any]:
        """
        Makes a GraphQL request using session credentials, refreshing them once if the server
        identifies them as expired.

        Determining whether a call has failed to a GraphQL endpoint is less axiomatic than for a
        REST interface, and so here we follow the pattern set by Rigetti services, which return an
        HTTP 200 response with an array of errors. If any of those errors cite an expired
        authentication token, we refresh the token to clear that error. Note that other error
        messages will not trigger a retry.
        """
        result = self._request_graphql(*args, **kwargs)
        errors = result.get("errors", [])
        token_is_expired = any(
            error.get("extensions", {}).get("code") == "AUTH_TOKEN_EXPIRED" for error in errors
        )
        if token_is_expired:
            if self._refresh_auth_token():
                result = self._request_graphql(*args, **kwargs)
        return result


class ForestConnection:
    @_record_call
    def __init__(
        self,
        sync_endpoint: Optional[str] = None,
        compiler_endpoint: Optional[str] = None,
        forest_cloud_endpoint: Optional[str] = None,
    ):
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

        assert sync_endpoint is not None
        self.sync_endpoint = sync_endpoint
        self.compiler_endpoint = compiler_endpoint
        self.forest_cloud_endpoint = forest_cloud_endpoint
        self.session = get_session(config=pyquil_config)

    @_record_call
    def _run_and_measure(
        self, quil_program: Program, qubits: Sequence[int], trials: int, random_seed: int
    ) -> np.ndarray:
        """
        Run a Forest ``run_and_measure`` job.

        Users should use :py:func:`WavefunctionSimulator.run_and_measure` instead of calling
        this directly.
        """
        payload = run_and_measure_payload(quil_program, qubits, trials, random_seed)
        response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
        return np.asarray(response.json())

    @_record_call
    def _wavefunction(self, quil_program: Program, random_seed: int) -> Wavefunction:
        """
        Run a Forest ``wavefunction`` job.

        Users should use :py:func:`WavefunctionSimulator.wavefunction` instead of calling
        this directly.
        """

        payload = wavefunction_payload(quil_program, random_seed)
        response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
        return Wavefunction.from_bit_packed_string(response.content)

    @_record_call
    def _expectation(
        self, prep_prog: Program, operator_programs: Iterable[Program], random_seed: int
    ) -> np.ndarray:
        """
        Run a Forest ``expectation`` job.

        Users should use :py:func:`WavefunctionSimulator.expectation` instead of calling
        this directly.
        """
        if isinstance(operator_programs, Program):
            warnings.warn(
                "You have provided a Program rather than a list of Programs. The results "
                "from expectation will be line-wise expectation values of the "
                "operator_programs.",
                SyntaxWarning,
            )

        payload = expectation_payload(prep_prog, operator_programs, random_seed)
        response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
        return np.asarray(response.json())

    @_record_call
    def _qvm_run(
        self,
        quil_program: Program,
        classical_addresses: Dict[str, Union[bool, Sequence[int]]],
        trials: int,
        measurement_noise: Optional[Tuple[float, float, float]],
        gate_noise: Optional[Tuple[float, float, float]],
        random_seed: Optional[int],
    ) -> np.ndarray:
        """
        Run a Forest ``run`` job on a QVM.

        Users should use :py:func:`QVM.run` instead of calling this directly.
        """
        payload = qvm_run_payload(
            quil_program, classical_addresses, trials, measurement_noise, gate_noise, random_seed
        )
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
        response = post_json(self.session, self.sync_endpoint, {"type": "version"})
        split_version_string = response.text.split()
        try:
            qvm_version = split_version_string[0]
        except ValueError:
            raise TypeError(f"Malformed version string returned by the QVM: {response.text}")
        return qvm_version


class Engagement:
    """
    An Engagement stores all the information retrieved via an engagement request sent to
      the dispatch server.
    """

    def __init__(
        self,
        client_public_key: bytes,
        client_secret_key: bytes,
        server_public_key: bytes,
        expires_at: Union[int, float, str],
        qpu_endpoint: str,
        qpu_compiler_endpoint: str,
    ):
        self.client_public_key = client_public_key
        self.client_secret_key = client_secret_key
        self.server_public_key = server_public_key
        self.expires_at = float(expires_at) if expires_at else None
        self.qpu_endpoint = qpu_endpoint
        self.qpu_compiler_endpoint = qpu_compiler_endpoint
        logger.debug("New engagement created: \n%s", self)

    def is_valid(self) -> bool:
        """
        Return true if an engagement is valid for use, false if it is missing required
          fields

        An 'invalid' engagement is one which will not grant access to the QPU.
        """
        return all(
            [
                self.client_public_key is not None,
                self.client_secret_key is not None,
                self.server_public_key is not None,
                (self.expires_at is None or self.expires_at > time.time()),
                self.qpu_endpoint is not None,
            ]
        )

    def __str__(self) -> str:
        return f"""Client public key: {self.client_public_key}
Client secret key: masked ({len(self.client_secret_key)} B)
Server public key: {self.server_public_key}
Expiration time: {self.expires_at}
QPU Endpoint: {self.qpu_endpoint}
QPU Compiler Endpoint: {self.qpu_compiler_endpoint}"""  # type: ignore

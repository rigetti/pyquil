##############################################################################
# Copyright 2016-2021 Rigetti Computing
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
from contextlib import contextmanager
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Iterator, Any, Dict, Union, Tuple, Optional, List, cast

import httpx
from qcs_api_client.client import QCSClientConfiguration

from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping


@dataclass
class RunProgramRequest:
    """
    Request to run a Quil program.
    """

    program: str
    """Quil program to run."""

    addresses: Dict[str, Union[bool, List[int]]]
    """Memory addresses to be read and returned after execution. Mapped by region names to either:
       - a numeric index, to return that particular value,
       - `True` to return all memory in that region, or
       - `False` to return no memory in that region (equivalent to missing entry)
    """

    trials: int
    """Number of times to run program and collect results."""

    measurement_noise: Optional[Tuple[float, float, float]]
    """Simulated measurement noise for X, Y, and Z axes."""

    gate_noise: Optional[Tuple[float, float, float]]
    """Simulated gate noise for X, Y, and Z axes."""

    seed: Optional[int]
    """PRNG seed. Set this to guarantee repeatable results."""


@dataclass
class RunProgramResponse:
    """
    Program run response.
    """

    results: Dict[str, List[List[int]]]
    """Run results, by memory region name. Values are multi-dimensional arrays of size <trials>-by-<slots>."""


@dataclass
class RunAndMeasureProgramRequest:
    """
    Request to run and measure a Quil program.
    """

    program: str
    """Quil program to run."""

    qubits: List[int]
    """Qubits to measure."""

    trials: int
    """Number of times to run program and collect results."""

    measurement_noise: Optional[Tuple[float, float, float]]
    """Simulated measurement noise for X, Y, and Z axes."""

    gate_noise: Optional[Tuple[float, float, float]]
    """Simulated gate noise for X, Y, and Z axes."""

    seed: Optional[int]
    """PRNG seed. Set this to guarantee repeatable results."""


@dataclass
class RunAndMeasureProgramResponse:
    """
    Program run and measure response.
    """

    results: List[List[int]]
    """Resulting memory region value, a multi-dimensional array of size <trials>-by-<slots>."""


@dataclass
class MeasureExpectationRequest:
    """
    Request to measure expectations of Pauli operators.
    """

    prep_program: str
    """Quil program to place QVM into a desired state before expectation measurement."""

    pauli_operators: List[str]
    """Quil programs representing Pauli operators for which to measure expectations."""

    seed: Optional[int]
    """PRNG seed. Set this to guarantee repeatable results."""


@dataclass
class MeasureExpectationResponse:
    """
    Expectation measurement response.
    """

    expectations: List[float]
    """Measured expectations, one for each Pauli operator in original request."""


@dataclass
class GetWavefunctionRequest:
    """
    Request to run a program and retrieve the resulting wavefunction.
    """

    program: str
    """Quil program to run."""

    measurement_noise: Optional[Tuple[float, float, float]]
    """Simulated measurement noise for X, Y, and Z axes."""

    gate_noise: Optional[Tuple[float, float, float]]
    """Simulated gate noise for X, Y, and Z axes."""

    seed: Optional[int]
    """PRNG seed. Set this to guarantee repeatable results."""


@dataclass
class GetWavefunctionResponse:
    """
    Get wavefunction response.
    """

    wavefunction: bytes
    """Bit-packed wavefunction string."""


class QVMClient:
    """
    Client for making requests to a Quantum Virtual Machine.
    """

    def __init__(self, *, client_configuration: QCSClientConfiguration, request_timeout: float = 10.0) -> None:
        """
        Instantiate a new compiler client.

        :param client_configuration: Configuration for client.
        :param request_timeout: Timeout for requests, in seconds.
        """
        self.base_url = client_configuration.profile.applications.pyquil.qvm_url
        self.timeout = request_timeout

    def get_version(self) -> str:
        """
        Get version info for QVM server.
        """
        return self._post_json({"type": "version"}).text.split()[0]

    def run_program(self, request: RunProgramRequest) -> RunProgramResponse:
        """
        Run a Quil program and return its results.
        """
        payload: Dict[str, Any] = {
            "type": "multishot",
            "compiled-quil": request.program,
            "addresses": request.addresses,
            "trials": request.trials,
        }

        if request.measurement_noise is not None:
            payload["measurement-noise"] = request.measurement_noise

        if request.gate_noise is not None:
            payload["gate-noise"] = request.gate_noise

        if request.seed is not None:
            payload["rng-seed"] = request.seed

        return RunProgramResponse(results=cast(Dict[str, List[List[int]]], self._post_json(payload).json()))

    def run_and_measure_program(self, request: RunAndMeasureProgramRequest) -> RunAndMeasureProgramResponse:
        """
        Run and measure a Quil program, and return its results.
        """
        payload: Dict[str, Any] = {
            "type": "multishot-measure",
            "compiled-quil": request.program,
            "qubits": request.qubits,
            "trials": request.trials,
        }

        if request.measurement_noise is not None:
            payload["measurement-noise"] = request.measurement_noise

        if request.gate_noise is not None:
            payload["gate-noise"] = request.gate_noise

        if request.seed is not None:
            payload["rng-seed"] = request.seed

        return RunAndMeasureProgramResponse(results=cast(List[List[int]], self._post_json(payload).json()))

    def measure_expectation(self, request: MeasureExpectationRequest) -> MeasureExpectationResponse:
        """
        Measure expectation value of Pauli operators given a defined state.
        """
        payload: Dict[str, Any] = {
            "type": "expectation",
            "state-preparation": request.prep_program,
            "operators": request.pauli_operators,
        }

        if request.seed is not None:
            payload["rng-seed"] = request.seed

        return MeasureExpectationResponse(expectations=cast(List[float], self._post_json(payload).json()))

    def get_wavefunction(self, request: GetWavefunctionRequest) -> GetWavefunctionResponse:
        """
        Run a program and retrieve the resulting wavefunction.
        """
        payload: Dict[str, Any] = {
            "type": "wavefunction",
            "compiled-quil": request.program,
        }

        if request.measurement_noise is not None:
            payload["measurement-noise"] = request.measurement_noise

        if request.gate_noise is not None:
            payload["gate-noise"] = request.gate_noise

        if request.seed is not None:
            payload["rng-seed"] = request.seed

        return GetWavefunctionResponse(wavefunction=self._post_json(payload).content)

    def _post_json(self, json: Dict[str, Any]) -> httpx.Response:
        with self._http_client() as http:  # type: httpx.Client
            response = http.post("/", json=json)
            if response.status_code >= 400:
                raise self._parse_error(response)
        return response

    @contextmanager
    def _http_client(self) -> Iterator[httpx.Client]:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            yield client

    @staticmethod
    def _parse_error(res: httpx.Response) -> ApiError:
        """
        Errors should contain a "status" field with a human readable explanation of
        what went wrong as well as a "error_type" field indicating the kind of error that can be mapped
        to a Python type.

        There's a fallback error UnknownApiError for other types of exceptions (network issues, api
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

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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from qcs_sdk import ExecutionData, QCSClient, ResultData, qvm
from qcs_sdk.qvm import QVMClient, QVMOptions, QVMResultData

from pyquil._version import pyquil_version
from pyquil.api import QAM, MemoryMap, QAMExecutionResult, QuantumExecutable
from pyquil.noise import NoiseModel, apply_noise_model
from pyquil.quil import Program


class QVMVersionMismatch(Exception):
    pass


class QVMNotRunning(Exception):
    pass


def check_qvm_version(version: str) -> None:
    """Verify that there is no mismatch between pyquil and QVM versions.

    :param version: The version of the QVM
    """
    major, minor = map(int, version.split(".")[:2])
    if major == 1 and minor < 8:
        raise QVMVersionMismatch(
            "Must use QVM >= 1.8.0 with pyquil >= 2.8.0, but you " f"have QVM {version} and pyquil {pyquil_version}"
        )


@dataclass
class QVMExecuteResponse:
    executable: Program
    data: QVMResultData

    @property
    def memory(self) -> dict[str, np.ndarray]:
        return {key: matrix.as_ndarray() for key, matrix in self.data.memory.items()}


class QVM(QAM[QVMExecuteResponse]):
    def __init__(
        self,
        noise_model: Optional[NoiseModel] = None,
        gate_noise: Optional[tuple[float, float, float]] = None,
        measurement_noise: Optional[tuple[float, float, float]] = None,
        random_seed: Optional[int] = None,
        timeout: float = 10.0,
        client: Optional[QVMClient] = None,
    ) -> None:
        """Return a virtual machine that classically emulates the execution of Quil programs.

        :param noise_model: A noise model that describes noise to apply when emulating a program's
            execution.
        :param gate_noise: A tuple of three numbers [Px, Py, Pz] indicating the probability of an X,
           Y, or Z gate getting applied to each qubit after a gate application or reset. The
           default value of None indicates no noise.
        :param measurement_noise: A tuple of three numbers [Px, Py, Pz] indicating the probability
            of an X, Y, or Z gate getting applied before a measurement. The default value of
            None indicates no noise.
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
            automatically generated seed) or a non-negative integer.
        :param timeout: Time limit for requests, in seconds.
        :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        """
        super().__init__()

        if (noise_model is not None) and (gate_noise is not None or measurement_noise is not None):
            raise ValueError(
                """
You have attempted to supply the QVM with both a Kraus noise model
(by supplying a `noise_model` argument), as well as either `gate_noise`
or `measurement_noise`. At this time, only one may be supplied.

To read more about supplying noise to the QVM, see
http://pyquil.readthedocs.io/en/latest/noise_models.html#support-for-noisy-gates-on-the-rigetti-qvm.
"""
            )

        self.noise_model = noise_model

        validate_noise_probabilities(gate_noise)
        validate_noise_probabilities(measurement_noise)
        self.gate_noise = gate_noise
        self.measurement_noise = measurement_noise

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, int) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int")

        self.timeout = timeout

        if client is None:
            client = QVMClient.new_http(QCSClient.load().qvm_url)
        self._client = client

        self.connect()

    def connect(self) -> None:
        try:
            version = self.get_version_info()
            check_qvm_version(version)
        except ConnectionError:
            raise QVMNotRunning(f"No QVM server running at {self._client.qvm_url}") from ConnectionError

    def execute_with_memory_map_batch(
        self, executable: QuantumExecutable, memory_maps: Iterable[MemoryMap], **__: Any
    ) -> list[QVMExecuteResponse]:
        """Execute a single program on the QVM with multiple memory maps.

        This method is a convenience wrapper around QVM#execute and isn't more efficient than making multiple separate
        requests to the QVM.
        """
        return [self.execute(executable, memory_map) for memory_map in memory_maps]

    def execute(
        self,
        executable: QuantumExecutable,
        memory_map: Optional[MemoryMap] = None,
        **__: Any,
    ) -> QVMExecuteResponse:
        """Execute the input program to completion."""
        if not isinstance(executable, Program):
            raise TypeError(f"`QVM#executable` argument must be a `Program`; got {type(executable)}")

        # Request all memory back from the QVM.
        addresses = {address: qvm.api.AddressRequest.include_all() for address in executable.declarations.keys()}

        trials = executable.num_shots
        if self.noise_model is not None:
            executable = apply_noise_model(executable, self.noise_model)

        result = qvm.run(
            executable.out(calibrations=False),
            trials,
            addresses,
            memory_map or {},
            self._client,
            self.measurement_noise,
            self.gate_noise,
            self.random_seed,
            options=QVMOptions(timeout_seconds=self.timeout),
        )

        return QVMExecuteResponse(executable=executable, data=result)

    def get_result(self, execute_response: QVMExecuteResponse) -> QAMExecutionResult:
        """Return the results of execution on the QVM."""
        result_data = ResultData(execute_response.data)
        data = ExecutionData(result_data=result_data, duration=None)
        return QAMExecutionResult(executable=execute_response.executable, data=data)

    def get_version_info(self) -> str:
        """Return version information for the QVM.

        :return: String with version information
        """
        return qvm.api.get_version_info(self._client, options=QVMOptions(timeout_seconds=self.timeout))


def validate_noise_probabilities(noise_parameter: Optional[tuple[float, float, float]]) -> None:
    """Validate the noise probabilities.

    This function checks that the provided noise parameters are in the correct format and within the expected ranges.

    :param noise_parameter: A tuple containing three float values representing noise probabilities. If None, the
    function returns without any validation.

    :type noise_parameter: Optional[tuple[float, float, float]]

    :raises TypeError: If noise_parameter is not a tuple or if any of its elements are not floats.
    :raises ValueError: If noise_parameter does not have exactly three elements, if the sum of the elements is not
        between 0 and 1 (inclusive), or if any element is negative.

    :returns: None
    """
    if not noise_parameter:
        return
    if not isinstance(noise_parameter, tuple):
        raise TypeError("noise_parameter must be a tuple")
    if any([not isinstance(value, float) for value in noise_parameter]):
        raise TypeError("noise_parameter values should all be floats")
    if len(noise_parameter) != 3:
        raise ValueError("noise_parameter tuple must be of length 3")
    if sum(noise_parameter) > 1 or sum(noise_parameter) < 0:
        raise ValueError("sum of entries in noise_parameter must be between 0 and 1 (inclusive)")
    if any([value < 0 for value in noise_parameter]):
        raise ValueError("noise_parameter values should all be non-negative")


def validate_qubit_list(qubit_list: Sequence[int]) -> Sequence[int]:
    """Check the validity of qubits for the payload.

    :param qubit_list: List of qubits to be validated.
    """
    if not isinstance(qubit_list, Sequence):
        raise TypeError("'qubit_list' must be of type 'Sequence'")
    if any(not isinstance(i, int) or i < 0 for i in qubit_list):
        raise TypeError("'qubit_list' must contain positive integer values")
    return qubit_list

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
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union, Tuple

import numpy as np
from qcs_api_client.client import QCSClientConfiguration

from pyquil._version import pyquil_version
from pyquil.api import QuantumExecutable
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qvm_client import (
    QVMClient,
    RunProgramRequest,
)
from pyquil.noise import NoiseModel, apply_noise_model
from pyquil.quil import Program, get_classical_addresses_from_program


class QVMVersionMismatch(Exception):
    pass


class QVMNotRunning(Exception):
    pass


def check_qvm_version(version: str) -> None:
    """
    Verify that there is no mismatch between pyquil and QVM versions.

    :param version: The version of the QVM
    """
    major, minor, patch = map(int, version.split("."))
    if major == 1 and minor < 8:
        raise QVMVersionMismatch(
            "Must use QVM >= 1.8.0 with pyquil >= 2.8.0, but you " f"have QVM {version} and pyquil {pyquil_version}"
        )


@dataclass
class QVMExecuteResponse:
    executable: Program
    memory: Mapping[str, np.ndarray]


class QVM(QAM[QVMExecuteResponse]):
    def __init__(
        self,
        noise_model: Optional[NoiseModel] = None,
        gate_noise: Optional[Tuple[float, float, float]] = None,
        measurement_noise: Optional[Tuple[float, float, float]] = None,
        random_seed: Optional[int] = None,
        timeout: float = 10.0,
        client_configuration: Optional[QCSClientConfiguration] = None,
    ) -> None:
        """
        A virtual machine that classically emulates the execution of Quil programs.

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

        client_configuration = client_configuration or QCSClientConfiguration.load()
        self._qvm_client = QVMClient(client_configuration=client_configuration, request_timeout=timeout)
        self.connect()

    def connect(self) -> None:
        try:
            version = self.get_version_info()
            check_qvm_version(version)
        except ConnectionError:
            raise QVMNotRunning(f"No QVM server running at {self._qvm_client.base_url}")

    def execute(self, executable: QuantumExecutable) -> QVMExecuteResponse:
        """
        Synchronously execute the input program to completion.
        """
        executable = executable.copy()

        if not isinstance(executable, Program):
            raise TypeError(f"`QVM#executable` argument must be a `Program`; got {type(executable)}")

        result_memory: dict = {}

        for region in executable.declarations.keys():
            result_memory[region] = np.ndarray((executable.num_shots, 0), dtype=np.int64)

        trials = executable.num_shots
        classical_addresses = get_classical_addresses_from_program(executable)

        if self.noise_model is not None:
            executable = apply_noise_model(executable, self.noise_model)

        executable._set_parameter_values_at_runtime()

        request = qvm_run_request(
            executable,
            classical_addresses,
            trials,
            self.measurement_noise,
            self.gate_noise,
            self.random_seed,
        )
        response = self._qvm_client.run_program(request)
        ram = {key: np.array(val) for key, val in response.results.items()}
        result_memory.update(ram)

        return QVMExecuteResponse(executable=executable, memory=result_memory)

    def get_result(self, execute_response: QVMExecuteResponse) -> QAMExecutionResult:
        """
        Return the results of execution on the QVM.

        Because QVM execution is synchronous, this is a no-op which returns its input.
        """
        return QAMExecutionResult(executable=execute_response.executable, readout_data=execute_response.memory)

    def get_version_info(self) -> str:
        """
        Return version information for the QVM.

        :return: String with version information
        """
        return self._qvm_client.get_version()


def validate_noise_probabilities(noise_parameter: Optional[Tuple[float, float, float]]) -> None:
    """
    Is noise_parameter a valid specification of noise probabilities for depolarizing noise?

    :param tuple noise_parameter: Tuple of 3 noise parameter values to be validated.
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
    register_dict: Mapping[str, Union[bool, Sequence[int]]]
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


def qvm_run_request(
    quil_program: Program,
    classical_addresses: Mapping[str, Union[bool, Sequence[int]]],
    trials: int,
    measurement_noise: Optional[Tuple[float, float, float]],
    gate_noise: Optional[Tuple[float, float, float]],
    random_seed: Optional[int],
) -> RunProgramRequest:
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

    return RunProgramRequest(
        program=quil_program.out(calibrations=False),
        addresses=classical_addresses,  # type: ignore
        trials=trials,
        measurement_noise=measurement_noise,
        gate_noise=gate_noise,
        seed=random_seed,
    )

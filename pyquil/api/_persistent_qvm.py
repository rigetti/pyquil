##############################################################################
# Copyright 2019 Rigetti Computing
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
from typing import Any, Dict, List, Optional

from six import integer_types
import numpy as np

from pyquil.api._base_connection import (ForestConnection, QVMAllocationMethod, QVMSimulationMethod,
                                         validate_simulation_method, validate_allocation_method,
                                         validate_persistent_qvm_token)
from pyquil.api._error_reporting import _record_call
from pyquil.api._qvm import QVMNotRunning, QVMVersionMismatch, validate_noise_probabilities
from pyquil.quil import Program, get_classical_addresses_from_program


def check_qvm_ng_version(version: str):
    """
    Verify that there is no mismatch between pyquil and QVM versions.

    :param version: The version of the QVM
    """
    major, minor, patch = map(int, version.split('.'))
    if major == 1 and minor < 11:
        raise QVMVersionMismatch('Must use QVM >= 1.11.0 with the PersistentQVM, but you '
                                 f'have QVM {version}.')


class PersistentQVM:
    """
    Represents a connection to a PersistentQVM.
    """
    @_record_call
    def __init__(self, num_qubits: int,
                 connection: ForestConnection = None,
                 simulation_method: QVMSimulationMethod = QVMSimulationMethod.PURE_STATE,
                 allocation_method: QVMAllocationMethod = QVMAllocationMethod.NATIVE,
                 measurement_noise: Optional[List[float]] = None,
                 gate_noise: Optional[List[float]] = None,
                 random_seed: Optional[int] = None) -> None:
        """
        A PersistentQVM that classically emulates the execution of Quil programs.

        :param num_qubits: The maximum number of qubits available to this QVM.
        :param connection: A connection to the Forest web API.
        :param simulation_method: The simulation method to use for this PersistentQVM.
            See the enum QVMSimulationmethod for valid values.
        :param allocation_method: The allocation method to use for this PersistentQVM.
            See the enum QVMAllocationmethod for valid values.
        :param measurement_noise: A list of three numbers [Px, Py, Pz] indicating the probability
            of an X, Y, or Z gate getting applied before a measurement. The default value of
            None indicates no noise.
        :param gate_noise: A list of three numbers [Px, Py, Pz] indicating the probability of an X,
           Y, or Z gate getting applied to each qubit after a gate application or reset. The
           default value of None indicates no noise.
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
            automatically generated seed) or a non-negative integer.
        """
        if not isinstance(num_qubits, integer_types) or num_qubits < 0:
            raise TypeError(f"num_qubits must be a positive integer. Got {num_qubits}.")

        validate_simulation_method(simulation_method)
        validate_allocation_method(allocation_method)
        validate_noise_probabilities(gate_noise)
        validate_noise_probabilities(measurement_noise)

        self.simulation_method = simulation_method
        self.allocation_method = allocation_method
        self.num_qubits = num_qubits
        self.gate_noise = gate_noise
        self.measurement_noise = measurement_noise

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, integer_types) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int")

        if connection is None:
            connection = ForestConnection()
        self.connection = connection
        self.connect()
        self.token = self.connection._qvm_ng_create_qvm(simulation_method,
                                                        allocation_method,
                                                        num_qubits,
                                                        measurement_noise,
                                                        gate_noise)

    def __del__(self) -> None:
        self.connection._qvm_ng_delete_qvm(self.token)

    def connect(self) -> None:
        try:
            version = self.get_version_info()
            check_qvm_ng_version(version)
        except ConnectionError:
            raise QVMNotRunning(f'No QVM-NG server running at {self.connection.qvm_ng_endpoint}')

    @_record_call
    def get_version_info(self) -> str:
        """
        Return version information for the connected QVM.

        :return: String with version information
        :rtype: str
        """
        return self.connection._qvm_ng_get_version_info()

    @_record_call
    def get_qvm_info(self) -> Dict[str, Any]:
        """
        Return configuration information about the PersistentQVM.

        :return: Dict with QVM information
        :rtype: Dict[str, Any]
        """
        return self.connection._qvm_ng_qvm_info(self.token)

    @_record_call
    def run_program(self, quil_program: Program) -> Dict[str, np.array]:
        """
        Run quil_program on this PersistentQVM instance, and return the values stored in all of the
        classical registers assigned to by the program.

        :param Program quil_program: the Quil program to run.

        :return: A Dict mapping classical memory names to values.
        :rtype: Dict[str, np.array]
        """
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil Program. Got {quil_program}.")

        classical_addresses = get_classical_addresses_from_program(quil_program)
        return self.connection._qvm_ng_run_program(quil_program=quil_program,
                                                   qvm_token=self.token,
                                                   simulation_method=None,
                                                   allocation_method=None,
                                                   classical_addresses=classical_addresses,
                                                   measurement_noise=None,
                                                   gate_noise=None)

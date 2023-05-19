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
from qcs_sdk import QCSClient
import qcs_sdk.qvm.api as qvm_api  # pyright: ignore [reportMissingModuleSource]

# TODO: Deprecate


class QVMClient:
    """
    Client for making requests to a Quantum Virtual Machine.
    """

    def __init__(self, *, client_configuration: QCSClient, request_timeout: float = 10.0) -> None:
        """
        Instantiate a new compiler client.

        :param client_configuration: Configuration for client.
        :param request_timeout: Timeout for requests, in seconds.
        """
        self.client = client_configuration
        self.timeout = request_timeout

    def get_version(self) -> str:
        """
        Get version info for QVM server.
        """
        return qvm_api.get_version_info(self.client)

    def run_program(self, request: qvm_api.MultishotRequest) -> qvm_api.MultishotResponse:
        """
        Run a Quil program and return its results.
        """
        return qvm_api.run(request, self.client)

    def run_and_measure_program(self, request: qvm_api.MultishotMeasureRequest) -> qvm_api.MultishotMeasureResponse:
        """
        Run and measure a Quil program, and return its results.
        """
        return qvm_api.run_and_measure(request, self.client)

    def measure_expectation(self, request: qvm_api.ExpectationRequest) -> qvm_api.ExpectationResponse:
        """
        Measure expectation value of Pauli operators given a defined state.
        """
        return qvm_api.measure_expectation(request, self.client)

    def get_wavefunction(self, request: qvm_api.WavefunctionRequest) -> qvm_api.WavefunctionResponse:
        """
        Run a program and retrieve the resulting wavefunction.
        """
        return qvm_api.get_wavefunction(request, self.client)

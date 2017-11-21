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

from six import integer_types

from pyquil.quil import Program
from ._base_connection import validate_run_items, TYPE_MULTISHOT, TYPE_MULTISHOT_MEASURE, get_job_id, BaseConnection


class QPUConnection(BaseConnection):
    """
    Represents a connection to the QPU (Quantum Processing Unit)
    """

    def __init__(self, async_endpoint='https://job.rigetti.com/beta', api_key=None, user_id=None):
        super(QPUConnection, self).__init__(async_endpoint=async_endpoint, api_key=api_key, user_id=user_id)

    def run(self, quil_program, classical_addresses, trials=1):
        """
        Run a pyQuil program on the QPU. This functionality is in beta.

        :param Program quil_program: Quil program to run on the QPU
        :param list classical_addresses: Currently unused
        :param int trials: Number of shots to take
        :return: A list of lists of bits. Each sublist corresponds to the values
                 in `classical_addresses`.
        :rtype: list
        """
        payload = self._run_payload(quil_program, classical_addresses, trials)

        response = self._post_json(self.async_endpoint + "/job", {"machine": "QPU", "program": payload})
        job = self.wait_for_job(get_job_id(response))
        return job.result()

    def run_async(self, quil_program, classical_addresses, trials=1):
        """
        Similar to run except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._run_payload(quil_program, classical_addresses, trials)
        response = self._post_json(self.async_endpoint + "/job", {"machine": "QPU", "program": payload})
        return get_job_id(response)

    def _run_payload(self, quil_program, classical_addresses, trials):
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        validate_run_items(classical_addresses)
        if not isinstance(trials, integer_types):
            raise TypeError("trials must be an integer")

        payload = {"type": TYPE_MULTISHOT,
                   "addresses": classical_addresses,
                   "trials": trials,
                   "quil-instructions": quil_program.out()}

        return payload

    def run_and_measure(self, quil_program, qubits, trials=1):
        """
        Run a pyQuil program on the QPU multiple times, measuring all the qubits in the QPU
        simultaneously at the end of the program each time. This functionality is in beta.

        :param Program quil_program: A Quil program.
        :param list qubits: The list of qubits to measure
        :param int trials: Number of shots to collect.
        :return: A list of a list of bits.
        :rtype: list
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)

        response = self._post_json(self.async_endpoint + "/job", {"machine": "QPU", "program": payload})
        job = self.wait_for_job(get_job_id(response))
        return job.result()

    def run_and_measure_async(self, quil_program, qubits, trials):
        """
        Similar to run_and_measure except that it returns a job id and doesn't wait for the program to be executed.
        See https://go.rigetti.com/connections for reasons to use this method.
        """
        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        response = self._post_json(self.async_endpoint + "/job", {"machine": "QPU", "program": payload})
        return get_job_id(response)

    def _run_and_measure_payload(self, quil_program, qubits, trials):
        if not isinstance(quil_program, Program):
            raise TypeError('quil_program must be a Quil program object')
        validate_run_items(qubits)
        if not isinstance(trials, int):
            raise TypeError('trials must be an integer')

        payload = {'type': TYPE_MULTISHOT_MEASURE,
                   'qubits': qubits,
                   'trials': trials,
                   'quil-instructions': quil_program.out()}

        return payload

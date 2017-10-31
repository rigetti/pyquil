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
Module for facilitating connections to the QPU.
"""

from pyquil.api import JobConnection, TYPE_MULTISHOT, TYPE_MULTISHOT_MEASURE, validate_run_items
from pyquil.quil import Program

ENDPOINT = 'https://job.rigetti.com/beta'


class NoParametersFoundException(Exception):
    pass


class QPUConnection(JobConnection):

    def __init__(self, device_name='Device', **kwargs):
        """
        :param str device_name: Unique identifier of the device in question
        """
        super(QPUConnection, self).__init__(**kwargs)
        self.device_name = device_name
        self.machine = 'QPU'
        self.session.headers = self.headers
        self.endpoint = ENDPOINT

    def run(self, quil_program, classical_addresses, trials=1):
        """
        Run a pyQuil program on the QPU. This functionality is in beta.

        :param Program quil_program: Quil program to run on the QPU
        :param list classical_addresses: Currently unused
        :param int trials: Number of shots to take
        :return: A job result
        :rtype: JobResult
        """
        if not isinstance(quil_program, Program):
            raise TypeError('quil_program must be a Quil program object')
        validate_run_items(classical_addresses)
        if not isinstance(trials, int):
            raise TypeError('trials must be an integer')

        payload = {'type': TYPE_MULTISHOT,
                   'addresses': classical_addresses,
                   'trials': trials,
                   'quil-instructions': quil_program.out(),
                   'device_id': self.device_name}

        res = self.post_job(payload, headers=self.headers)
        return self.process_response(res)

    def run_and_measure(self, quil_program, qubits, trials=1):
        """
        Run a pyQuil program on the QPU multiple times, measuring all the qubits in the QPU
        simultaneously at the end of the program each time. This functionality is in beta.

        :param Program quil_program: Quil program to run on the QPU
        :param list qubits: The list of qubits to return results for
        :param int trials: Number of shots to take
        :return: A job result
        :rtype: JobResult
        """
        if not isinstance(quil_program, Program):
            raise TypeError('quil_program must be a Quil program object')
        validate_run_items(qubits)
        if not isinstance(trials, int):
            raise TypeError('trials must be an integer')

        payload = {'type': TYPE_MULTISHOT_MEASURE,
                   'qubits': qubits,
                   'trials': trials,
                   'quil-instructions': quil_program.out(),
                   'device_id': self.device_name}

        res = self.post_job(payload, headers=self.headers)
        return self.process_response(res)

    def wavefunction(self, quil_program, classical_addresses=None):
        raise NotImplementedError("It's physically impossible to to retrieve a wavefunction from a real device")

    def expectation(self, prep_prog, operator_programs=None):
        raise NotImplementedError

    def bit_string_probabilities(self, quil_program):
        raise NotImplementedError

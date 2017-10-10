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

import requests

from pyquil.api import JobConnection, TYPE_MULTISHOT, TYPE_MULTISHOT_MEASURE, validate_run_items
from pyquil.job_results import RamseyResult, RabiResult, T1Result
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

    def get_qubits(self):
        """
        :return: A list of active qubit ids on this device.
        :rtype: list
        """
        config_dict = self.get_info()
        device_config = config_dict[self.device_name]
        return [qq['num'] for qq in device_config['qubits']]

    def rabi(self, qubit_id):
        """
        Runs a Rabi experiment on the given qubit.

        :param int qubit_id: Unique identifier of the qubit in question
        :return: A RabiResult object
        """
        payload = self.get_rabi_params(qubit_id)
        payload.update({
            'type': 'pyquillow',
            'experiment': 'rabi',
            'qcid': qubit_id,
            'device_id': self.device_name,
        })
        res = self.post_job(payload)
        return RabiResult.load_res(self, res)

    def ramsey(self, qubit_id):
        """
        Runs a Ramsey experiment on the given qubit.

        :param int qubit_id: Unique identifier of the qubit in question
        :return: A RamseyResult object
        """
        payload = self.get_ramsey_params(qubit_id)
        payload.update({
            'type': 'pyquillow',
            'experiment': 'ramsey',
            'qcid': qubit_id,
            'device_id': self.device_name,
        })
        res = self.post_job(payload)
        return RamseyResult.load_res(self, res)

    def t1(self, qubit_id):
        """
        Runs a T1 experiment on the given qubit.

        :param int qubit_id: Unique identifier of the qubit in question
        :return: A T1Result object
        """
        payload = self.get_t1_params(qubit_id)
        payload.update({
            'type': 'pyquillow',
            'experiment': 't1',
            'qcid': qubit_id,
            'device_id': self.device_name,
        })
        res = self.post_job(payload)
        return T1Result.load_res(self, res)

    def wavefunction(self, quil_program, classical_addresses=[]):
        return NotImplementedError

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

        res = self.post_job(payload, headers=self.json_headers)
        return self.process_response(res)

    def expectation(self, prep_prog, operator_programs=[Program()]):
        return NotImplementedError

    def bit_string_probabilities(self, quil_program):
        return NotImplementedError

    def get_info(self):
        """
        Gets information about what devices are currently available through the Forest API.

        :return: A JSON dictionary with configuration information.
        :rtype: dict
        """
        url = self.endpoint + '/config'
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'x-api-key': self.api_key,
            'x-user-id': self.user_id,
        }
        res = requests.get(url, headers=headers)
        config_json = res.json()
        return config_json

    def get_params(self, qcid, func):
        """
        Get and parse the configuration information from the Forest API.

        :param int qcid: Unique identifier of the qubit in question
        :param func: A function to apply to the qubit specific JSON dictionary of config info
        :return: A dictionary with the parameter info
        :rtype: dict
        """
        config_dict = self.get_info()
        try:
            device_config = [d for d in config_dict['devices'] if d['name'] == self.device_name][0]
            device_config = device_config['qubits']
        except (KeyError, IndexError):
            raise NoParametersFoundException('Device {} not found.'.format(self.device_name))
        for qc in device_config:
            if qc['num'] == qcid:
                return func(qc)
        raise NoParametersFoundException

    def get_rabi_params(self, qcid):
        """
        Gets the current Rabi experiment parameters for a specific qubit on a specific device.

        :param int qcid: Unique identifier of the qubit in question
        :return: A dictionary with the parameter info
        :rtype: dict
        """
        def rabi_parse(qc):
            rabi_params = qc['rabi_params']
            return {
                'start': rabi_params['start'],
                'stop': rabi_params['stop'],
                'step': rabi_params['step'],
                'time': rabi_params['time'],
            }
        return self.get_params(qcid, rabi_parse)

    def get_ramsey_params(self, qcid):
        """
        Gets the current Ramsey experiment parameters for a specific qubit on a specific device.

        :param int qcid: Unique identifier of the qubit in question
        :return: A dictionary with the parameter info
        :rtype: dict
        """
        def ramsey_parse(qc):
            ramsey_params = qc['ramsey_params']
            return {
                'start': ramsey_params['start'],
                'stop': ramsey_params['stop'],
                'step': ramsey_params['step'],
                'detuning': ramsey_params['detuning'],
            }
        return self.get_params(qcid, ramsey_parse)

    def get_t1_params(self, qcid):
        """
        Gets the current T1 experiment parameters for a specific qubit on a specific device.

        :param int qcid: Unique identifier of the qubit in question
        :return: A dictionary with the parameter info
        :rtype: dict
        """
        def t1_parse(qc):
            t1_params = qc['t1_params']
            return {
                'start': t1_params['start'],
                'stop': t1_params['stop'],
                'num_pts': t1_params['num_pts'],
            }
        return self.get_params(qcid, t1_parse)

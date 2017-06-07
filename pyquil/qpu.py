from copy import deepcopy
import json

import requests

from pyquil.forest import Connection, ENDPOINT, USER_ID, API_KEY
import pyquil.quil as pq
from pyquil.job_results import RamseyResult, RabiResult, T1Result


class QPUConnection(Connection):

    def __init__(self, device_name, *args):
        super(Connection, self).__init__(*args)
        self.device_name = device_name
        # overloads the connection information with endpoints for the jobqueue
        self.api_key = ""
        self.json_headers = {
            'Content-Type' : 'application/json; charset=utf-8',
            'x-api-key' : self.api_key,
            'x-user-id' : self.user_id,
        }
        self.text_headers = deepcopy(self.json_headers)
        self.text_headers['Content-Type'] = 'application/text; charset=utf-8'

    def get_qubits(self):
        """
        :return: A list of active qubit ids on this device.
        :rtype: list
        """
        config_dict = get_info()
        device_config = config_dict[self.device_name]
        return [qq['num'] for qq in device_config['qubits']]

    def rabi(self, qubit_id):
        payload = get_rabi_params(self.device_name, qubit_id)
        payload.update({
            'type': 'pyquillow',
            'experiment': 'rabi',
            'qcid': qubit_id

        })
        res = self.post_job(payload)
        return RabiResult.load_res(self, res)

    def ramsey(self, qubit_id):
        payload = get_ramsey_params(self.device_name, qubit_id)
        payload.update({
            'type': 'pyquillow',
            'experiment': 'ramsey',
            'qcid': qubit_id

        })
        res = self.post_job(payload)
        return RamseyResult.load_res(self, res)

    def t1(self, qubit_id):
        payload = get_t1_params(self.device_name, qubit_id)
        payload.update({
            'type': 'pyquillow',
            'experiment': 't1',
            'qcid': qubit_id
        })
        res = self.post_job(payload)
        return T1Result.load_res(self, res)

    def version(self):
        """
        This returns a JSON blob with some information about the currently available chip, e.g.
        number of qubits, t1, t2, and whether the chip is live for execution or not.
        :return:
        """
        payload = {
            'type': 'pyquillow',
            'experiment': 'version_query'
        }
        res = self.post_job(payload)
        return res

    def ping(self):
        """
        This returns a JSON blob with some information about the currently available chip, e.g.
        number of qubits, t1, t2, and whether the chip is live for execution or not.
        :return:
        """
        return self.version()

    def wavefunction(self, quil_program, classical_addresses=[]):
        return NotImplementedError

    def run(self, quil_program, classical_addresses, trials=1):
        return NotImplementedError

    def run_and_measure(self, quil_program, qubits, trials=1):
        return NotImplementedError

    def expectation(self, prep_prog, operator_programs=[pq.Program()]):
        return NotImplementedError

    def bit_string_probabilities(self, quil_program):
        return NotImplementedError


def get_info():
    url = ENDPOINT + "/config"
    headers = {
            'Content-Type' : 'application/json; charset=utf-8',
            'x-api-key' : API_KEY,
            'x-user-id' : USER_ID,
    }
    res = requests.get(url, headers=headers)
    config_json = json.loads(res.content.decode("utf-8"))
    return config_json


def get_rabi_params(device_name, qcid):
    config_dict = get_info()
    for qc in config_dict[device_name]['qubits']:
        if qc['num'] == qcid:
            rabi_params = qc['rabi_params']
            return {
                'start': rabi_params['start'],
                'stop': rabi_params['stop'],
                'step': rabi_params['step'],
                'time': rabi_params['time'],
            }


def get_ramsey_params(device_name, qcid):
    config_dict = get_info()
    for qc in config_dict[device_name]['qubits']:
        if qc['num'] == qcid:
            ramsey_params = qc['ramsey_params']
            return {
                'start': ramsey_params['start'],
                'stop': ramsey_params['stop'],
                'step': ramsey_params['step'],
                'detuning': ramsey_params['detuning'],
            }


def get_t1_params(device_name, qcid):
    config_dict = get_info()
    for qc in config_dict[device_name]['qubits']:
        if qc['num'] == qcid:
            t1_params = qc['t1_params']
            return {
                'start': t1_params['start'],
                'stop': t1_params['stop'],
                'num_pts': t1_params['num_pts'],
            }

from copy import deepcopy
import json
import requests

from pyquil.api import JobConnection, ENDPOINT, USER_ID, API_KEY
import pyquil.quil as pq
from pyquil.job_results import RamseyResult, RabiResult, T1Result


class NoParametersFoundException(Exception):
    pass


class QPUConnection(JobConnection):

    def __init__(self, device_name, *args):
        super(QPUConnection, self).__init__(*args)
        self.device_name = device_name
        self.machine = "QPU"

    def get_qubits(self):
        """
        :return: A list of active qubit ids on this device.
        :rtype: list
        """
        config_dict = get_info()
        device_config = config_dict[self.device_name]
        return [qq['num'] for qq in device_config['qubits']]

    def rabi(self, qubit_id):
        """
        Runs a Rabi experiment on the given qubit
        :param int qubit_id:
        :return: A RabiResult object
        """
        payload = get_rabi_params(self.device_name, qubit_id)
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
         Runs a Ramsey experiment on the given qubit
         :param int qubit_id:
         :return: A RamseyResult object
         """
        payload = get_ramsey_params(self.device_name, qubit_id)
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
         Runs a T1 experiment on the given qubit
         :param int qubit_id:
         :return: A T1Result object
         """
        payload = get_t1_params(self.device_name, qubit_id)
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
        return NotImplementedError

    def run_and_measure(self, quil_program, qubits, trials=1):
        return NotImplementedError

    def expectation(self, prep_prog, operator_programs=[pq.Program()]):
        return NotImplementedError

    def bit_string_probabilities(self, quil_program):
        return NotImplementedError


def get_info():
    """
    Gets information about what devices are currently available through the Forest API.
    :return: A JSON dictionary with configuration information.
    """
    url = ENDPOINT + "/config"
    headers = {
            'Content-Type' : 'application/json; charset=utf-8',
            'x-api-key' : API_KEY,
            'x-user-id' : USER_ID,
    }
    res = requests.get(url, headers=headers)
    config_json = json.loads(res.content.decode("utf-8"))
    return config_json


def get_params(device_name, qcid, func):
    """
    Get and parse the configuration information from the Forest API.
    :param str device_name: The device to get info for
    :param int qcid: The qubit number to look at
    :param func: A function to apply to the qubit specific JSON dictionary of config info
    :return:
    """
    config_dict = get_info()
    try:
        device_config = filter(lambda dd: dd['name'] == device_name, config_dict['devices'])[0]
        device_config = device_config['qubits']
    except (KeyError, IndexError):
        raise NoParametersFoundException, "Device with name {} not found.".format(device_name)
    for qc in device_config:
        if qc['num'] == qcid:
            return func(qc)
    raise NoParametersFoundException


def get_rabi_params(device_name, qcid):
    """
    Gets the current Rabi experiment parameters for a specific qubit on a specific device
    :param str device_name:
    :param int qcid:
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
    return get_params(device_name, qcid, rabi_parse)


def get_ramsey_params(device_name, qcid):
    """
    Gets the current Ramsey experiment parameters for a specific qubit on a specific device
    :param str device_name:
    :param int qcid:
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
    return get_params(device_name, qcid, ramsey_parse)


def get_t1_params(device_name, qcid):
    """
    Gets the current T1 experiment parameters for a specific qubit on a specific device
    :param str device_name:
    :param int qcid:
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
    return get_params(device_name, qcid, t1_parse)

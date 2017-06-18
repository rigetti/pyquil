import mock
import pytest
import json

from pyquil.qpu import get_ramsey_params, get_rabi_params, get_t1_params, \
    NoParametersFoundException, QPUConnection

@pytest.fixture()
def sample_config():
    return {'devices': [
        {'name': "Z12-13-C4a2",
         'qubits': [
             {'num': 2, 't1': 20.e-6, 't2': 15.e-6, 'ssr_fidelity': 0.923,
              'rabi_params': {
                  'start': 0.01,
                  'stop': 20,
                  'step': 0.2,
                  'time': 160.},
              'ramsey_params': {
                  'start': 0.01,
                  'stop': 20,
                  'step': 0.2,
                  'detuning': 0.5},
              't1_params': {
                  'start': 0.01,
                  'stop': 20,
                  'num_pts': 25}
              },
             {'num': 3, 't1': 20.e-6, 't2': 15.e-6, 'ssr_fidelity': 0.923,
              'rabi_params': {
                  'start': 0.01,
                  'stop': 20,
                  'step': 0.2,
                  'time': 160.},
              'ramsey_params': {
                  'start': 0.01,
                  'stop': 20,
                  'step': 0.2,
                  'detuning': 0.5},
              't1_params': {
                  'start': 0.01,
                  'stop': 20,
                  'num_pts': 25}
              }
         ]}
    ]}


def test_qpu_rabi(sample_config):
    with mock.patch('pyquil.qpu.get_info') as m_get_info:
        m_get_info.return_value = sample_config
        with mock.patch.object(QPUConnection, "post_json") as m_post:
            mocked_response = {'jobId': "ASHJKSDUK", 'result': [[1]]}
            m_post.return_value.content = json.dumps(mocked_response)
            device_name = "Z12-13-C4a2"
            qpu = QPUConnection(device_name)
            qpu.rabi(2)
            message = {'machine': 'QPU',
                       'program': {'start': 0.01, 'step': 0.2, 'experiment': 'rabi', 'time': 160.0,
                                   'type': 'pyquillow', 'qcid': 2, 'stop': 20,
                                   'device_id': device_name},
                       'results': '', 'jobId': ''}
            # userIds are passed in the message, but shouldn't be tested against
            assert message == anon_message(m_post.call_args[0][0])


def test_qpu_ramsey(sample_config):
    with mock.patch('pyquil.qpu.get_info') as m_get_info:
        m_get_info.return_value = sample_config
        with mock.patch.object(QPUConnection, "post_json") as m_post:
            mocked_response = {'jobId': "ASHJKSDUK", 'result': [[1]]}
            m_post.return_value.content = json.dumps(mocked_response)
            device_name = "Z12-13-C4a2"
            qpu = QPUConnection(device_name)
            qpu.ramsey(3)
            message = {'machine': 'QPU',
                       'program': {'start': 0.01, 'step': 0.2, 'experiment': 'ramsey',
                                   'type': 'pyquillow', 'qcid': 3, 'stop': 20, 'detuning': 0.5,
                                   'device_id': device_name},
                       'results': '', 'jobId': ''}
            # userIds are passed in the message, but shouldn't be tested against
            assert message == anon_message(m_post.call_args[0][0])


def test_qpu_t1(sample_config):
    with mock.patch('pyquil.qpu.get_info') as m_get_info:
        m_get_info.return_value = sample_config
        with mock.patch.object(QPUConnection, "post_json") as m_post:
            mocked_response = {'jobId': "ASHJKSDUK", 'result': [[1]]}
            m_post.return_value.content = json.dumps(mocked_response)
            device_name = "Z12-13-C4a2"
            qpu = QPUConnection(device_name)
            qpu.t1(2)
            message = {'machine': 'QPU',
                       'program': {'start': 0.01, 'experiment': 't1', 'type': 'pyquillow',
                                   'qcid': 2, 'stop': 20, 'num_pts': 25, 'device_id':device_name},
                       'results': '', 'jobId': ''}
            # userIds are passed in the message, but shouldn't be tested against
            assert message == anon_message(m_post.call_args[0][0])


def anon_message(message):
    message.pop('userId', None)
    return message


def test_get_params(sample_config):
    with mock.patch('pyquil.qpu.get_info') as m_get_info:
        m_get_info.return_value = sample_config
        assert get_rabi_params("Z12-13-C4a2", 2) == {'start': 0.01, 'stop': 20, 'step': 0.2,
                                                     'time': 160.}
        assert get_ramsey_params("Z12-13-C4a2", 2) == {'start': 0.01, 'stop': 20, 'step': 0.2,
                                                       'detuning': 0.5}
        assert get_t1_params("Z12-13-C4a2", 3) == {'start': 0.01, 'stop': 20, 'num_pts': 25}

        getters = [get_rabi_params, get_ramsey_params, get_t1_params]
        for getter in getters:
            with pytest.raises(NoParametersFoundException):
                getter("Z12-13-C4a2", 5)
            with pytest.raises(NoParametersFoundException):
                getter("Z12-13-C0a2", 2)

import pytest
import os

from pyquil.api._config import PyquilConfig
from pyquil.api._errors import UserMessageError
from pyquil.tests.utils import api_fixture_path

test_config_paths = {
    'QCS_CONFIG': api_fixture_path('qcs_config.ini'),
    'FOREST_CONFIG': api_fixture_path('forest_config.ini'),
}


def test_config_qcs_auth_headers_valid_user_token():
    config = PyquilConfig(test_config_paths)
    config.user_auth_token = {
        'access_token': 'secret',
        'refresh_token': 'supersecret',
        'scope': 'openid profile'
    }
    config.qmi_auth_token = None
    assert 'Authorization' in config.qcs_auth_headers
    assert 'X-QMI-AUTH-TOKEN' not in config.qcs_auth_headers
    assert config.qcs_auth_headers['Authorization'] == 'Bearer secret'


def test_config_qcs_auth_headers_valid_qmi_token():
    config = PyquilConfig(test_config_paths)
    config.user_auth_token = None
    config.qmi_auth_token = {'access_token': 'secret', 'refresh_token': 'supersecret'}
    assert 'Authorization' not in config.qcs_auth_headers
    assert 'X-QMI-AUTH-TOKEN' in config.qcs_auth_headers
    assert config.qcs_auth_headers['X-QMI-AUTH-TOKEN'] == 'secret'


def test_config_assert_valid_auth_credential():
    print("Fixture path: ", api_fixture_path('qmi_auth_token_invalid.json'))
    config = PyquilConfig(test_config_paths)
    config.config_parsers['QCS_CONFIG'].set('Rigetti Forest',
                                            'qmi_auth_token_path',
                                            api_fixture_path('qmi_auth_token_invalid.json'))
    config.config_parsers['QCS_CONFIG'].set('Rigetti Forest',
                                            'user_auth_token_path',
                                            api_fixture_path('user_auth_token_invalid.json'))
    config._parse_auth_tokens()
    assert config.user_auth_token is None
    assert config.qmi_auth_token is None
    with pytest.raises(UserMessageError):
        config.assert_valid_auth_credential()

    config.config_parsers['QCS_CONFIG'].set('Rigetti Forest',
                                            'qmi_auth_token_path',
                                            api_fixture_path('qmi_auth_token_valid.json'))
    config.config_parsers['QCS_CONFIG'].set('Rigetti Forest',
                                            'user_auth_token_path',
                                            api_fixture_path('user_auth_token_valid.json'))
    config._parse_auth_tokens()
    assert config.user_auth_token is not None
    assert config.qmi_auth_token is not None
    config.assert_valid_auth_credential()

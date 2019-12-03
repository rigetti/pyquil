import pytest
import os
import requests_mock
from pyquil.api._config import PyquilConfig


def fixture_path(path: str) -> str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, 'data', path)


test_config_paths = {
    'QCS_CONFIG': fixture_path('qcs_config.test'),
    'FOREST_CONFIG': fixture_path('forest_config.test'),
}


def test_config_qcs_auth_headers_valid_user_token():
    config = PyquilConfig(test_config_paths)
    config.user_auth_token = {
        'access_token': 'secret',
        'refresh_token': 'supersecret',
        'scope': 'openid profile'}
    config.qmi_auth_token = None
    assert 'Authorization' in config.qcs_auth_headers
    assert 'X-QMI-AUTH-TOKEN' not in config.qcs_auth_headers
    assert config.qcs_auth_headers['Authorization'] == 'Bearer secret'


def test_config_qcs_auth_headers_valid_qmi_token():
    config = PyquilConfig(test_config_paths)
    config.user_auth_token = None
    config.qmi_auth_token = {
        'access_token': 'secret',
        'refresh_token': 'supersecret'}
    assert 'Authorization' not in config.qcs_auth_headers
    assert 'X-QMI-AUTH-TOKEN' in config.qcs_auth_headers
    assert config.qcs_auth_headers['X-QMI-AUTH-TOKEN'] == 'secret'


def test_config_assert_valid_auth_credential():
    config = PyquilConfig(test_config_paths)
    config.configparsers['QCS_CONFIG'].set(
        'Rigetti Forest', 'qmi_auth_token_path',
        fixture_path('qmi_auth_token_invalid.json'))
    config.configparsers['QCS_CONFIG'].set(
        'Rigetti Forest', 'user_auth_token_path',
        fixture_path('user_auth_token_invalid.json'))
    config._parse_auth_tokens()
    assert config.user_auth_token is None
    assert config.qmi_auth_token is None
    with pytest.raises(ValueError) as excinfo:
        config.assert_valid_auth_credential()

    config.configparsers['QCS_CONFIG'].set(
        'Rigetti Forest', 'qmi_auth_token_path',
        fixture_path('qmi_auth_token_valid.json'))
    config.configparsers['QCS_CONFIG'].set(
        'Rigetti Forest', 'user_auth_token_path',
        fixture_path('user_auth_token_valid.json'))
    config._parse_auth_tokens()
    assert config.user_auth_token is not None
    assert config.qmi_auth_token is not None
    config.assert_valid_auth_credential()

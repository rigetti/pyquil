import pytest

from pyquil.api._config import PyquilConfig
from pyquil.api._errors import UserMessageError
from pyquil.tests.utils import api_fixture_path

TEST_CONFIG_PATHS = {
    "QCS_CONFIG": api_fixture_path("qcs_config.ini"),
    "FOREST_CONFIG": api_fixture_path("forest_config.ini"),
}


def test_config_qcs_auth_headers_valid_user_token():
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.user_auth_token = {
        "access_token": "secret",
        "refresh_token": "supersecret",
        "scope": "openid profile",
    }
    config.qmi_auth_token = None
    assert "Authorization" in config.qcs_auth_headers
    assert "X-QMI-AUTH-TOKEN" not in config.qcs_auth_headers
    assert config.qcs_auth_headers["Authorization"] == "Bearer secret"


def test_config_qcs_auth_headers_valid_qmi_token():
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.user_auth_token = None
    config.qmi_auth_token = {"access_token": "secret", "refresh_token": "supersecret"}
    assert "Authorization" not in config.qcs_auth_headers
    assert "X-QMI-AUTH-TOKEN" in config.qcs_auth_headers
    assert config.qcs_auth_headers["X-QMI-AUTH-TOKEN"] == "secret"


def test_config_assert_valid_auth_credential():
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "qmi_auth_token_path", api_fixture_path("qmi_auth_token_invalid.json")
    )
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", api_fixture_path("user_auth_token_invalid.json")
    )
    config._parse_auth_tokens()
    assert config.user_auth_token is None
    assert config.qmi_auth_token is None
    with pytest.raises(UserMessageError):
        config.assert_valid_auth_credential()

    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "qmi_auth_token_path", api_fixture_path("qmi_auth_token_valid.json")
    )
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", api_fixture_path("user_auth_token_valid.json")
    )
    config._parse_auth_tokens()
    assert config.user_auth_token is not None
    assert config.qmi_auth_token is not None
    config.assert_valid_auth_credential()


def test_engagement_not_requested_when_unnecessary():
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.config_parsers["FOREST_CONFIG"].set(
        "Rigetti Forest", "qpu_compiler_address", "tcp://fake_compiler:5555"
    )
    config.config_parsers["FOREST_CONFIG"].set(
        "Rigetti Forest", "qpu_endpoint_address", "tcp://fake_qpu:5555"
    )
    assert config.qpu_compiler_url == "tcp://fake_compiler:5555"
    assert config.qpu_url == "tcp://fake_qpu:5555"
    assert config.engagement is None

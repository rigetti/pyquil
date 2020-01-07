import pytest
import requests_mock
import urllib.parse

from pyquil.api._base_connection import ForestSession
from pyquil.api._config import PyquilConfig
from pyquil.api._errors import UserMessageError
from pyquil.tests.utils import api_fixture_path


SUCCESSFUL_ENGAGEMENT_RESPONSE = {
    "engage": {
        "success": True,
        "message": "Good job, you engaged all by yourself",
        "engagement": {
            "type": "RESERVATION",
            "qpu": {
                "endpoint": "tcp://fake.url:12345",
                "credentials": {
                    "clientPublic": "abc123",
                    "clientSecret": "abc123",
                    "serverPublic": "abc123",
                },
            },
            "compiler": {"endpoint": "tcp://fake.url:12346"},
            "expiresAt": "9999999999.0",
        },
    }
}


FAILED_ENGAGEMENT_RESPONSE = {
    "engage": {"success": False, "message": "That did not work", "engagement": None}
}


TEST_CONFIG_PATHS = {
    "QCS_CONFIG": api_fixture_path("qcs_config.ini"),
    "FOREST_CONFIG": api_fixture_path("forest_config.ini"),
}


def test_forest_session_request_authenticated_with_user_token():
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "qmi_auth_token_path", api_fixture_path("qmi_auth_token_invalid.json")
    )
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", api_fixture_path("user_auth_token_valid.json")
    )
    config.config_parsers["QCS_CONFIG"].set("Rigetti Forest", "url", "mock://forest")
    config._parse_auth_tokens()

    session = ForestSession(config=config)
    mock_adapter = requests_mock.Adapter()
    session.mount("mock", mock_adapter)

    url = "%s/devices" % config.forest_url
    headers = {
        # access token from ./data/user_auth_token_valid.json.
        "Authorization": "Bearer secret"
    }
    mock_adapter.register_uri("GET", url, status_code=200, json=[{"id": 0}], headers=headers)

    devices = session.get(url).json()
    assert len(devices) == 1
    assert devices[0]["id"] == 0


def test_forest_session_request_authenticated_with_qmi_auth():
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "qmi_auth_token_path", api_fixture_path("qmi_auth_token_valid.json")
    )
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", api_fixture_path("user_auth_token_invalid.json")
    )
    config.config_parsers["QCS_CONFIG"].set("Rigetti Forest", "url", "mock://forest")
    config._parse_auth_tokens()

    session = ForestSession(config=config)
    mock_adapter = requests_mock.Adapter()
    session.mount("mock", mock_adapter)

    url = "%s/devices" % config.forest_url
    headers = {
        # access token from ./data/qmi_auth_token_valid.json.
        "X-QMI-AUTH-TOKEN": "secret"
    }
    mock_adapter.register_uri("GET", url, status_code=200, json=[{"id": 0}], headers=headers)

    devices = session.get(url).json()
    assert len(devices) == 1
    assert devices[0]["id"] == 0


def test_forest_session_request_refresh_user_auth_token():
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "qmi_auth_token_path", api_fixture_path("qmi_auth_token_invalid.json")
    )
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", api_fixture_path("user_auth_token_valid.json")
    )
    config.config_parsers["QCS_CONFIG"].set("Rigetti Forest", "url", "mock://forest")
    config._parse_auth_tokens()

    session = ForestSession(config=config)
    mock_adapter = requests_mock.Adapter()
    session.mount("mock", mock_adapter)

    url = "%s/devices" % config.forest_url
    response_list = [
        # access token from ./data/user_auth_token_valid.json.
        {
            "status_code": 401,
            "json": {"error": "user_unauthorized"},
            "headers": {"Authorization": "Bearer secret"},
        },
        # access token from new_user_auth_token.
        {"status_code": 200, "json": [{"id": 0}], "headers": {"Authorization": "Bearer secret2"}},
    ]
    mock_adapter.register_uri("GET", url, response_list=response_list)

    refresh_url = "%s/auth/idp/oauth2/v1/token" % config.forest_url

    def refresh_matcher(request):
        body = dict(urllib.parse.parse_qsl(request.text))
        return (body["refresh_token"] == "supersecret") and (body["grant_type"] == "refresh_token")

    new_user_auth_token = {
        "access_token": "secret2",
        "refresh_token": "supersecret2",
        "scope": "openid offline_access profile",
    }
    mock_adapter.register_uri(
        "POST",
        refresh_url,
        status_code=200,
        json=new_user_auth_token,
        additional_matcher=refresh_matcher,
    )

    # refresh will write the new auth tokens to file. Do not over-write text fixture data.
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "qmi_auth_token_path", "/tmp/qmi_auth_token_invalid.json"
    )
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", "/tmp/user_auth_token_valid.json"
    )
    devices = session.get(url).json()
    assert len(devices) == 1
    assert devices[0]["id"] == 0


def test_forest_session_request_refresh_qmi_auth_token():
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "qmi_auth_token_path", api_fixture_path("qmi_auth_token_valid.json")
    )
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", api_fixture_path("user_auth_token_invalid.json")
    )
    config.config_parsers["QCS_CONFIG"].set("Rigetti Forest", "url", "mock://forest")
    config._parse_auth_tokens()

    session = ForestSession(config=config)
    mock_adapter = requests_mock.Adapter()
    session.mount("mock", mock_adapter)

    url = "%s/devices" % config.forest_url
    response_list = [
        # access token from ./data/user_auth_token_valid.json.
        {
            "status_code": 401,
            "json": {"error": "user_unauthorized"},
            "headers": {"X-QMI-AUTH-TOKEN": "ok"},
        },
        # access token from new_user_auth_token.
        {"status_code": 200, "json": [{"id": 0}], "headers": {"X-QMI-AUTH-TOKEN": "ok"}},
    ]
    mock_adapter.register_uri("GET", url, response_list=response_list)

    refresh_url = "%s/auth/qmi/refresh" % config.forest_url

    def refresh_matcher(request):
        body = request.json()
        return (body["refresh_token"] == "supersecret") and (body["access_token"] == "ok")

    new_user_auth_token = {"access_token": "secret2", "refresh_token": "supersecret2"}
    mock_adapter.register_uri(
        "POST",
        refresh_url,
        status_code=200,
        json=new_user_auth_token,
        additional_matcher=refresh_matcher,
    )

    # refresh will write the new auth tokens to file. Do not over-write text fixture data.
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "qmi_auth_token_path", "/tmp/qmi_auth_token_invalid.json"
    )
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", "/tmp/user_auth_token_valid.json"
    )
    devices = session.get(url).json()
    assert len(devices) == 1
    assert devices[0]["id"] == 0


def test_forest_session_request_engagement():
    """
    The QPU Endpoint address provided by engagement should be available to the
      PyQuilConfig object.
    """
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.config_parsers["FOREST_CONFIG"].remove_section("Rigetti Forest")
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", api_fixture_path("user_auth_token_invalid.json")
    )
    config.config_parsers["QCS_CONFIG"].set("Rigetti Forest", "url", "mock://forest")
    config.config_parsers["QCS_CONFIG"].set("Rigetti Forest", "dispatch_url", "mock://dispatch")
    config._parse_auth_tokens()

    session = ForestSession(config=config, lattice_name="fake-lattice")
    mock_adapter = requests_mock.Adapter()
    session.mount("mock", mock_adapter)

    url = config.dispatch_url
    response_list = [
        # access token from ./data/user_auth_token_valid.json.
        {"status_code": 200, "json": {"data": SUCCESSFUL_ENGAGEMENT_RESPONSE}}
    ]
    mock_adapter.register_uri("POST", url, response_list=response_list)

    assert (
        session.qpu_url == SUCCESSFUL_ENGAGEMENT_RESPONSE["engage"]["engagement"]["qpu"]["endpoint"]
    )
    assert (
        session.qpu_compiler_url
        == SUCCESSFUL_ENGAGEMENT_RESPONSE["engage"]["engagement"]["compiler"]["endpoint"]
    )


def test_forest_session_engagement_not_requested_if_config_present():
    """
    Engagement is the source-of-last-resort for configuration data. If all endpoints are
      provided elsewhere, then engagement should never be requested.
    """
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", api_fixture_path("user_auth_token_invalid.json")
    )
    config.config_parsers["QCS_CONFIG"].set("Rigetti Forest", "url", "mock://forest")
    config.config_parsers["QCS_CONFIG"].set("Rigetti Forest", "dispatch_url", "mock://dispatch")
    config._parse_auth_tokens()

    session = ForestSession(config=config, lattice_name="fake-lattice")
    mock_adapter = requests_mock.Adapter()
    session.mount("mock", mock_adapter)

    url = config.dispatch_url
    response_list = [
        # access token from ./data/user_auth_token_valid.json.
        {"status_code": 200, "json": {"data": SUCCESSFUL_ENGAGEMENT_RESPONSE}}
    ]
    mock_adapter.register_uri("POST", url, response_list=response_list)

    assert len(mock_adapter.request_history) == 0
    assert session.qpu_url == config.config_parsers["FOREST_CONFIG"].get(
        "Rigetti Forest", "qpu_endpoint_address"
    )


def test_forest_session_request_engagement_failure():
    """
    If engagement fails, no QPU URL is available to the client.
    """
    config = PyquilConfig(TEST_CONFIG_PATHS)
    config.config_parsers["FOREST_CONFIG"].remove_section("Rigetti Forest")
    config.config_parsers["QCS_CONFIG"].set(
        "Rigetti Forest", "user_auth_token_path", api_fixture_path("user_auth_token_invalid.json")
    )
    config.config_parsers["QCS_CONFIG"].set("Rigetti Forest", "url", "mock://forest")
    config.config_parsers["QCS_CONFIG"].set("Rigetti Forest", "dispatch_url", "mock://dispatch")
    config._parse_auth_tokens()

    session = ForestSession(config=config, lattice_name="fake-lattice")
    mock_adapter = requests_mock.Adapter()
    session.mount("mock", mock_adapter)

    url = config.dispatch_url
    response_list = [
        # access token from ./data/user_auth_token_valid.json.
        {"status_code": 200, "json": {"data": FAILED_ENGAGEMENT_RESPONSE}}
    ]
    mock_adapter.register_uri("POST", url, response_list=response_list)

    with pytest.raises(UserMessageError):
        assert session.qpu_url is None

##############################################################################
# Copyright 2016-2018 Rigetti Computing
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
import json
import warnings
from configparser import ConfigParser, NoSectionError, NoOptionError
from os import environ, path
from os.path import expanduser, abspath
from types import MappingProxyType
from typing import Callable, Dict, Iterable, Mapping, Optional, TYPE_CHECKING, cast

from pyquil.api._errors import UserMessageError
from pyquil.api._logger import logger

if TYPE_CHECKING:
    from pyquil.api._base_connection import Engagement

# `.qcs_config` is for content (mostly) related to QCS: the QCS front end stacks endpoint (`url`)
# for querying all QCS data: devices, reservations, etc. etc., and the `exec_on_engage` that the
# daemon will call during an engage event
QCS_CONFIG = "QCS_CONFIG"

# `.forest_config`, for content related to the Forest SDK, such as ip addresses for the various
# servers to which users submit quil & jobs (qvm, compiler, qpu, etc.)
FOREST_CONFIG = "FOREST_CONFIG"
CONFIG_PATHS = cast(
    Mapping[str, str],
    MappingProxyType({QCS_CONFIG: "~/.qcs_config", FOREST_CONFIG: "~/.forest_config"}),
)


class PyquilConfig(object):
    """
    The PyQuilConfig object holds the configuration necessary to communicate with Rigetti systems
        such as authentication tokens, endpoint URLs, and file paths for configuration files
        on disk.

    :attribute get_engagement: A callback to fetch a currently valid engagement from which to read
        configuration parameters (i.e., QPU_URL) as needed. This allows the engagement to be fetched
        and maintained elsewhere (i.e., by ForestSession or manually).
    """

    FOREST_URL = {
        "env": "FOREST_SERVER_URL",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "url",
        "default": "https://forest-server.qcs.rigetti.com",
    }

    API_KEY = {
        "env": "FOREST_API_KEY",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "key",
        "default": None,
    }

    USER_ID = {
        "env": "FOREST_USER_ID",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "user_id",
        "default": None,
    }

    DISPATCH_URL = {
        "env": "FOREST_DISPATCH_URL",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "dispatch_url",
        "default": "https://dispatch.services.qcs.rigetti.com/graphql",
    }

    ENGAGE_CMD = {
        "env": "QMI_ENGAGE_CMD",
        "file": QCS_CONFIG,
        "section": "QPU",
        "name": "exec_on_engage",
        "default": "",
    }

    QMI_AUTH_TOKEN_PATH = {
        "env": "QMI_AUTH_TOKEN_PATH",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "qmi_auth_token_path",
        "default": "~/.qcs/qmi_auth_token",
    }

    USER_AUTH_TOKEN_PATH = {
        "env": "USER_AUTH_TOKEN_PATH",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "user_auth_token_path",
        "default": "~/.qcs/user_auth_token",
    }

    QCS_URL = {
        "env": "QCS_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "qcs_url",
        "default": "https://qcs.rigetti.com",
    }

    QPU_URL = {
        "env": "QPU_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "qpu_endpoint_address",
        "engagement_key": "qpu_endpoint",
        "default": None,
    }

    QVM_URL = {
        "env": "QVM_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "qvm_address",
        "default": "http://127.0.0.1:5000",
    }

    QUILC_URL = {
        "env": "QUILC_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "quilc_address",
        "default": "tcp://127.0.0.1:5555",
    }

    QPU_COMPILER_URL = {
        "env": "QPU_COMPILER_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "qpu_compiler_address",
        "engagement_key": "qpu_compiler_endpoint",
        "default": None,
    }

    def __init__(self, config_paths: Mapping[str, str] = CONFIG_PATHS):
        """
        :param config_paths: The paths to the various configuration files read by pyQuil.
        """

        # The engagement callback can be added by config consumers after construction
        self.get_engagement: Optional[Callable[..., Optional[Engagement]]] = lambda: None

        # Whether engagement has been requested in order to provide any config values
        self._engagement_requested = False

        self.config_parsers: Dict[str, ConfigParser] = {}
        for env_name, default_path in config_paths.items():
            default_path = expanduser(default_path)
            path = environ.get(env_name, default_path)

            cp = ConfigParser()
            cp.read(abspath(path))
            self.config_parsers[env_name] = cp
        self._parse_auth_tokens()

    def _parse_auth_tokens(self) -> None:
        self.user_auth_token = _parse_auth_token(
            self.user_auth_token_path, ["access_token", "refresh_token"]
        )
        self.qmi_auth_token = _parse_auth_token(
            self.qmi_auth_token_path, ["access_token", "refresh_token"]
        )

    def _env_or_config_or_default(
        self,
        env: Optional[str] = None,
        file: Optional[str] = None,
        section: Optional[str] = None,
        name: Optional[str] = None,
        default: Optional[str] = None,
        engagement_key: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get the value of the environment variable or config file value.
        The environment variable takes precedence.

        :param env: The environment variable name.
        :param name: The config file key.
        :param engagement_key: The attribute name by which this value can be read from
            an engagement. If None, then this value is not provided by engagement.
        :return: The value or None if not found
        """

        # If it's an envvar, that overrides all other options
        assert env is not None
        env_val = environ.get(env)
        if env_val is not None:
            return env_val

        # Otherwise, use the values in the config files from disk
        assert file is not None and section is not None and name is not None
        try:
            return self.config_parsers[file].get(section, name)
        except (NoSectionError, NoOptionError, KeyError):
            pass
        """
        If no local configuration is available, certain values are provided
            by the dispatch service.
        """
        assert self.get_engagement is not None
        try:
            if engagement_key is not None and self.get_engagement() is not None:
                self._engagement_requested = True
                return cast(str, getattr(self.get_engagement(), engagement_key))
        except AttributeError:
            pass

        return default

    @property
    def api_key(self) -> Optional[str]:
        warnings.warn(
            "PyQuilConfig#api_key is deprecated, unused by Rigetti services, and will "
            "be removed in a future release.",
            category=FutureWarning,
        )
        return self._env_or_config_or_default(**self.API_KEY)

    @property
    def dispatch_url(self) -> Optional[str]:
        return self._env_or_config_or_default(**self.DISPATCH_URL)

    @property
    def engage_cmd(self) -> Optional[str]:
        return self._env_or_config_or_default(**self.ENGAGE_CMD)

    @property
    def engagement(self) -> Optional["Engagement"]:
        """
        An Engagement should only be made available to consumers if it was used to retrieve a
        configuration value.
        """
        if not self._engagement_requested:
            return None
        assert self.get_engagement is not None
        return self.get_engagement()

    @property
    def forest_url(self) -> Optional[str]:
        return self._env_or_config_or_default(**self.FOREST_URL)

    @property
    def qmi_auth_token_path(self) -> str:
        value = self._env_or_config_or_default(**self.QMI_AUTH_TOKEN_PATH)
        assert value is not None
        return path.expanduser(value)

    @property
    def qcs_auth_headers(self) -> Dict[str, str]:
        if self.user_auth_token is not None:
            return {"Authorization": f'Bearer {self.user_auth_token["access_token"]}'}
        if self.qmi_auth_token is not None:
            return {"X-QMI-AUTH-TOKEN": self.qmi_auth_token["access_token"]}
        return {}

    @property
    def qcs_url(self) -> Optional[str]:
        return self._env_or_config_or_default(**self.QCS_URL)

    @property
    def qpu_compiler_url(self) -> Optional[str]:
        return self._env_or_config_or_default(**self.QPU_COMPILER_URL)

    @property
    def qpu_url(self) -> Optional[str]:
        return self._env_or_config_or_default(**self.QPU_URL)

    @property
    def quilc_url(self) -> Optional[str]:
        return self._env_or_config_or_default(**self.QUILC_URL)

    @property
    def qvm_url(self) -> Optional[str]:
        return self._env_or_config_or_default(**self.QVM_URL)

    def update_user_auth_token(self, user_auth_token: Dict[str, str]) -> None:
        self.user_auth_token = user_auth_token
        with open(self.user_auth_token_path, "w") as f:
            json.dump(user_auth_token, f)

    def update_qmi_auth_token(self, qmi_auth_token: Dict[str, str]) -> None:
        self.qmi_auth_token = qmi_auth_token
        with open(self.qmi_auth_token_path, "w") as f:
            json.dump(qmi_auth_token, f)

    @property
    def user_auth_token_path(self) -> str:
        value = self._env_or_config_or_default(**self.USER_AUTH_TOKEN_PATH)
        assert value is not None
        return path.expanduser(value)

    @property
    def user_id(self) -> Optional[str]:
        return self._env_or_config_or_default(**self.USER_ID)

    def assert_valid_auth_credential(self) -> None:
        """
        assert_valid_auth_credential will check to make sure the user has a valid
        auth credential configured. This assertion is made lazily - it is called
        only after the user has received a 401 or 403 from forest server. See
        _base_connection.py::ForestSession#_refresh_auth_token.
        """
        if self.user_auth_token is None and self.qmi_auth_token is None:
            raise UserMessageError(
                f"Your configuration does not have valid authentication credentials."
                f"Please visit {self.qcs_url}/auth/token to download credentials"
                f"and save to {self.user_auth_token_path}."
            )


def _parse_auth_token(path: str, required_keys: Iterable[str]) -> Optional[Dict[str, str]]:
    try:
        with open(abspath(expanduser(path)), "r") as f:
            token: Dict[str, str] = json.load(f)
            invalid_values = [k for k in required_keys if not isinstance(token.get(k), str)]
            if len(invalid_values) == 0:
                return token
            logger.warning(f"Failed to parse auth token at {path}.")
            logger.warning(f"Invalid {invalid_values}.")
    except json.decoder.JSONDecodeError:
        logger.warning(f"Failed to parse auth token at {path}. Invalid JSON.")
    except FileNotFoundError:
        logger.debug("Auth token at %s not found.", path)
    return None

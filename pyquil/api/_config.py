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
"""
Module for reading configuration information about api keys and user ids.
"""
from configparser import ConfigParser, NoSectionError, NoOptionError
from os.path import expanduser, abspath
from os import environ, path
from pyquil.api._auth import AuthClient

# `.qcs_config` is for content (mostly) related to QCS: the QCS front end stacks endpoint (`url`)
# for querying all QCS data: devices, reservations, etc. etc., and the `exec_on_engage` that the
# daemon will call during an engage event
QCS_CONFIG = "QCS_CONFIG"
# `.forest_config`, for content related to the Forest SDK, such as ip addresses for the various
# servers to which users submit quil & jobs (qvm, compiler, qpu, etc.)
FOREST_CONFIG = "FOREST_CONFIG"
CONFIG_PATHS = {
    "QCS_CONFIG": "~/.qcs_config",
    "FOREST_CONFIG": "~/.forest_config"
}


class PyquilConfig(object):
    FOREST_URL = {
        "env": "FOREST_SERVER_URL",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "url",
        "engagement_key": None,
        "default": "https://forest-server.qcs.rigetti.com"
    }

    # The path at which the current user's auth token is stored
    AUTH_TOKEN_PATH = {
        "env": "FOREST_AUTH_TOKEN_PATH",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "auth_token_path",
        "engagement_key": None,
        "default": path.expanduser("~/.qcs/user_auth_token")
    }

    # The endpoint to send QPU engagement requests
    DISPATCH_URL = {
        "env": "FOREST_DISPATCH_URL",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "dispatch_url",
        "engagement_key": None,
        "default": "https://dispatch.qcs.rigetti.com/graphql/"
    }

    # The url to the website, for use with OAuth redirect
    QCS_UI_URL = {
        "env": "QCS_UI_URL",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "qcs_ui_url",
        "engagement_key": None,
        "default": "https://qcs.rigetti.com"
    }

    API_KEY = {
        "env": "FOREST_API_KEY",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "key",
        "engagement_key": None,
        "default": None
    }

    USER_ID = {
        "env": "FOREST_USER_ID",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "user_id",
        "engagement_key": None,
        "default": None
    }

    ENGAGE_CMD = {
        "env": "QMI_ENGAGE_CMD",
        "file": QCS_CONFIG,
        "section": "QPU",
        "name": "exec_on_engage",
        "engagement_key": None,
        "default": ""
    }

    QPU_URL = {
        "env": "QPU_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "qpu_endpoint_address",
        "engagement_key": 'qpu_endpoint',
        "default": None
    }

    QVM_URL = {
        "env": "QVM_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "qvm_address",
        "engagement_key": None,
        "default": "http://127.0.0.1:5000"
    }

    QUILC_URL = {
        "env": "QUILC_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "quilc_address",
        "engagement_key": None,
        "default": "tcp://127.0.0.1:5555"
    }

    QPU_COMPILER_URL = {
        "env": "QPU_COMPILER_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "qpu_compiler_address",
        "engagement_key": 'qpu_compiler_endpoint',
        "default": None
    }

    def __init__(self):
        self._attempt_engagement = False
        self._engagement = None
        self._lattice_name = None

        self.configparsers = {}
        for env_name, default_path in CONFIG_PATHS.items():
            default_path = expanduser(default_path)
            path = environ.get(env_name, default_path)

            cp = ConfigParser()
            cp.read(abspath(path))
            self.configparsers[env_name] = cp

    def set_lattice(self, name):
        if name != self.lattice_name:
            self._engagement = None
        self._lattice_name = name

    @property
    def lattice_name(self):
        return self._lattice_name

    def _env_or_config_or_default(self,
                                  env=None,
                                  file=None,
                                  section=None,
                                  name=None,
                                  engagement_key=None,
                                  default=None):
        """
        Get the value of the environment variable or config file value.
        The environment variable takes precedence.

        :param env: The environment variable name.
        :param name: The config file key.
        :param engagement: The path to read this value from an engagement. 
            If None, then this value is not provided by engagement
        :return: The value or None if not found
        """

        # If it's an envvar, that overrides all other options
        env_val = environ.get(env)
        if env_val is not None:
            return env_val

        # Otherwise, use the values in the config files from disk
        try:
            return self.configparsers[file].get(section, name)
        except (NoSectionError, NoOptionError, KeyError):
            pass

        # If no local configuration is available, certain values are provided
        #   by the dispatch server. 
        try:
            if engagement_key is not None:
                self._attempt_engagement = True
                return getattr(self.engagement, engagement_key)
        except AttributeError:
            pass

        return default

    @property
    def auth_client(self):
        if self._auth_client:
            self.auth_client = AuthClient(config=self)
        return self._auth_client

    @property
    def can_engage(self):
        return self.lattice_name is not None

    @property
    def engagement(self):
        if not self.can_engage or not self._attempt_engagement:
            return

        if not (self._engagement and self._engagement.is_valid()):
            self._engagement = AuthClient(config=self).engage(
                self.lattice_name)

        return self._engagement

    @property
    def api_key(self):
        return self._env_or_config_or_default(**self.API_KEY)

    @property
    def auth_token_path(self):
        return self._env_or_config_or_default(**self.AUTH_TOKEN_PATH)

    @property
    def dispatch_url(self):
        return self._env_or_config_or_default(**self.DISPATCH_URL)

    @property
    def qcs_ui_url(self):
        return self._env_or_config_or_default(**self.QCS_UI_URL)

    @property
    def user_id(self):
        return self._env_or_config_or_default(**self.USER_ID)

    @property
    def forest_url(self):
        return self._env_or_config_or_default(**self.FOREST_URL)

    @property
    def engage_cmd(self):
        return self._env_or_config_or_default(**self.ENGAGE_CMD)

    @property
    def qpu_url(self):
        return self._env_or_config_or_default(**self.QPU_URL)

    @property
    def qvm_url(self):
        return self._env_or_config_or_default(**self.QVM_URL)

    @property
    def quilc_url(self):
        return self._env_or_config_or_default(**self.QUILC_URL)

    @property
    def qpu_compiler_url(self):
        return self._env_or_config_or_default(**self.QPU_COMPILER_URL)

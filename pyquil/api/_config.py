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
from __future__ import print_function
from configparser import ConfigParser, NoSectionError, NoOptionError
from os.path import expanduser, abspath
from os import environ

# `.qcs_config` is for content (mostly) related to QCS: the QCS front end stacks endpoint (`url`)
# for querying all QCS data: devices, reservations, etc. etc., and the `exec_on_engage` that the
# daemon will call during an engage event
QCS_CONFIG = "QCS_CONFIG"
# `.forest_config`, for content related to the Forest SDK, such as ip addresses for the various
# servers to which users submit quil & jobs (qvm, compiler, qpu, etc.)
FOREST_CONFIG = "FOREST_CONFIG"
CONFIG_PATHS = {"QCS_CONFIG": "~/.qcs_config",
                "FOREST_CONFIG": "~/.forest_config"}


class PyquilConfig(object):
    FOREST_URL = {
        "env": "FOREST_SERVER_URL",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "url",
        "default": "https://forest-server.qcs.rigetti.com"
    }

    API_KEY = {
        "env": "FOREST_API_KEY",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "key",
        "default": None
    }

    USER_ID = {
        "env": "FOREST_USER_ID",
        "file": QCS_CONFIG,
        "section": "Rigetti Forest",
        "name": "user_id",
        "default": None
    }

    ENGAGE_CMD = {
        "env": "QMI_ENGAGE_CMD",
        "file": QCS_CONFIG,
        "section": "QPU",
        "name": "exec_on_engage",
        "default": ""
    }

    QPU_URL = {
        "env": "QPU_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "qpu_endpoint_address",
        "default": None
    }

    QVM_URL = {
        "env": "QVM_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "qvm_address",
        "default": "http://127.0.0.1:5000"
    }

    COMPILER_URL = {
        "env": "COMPILER_URL",
        "file": FOREST_CONFIG,
        "section": "Rigetti Forest",
        "name": "compiler_server_address",
        "default": "http://127.0.0.1:6000"
    }

    def __init__(self):
        self.configparsers = {}
        for env_name, default_path in CONFIG_PATHS.items():
            default_path = expanduser(default_path)
            path = environ.get(env_name, default_path)

            cp = ConfigParser()
            cp.read(abspath(path))
            self.configparsers[env_name] = cp

    def _env_or_config_or_default(self, env=None, file=None, section=None, name=None, default=None):
        """
        Get the value of the environment variable or config file value.
        The environment variable takes precedence.

        :param env: The environment variable name.
        :param name: The config file key.
        :return: The value or None if not found
        """
        env_val = environ.get(env)
        if env_val is not None:
            return env_val

        try:
            return self.configparsers[file].get(section, name)
        except (NoSectionError, NoOptionError, KeyError):
            return default

    @property
    def api_key(self):
        return self._env_or_config_or_default(**self.API_KEY)

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
    def compiler_url(self):
        return self._env_or_config_or_default(**self.COMPILER_URL)

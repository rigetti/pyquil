##############################################################################
# Copyright 2016-2017 Rigetti Computing
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
from six.moves.configparser import ConfigParser, NoSectionError, NoOptionError
from os.path import expanduser

from os import getenv

import sys


class PyquilConfig(object):
    DEFAULT_PYQUIL_CONFIG_PATH = expanduser('~/.pyquil_config')
    PYQUIL_CONFIG_PATH = getenv('PYQUIL_CONFIG', DEFAULT_PYQUIL_CONFIG_PATH)

    SECTION = "Rigetti Forest"
    API_KEY = "key"
    USER_ID = "user_id"

    def __init__(self):
        self.configparser = ConfigParser()

        if len(self.configparser.read(self.PYQUIL_CONFIG_PATH)) == 0:
            print("! WARNING:\n"
                  "!   There was an issue finding your pyQuil config file.\n"
                  "!   Have you run the pyquil-config-setup command yet?\n"
                  "! See the getting started guide at https://go.rigetti.com/getting-started",
                  file=sys.stderr)

    @property
    def api_key(self):
        return self._env_or_config('QVM_API_KEY', self.API_KEY)

    @property
    def user_id(self):
        return self._env_or_config('QVM_USER_ID', self.USER_ID)

    def _env_or_config(self, env, name):
        """
        Get the value of the environment variable or config file value.
        The environment variable takes precedence.

        :param env: The environment variable name.
        :param name: The config file key.
        :return: The value or None if not found
        """
        env_val = getenv(env)
        if env_val is not None:
            return env_val

        try:
            return self.configparser.get(self.SECTION, name)
        except (NoSectionError, NoOptionError, KeyError):
            return None

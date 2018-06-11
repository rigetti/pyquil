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

import re
import os

from six.moves import input

from pyquil.api._config import PyquilConfig


def main():
    print("Welcome to PyQuil!")
    print("Enter the required information below for Forest connections.")
    print("If you haven't signed up yet you will need to do so first at http://forest.rigetti.com")

    while True:
        key = input("Forest API Key: ")
        key_ma = re.match(r'^\s*(\w{40})\s*$', key)
        if key_ma:
            # Looks like a real key
            key = key_ma.group(1)
            break

        print("That doesn't look like a valid API key. Try again or use Ctrl-C to quit")

    while True:
        user = input("User ID: ")
        user_ma = re.match(r'^\s*([a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-'
                           r'[a-fA-F0-9]{4}-[a-fA-F0-9]{12})\s*$', user)
        if user_ma:
            # Looks like a real user id
            user = user_ma.group(1)
            break

        print("That doesn't look like a valid User ID. Try again or use Ctrl-C to quit")

    path = PyquilConfig.DEFAULT_PYQUIL_CONFIG_PATH

    if os.path.exists(path):
        # Make a backup by appending .bak(i) where we increment
        # i until the desired backup filename doesn't already exist
        i = 1
        dn = os.path.dirname(path)
        bn = os.path.basename(path)
        while True:
            bak_path = os.path.join(dn, "{bn}.bak{i}".format(bn=bn, i=i))
            if not os.path.exists(bak_path):
                break
            i += 1

        print("I already found a file at {path}. Creating a backup at {bak_path}."
              .format(path=path, bak_path=bak_path))

        os.rename(path, bak_path)

    with open(path, 'w') as f:
        f.write("[" + PyquilConfig.SECTION + "]\n")
        f.write(PyquilConfig.API_KEY + ": " + key + "\n")
        f.write(PyquilConfig.USER_ID + ": " + user + "\n")

    print("Pyquil config file created at '%s'" % path)
    print("If you experience any problems see the guide at https://go.rigetti.com/getting-started")

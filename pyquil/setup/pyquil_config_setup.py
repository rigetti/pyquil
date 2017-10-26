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

from six.moves import input

from pyquil.config import PyquilConfig


def main():
    print("Welcome to PyQuil!")
    print("Enter the required information below for Forest connections.")
    print("If you haven't signed up yet you will need to do so first at https://forest.rigetti.com")

    key = input("Forest API Key: ")
    user = input("User ID: ")

    path = PyquilConfig.DEFAULT_PYQUIL_CONFIG_PATH
    with open(path, 'w') as f:
        f.write("[" + PyquilConfig.SECTION + "]\n")
        f.write(PyquilConfig.API_KEY + ": " + key + "\n")
        f.write(PyquilConfig.USER_ID + ": " + user + "\n")

    print("Pyquil config file created at '%s'" % path)
    print("If you experience any problems see the guide at https://go.rigetti.com/getting-started")

#!/usr/bin/python
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

from os.path import expanduser

def main():
    print("Welcome to the pyquil config setup!")
    print("Enter the required information below for Forest connections.")

    # Input for python2 and python3
    try:
       input = raw_input
    except NameError:
       pass

    endpoint = input("Forest URL (https://api.rigetti.com/qvm): ")
    if len(endpoint) == 0:
        endpoint = "https://api.rigetti.com/qvm"
    key = input("Forest API Key: ")
    user = input("User ID: ")

    path = expanduser("~/.pyquil_config")
    with open(path, 'a+') as f:
        f.seek(0)
        f.truncate()
        f.write("[Rigetti Forest]\n")
        f.write("url: %s\n" % endpoint)
        f.write("key: %s\n" % key)
        f.write("user id: %s" % user)

    print("File created at '%s'" % path)
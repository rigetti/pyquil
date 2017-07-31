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
file = open(path, 'a+')
contents = "[Rigetti Forest]\n"
contents += "url: %s\n" % endpoint
contents += "key: %s\n" % key
contents += "user id: %s" % user
file.truncate()
file.write(contents)

print("File created at '%s' with the following contents:\n" % path)
print(contents)

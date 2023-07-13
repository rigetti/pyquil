##############################################################################
# Copyright 2018 Rigetti Computing
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
from importlib.metadata import version
from packaging.version import parse

pyquil_version = version(__package__)
pyquil_docs_version = parse(pyquil_version).base_version or "stable"

DOCS_URL = f"https://pyquil-docs.rigetti.com/en/{pyquil_docs_version}"
"""
The URL of the hosted docs for this package version.
"""

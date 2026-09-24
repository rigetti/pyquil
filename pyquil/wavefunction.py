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
"""Module containing the Wavefunction object and methods for working with wavefunctions.

.. deprecated:: 4.22.0
    To be removed in pyQuil v5 in favor of the Quax-based simulators. Those simulators are
    available in pyQuil v4 (pyquil.simulation._simulator), but are private, experimental and subject
    to change, so we recommend continuing to use this API until you upgrade to pyQuil v5.
"""

# The implementation lives in a private module, which pyQuil imports internally, so that only user
# imports of this module warn.
import warnings

from pyquil._deprecation import DEPRECATED_IN_VERSION, SIMULATOR_REASON, PyQuilDeprecationWarning
from pyquil._wavefunction import (
    OCTETS_PER_COMPLEX_DOUBLE,
    OCTETS_PER_DOUBLE_FLOAT,
    Wavefunction,
    _octet_bits,  # noqa: F401 -- private, but tests and existing code import it from here
    get_bitstring_from_index,
)

__all__ = ["OCTETS_PER_COMPLEX_DOUBLE", "OCTETS_PER_DOUBLE_FLOAT", "Wavefunction", "get_bitstring_from_index"]

warnings.warn(
    f"The module {__name__} is deprecated. ({SIMULATOR_REASON}) -- Deprecated since version {DEPRECATED_IN_VERSION}.",
    PyQuilDeprecationWarning,
    stacklevel=2,
)

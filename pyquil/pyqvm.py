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
"""A pure Python implementation of the Quantum Virtual Machine (QVM).

.. deprecated:: PENDING_DEPRECATION_RELEASE
    To be removed in pyQuil v5 in favor of the Quax-based simulators. They are already in pyQuil v4,
    but private and experimental (see the :ref:`simulation architecture documentation
    <simulation_architecture>`), so we recommend continuing to use this API until you upgrade to
    pyQuil v5.
"""

# The implementation lives in a private module, which pyQuil imports internally, so that only user
# imports of this module warn.
import warnings

from pyquil._deprecation import PENDING_DEPRECATION_RELEASE, QUAX_REPLACEMENT_NOTE, PyQuilDeprecationWarning
from pyquil._pyqvm import QUIL_TO_NUMPY_DTYPE, AbstractQuantumSimulator, PyQVM

__all__ = ["AbstractQuantumSimulator", "PyQVM", "QUIL_TO_NUMPY_DTYPE"]

warnings.warn(
    f"The module {__name__} is deprecated and will be removed in pyQuil v5, where TrajectorySimulator "
    f"replaces PyQVM. {QUAX_REPLACEMENT_NOTE} -- Deprecated since version {PENDING_DEPRECATION_RELEASE}.",
    PyQuilDeprecationWarning,
    stacklevel=2,
)

##############################################################################
# Copyright 2016-2019 Rigetti Computing
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

import warnings

from pyquil.simulation import (  # noqa: F401
    ReferenceDensitySimulator,
    ReferenceWavefunctionSimulator,
    zero_state_matrix,
)

warnings.warn(
    "The code in pyquil.reference_simulator has been moved to pyquil.simulation, "
    "please update your import statements.",
    FutureWarning,
)

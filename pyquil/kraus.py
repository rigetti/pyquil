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
"""
Module for creating and verifying noisy gate definitions in terms of Kraus maps.
"""
import warnings

warnings.warn("pyquil.kraus is deprecated, please use pyquil.noise instead.", DeprecationWarning)


# for backwards compatibility
# noinspection PyUnresolvedReferences
from pyquil.noise import (
    append_kraus_to_gate,
    damping_kraus_map,
    dephasing_kraus_map,
    tensor_kraus_maps,
    combine_kraus_maps,
    damping_after_dephasing,
    add_noise_to_program)



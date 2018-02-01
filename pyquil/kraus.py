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

# for backwards compatibility
# noinspection PyUnresolvedReferences
from pyquil.noise import (
    append_kraus_to_gate,
    damping_kraus_map,
    dephasing_kraus_map,
    tensor_kraus_maps,
    combine_kraus_maps,
    damping_after_dephasing, add_decoherence_noise)

warnings.warn("pyquil.kraus is deprecated, please use pyquil.noise instead.", DeprecationWarning)


def add_noise_to_program(prog, T1=30e-6, T2=30e-6, gate_time_1q=50e-9, gate_time_2q=150e-09,
                         ro_fidelity=0.95):
    """
    Add generic damping and dephasing noise to a program.

    .. warning::

        This function is deprecated. Please use :py:func:`add_decoherence_noise` instead.

    :param prog: A pyquil program consisting of I, RZ, CZ, and RX(+-pi/2) instructions
    :param Union[Dict[int,float],float] T1: The T1 amplitude damping time either globally or in a
        dictionary indexed by qubit id. By default, this is 30 us.
    :param Union[Dict[int,float],float] T2: The T2 dephasing time either globally or in a
        dictionary indexed by qubit id. By default, this is also 30 us.
    :param float gate_time_1q: The duration of the one-qubit gates, namely RX(+pi/2) and RX(-pi/2).
        By default, this is 50 ns.
    :param float gate_time_2q: The duration of the two-qubit gates, namely CZ.
        By default, this is 150 ns.
    :param Union[Dict[int,float],float] ro_fidelity: The readout assignment fidelity
        :math:`F = (p(0|0) + p(1|1))/2` either globally or in a dictionary indexed by qubit id.
    :return: A new program with noisy operators.
    """
    warnings.warn("pyquil.kraus.add_noise_to_program is deprecated, please use "
                  "pyquil.noise.add_decoherence_noise instead.",
                  DeprecationWarning)
    return add_decoherence_noise(prog, T1=T1, T2=T2, gate_time_1q=gate_time_1q,
                                 gate_time_2q=gate_time_2q, ro_fidelity=ro_fidelity)

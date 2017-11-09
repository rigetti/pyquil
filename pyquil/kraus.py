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
"""
Module for creating and verifying noisy gate definitions in terms of Kraus maps.
"""
import numpy as np

from pyquil.quilbase import format_parameter, Pragma


def _check_kraus_ops(n, kraus_ops):
    """
    Verify that the Kraus operators are of the correct shape and satisfy the correct normalization.

    :param int n: Number of qubits
    :param list|tuple kraus_ops: The Kraus operators as numpy.ndarrays.
    """
    for k in kraus_ops:
        if not np.shape(k) == (2 ** n, 2 ** n):
            raise ValueError(
                "Kraus operators for {0} qubits must have shape {1}x{1}: {2}".format(n, 2 ** n, k))

    kdk_sum = sum(np.transpose(k).conjugate().dot(k) for k in kraus_ops)
    if not np.allclose(kdk_sum, np.eye(2**n), atol=1e-5):
        raise ValueError(
            "Kraus operator not correctly normalized: sum_j K_j^*K_j == {}".format(kdk_sum))


def _create_kraus_pragmas(name, qubit_indices, kraus_ops):
    """
    Generate the pragmas to define a Kraus map for a specific gate on some qubits.

    :param str name: The name of the gate.
    :param list|tuple qubit_indices: The qubits
    :param list|tuple kraus_ops: The Kraus operators as matrices.
    :return: A QUIL string with PRAGMA ADD-KRAUS ... statements.
    :rtype: str
    """

    pragmas = [Pragma("ADD-KRAUS",
                      [name] + list(qubit_indices),
                      "({})".format(" ".join(map(format_parameter, np.ravel(k)))))
               for k in kraus_ops]
    return pragmas

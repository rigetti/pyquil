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
Module containing the Wavefunction object and methods for working with wavefunctions.
"""
import itertools
import struct
import warnings
from typing import Dict, Iterator, List, Optional, Sequence, cast

import numpy as np

OCTETS_PER_DOUBLE_FLOAT = 8
OCTETS_PER_COMPLEX_DOUBLE = 2 * OCTETS_PER_DOUBLE_FLOAT


class Wavefunction(object):
    """
    Encapsulate a wavefunction representing a quantum state
    as returned by :py:class:`~pyquil.api.WavefunctionSimulator`.

    .. note::

        The elements of the wavefunction are ordered by bitstring. E.g., for two qubits the order
        is ``00, 01, 10, 11``, where the the bits **are ordered in reverse** by the qubit index,
        i.e., for qubits 0 and 1 the bitstring ``01`` indicates that qubit 0 is in the state 1.
        See also :ref:`the related documentation section in the WavefunctionSimulator Overview
        <basis_ordering>`.
    """

    def __init__(self, amplitude_vector: np.ndarray):
        """
        Initializes a wavefunction

        :param amplitude_vector: A numpy array of complex amplitudes
        """
        if len(amplitude_vector) == 0 or len(amplitude_vector) & (len(amplitude_vector) - 1) != 0:
            raise TypeError("Amplitude vector must have a length that is a power of two")

        self.amplitudes = np.asarray(amplitude_vector)
        sumprob = np.sum(self.probabilities())
        if not np.isclose(sumprob, 1.0):
            raise ValueError(
                "The wavefunction is not normalized. "
                "The probabilities sum to {} instead of 1".format(sumprob)
            )

    @staticmethod
    def ground(qubit_num: int) -> "Wavefunction":
        warnings.warn("ground() has been deprecated in favor of zeros()", stacklevel=2)
        return Wavefunction.zeros(qubit_num)

    @staticmethod
    def zeros(qubit_num: int) -> "Wavefunction":
        """
        Constructs the groundstate wavefunction for a given number of qubits.

        :param qubit_num:
        :return: A Wavefunction in the ground state
        """
        amplitude_vector = np.zeros(2 ** qubit_num)
        amplitude_vector[0] = 1.0
        return Wavefunction(amplitude_vector)

    @staticmethod
    def from_bit_packed_string(coef_string: bytes) -> "Wavefunction":
        """
        From a bit packed string, unpacks to get the wavefunction
        :param coef_string:
        """
        num_octets = len(coef_string)

        # Parse the wavefunction
        wf = np.zeros(num_octets // OCTETS_PER_COMPLEX_DOUBLE, dtype=np.cfloat)
        for i, p in enumerate(range(0, num_octets, OCTETS_PER_COMPLEX_DOUBLE)):
            re_be = coef_string[p : p + OCTETS_PER_DOUBLE_FLOAT]
            im_be = coef_string[p + OCTETS_PER_DOUBLE_FLOAT : p + OCTETS_PER_COMPLEX_DOUBLE]
            re = struct.unpack(">d", re_be)[0]
            im = struct.unpack(">d", im_be)[0]
            wf[i] = complex(re, im)

        return Wavefunction(wf)

    def __len__(self) -> int:
        return len(self.amplitudes).bit_length() - 1

    def __iter__(self) -> Iterator[complex]:
        return cast(Iterator[complex], self.amplitudes.__iter__())

    def __getitem__(self, index: int) -> complex:
        return cast(complex, self.amplitudes[index])

    def __setitem__(self, key: int, value: complex) -> None:
        self.amplitudes[key] = value

    def __str__(self) -> str:
        return self.pretty_print(decimal_digits=10)

    def probabilities(self) -> np.ndarray:
        """Returns an array of probabilities in lexicographical order"""
        return np.abs(self.amplitudes) ** 2

    def get_outcome_probs(self) -> Dict[str, float]:
        """
        Parses a wavefunction (array of complex amplitudes) and returns a dictionary of
        outcomes and associated probabilities.

        :return: A dict with outcomes as keys and probabilities as values.
        :rtype: dict
        """
        outcome_dict = {}
        qubit_num = len(self)
        for index, amplitude in enumerate(self.amplitudes):
            outcome = get_bitstring_from_index(index, qubit_num)
            outcome_dict[outcome] = abs(amplitude) ** 2
        return outcome_dict

    def pretty_print_probabilities(self, decimal_digits: int = 2) -> Dict[str, float]:
        """
        TODO: This doesn't seem like it is named correctly...

        Prints outcome probabilities, ignoring all outcomes with approximately zero probabilities
        (up to a certain number of decimal digits) and rounding the probabilities to decimal_digits.

        :param int decimal_digits: The number of digits to truncate to.
        :return: A dict with outcomes as keys and probabilities as values.
        """
        outcome_dict = {}
        qubit_num = len(self)
        for index, amplitude in enumerate(self.amplitudes):
            outcome = get_bitstring_from_index(index, qubit_num)
            prob = round(abs(amplitude) ** 2, decimal_digits)
            if prob != 0.0:
                outcome_dict[outcome] = prob
        return outcome_dict

    def pretty_print(self, decimal_digits: int = 2) -> str:
        """
        Returns a string repr of the wavefunction, ignoring all outcomes with approximately zero
        amplitude (up to a certain number of decimal digits) and rounding the amplitudes to
        decimal_digits.

        :param int decimal_digits: The number of digits to truncate to.
        :return: A string representation of the wavefunction.
        """
        outcome_dict = {}
        qubit_num = len(self)
        pp_string = ""
        for index, amplitude in enumerate(self.amplitudes):
            outcome = get_bitstring_from_index(index, qubit_num)
            amplitude = (
                round(amplitude.real, decimal_digits) + round(amplitude.imag, decimal_digits) * 1.0j
            )
            if amplitude != 0.0:
                outcome_dict[outcome] = amplitude
                pp_string += str(amplitude) + "|{}> + ".format(outcome)
        if len(pp_string) >= 3:
            pp_string = pp_string[:-3]  # remove the dangling + if it is there
        return pp_string

    def plot(self, qubit_subset: Optional[Sequence[int]] = None) -> None:
        """
        TODO: calling this will error because of matplotlib

        Plots a bar chart with bitstring on the x axis and probability on the y axis.

        :param qubit_subset: Optional parameter used for plotting a subset of the Hilbert space.
        """
        import matplotlib.pyplot as plt

        prob_dict = self.get_outcome_probs()
        if qubit_subset:
            sub_dict = {}
            qubit_num = len(self)
            for i in qubit_subset:
                if i > (2 ** qubit_num - 1):
                    raise IndexError("Index {} too large for {} qubits.".format(i, qubit_num))
                else:
                    sub_dict[get_bitstring_from_index(i, qubit_num)] = prob_dict[
                        get_bitstring_from_index(i, qubit_num)
                    ]
            prob_dict = sub_dict
        plt.bar(range(len(prob_dict)), prob_dict.values(), align="center", color="#6CAFB7")
        plt.xticks(range(len(prob_dict)), prob_dict.keys())
        plt.show()

    def sample_bitstrings(self, n_samples: int) -> np.ndarray:
        """
        Sample bitstrings from the distribution defined by the wavefunction.

        :param n_samples: The number of bitstrings to sample
        :return: An array of shape (n_samples, n_qubits)
        """
        possible_bitstrings = np.array(list(itertools.product((0, 1), repeat=len(self))))
        inds = np.random.choice(2 ** len(self), n_samples, p=self.probabilities())
        bitstrings = possible_bitstrings[inds, :]
        return bitstrings


def get_bitstring_from_index(index: int, qubit_num: int) -> str:
    """
    Returns the bitstring in lexical order that corresponds to the given index in 0 to 2^(qubit_num)
    :param int index:
    :param int qubit_num:
    :return: the bitstring
    :rtype: str
    """
    if index > (2 ** qubit_num - 1):
        raise IndexError("Index {} too large for {} qubits.".format(index, qubit_num))
    return bin(index)[2:].rjust(qubit_num, "0")


def _octet_bits(o: int) -> List[int]:
    """
    Get the bits of an octet.

    :param o: The octets.
    :return: The bits as a list in LSB-to-MSB order.
    """
    if not isinstance(o, int):
        raise TypeError("o should be an int")
    if not (0 <= o <= 255):
        raise ValueError("o should be between 0 and 255 inclusive")
    bits = [0] * 8
    for i in range(8):
        if 1 == o & 1:
            bits[i] = 1
        o = o >> 1
    return bits

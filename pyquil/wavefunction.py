"""
Module containing the Wavefunction object and methods for working with wavefunctions.
"""
import struct
import warnings

import numpy as np
from six import integer_types

OCTETS_PER_DOUBLE_FLOAT = 8
OCTETS_PER_COMPLEX_DOUBLE = 2 * OCTETS_PER_DOUBLE_FLOAT


class Wavefunction(object):

    def __init__(self, amplitude_vector, classical_memory=None):
        """
        Initializes a wavefunction
        :param amplitude_vector: A numpy array of complex amplitudes
        :param classical_memory: Any classical memory associated with this wavefunction result
        """
        if len(amplitude_vector) == 0 or len(amplitude_vector) & (len(amplitude_vector) - 1) != 0:
            raise TypeError("Amplitude vector must have a length that is a power of two")

        self.amplitudes = amplitude_vector
        self.classical_memory = classical_memory

    @staticmethod
    def ground(qubit_num):
        warnings.warn("ground() has been deprecated in favor of zeros()", stacklevel=2)
        return Wavefunction.zeros(qubit_num)

    @staticmethod
    def zeros(qubit_num):
        """
        Constructs the groundstate wavefunction for a given number of qubits.

        :param int qubit_num:
        :return: A Wavefunction in the ground state
        :rtype: Wavefunction
        """
        container = [0] * (2**qubit_num)
        container[0] = 1.0
        return Wavefunction(container)

    @staticmethod
    def from_bit_packed_string(coef_string, classical_addresses):
        """
        From a bit packed string, unpacks to get the wavefunction and classical measurement results
        :param bytes coef_string:
        :param list classical_addresses:
        :return:
        """
        num_octets = len(coef_string)
        num_addresses = len(classical_addresses)
        num_memory_octets = _round_to_next_multiple(num_addresses, 8) // 8
        num_wavefunction_octets = num_octets - num_memory_octets

        # Parse the classical memory
        mem = []
        for i in range(num_memory_octets):
            octet = struct.unpack('B', coef_string[i:i + 1])[0]
            mem.extend(_octet_bits(octet))

        mem = mem[0:num_addresses]

        # Parse the wavefunction
        wf = np.zeros(num_wavefunction_octets // OCTETS_PER_COMPLEX_DOUBLE, dtype=np.cfloat)
        for i, p in enumerate(range(num_memory_octets, num_octets, OCTETS_PER_COMPLEX_DOUBLE)):
            re_be = coef_string[p: p + OCTETS_PER_DOUBLE_FLOAT]
            im_be = coef_string[p + OCTETS_PER_DOUBLE_FLOAT: p + OCTETS_PER_COMPLEX_DOUBLE]
            re = struct.unpack('>d', re_be)[0]
            im = struct.unpack('>d', im_be)[0]
            wf[i] = complex(re, im)

        return Wavefunction(wf, mem)

    def __len__(self):
        return len(self.amplitudes).bit_length() - 1

    def __iter__(self):
        warnings.warn("""
Previously, qvm.wavefunction returned both classical memory and wavefunction
as a pair. Now it just returns a Wavefunction object.
You likely need to change this:
    wf, _ = qvm.wavefunction(program, ...)
To just this:
    wf = qvm.wavefunction(program, ...)\n""", stacklevel=2)
        return self.amplitudes.__iter__()

    def __getitem__(self, index):
        warnings.warn("""
Previously, qvm.wavefunction returned both classical memory and wavefunction
as a pair. Now it just returns a Wavefunction object.
You likely need to change this:
    wf, _ = qvm.wavefunction(program, ...)
To just this:
    wf = qvm.wavefunction(program, ...)\n""", stacklevel=2)
        return self.amplitudes[index]

    def __setitem__(self, key, value):
        self.amplitudes[key] = value

    def __str__(self):
        return self.pretty_print(decimal_digits=10)

    def get_outcome_probs(self):
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

    def pretty_print_probabilities(self, decimal_digits=2):
        """
        Prints outcome probabilities, ignoring all outcomes with approximately zero probabilities
        (up to a certain number of decimal digits) and rounding the probabilities to decimal_digits.

        :param int decimal_digits: The number of digits to truncate to.
        :return: A dict with outcomes as keys and probabilities as values.
        :rtype: dict
        """
        outcome_dict = {}
        qubit_num = len(self)
        for index, amplitude in enumerate(self.amplitudes):
            outcome = get_bitstring_from_index(index, qubit_num)
            prob = round(abs(amplitude) ** 2, decimal_digits)
            if prob != 0.:
                outcome_dict[outcome] = prob
        return outcome_dict

    def pretty_print(self, decimal_digits=2):
        """
        Returns a string repr of the wavefunction, ignoring all outcomes with approximately zero
        amplitude (up to a certain number of decimal digits) and rounding the amplitudes to
        decimal_digits.

        :param int decimal_digits: The number of digits to truncate to.
        :return: A dict with outcomes as keys and complex amplitudes as values.
        :rtype: str
        """
        outcome_dict = {}
        qubit_num = len(self)
        pp_string = ""
        for index, amplitude in enumerate(self.amplitudes):
            outcome = get_bitstring_from_index(index, qubit_num)
            amplitude = round(amplitude.real, decimal_digits) + \
                round(amplitude.imag, decimal_digits) * 1.j
            if amplitude != 0.:
                outcome_dict[outcome] = amplitude
                pp_string += str(amplitude) + "|{}> + ".format(outcome)
        if len(pp_string) >= 3:
            pp_string = pp_string[:-3]  # remove the dangling + if it is there
        return pp_string

    def plot(self, qubit_subset=None):
        """
        Plots a bar chart with bitstring on the x axis and probability on the y axis.

        :param list qubit_subset: Optional parameter used for plotting a subset of the Hilbert space.
        """
        import matplotlib.pyplot as plt
        prob_dict = self.get_outcome_probs()
        if qubit_subset:
            sub_dict = {}
            qubit_num = len(self)
            for i in qubit_subset:
                if i > (2**qubit_num - 1):
                    raise IndexError("Index {} too large for {} qubits.".format(i, qubit_num))
                else:
                    sub_dict[get_bitstring_from_index(i, qubit_num)] = prob_dict[get_bitstring_from_index(i, qubit_num)]
            prob_dict = sub_dict
        plt.bar(range(len(prob_dict)), prob_dict.values(), align='center', color='#6CAFB7')
        plt.xticks(range(len(prob_dict)), prob_dict.keys())
        plt.show()


def get_bitstring_from_index(index, qubit_num):
    """
    Returns the bitstring in lexical order that corresponds to the given index in 0 to 2^(qubit_num)
    :param int index:
    :param int qubit_num:
    :return: the bitstring
    :rtype: str
    """
    if index > (2**qubit_num - 1):
        raise IndexError("Index {} too large for {} qubits.".format(index, qubit_num))
    return bin(index)[2:].rjust(qubit_num, '0')


def _round_to_next_multiple(n, m):
    """
    Round up the the next multiple.

    :param n: The number to round up.
    :param m: The multiple.
    :return: The rounded number
    """
    return n if n % m == 0 else n + m - n % m


def _octet_bits(o):
    """
    Get the bits of an octet.

    :param o: The octets.
    :return: The bits as a list in LSB-to-MSB order.
    :rtype: list
    """
    if not isinstance(o, integer_types):
        raise TypeError("o should be an int")
    if not (0 <= o <= 255):
        raise ValueError("o should be between 0 and 255 inclusive")
    bits = [0] * 8
    for i in range(8):
        if 1 == o & 1:
            bits[i] = 1
        o = o >> 1
    return bits

"""
Module containing the Wavefunction object and methods for working with wavefunctions.
"""


class Wavefunction(object):

    def __init__(self, amplitude_vector):
        """
        Initializes a wavefunction
        :param amplitude_vector: A numpy array of complex amplitudes
        """
        if len(amplitude_vector) == 0 or len(amplitude_vector) & (len(amplitude_vector) - 1) != 0:
            raise TypeError("Amplitude vector must have a length that is a power of two")

        self.amplitudes = amplitude_vector

    def __len__(self):
        return len(self.amplitudes).bit_length() - 1

    def __iter__(self):
        return self.amplitudes.__iter__()

    def __getitem__(self, index):
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
        for index, amplitude in enumerate(self):
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
        for index, amplitude in enumerate(self):
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
        for index, amplitude in enumerate(self):
            outcome = get_bitstring_from_index(index, qubit_num)
            amplitude = round(amplitude.real, decimal_digits) + \
                        round(amplitude.imag, decimal_digits) * 1.j
            if amplitude != 0.:
                outcome_dict[outcome] = amplitude
                pp_string += str(amplitude) + "|{}> + ".format(outcome)
        if len(pp_string) >= 3:
            pp_string = pp_string[:-3]  # remove the dangling + if it is there
        return pp_string

    @classmethod
    def ground(cls, qubit_num):
        """
        Constructs the groundstate wavefunction for a given number of qubits.

        :param int qubit_num:
        :return: A Wavefunction in the ground state
        :rtype: Wavefunction
        """
        container = [0] * (2**qubit_num)
        container[0] = 1.0
        return cls(container)


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

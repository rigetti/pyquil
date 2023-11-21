import numpy as np

from pyquil.gates import RX
from pyquil.quil import Program
from pyquil.quilatom import Pi


class TestPi:
    def test_print(self):
        assert str(Pi()) == "pi"
        expr = Pi() / 2
        print(type(expr))
        assert str(expr) == "pi/2"

    def test_program(self):
        program = Program(RX(Pi(), 0), RX(Pi() / 2, 1))
        assert program.out() == "RX(pi) 0\nRX(pi/2) 1\n"
        assert program[0] == RX(Pi(), 0)
        assert program[1] == RX(Pi() / 2, 1)

    def test_numpy(self):
        pi_class_matrix = np.asmatrix([[Pi(), Pi() / 2], [Pi() / 3, Pi() / 4]])
        pi_float_matrix = np.asmatrix([[np.pi, np.pi / 2], [np.pi / 3, np.pi / 4]])
        assert np.allclose(pi_class_matrix, pi_float_matrix)

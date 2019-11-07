import numpy as np

from pyquil.experiment._result import bitstrings_to_expectations


def test_bitstrings_to_expectations():
    bitstrings = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])

    assert np.allclose(bitstrings_to_expectations(bitstrings),
                       np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]))
    assert np.allclose(bitstrings_to_expectations(bitstrings, correlations=[[0]]),
                       np.array([[1], [1], [-1], [-1]]))
    assert np.allclose(bitstrings_to_expectations(bitstrings, correlations=[[1]]),
                       np.array([[1], [-1], [1], [-1]]))
    assert np.allclose(bitstrings_to_expectations(bitstrings, correlations=[[0, 1]]),
                       np.array([[1], [-1], [-1], [1]]))

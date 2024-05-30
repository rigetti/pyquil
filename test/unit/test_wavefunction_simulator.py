import numpy as np
import pytest

from pyquil import Program
from pyquil.api import QCSClient, WavefunctionSimulator
from pyquil.gates import CNOT, H
from pyquil.paulis import PauliSum, sX, sZ


def test_wavefunction(client_configuration: QCSClient):
    wfnsim = WavefunctionSimulator(client_configuration=client_configuration)
    bell = Program(H(0), CNOT(0, 1))
    wfn = wfnsim.wavefunction(bell)
    np.testing.assert_allclose(wfn.amplitudes, 1 / np.sqrt(2) * np.array([1, 0, 0, 1]))
    np.testing.assert_allclose(wfn.probabilities(), [0.5, 0, 0, 0.5])
    assert wfn.pretty_print() == "(0.71+0j)|00> + (0.71+0j)|11>"

    bitstrings = wfn.sample_bitstrings(1000)
    parity = np.sum(bitstrings, axis=1) % 2
    assert np.all(parity == 0)


def test_random_seed(client_configuration: QCSClient):
    wfnsim = WavefunctionSimulator(client_configuration=client_configuration, random_seed=100)
    assert wfnsim.random_seed == 100

    with pytest.raises(TypeError):
        WavefunctionSimulator(client_configuration=client_configuration, random_seed="NOT AN INTEGER")


def test_noise(client_configuration: QCSClient):
    wfnsim = WavefunctionSimulator(
        client_configuration=client_configuration,
        gate_noise=(0.2, 0.3, 0.5),
        measurement_noise=(0.5, 0.2, 0.3),
    )

    assert wfnsim.gate_noise == (0.2, 0.3, 0.5)
    assert wfnsim.measurement_noise == (0.5, 0.2, 0.3)

    with pytest.raises(TypeError):
        WavefunctionSimulator(client_configuration=client_configuration, gate_noise="NOT A TUPLE")

    with pytest.raises(TypeError):
        WavefunctionSimulator(client_configuration=client_configuration, measurement_noise="NOT A TUPLE")


def test_expectation(client_configuration: QCSClient):
    wfnsim = WavefunctionSimulator(client_configuration=client_configuration)
    bell = Program(H(0), CNOT(0, 1))
    expects = wfnsim.expectation(bell, [sZ(0) * sZ(1), sZ(0), sZ(1), sX(0) * sX(1)])
    assert expects.size == 4
    np.testing.assert_allclose(expects, [1, 0, 0, 1])

    pauli_sum = PauliSum([sZ(0) * sZ(1)])
    expects = wfnsim.expectation(bell, pauli_sum)
    assert expects.size == 1
    np.testing.assert_allclose(expects, [1])


def test_run_and_measure(client_configuration: QCSClient):
    wfnsim = WavefunctionSimulator(client_configuration=client_configuration)
    bell = Program(H(0), CNOT(0, 1))
    bitstrings = wfnsim.run_and_measure(bell, trials=1000)
    parity = np.sum(bitstrings, axis=1) % 2
    assert np.all(parity == 0)


def test_run_and_measure_qubits(client_configuration: QCSClient):
    wfnsim = WavefunctionSimulator(client_configuration=client_configuration)
    bell = Program(H(0), CNOT(0, 1))
    bitstrings = wfnsim.run_and_measure(bell, qubits=[0, 100], trials=1000)
    assert np.all(bitstrings[:, 1] == 0)
    assert 0.4 < np.mean(bitstrings[:, 0]) < 0.6

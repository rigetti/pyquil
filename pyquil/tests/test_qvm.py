import numpy as np
import pytest

from pyquil import Program
from pyquil.api import QVM, Client
from pyquil.api._errors import QVMError
from pyquil.api._qvm import validate_noise_probabilities, validate_qubit_list, prepare_register_list
from pyquil.gates import MEASURE, X
from pyquil.quilbase import Declare, MemoryReference


def test_qvm__default_client(client):
    qvm = QVM(client=client)
    p = Program(Declare("ro", "BIT"), X(0), MEASURE(0, MemoryReference("ro")))
    qvm.load(p.wrap_in_numshots_loop(1000))
    qvm.run()
    qvm.wait()
    bitstrings = qvm.read_memory(region_name="ro")
    assert bitstrings.shape == (1000, 1)


def test_qvm_run_pqer(client: Client):
    qvm = QVM(client=client, gate_noise=(0.01, 0.01, 0.01))
    p = Program(Declare("ro", "BIT"), X(0), MEASURE(0, MemoryReference("ro")))
    qvm.load(p.wrap_in_numshots_loop(1000))
    qvm.run()
    qvm.wait()
    bitstrings = qvm.read_memory(region_name="ro")
    assert bitstrings.shape == (1000, 1)
    assert np.mean(bitstrings) > 0.8


def test_qvm_run_just_program(client: Client):
    qvm = QVM(client=client, gate_noise=(0.01, 0.01, 0.01))
    p = Program(Declare("ro", "BIT"), X(0), MEASURE(0, MemoryReference("ro")))
    qvm.load(p.wrap_in_numshots_loop(1000))
    qvm.run()
    qvm.wait()
    bitstrings = qvm.read_memory(region_name="ro")
    assert bitstrings.shape == (1000, 1)
    assert np.mean(bitstrings) > 0.8


def test_qvm_run_only_pqer(client: Client):
    qvm = QVM(client=client, gate_noise=(0.01, 0.01, 0.01))
    p = Program(Declare("ro", "BIT"), X(0), MEASURE(0, MemoryReference("ro")))

    qvm.load(p.wrap_in_numshots_loop(1000))
    qvm.run()
    qvm.wait()
    bitstrings = qvm.read_memory(region_name="ro")
    assert bitstrings.shape == (1000, 1)
    assert np.mean(bitstrings) > 0.8


def test_qvm_run_region_declared_and_measured(client: Client):
    qvm = QVM(client=client)
    p = Program(Declare("reg", "BIT"), X(0), MEASURE(0, MemoryReference("reg")))
    qvm.load(p.wrap_in_numshots_loop(100)).run().wait()
    bitstrings = qvm.read_memory(region_name="reg")
    assert bitstrings.shape == (100, 1)


def test_qvm_run_region_declared_not_measured(client: Client):
    qvm = QVM(client=client)
    p = Program(Declare("reg", "BIT"), X(0))
    qvm.load(p.wrap_in_numshots_loop(100)).run().wait()
    bitstrings = qvm.read_memory(region_name="reg")
    assert bitstrings.shape == (100, 0)


# For backwards compatibility, we support omitting the declaration for "ro" specifically
# TODO(andrew): we should remove this lenient behavior
def test_qvm_run_region_not_declared_is_measured_ro(client: Client):
    qvm = QVM(client=client)
    p = Program(X(0), MEASURE(0, MemoryReference("ro")))
    qvm.load(p.wrap_in_numshots_loop(100)).run().wait()
    bitstrings = qvm.read_memory(region_name="ro")
    assert bitstrings.shape == (100, 1)


def test_qvm_run_region_not_declared_is_measured_non_ro(client: Client):
    qvm = QVM(client=client)
    p = Program(X(0), MEASURE(0, MemoryReference("reg")))

    with pytest.raises(QVMError, match='Bad memory region name "reg" in MEASURE'):
        qvm.load(p).run().wait()


def test_qvm_run_region_not_declared_not_measured_ro(client: Client):
    qvm = QVM(client=client)
    p = Program(X(0))
    qvm.load(p.wrap_in_numshots_loop(100)).run().wait()
    bitstrings = qvm.read_memory(region_name="ro")
    assert bitstrings.shape == (100, 0)


def test_qvm_run_region_not_declared_not_measured_non_ro(client: Client):
    qvm = QVM(client=client)
    p = Program(X(0))
    qvm.load(p.wrap_in_numshots_loop(100)).run().wait()
    assert qvm.read_memory(region_name="reg") is None


def test_qvm_version(client: Client):
    qvm = QVM(client=client)
    version = qvm.get_version_info()

    def is_a_version_string(version_string: str):
        parts = version_string.split(".")
        try:
            map(int, parts)
        except ValueError:
            return False
        return True

    assert is_a_version_string(version)


def test_validate_noise_probabilities():
    with pytest.raises(TypeError, match="noise_parameter must be a tuple"):
        validate_noise_probabilities(1)
    with pytest.raises(TypeError, match="noise_parameter values should all be floats"):
        validate_noise_probabilities(("a", "b", "c"))
    with pytest.raises(ValueError, match="noise_parameter tuple must be of length 3"):
        validate_noise_probabilities((0.0, 0.0, 0.0, 0.0))
    with pytest.raises(
        ValueError,
        match="sum of entries in noise_parameter must be between 0 and 1 \\(inclusive\\)",
    ):
        validate_noise_probabilities((0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="noise_parameter values should all be non-negative"):
        validate_noise_probabilities((-0.5, -0.5, 1.0))


def test_validate_qubit_list():
    with pytest.raises(TypeError):
        validate_qubit_list([-1, 1])
    with pytest.raises(TypeError):
        validate_qubit_list(["a", 0], 1)


def test_prepare_register_list():
    with pytest.raises(TypeError):
        prepare_register_list({"ro": [-1, 1]})

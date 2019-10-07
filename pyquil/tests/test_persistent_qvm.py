from typing import Dict

import pytest
import numpy as np

from pyquil import Program
from pyquil.api import (ForestConnection, PersistentQVM, QVMSimulationMethod, QVMAllocationMethod,
                        get_qvm_memory_estimate)
from pyquil.api._errors import QVMError
from pyquil.gates import MEASURE, X
from pyquil.tests.utils import is_qvm_version_string


MemoryContentsDict = Dict[str, np.array]


def _check_mem_equal(a: MemoryContentsDict, b: MemoryContentsDict) -> None:
    """Assert that the MemoryContentsDicts a and b are equal

    A MemoryContentsDict is a mapping where the keys are ``str`` and the values are ``np.array``s.
    A MemoryContentsDict corresponds to the data structure returned by ``PersistentQVM.run_program``
    and ``PersistentQVM.read_memory``.

    :param a: a dict of memory contents
    :param b: a dict of memory contents
    """
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_pqvm_version(forest_app_ng: ForestConnection):
    pqvm = PersistentQVM(num_qubits=0, connection=forest_app_ng)
    version = pqvm.get_version_info()
    assert is_qvm_version_string(version)


def test_qvm_memory_estimate(forest_app_ng: ForestConnection):
    def is_valid_mem_estimate(estimate):
        return isinstance(estimate, int) and estimate >= 0

    assert is_valid_mem_estimate(get_qvm_memory_estimate(0, connection=forest_app_ng))
    assert is_valid_mem_estimate(get_qvm_memory_estimate(1, connection=forest_app_ng))
    assert is_valid_mem_estimate(get_qvm_memory_estimate(10, connection=forest_app_ng))
    assert is_valid_mem_estimate(get_qvm_memory_estimate(
        11, simulation_method=QVMSimulationMethod.FULL_DENSITY_MATRIX, connection=forest_app_ng
    ))
    assert is_valid_mem_estimate(get_qvm_memory_estimate(
        12, allocation_method=QVMAllocationMethod.FOREIGN, connection=forest_app_ng
    ))
    assert is_valid_mem_estimate(get_qvm_memory_estimate(
        13, measurement_noise=[1.0, 0.0, 0.0], connection=forest_app_ng
    ))
    assert is_valid_mem_estimate(get_qvm_memory_estimate(
        14, gate_noise=[1.0, 0.0, 0.0], connection=forest_app_ng
    ))

    with pytest.raises(TypeError):
        get_qvm_memory_estimate(-1, connection=forest_app_ng)
    with pytest.raises(TypeError):
        get_qvm_memory_estimate(0, connection=forest_app_ng, simulation_method="pure-state")
    with pytest.raises(TypeError):
        get_qvm_memory_estimate(0, connection=forest_app_ng, allocation_method="native")
    with pytest.raises(ValueError):
        get_qvm_memory_estimate(0, connection=forest_app_ng, measurement_noise=[1.0])
    with pytest.raises(ValueError):
        get_qvm_memory_estimate(0, connection=forest_app_ng, gate_noise=[1.0])


def test_pqvm_create_and_info(forest_app_ng: ForestConnection):
    pqvm = PersistentQVM(num_qubits=0, connection=forest_app_ng)
    info = pqvm.get_qvm_info()
    assert info['qvm-type'] == 'PURE-STATE-QVM'
    assert info['num-qubits'] == 0
    assert info['metadata']['allocation-method'] == 'NATIVE'

    pqvm = PersistentQVM(num_qubits=1, connection=forest_app_ng,
                         allocation_method=QVMAllocationMethod.FOREIGN)
    info = pqvm.get_qvm_info()
    assert info['qvm-type'] == 'PURE-STATE-QVM'
    assert info['num-qubits'] == 1
    assert info['metadata']['allocation-method'] == 'FOREIGN'

    pqvm = PersistentQVM(num_qubits=2, connection=forest_app_ng,
                         simulation_method=QVMSimulationMethod.FULL_DENSITY_MATRIX)
    info = pqvm.get_qvm_info()
    assert info['qvm-type'] == 'DENSITY-QVM'
    assert info['num-qubits'] == 2
    assert info['metadata']['allocation-method'] == 'NATIVE'

    pqvm = PersistentQVM(num_qubits=3, connection=forest_app_ng,
                         simulation_method=QVMSimulationMethod.FULL_DENSITY_MATRIX,
                         allocation_method=QVMAllocationMethod.FOREIGN)
    info = pqvm.get_qvm_info()
    assert info['qvm-type'] == 'DENSITY-QVM'
    assert info['num-qubits'] == 3
    assert info['metadata']['allocation-method'] == 'FOREIGN'


def test_pqvm_read_memory(forest_app_ng: ForestConnection):
    pqvm = PersistentQVM(num_qubits=2, connection=forest_app_ng)
    _check_mem_equal(pqvm.read_memory({}), {})

    # No classical memory has been configured yet.
    with pytest.raises(QVMError):
        pqvm.read_memory({"ro": True})

    pqvm.run_program(Program("DECLARE ro BIT"))
    _check_mem_equal(pqvm.read_memory({"ro": True}), {"ro": [[0]]})

    # The ro register exists, but nothing else.
    with pytest.raises(QVMError):
        pqvm.read_memory({"foo": True})

    # Request memory at a specific offset.
    pqvm.run_program(Program("DECLARE byte BIT[8]\nX 0\nMEASURE 0 byte[4]"))
    _check_mem_equal(pqvm.read_memory({"byte": [4]}), {"byte": [[1]]})


def test_write_memory(forest_app_ng: ForestConnection):
    pqvm = PersistentQVM(num_qubits=2, connection=forest_app_ng)
    p = Program()
    p.declare("ro", "BIT", 2)
    p.declare("theta", "REAL", 4)
    pqvm.run_program(p)

    with pytest.raises(TypeError):
        pqvm.write_memory("ro")

    with pytest.raises(TypeError):
        pqvm.write_memory(("ro", 2))

    with pytest.raises(TypeError):
        pqvm.write_memory({"ro": 1})

    with pytest.raises(TypeError):
        pqvm.write_memory({"ro": "2"})

    with pytest.raises(TypeError):
        pqvm.write_memory({"ro": object()})

    with pytest.raises(TypeError):
        # str is invalid value type
        pqvm.write_memory({"ro": [(1, "hey")]})

    with pytest.raises(TypeError):
        # index cannot be negative
        pqvm.write_memory({"ro": [(-1, 1)]})

    with pytest.raises(ValueError):
        # empty values not allowed
        pqvm.write_memory({"ro": []})

    with pytest.raises(ValueError):
        # value must be 2-tuple
        pqvm.write_memory({"ro": [(1, 2, 3)]})

    _check_mem_equal(pqvm.read_memory({"ro": True}), {"ro": [[0, 0]]})
    _check_mem_equal(pqvm.read_memory({"theta": True}), {"theta": [[0.0, 0.0, 0.0, 0.0]]})

    # sparse (index, value) encoding
    pqvm.write_memory({"ro": [(1, 1)]})
    _check_mem_equal(pqvm.read_memory({"ro": True}), {"ro": [[0, 1]]})

    # range
    pqvm.write_memory({"ro": reversed(range(2))})
    _check_mem_equal(pqvm.read_memory({"ro": True}), {"ro": [[1, 0]]})

    # numpy array
    pqvm.write_memory({"theta": np.arange(0.0, 4.0, 1.0)})
    _check_mem_equal(pqvm.read_memory({"theta": True}), {"theta": [[0.0, 1.0, 2.0, 3.0]]})

    # tuple of values
    pqvm.write_memory({"theta": (4.1, 3.1, 2.1, 1.1)})
    _check_mem_equal(pqvm.read_memory({"theta": True}), {"theta": [[4.1, 3.1, 2.1, 1.1]]})


def test_pqvm_run_program(forest_app_ng: ForestConnection):
    p = Program()
    p.declare('ro')
    p += X(0)
    p += MEASURE(0, "ro")

    for simulation_method in QVMSimulationMethod:
        for allocation_method in QVMAllocationMethod:
            pqvm = PersistentQVM(num_qubits=2, connection=forest_app_ng,
                                 simulation_method=simulation_method,
                                 allocation_method=allocation_method)
            mem = pqvm.run_program(p)
            _check_mem_equal(mem, {'ro': [[1]]})


def test_pqvm_run_program_with_pauli_noise(forest_app_ng: ForestConnection):
    p = Program()
    p.declare('ro')
    p += X(0)
    p += MEASURE(0, "ro")

    pqvm = PersistentQVM(num_qubits=2, connection=forest_app_ng,
                         measurement_noise=[1.0, 0.0, 0.0])
    mem = pqvm.run_program(p)
    assert mem == {'ro': [[0]]}

    pqvm = PersistentQVM(num_qubits=2, connection=forest_app_ng,
                         measurement_noise=[1.0, 0.0, 0.0],
                         gate_noise=[1.0, 0.0, 0.0])
    mem = pqvm.run_program(p)
    _check_mem_equal(mem, {'ro': [[1]]})

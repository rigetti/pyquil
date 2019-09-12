import pytest

from pyquil import Program
from pyquil.api import ForestConnection, PersistentQVM, QVMSimulationMethod, QVMAllocationMethod
from pyquil.gates import MEASURE, X
from pyquil.tests.utils import is_qvm_version_string


def test_pqvm_version(forest_app_ng: ForestConnection):
    pqvm = PersistentQVM(num_qubits=0, connection=forest_app_ng)
    version = pqvm.get_version_info()
    assert is_qvm_version_string(version)


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
            assert mem == {'ro': [[1]]}


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
    assert mem == {'ro': [[1]]}

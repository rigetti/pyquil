import networkx as nx
import numpy as np
import pytest

from rpcq.messages import PyQuilExecutableResponse

from pyquil import Program
from pyquil.api import QVM, ForestConnection, LocalQVMCompiler
from pyquil.api._compiler import _extract_program_from_pyquil_executable_response
from pyquil.device import NxDevice
from pyquil.gates import MEASURE, X, CNOT, H


def test_qvm_run_pqer(forest: ForestConnection):
    qvm = QVM(connection=forest, gate_noise=[0.01] * 3)
    p = Program(X(0), MEASURE(0, 0))
    p.wrap_in_numshots_loop(1000)
    nq = PyQuilExecutableResponse(program=p.out(), attributes={'num_shots': 1000})
    qvm.load(nq)
    qvm.run()
    qvm.wait()
    bitstrings = qvm.read_memory(region_name="ro")
    assert bitstrings.shape == (1000, 1)
    assert np.mean(bitstrings) > 0.8


def test_qvm_run_just_program(forest: ForestConnection):
    qvm = QVM(connection=forest, gate_noise=[0.01] * 3)
    p = Program(X(0), MEASURE(0, 0))
    p.wrap_in_numshots_loop(1000)
    qvm.load(p)
    qvm.run()
    qvm.wait()
    bitstrings = qvm.read_memory(region_name="ro")
    assert bitstrings.shape == (1000, 1)
    assert np.mean(bitstrings) > 0.8


def test_qvm_run_only_pqer(forest: ForestConnection):
    qvm = QVM(connection=forest, gate_noise=[0.01] * 3, requires_executable=True)
    p = Program(X(0), MEASURE(0, 0))
    p.wrap_in_numshots_loop(1000)

    with pytest.raises(TypeError) as e:
        qvm.load(p)
        qvm.run()
        qvm.wait()
    assert e.match(r'.*Make sure you have explicitly compiled your program.*')

    nq = PyQuilExecutableResponse(program=p.out(), attributes={'num_shots': 1000})
    qvm.load(nq)
    qvm.run()
    qvm.wait()
    bitstrings = qvm.read_memory(region_name="ro")
    assert bitstrings.shape == (1000, 1)
    assert np.mean(bitstrings) > 0.8


def test_qvm_run_no_measure(forest: ForestConnection):
    qvm = QVM(connection=forest)
    p = Program(X(0))
    nq = PyQuilExecutableResponse(program=p.out(), attributes={'num_shots': 100})
    qvm.load(nq).run().wait()
    bitstrings = qvm.read_memory(region_name="ro")
    assert bitstrings.shape == (100, 0)


def test_roundtrip_pyquilexecutableresponse():
    p = Program(H(10), CNOT(10, 11))
    lcqvm = LocalQVMCompiler(endpoint=None, device=NxDevice(nx.complete_graph(3)))
    pqer = lcqvm.native_quil_to_executable(p)
    p2 = _extract_program_from_pyquil_executable_response(pqer)
    for i1, i2 in zip(p, p2):
        assert i1 == i2


def test_qvm_version(forest: ForestConnection):
    qvm = QVM(connection=forest)
    version = qvm.get_version_info()

    def is_a_version_string(version_string: str):
        parts = version_string.split('.')
        try:
            map(int, parts)
        except ValueError:
            return False
        return True

    assert is_a_version_string(version)

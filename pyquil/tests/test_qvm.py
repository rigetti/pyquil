import networkx as nx
import numpy as np

from rpcq.core_messages import PyQuilExecutableResponse

from pyquil import Program
from pyquil.api import QVM, ForestConnection, LocalQVMCompiler
from pyquil.api._compiler import _extract_program_from_pyquil_executable_response
from pyquil.device import NxDevice
from pyquil.gates import MEASURE, X, CNOT, H


def test_qvm_run(forest: ForestConnection):
    qvm = QVM(connection=forest, gate_noise=[0.01] * 3)
    p = Program(X(0), MEASURE(0, 0))
    p.wrap_in_numshots_loop(1000)
    nq = PyQuilExecutableResponse(program=p.out(), attributes={'num_shots': 1000})
    qvm.load(nq)
    qvm.run()
    qvm.wait()
    bitstrings = qvm.read_from_memory_region(region_name="ro", offsets=True)
    assert bitstrings.shape == (1000, 1)
    assert np.mean(bitstrings) > 0.8


def test_roundtrip_pyquilexecutableresponse():
    p = Program(H(10), CNOT(10, 11))
    lcqvm = LocalQVMCompiler(endpoint=None, device=NxDevice(nx.complete_graph(3)))
    pqer = lcqvm.native_quil_to_executable(p)
    p2 = _extract_program_from_pyquil_executable_response(pqer)
    for i1, i2 in zip(p, p2):
        assert i1 == i2

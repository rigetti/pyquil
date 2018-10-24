import networkx as nx
import numpy as np
import pytest

from pyquil import Program
from pyquil.api._compiler import _collect_classical_memory_write_locations
from pyquil.api._config import PyquilConfig
from pyquil.api import QuantumComputer, QPU, QPUCompiler
from pyquil.api._qpu import _extract_bitstrings
from pyquil.device import NxDevice
from pyquil.gates import X


def test_qpu_run():
    config = PyquilConfig()
    if config.qpu_url and config.compiler_url:
        g = nx.Graph()
        g.add_node(0)
        device = NxDevice(g)

        qc = QuantumComputer(name="pyQuil test QC",
                             qam=QPU(endpoint=config.qpu_url,
                                     user="pyQuil test suite"),
                             device=device,
                             compiler=QPUCompiler(endpoint=config.compiler_url,
                                                  device=device))
        bitstrings = qc.run_and_measure(
            quil_program=Program(X(0)),
            trials=1000,
        )
        assert bitstrings.shape == (1000, 1)
        assert np.mean(bitstrings) > 0.8
    else:
        pytest.skip("QPU or compiler-server not available; skipping QPU run test.")


def test_readout_demux():
    p = Program("""DECLARE ro BIT[6]
RESET
RX(pi/2) 0
RX(pi/2) 1
RX(pi/2) 2
RX(pi/2) 3
MEASURE 0 ro[0]
MEASURE 1 ro[1]
MEASURE 2
RX(pi/2) 0
RX(pi/2) 1
RX(pi/2) 2
RX(pi/2) 3
MEASURE 0 ro[2]
MEASURE 1 ro[3]
MEASURE 2 ro[4]
MEASURE 3 ro[5]
""")
    ro_sources = _collect_classical_memory_write_locations(p)

    assert ro_sources == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 0)
    ]

    num_shots = 1000
    buffers = {
        # 0 measured, stored twice
        "q0": np.random.randint(0, 2, size=(num_shots, 2)),
        # 1 measured, stored twice
        "q1": np.random.randint(0, 2, size=(num_shots, 2)),
        # 2 measured twice, stored once
        "q2": np.random.randint(0, 2, size=(num_shots, 2)),
        # 3 measured once
        "q3": np.random.randint(0, 2, size=num_shots),
    }

    bitstrings = _extract_bitstrings(ro_sources, buffers=buffers)
    assert bitstrings.dtype == np.int64
    assert np.allclose(bitstrings[:, 0], buffers["q0"][:, 0])
    assert np.allclose(bitstrings[:, 1], buffers["q1"][:, 0])
    assert np.allclose(bitstrings[:, 2], buffers["q0"][:, 1])
    assert np.allclose(bitstrings[:, 3], buffers["q1"][:, 1])
    assert np.allclose(bitstrings[:, 4], buffers["q2"][:, 1])
    assert np.allclose(bitstrings[:, 5], buffers["q3"])

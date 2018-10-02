import networkx as nx
import numpy as np
import pytest

from pyquil import Program
from pyquil.api._config import PyquilConfig
from pyquil.api import QuantumComputer, QPU, QPUCompiler
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

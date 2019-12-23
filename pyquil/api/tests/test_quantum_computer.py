import networkx as nx
import numpy as np

from pyquil import Program
from pyquil.api import QVM, QuantumComputer
from pyquil.device import NxDevice
from pyquil.experiment import ExperimentSetting, TomographyExperiment
from pyquil.gates import CNOT, H, RESET
from pyquil.paulis import sX, sY, sZ
from pyquil.tests.utils import DummyCompiler


def test_qc_expectation(forest):
    device = NxDevice(nx.complete_graph(2))
    qc = QuantumComputer(
        name="testy!", qam=QVM(connection=forest), device=device, compiler=DummyCompiler()
    )

    # bell state program
    p = Program()
    p += RESET()
    p += H(0)
    p += CNOT(0, 1)
    p.wrap_in_numshots_loop(10)

    # XX, YY, ZZ experiment
    sx = ExperimentSetting(in_state=sZ(0) * sZ(1), out_operator=sX(0) * sX(1))
    sy = ExperimentSetting(in_state=sZ(0) * sZ(1), out_operator=sY(0) * sY(1))
    sz = ExperimentSetting(in_state=sZ(0) * sZ(1), out_operator=sZ(0) * sZ(1))

    e = TomographyExperiment(settings=[sx, sy, sz], program=p)

    results = qc.experiment(e)

    # XX expectation value for bell state |00> + |11> is 1
    assert np.isclose(results[0].expectation, 1)
    assert np.isclose(results[0].std_err, 0)
    assert results[0].total_counts == 40

    # YY expectation value for bell state |00> + |11> is -1
    assert np.isclose(results[1].expectation, -1)
    assert np.isclose(results[1].std_err, 0)
    assert results[1].total_counts == 40

    # ZZ expectation value for bell state |00> + |11> is 1
    assert np.isclose(results[2].expectation, 1)
    assert np.isclose(results[2].std_err, 0)
    assert results[2].total_counts == 40


def test_qc_expectation_larger_lattice(forest):
    device = NxDevice(nx.complete_graph(4))
    qc = QuantumComputer(
        name="testy!", qam=QVM(connection=forest), device=device, compiler=DummyCompiler()
    )

    q0 = 2
    q1 = 3

    # bell state program
    p = Program()
    p += RESET()
    p += H(q0)
    p += CNOT(q0, q1)
    p.wrap_in_numshots_loop(10)

    # XX, YY, ZZ experiment
    sx = ExperimentSetting(in_state=sZ(q0) * sZ(q1), out_operator=sX(q0) * sX(q1))
    sy = ExperimentSetting(in_state=sZ(q0) * sZ(q1), out_operator=sY(q0) * sY(q1))
    sz = ExperimentSetting(in_state=sZ(q0) * sZ(q1), out_operator=sZ(q0) * sZ(q1))

    e = TomographyExperiment(settings=[sx, sy, sz], program=p)

    results = qc.experiment(e)

    # XX expectation value for bell state |00> + |11> is 1
    assert np.isclose(results[0].expectation, 1)
    assert np.isclose(results[0].std_err, 0)
    assert results[0].total_counts == 40

    # YY expectation value for bell state |00> + |11> is -1
    assert np.isclose(results[1].expectation, -1)
    assert np.isclose(results[1].std_err, 0)
    assert results[1].total_counts == 40

    # ZZ expectation value for bell state |00> + |11> is 1
    assert np.isclose(results[2].expectation, 1)
    assert np.isclose(results[2].std_err, 0)
    assert results[2].total_counts == 40

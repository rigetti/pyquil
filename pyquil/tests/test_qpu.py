import networkx as nx
import numpy as np
import pytest

from rpcq.messages import ParameterAref

from pyquil.parser import parse
from pyquil import Program
from pyquil.api._compiler import _collect_classical_memory_write_locations
from pyquil.api._config import PyquilConfig
from pyquil.api import QuantumComputer, QPU, QPUCompiler
from pyquil.api._qpu import _extract_bitstrings
from pyquil.device import NxDevice
from pyquil.gates import X
from pyquil.quilatom import Expression


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
            program=Program(X(0)),
            trials=1000,
        )
        assert bitstrings[0].shape == (1000,)
        assert np.mean(bitstrings[0]) > 0.8
        bitstrings = qc.run(qc.compile(Program(X(0))))
        assert bitstrings.shape == (0, 0)
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


GATE_ARITHMETIC_PROGRAMS = [
    Program("""
DECLARE theta REAL[1]
DECLARE beta REAL[1]
DECLARE ro BIT[3]
RX(pi/2) 0
RZ(3*theta) 0
RZ(beta+theta) 0
RX(-pi/2) 0
MEASURE 0 ro[0]
MEASURE 1 ro[1]
"""),
    Program("""
RESET
DECLARE theta REAL[1]
DECLARE beta REAL[1]
DECLARE ro BIT[2]
RX(pi/2) 0
RZ(theta) 0
    """),
    Program("""
DECLARE theta REAL[1]
DECLARE beta REAL[1]
DECLARE ro BIT[3]
RX(pi/2) 0
RZ(0.79*theta) 0
RZ(2*beta+theta*0.5+beta+beta) 0
RX(-pi/2) 0
MEASURE 0 ro[0]
MEASURE 1 ro[1]
"""),
    Program("""
RX(pi) 0
"""),
]


@pytest.fixture(scope='session')
def mock_qpu():
    return QPU(endpoint='tcp://not-needed:00000',
               user="pyQuil test suite")


@pytest.fixture(scope='session')
def gate_arithmetic_binaries(qpu_compiler: QPUCompiler):
    return [qpu_compiler.native_quil_to_executable(p) for p in GATE_ARITHMETIC_PROGRAMS]


def test_load(gate_arithmetic_binaries, mock_qpu):
    def test_binary(binary):
        assert hasattr(binary, "recalculation_table")
        mock_qpu.load(binary)
        assert mock_qpu.status == 'loaded'
        for mref, rule in mock_qpu._executable.recalculation_table.items():
            assert isinstance(mref, ParameterAref)
            assert isinstance(rule, Expression)
        assert len(mock_qpu._executable.recalculation_table) in [0, 2]
    for bin in gate_arithmetic_binaries:
        test_binary(bin)


def test_build_patch_tables(gate_arithmetic_binaries, mock_qpu):
    for idx, bin in enumerate(gate_arithmetic_binaries[:-1]):
        mock_qpu.load(bin)
        theta = np.random.randint(-100, 100) + np.random.random()
        beta = np.random.randint(-100, 100) + np.random.random()
        mock_qpu.write_memory(region_name='theta', value=theta)
        mock_qpu.write_memory(region_name='beta', value=beta)
        patch_table = mock_qpu._build_patch_values()
        assert 'theta' in patch_table.keys()
        assert 'beta' in patch_table.keys()
        if idx == 0 or idx == 2:
            assert len(patch_table) == 3
        for parameter_name, values in patch_table.items():
            assert isinstance(parameter_name, str)
            assert isinstance(values, list)
            for v in values:
                assert isinstance(v, float) or isinstance(v, int)
            if (idx == 0 or idx == 2) and parameter_name not in ('theta', 'beta'):
                assert len(values) == 2


def test_recalculation(gate_arithmetic_binaries, mock_qpu):
    bin = gate_arithmetic_binaries[0]
    mock_qpu.load(bin)
    for theta in np.linspace(0, 1, 50):
        beta = -1 * np.random.random()
        mock_qpu.write_memory(region_name='beta', value=beta)
        mock_qpu.write_memory(region_name='theta', value=theta)
        mock_qpu._update_variables_shim_with_recalculation_table()
        assert any(np.isclose(v, 3 * theta) for v in mock_qpu._variables_shim.values())
        assert any(np.isclose(v, theta + beta) for v in mock_qpu._variables_shim.values())
        assert any(np.isclose(v, theta) for v in mock_qpu._variables_shim.values())
    bin = gate_arithmetic_binaries[2]
    mock_qpu.load(bin)
    beta = np.random.random()
    mock_qpu.write_memory(region_name='beta', value=beta)
    for theta in np.linspace(0, 1, 10):
        mock_qpu.write_memory(region_name='theta', value=theta)
        mock_qpu._update_variables_shim_with_recalculation_table()
        assert any(np.isclose(v, 4 * beta + 0.5 * theta) for v in mock_qpu._variables_shim.values())


def test_resolve_mem_references(gate_arithmetic_binaries, mock_qpu):
    def expression_test(expression, expected_val):
        expression = parse_expression(expression)
        assert np.isclose(mock_qpu._resolve_memory_references(expression), expected_val)

    def test_theta_and_beta(theta, beta):
        mock_qpu.write_memory(region_name='theta', value=theta)
        mock_qpu.write_memory(region_name='beta', value=beta)
        expression_test('sqrt(2) + theta', np.sqrt(2) + theta)
        expression_test('beta*2 + 1', beta * 2 + 1)
        expression_test('(beta + 2) * (1 + theta)', (beta + 2) * (1 + theta))
        expression_test('cos(beta)*sin(theta)', np.cos(beta) * np.sin(theta))
        expression_test('beta * theta', beta * theta)
        expression_test('theta - beta', theta - beta)
    # We just need the status to be loaded so we can write memory
    mock_qpu.load(gate_arithmetic_binaries[0])
    test_theta_and_beta(0.4, 3.1)
    test_theta_and_beta(5, 0)
    for _ in range(10):
        test_theta_and_beta(np.random.random(), np.random.random() + np.random.randint(-100, 100))


def parse_expression(expression):
    """ We have to use this as a hack for now, RZ is meaningless. """
    return parse(f"RZ({expression}) 0")[0].params[0]

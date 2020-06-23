import networkx as nx
import numpy as np
import pytest

from rpcq.messages import ParameterAref

from pyquil.parser import parse
from pyquil import Program, get_qc
from pyquil.api import QuantumComputer, QPU, QPUCompiler
from pyquil.api._base_connection import Engagement, get_session
from pyquil.api._config import PyquilConfig
from pyquil.device import NxDevice
from pyquil.gates import I, X
from pyquil.quilatom import Expression


def test_qpu_run():
    config = PyquilConfig()
    if config.qpu_url and config.qpu_compiler_url:
        g = nx.Graph()
        g.add_node(0)
        device = NxDevice(g)

        qc = QuantumComputer(
            name="pyQuil test QC",
            qam=QPU(endpoint=config.qpu_url, user="pyQuil test suite"),
            device=device,
            compiler=QPUCompiler(
                quilc_endpoint=config.quilc_url,
                qpu_compiler_endpoint=config.qpu_compiler_url,
                device=device,
            ),
        )
        bitstrings = qc.run_and_measure(program=Program(X(0)), trials=1000)
        assert bitstrings[0].shape == (1000,)
        assert np.mean(bitstrings[0]) > 0.8
        bitstrings = qc.run(qc.compile(Program(X(0))))
        assert bitstrings.shape == (0, 0)
    else:
        pytest.skip("QPU or compiler-server not available; skipping QPU run test.")


GATE_ARITHMETIC_PROGRAMS = [
    Program(
        """
DECLARE theta REAL[1]
DECLARE beta REAL[1]
DECLARE ro BIT[3]
RX(pi/2) 0
RZ(3*theta) 0
RZ(beta+theta) 0
RX(-pi/2) 0
MEASURE 0 ro[0]
MEASURE 1 ro[1]
"""
    ),
    Program(
        """
RESET
DECLARE theta REAL[1]
DECLARE beta REAL[1]
DECLARE ro BIT[2]
RX(pi/2) 0
RZ(theta) 0
    """
    ),
    Program(
        """
DECLARE theta REAL[1]
DECLARE beta REAL[1]
DECLARE ro BIT[3]
RX(pi/2) 0
RZ(0.79*theta) 0
RZ(2*beta+theta*0.5+beta+beta) 0
RX(-pi/2) 0
MEASURE 0 ro[0]
MEASURE 1 ro[1]
"""
    ),
    Program(
        """
RX(pi) 0
"""
    ),
]


@pytest.fixture
def mock_qpu():
    return QPU(endpoint="tcp://not-needed:00000", user="pyQuil test suite")


@pytest.fixture
def qpu_compiler(test_device):
    try:
        config = PyquilConfig()
        compiler = QPUCompiler(
            quilc_endpoint=config.quilc_url,
            qpu_compiler_endpoint=config.qpu_compiler_url,
            device=test_device,
            timeout=0.5,
        )
        compiler.quil_to_native_quil(Program(I(0)))
        return compiler
    except Exception as e:
        return pytest.skip(f"This test requires compiler connection: {e}")


@pytest.fixture
def gate_arithmetic_binaries(qpu_compiler: QPUCompiler):
    return [qpu_compiler.native_quil_to_executable(p) for p in GATE_ARITHMETIC_PROGRAMS]


def test_load(gate_arithmetic_binaries, mock_qpu):
    def test_binary(binary):
        assert hasattr(binary, "recalculation_table")
        mock_qpu.load(binary)
        assert mock_qpu.status == "loaded"
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
        mock_qpu.write_memory(region_name="theta", value=theta)
        mock_qpu.write_memory(region_name="beta", value=beta)
        patch_table = mock_qpu._build_patch_values()
        assert "theta" in patch_table.keys()
        assert "beta" in patch_table.keys()
        if idx == 0 or idx == 2:
            assert len(patch_table) == 3
        for parameter_name, values in patch_table.items():
            assert isinstance(parameter_name, str)
            assert isinstance(values, list)
            for v in values:
                assert isinstance(v, float) or isinstance(v, int)
            if (idx == 0 or idx == 2) and parameter_name not in ("theta", "beta"):
                assert len(values) == 2


def test_recalculation(gate_arithmetic_binaries, mock_qpu):
    bin = gate_arithmetic_binaries[0]
    mock_qpu.load(bin)
    for theta in np.linspace(0, 1, 50):
        beta = -1 * np.random.random()
        mock_qpu.write_memory(region_name="beta", value=beta)
        mock_qpu.write_memory(region_name="theta", value=theta)
        mock_qpu._update_variables_shim_with_recalculation_table()
        assert any(np.isclose(v, 3 * theta) for v in mock_qpu._variables_shim.values())
        assert any(np.isclose(v, theta + beta) for v in mock_qpu._variables_shim.values())
        assert any(np.isclose(v, theta) for v in mock_qpu._variables_shim.values())
    bin = gate_arithmetic_binaries[2]
    mock_qpu.load(bin)
    beta = np.random.random()
    mock_qpu.write_memory(region_name="beta", value=beta)
    for theta in np.linspace(0, 1, 10):
        mock_qpu.write_memory(region_name="theta", value=theta)
        mock_qpu._update_variables_shim_with_recalculation_table()
        assert any(np.isclose(v, 4 * beta + 0.5 * theta) for v in mock_qpu._variables_shim.values())


def test_resolve_mem_references(gate_arithmetic_binaries, mock_qpu):
    def expression_test(expression, expected_val):
        expression = parse_expression(expression)
        assert np.isclose(mock_qpu._resolve_memory_references(expression), expected_val)

    def test_theta_and_beta(theta, beta):
        mock_qpu.write_memory(region_name="theta", value=theta)
        mock_qpu.write_memory(region_name="beta", value=beta)
        expression_test("SQRT(2) + theta", np.sqrt(2) + theta)
        expression_test("beta*2 + 1", beta * 2 + 1)
        expression_test("(beta + 2) * (1 + theta)", (beta + 2) * (1 + theta))
        expression_test("COS(beta)*SIN(theta)", np.cos(beta) * np.sin(theta))
        expression_test("beta * theta", beta * theta)
        expression_test("theta - beta", theta - beta)

    # We just need the status to be loaded so we can write memory
    mock_qpu.load(gate_arithmetic_binaries[0])
    test_theta_and_beta(0.4, 3.1)
    test_theta_and_beta(5, 0)
    for _ in range(10):
        test_theta_and_beta(np.random.random(), np.random.random() + np.random.randint(-100, 100))


def parse_expression(expression):
    """ We have to use this as a hack for now, RZ is meaningless. """
    return parse(f"RZ({expression}) 0")[0].params[0]


def test_run_expects_executable(qvm, qpu_compiler):
    # https://github.com/rigetti/pyquil/issues/740

    # This test might need some more knowledgeable eyes. Not sure how
    # to best mock a qpu.
    qc = get_qc("1q-qvm")
    qc.qam = QPU(endpoint="tcp://not-needed:00000", user="pyQuil test suite")

    p = Program(X(0))
    with pytest.raises(TypeError):
        qc.run(p)


def test_qpu_not_engaged_error():
    with pytest.raises(ValueError):
        QPU()


def test_qpu_does_not_engage_without_session():
    qpu = QPU(endpoint="tcp://fake.qpu:50052")

    assert qpu._get_client_auth_config() is None


def test_qpu_reengage_when_invalid():
    config = PyquilConfig()
    engagement = Engagement(
        server_public_key=b"abc123",
        client_public_key=b"abc123",
        client_secret_key=b"abc123",
        expires_at=9999999999.0,
        qpu_endpoint="tcp://fake.qpu:50053",
        qpu_compiler_endpoint="tcp://fake.compiler:5555",
    )

    assert engagement.is_valid()

    session = get_session(config=config)
    config._engagement_requested = True
    config.get_engagement = lambda: engagement

    qpu = QPU(session=session)

    assert qpu._client_engagement is None
    assert qpu._get_client_auth_config() is not None
    assert qpu._client_engagement is engagement

    # By expiring the previous engagement, we expect QPU to attempt to re-engage
    engagement.expires_at = 0.0
    assert not engagement.is_valid()

    new_engagement = Engagement(
        server_public_key=b"abc12345",
        client_public_key=b"abc12345",
        client_secret_key=b"abc12345",
        expires_at=9999999999.0,
        qpu_endpoint="tcp://fake.qpu:50053",
        qpu_compiler_endpoint="tcp://fake.compiler:5555",
    )

    config.get_engagement = lambda: new_engagement

    new_auth_config = qpu._get_client_auth_config()
    assert new_auth_config is not None
    assert new_auth_config.client_public_key == new_engagement.client_public_key
    assert qpu._client_engagement is new_engagement

    new_engagement.expires_at = 0.0
    config.get_engagement = lambda: None

    assert qpu._get_client_auth_config() is None

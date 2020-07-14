import math
import pytest
import requests_mock

from rpcq.core_messages import QuiltBinaryExecutableResponse

from pyquil import Program
from pyquil.quilatom import FormalArgument
from pyquil.quilbase import DefCalibration
from pyquil.api._base_connection import get_session
from pyquil.api._compiler import QPUCompiler
from pyquil.api._config import PyquilConfig
from pyquil.api._errors import UserMessageError
from pyquil.device import Device
from pyquil.gates import RX, MEASURE, RZ
from pyquil.tests.utils import api_fixture_path


def simple_program():
    program = Program()
    readout = program.declare("ro", "BIT", 3)
    program += RX(math.pi / 2, 0)
    program += MEASURE(0, readout[0])
    return program


SIMPLE_RESPONSE = {
    "program": "bAsE64==",
    "debug": {},
    "memory_descriptors": {},
    "ro_sources": [],
    "_type": "QuiltBinaryExecutableResponse",
}

DUMMY_ISA_DICT = {"1Q": {"0": {}, "1": {}}, "2Q": {"0-1": {}}}

TEST_CONFIG_PATHS = {
    "QCS_CONFIG": api_fixture_path("qcs_config.ini"),
    "FOREST_CONFIG": api_fixture_path("forest_config.ini"),
}


def test_http_compilation(compiler):
    device_name = "test_device"
    mock_url = "http://mock-qpu-compiler"

    config = PyquilConfig(TEST_CONFIG_PATHS)
    session = get_session(config=config)
    mock_adapter = requests_mock.Adapter()
    session.mount("http://", mock_adapter)

    headers = {
        # access token from ./data/user_auth_token_valid.json.
        "Authorization": "Bearer secret"
    }
    mock_adapter.register_uri(
        "POST",
        f"{mock_url}/devices/{device_name}/get_version_info",
        status_code=200,
        json={},
        headers=headers,
    )

    mock_adapter.register_uri(
        "POST",
        f"{mock_url}/devices/{device_name}/native_quilt_to_binary",
        status_code=200,
        json=SIMPLE_RESPONSE,
        headers=headers,
    )

    device = Device(
        name="not_actually_device_name", raw={"device_name": device_name, "isa": DUMMY_ISA_DICT}
    )
    compiler = QPUCompiler(
        quilc_endpoint=session.config.quilc_url,
        qpu_compiler_endpoint=mock_url,
        device=device,
        session=session,
    )

    compilation_result = compiler.native_quil_to_executable(
        compiler.quil_to_native_quil(simple_program())
    )

    assert isinstance(compilation_result, QuiltBinaryExecutableResponse)
    assert compilation_result.program == SIMPLE_RESPONSE["program"]


def test_http_compilation_failure(compiler):
    device_name = "test_device"
    mock_url = "http://mock-qpu-compiler"

    config = PyquilConfig(TEST_CONFIG_PATHS)
    session = get_session(config=config)
    mock_adapter = requests_mock.Adapter()
    session.mount("http://", mock_adapter)

    headers = {
        # access token from ./data/user_auth_token_valid.json.
        "Authorization": "Bearer secret"
    }

    mock_adapter.register_uri(
        "POST",
        f"{mock_url}/devices/{device_name}/get_version_info",
        status_code=200,
        json={},
        headers=headers,
    )

    mock_adapter.register_uri(
        "POST",
        f"{mock_url}/devices/{device_name}/native_quilt_to_binary",
        status_code=500,
        json={"message": "test compilation failed"},
        headers=headers,
    )

    device = Device(
        name="not_actually_device_name", raw={"device_name": device_name, "isa": DUMMY_ISA_DICT}
    )

    compiler = QPUCompiler(
        quilc_endpoint=session.config.quilc_url,
        qpu_compiler_endpoint=mock_url,
        device=device,
        session=session,
    )

    native_quil = compiler.quil_to_native_quil(simple_program())

    with pytest.raises(UserMessageError) as excinfo:
        compiler.native_quil_to_executable(native_quil)

    assert "test compilation failed" in str(excinfo.value)


def test_invalid_protocol():
    device_name = "test_device"
    mock_url = "not-http-or-tcp://mock-qpu-compiler"

    config = PyquilConfig(TEST_CONFIG_PATHS)
    session = get_session(config=config)
    mock_adapter = requests_mock.Adapter()
    session.mount("", mock_adapter)
    device = Device(
        name="not_actually_device_name", raw={"device_name": device_name, "isa": DUMMY_ISA_DICT}
    )

    with pytest.raises(UserMessageError):
        QPUCompiler(
            quilc_endpoint=session.config.quilc_url,
            qpu_compiler_endpoint=mock_url,
            device=device,
            session=session,
        )


def test_compile_with_quilt_calibrations(compiler):
    device_name = "test_device"
    mock_url = "http://mock-qpu-compiler"

    config = PyquilConfig(TEST_CONFIG_PATHS)
    session = get_session(config=config)
    mock_adapter = requests_mock.Adapter()
    session.mount("http://", mock_adapter)

    headers = {
        # access token from ./data/user_auth_token_valid.json.
        "Authorization": "Bearer secret"
    }
    mock_adapter.register_uri(
        "POST",
        f"{mock_url}/devices/{device_name}/get_version_info",
        status_code=200,
        json={},
        headers=headers,
    )

    mock_adapter.register_uri(
        "POST",
        f"{mock_url}/devices/{device_name}/native_quilt_to_binary",
        status_code=200,
        json=SIMPLE_RESPONSE,
        headers=headers,
    )

    device = Device(
        name="not_actually_device_name", raw={"device_name": device_name, "isa": DUMMY_ISA_DICT}
    )
    compiler = QPUCompiler(
        quilc_endpoint=session.config.quilc_url,
        qpu_compiler_endpoint=mock_url,
        device=device,
        session=session,
    )

    program = simple_program()
    q = FormalArgument("q")
    defn = DefCalibration(
        "H", [], [q], [RZ(math.pi / 2, q), RX(math.pi / 2, q), RZ(math.pi / 2, q)]
    )
    cals = [defn]
    program._calibrations = cals
    # this should more or less pass through
    compilation_result = compiler.quil_to_native_quil(program, protoquil=True)
    assert compilation_result.calibrations == cals
    assert program.calibrations == cals
    assert compilation_result == program

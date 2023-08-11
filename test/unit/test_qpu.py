import pytest
from unittest.mock import patch, MagicMock

import numpy as np

from pyquil.api import ConnectionStrategy, ExecutionOptions
from pyquil.api._qpu import QPU
from pyquil.api._abstract_compiler import EncryptedProgram
from pyquil.quil import Program
from pytest_mock import MockerFixture
from qcs_sdk import RegisterMatrixConversionError
from qcs_sdk.qpu.api import Register, ExecutionResult, ExecutionResults
from rpcq.messages import ParameterSpec

from pyquil.quilatom import MemoryReference

import pyquil.api


@pytest.fixture
def mock_encrypted_program():
    program = Program(
        "DECLARE ro BIT[2]",
        "H 0",
        "CNOT 0 1",
        "MEASURE 0 ro[0]",
        "MEASURE 1 ro[2]",
    )
    return EncryptedProgram(
        program=program.out(),
        memory_descriptors={"ro": ParameterSpec(length=2, type="BIT")},
        ro_sources={
            MemoryReference("q1_classified"): "q1_classified",
            MemoryReference("q1_unclassified"): "q1_unclassified",
            MemoryReference("ro", 1): "q1",
            MemoryReference("ro"): "q0",
        },
        recalculation_table=[],
    )


def test_default_execution_options():
    qpu = QPU(quantum_processor_id="test", timeout=15.0, endpoint_id="endpoint-id")

    builder = ExecutionOptions.builder()
    builder.timeout_seconds = 15.0
    builder.connection_strategy = ConnectionStrategy.endpoint_id("endpoint-id")
    expected = builder.build()

    assert qpu.execution_options == expected


def test_provided_execution_options():
    builder = ExecutionOptions.builder()
    builder.timeout_seconds = 15.0
    builder.connection_strategy = ConnectionStrategy.direct_access()
    options = builder.build()

    qpu = QPU(quantum_processor_id="test", execution_options=options)
    assert qpu.execution_options == options


@patch("pyquil.api._qpu.retrieve_results")
@patch("pyquil.api._qpu.submit")
def test_qpu_execute(mock_submit, mock_retrieve_results, mock_encrypted_program):
    qpu = QPU(quantum_processor_id="test")

    mock_submit.return_value = "some-job-id"
    execute_response = qpu.execute(mock_encrypted_program)

    mock_retrieve_results.return_value = ExecutionResults(
        {
            "q0": ExecutionResult.from_register(Register.from_i32([1, 1, 1, 1])),
            "q1": ExecutionResult.from_register(Register.from_i32([1, 1, 1, 1])),
        }
    )

    result = qpu.get_result(execute_response)

    assert np.all(result.register_map["ro"] == np.array([[1, 1], [1, 1], [1, 1], [1, 1]]))
    assert np.all(result.register_map["ro"] == result.readout_data["ro"])


@patch("pyquil.api._qpu.retrieve_results")
@patch("pyquil.api._qpu.submit")
def test_qpu_execute_jagged_results(mock_submit, mock_retrieve_results, mock_encrypted_program):
    qpu = QPU(quantum_processor_id="test")

    mock_submit.return_value = "some-job-id"
    execute_response = qpu.execute(mock_encrypted_program)

    mock_retrieve_results.return_value = ExecutionResults(
        {
            "q0": ExecutionResult.from_register(Register.from_i32([1, 1])),
            "q1": ExecutionResult.from_register(Register.from_i32([1, 1, 1, 1])),
        }
    )

    result = qpu.get_result(execute_response)

    with pytest.raises(RegisterMatrixConversionError):
        result.register_map

    assert result.raw_readout_data == {
        "mappings": {"ro[0]": "q0", "ro[1]": "q1"},
        "readout_values": {"q0": [1, 1], "q1": [1, 1, 1, 1]},
    }

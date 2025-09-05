import pickle
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qcs_sdk import ExecutionData, ResultData
from qcs_sdk.qpu import MemoryValues, QPUResultData, ReadoutValues
from qcs_sdk.qpu.api import ExecutionResult, ExecutionResults, Register
from rpcq.messages import ParameterSpec

from pyquil.api import (
    ConnectionStrategy,
    ExecutionOptions,
    ExecutionOptionsBuilder,
    QAMExecutionResult,
    QPUExecuteResponse,
    RegisterMatrixConversionError,
)
from pyquil.api._abstract_compiler import EncryptedProgram
from pyquil.api._qpu import QPU
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference


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
@patch("pyquil.api._qpu.submit_with_parameter_batch")
def test_qpu_execute(
    mock_submit: MagicMock, mock_retrieve_results: MagicMock, mock_encrypted_program: EncryptedProgram
):
    qpu = QPU(quantum_processor_id="test")

    mock_submit.return_value = ["some-job-id"]
    execute_response = qpu.execute(mock_encrypted_program)

    mock_retrieve_results.return_value = ExecutionResults(
        {
            "q0": ExecutionResult.from_register(Register.from_i32([1, 1, 1, 1])),
            "q1": ExecutionResult.from_register(Register.from_i32([1, 1, 1, 1])),
        },
        {
            "binary": MemoryValues.from_binary([0, 1, 0, 1]),
            "int": MemoryValues.from_integer([2, 3, 4]),
            "real": MemoryValues.from_real([5.0, 6.0, 7.0]),
        },
    )

    result = qpu.get_result(execute_response)

    assert np.all(result.get_register_map()["ro"] == np.array([[1, 1], [1, 1], [1, 1], [1, 1]]))
    assert np.all(result.get_register_map()["ro"] == result.readout_data["ro"])
    assert result.get_memory_values() == {
        "binary": MemoryValues.from_binary([0, 1, 0, 1]),
        "int": MemoryValues.from_integer([2, 3, 4]),
        "real": MemoryValues.from_real([5.0, 6.0, 7.0]),
    }


@patch("pyquil.api._qpu.retrieve_results")
@patch("pyquil.api._qpu.submit_with_parameter_batch")
def test_qpu_execute_jagged_results(
    mock_submit: MagicMock, mock_retrieve_results: MagicMock, mock_encrypted_program: EncryptedProgram
):
    qpu = QPU(quantum_processor_id="test")

    mock_submit.return_value = ["some-job-id"]
    execute_response = qpu.execute(mock_encrypted_program)

    mock_retrieve_results.return_value = ExecutionResults(
        {
            "q0": ExecutionResult.from_register(Register.from_i32([1, 1])),
            "q1": ExecutionResult.from_register(Register.from_i32([1, 1, 1, 1])),
        },
        {
            "binary": MemoryValues.from_binary([0, 1, 0, 1]),
            "int": MemoryValues.from_integer([2, 3, 4]),
            "real": MemoryValues.from_real([5.0, 6.0, 7.0]),
        },
    )

    result = qpu.get_result(execute_response)

    with pytest.raises(RegisterMatrixConversionError):
        result.get_register_map()

    raw_readout_data = result.get_raw_readout_data()

    assert raw_readout_data.mappings == {"ro[0]": "q0", "ro[1]": "q1"}
    assert raw_readout_data.readout_values == {"q0": [1, 1], "q1": [1, 1, 1, 1]}
    assert raw_readout_data.memory_values == {
        "binary": [0, 1, 0, 1],
        "int": [2, 3, 4],
        "real": [5.0, 6.0, 7.0],
    }


class TestQPUExecutionOptions:
    @patch("pyquil.api._qpu.retrieve_results")
    @patch("pyquil.api._qpu.submit_with_parameter_batch")
    def test_submit_with_class_options(
        self, mock_submit: MagicMock, mock_retrieve_results: MagicMock, mock_encrypted_program: EncryptedProgram
    ):
        """Asserts that a ``QPU``'s execution_options property is used for submission, appears in the returned
        ``QPUExecuteResponse``, and is used for retrieval of results when execution options are not provided to
        ``QPU.execute``.
        """
        qpu = QPU(quantum_processor_id="test")
        execution_options_builder = ExecutionOptionsBuilder()
        execution_options_builder.timeout_seconds = 10.0
        execution_options_builder.connection_strategy = ConnectionStrategy.endpoint_id("some-endpoint-id")
        execution_options = execution_options_builder.build()
        qpu.execution_options = execution_options

        mock_submit.return_value = ["some-job-id"]
        execute_response = qpu.execute(mock_encrypted_program)
        assert execute_response.execution_options == qpu.execution_options

        mock_retrieve_results.return_value = ExecutionResults(
            {
                "q0": ExecutionResult.from_register(Register.from_i32([1, 1])),
                "q1": ExecutionResult.from_register(Register.from_i32([1, 1, 1, 1])),
            },
            {"stash": MemoryValues.from_binary([0, 1, 0, 1])},
        )

        qpu.get_result(execute_response)

        mock_retrieve_results.assert_called_once_with(
            job_id="some-job-id",
            quantum_processor_id="test",
            client=qpu._client_configuration,
            execution_options=qpu.execution_options,
        )

    @patch("pyquil.api._qpu.retrieve_results")
    @patch("pyquil.api._qpu.submit_with_parameter_batch")
    def test_submit_with_options(
        self, mock_submit: MagicMock, mock_retrieve_results: MagicMock, mock_encrypted_program: EncryptedProgram
    ):
        """Asserts that execution_options provided to ``QPU.execute`` are used for submission, appear in the returned
        ``QPUExecuteResponse``, and are used for retrieval of results.
        """
        qpu = QPU(quantum_processor_id="test")

        mock_submit.return_value = ["some-job-id"]
        execution_options_builder = ExecutionOptionsBuilder()
        execution_options_builder.timeout_seconds = 10.0
        execution_options_builder.connection_strategy = ConnectionStrategy.endpoint_id("some-endpoint-id")
        execution_options = execution_options_builder.build()
        execute_response = qpu.execute(mock_encrypted_program, execution_options=execution_options)
        assert execute_response.execution_options == execution_options

        mock_retrieve_results.return_value = ExecutionResults(
            {
                "q0": ExecutionResult.from_register(Register.from_i32([1, 1])),
                "q1": ExecutionResult.from_register(Register.from_i32([1, 1, 1, 1])),
            },
            {"stash": MemoryValues.from_binary([0, 1, 0, 1])},
        )

        qpu.get_result(execute_response)

        mock_retrieve_results.assert_called_once_with(
            job_id="some-job-id",
            quantum_processor_id="test",
            client=qpu._client_configuration,
            execution_options=execution_options,
        )


@pytest.mark.parametrize(
    "input",
    [
        (
            QAMExecutionResult(
                executable=mock_encrypted_program(),
                data=ExecutionData(
                    result_data=ResultData.from_qpu(
                        QPUResultData(
                            mappings={"ro[0]": "q0", "ro[1]": "q1"},
                            readout_values={
                                "q0": ReadoutValues.from_integer([1, 1]),
                                "q1": ReadoutValues.from_real([1.1, 1.2]),
                                "q2": ReadoutValues.from_complex([complex(3, 4), complex(2.35, 4.21)]),
                            },
                            memory_values={"int": MemoryValues([2, 3, 4]), "real": MemoryValues([5.0, 6.0, 7.0])},
                        )
                    )
                ),
            )
        ),
        (
            QPUExecuteResponse(
                job_id="some-job-id", _executable=mock_encrypted_program(), execution_options=ExecutionOptions.default()
            )
        ),
    ],
)
def test_pickle_execute_responses(input: Any):
    pickled_response = pickle.dumps(input)
    unpickled_response = pickle.loads(pickled_response)
    assert unpickled_response == input

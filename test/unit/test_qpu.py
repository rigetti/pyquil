from pyquil.api import ConnectionStrategy, ExecutionOptions
from pyquil.api._qpu import QPU


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

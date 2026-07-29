from typing import Union

import numpy as np
import pytest
from numpy.typing import NDArray
from qcs_sdk.client import QCSClient
from qcs_sdk.qpu.api import ExecutionOptionsBuilder, retrieve_results, submit
from qcs_sdk.qpu.translation import TranslationResult, translate

from pyquil._qpu import randomized_compiling as rc

from ..unit import test_qpu_randomized_compiling as trc


def _get_bitstrings_and_final_memory(
    live_quantum_processor_id: str,
    translation_result: TranslationResult,
    memory_map: dict[str, Union[list[int], list[float]]],
    execution_options: ExecutionOptionsBuilder,
    qcs_client: QCSClient,
) -> tuple[NDArray[np.int8], dict[str, Union[list[int], list[float]]]]:
    job_id = submit(
        translation_result.program, memory_map, live_quantum_processor_id, qcs_client, execution_options.build()
    )
    results = retrieve_results(
        job_id,
        quantum_processor_id=live_quantum_processor_id,
        execution_options=execution_options.build(),
        client=qcs_client,
    )
    final_memory: dict[str, Union[list[int], list[float]]] = {k: v.inner() for k, v in results.memory.items()}
    return np.array(
        [execution_result.data.to_i32() for name, execution_result in results.buffers.items() if "_classified" in name]
    ).transpose(), final_memory


@pytest.fixture
def execution_options() -> ExecutionOptionsBuilder:
    return ExecutionOptionsBuilder()


@pytest.mark.parametrize("test_case", trc.CONFIGURATION_TEST_CASES)
def test_randomized_compiling_configuration(
    quantum_processor_id: str,
    client_configuration: QCSClient,
    execution_options: ExecutionOptionsBuilder,
    test_case: trc.ConfigurationTestCase,
    live_qpu_access: bool,
) -> None:
    if not live_qpu_access:
        pytest.skip(reason="skipping this test since it requires live access to a QPU (use --live-qpu-access to run)")
    configuration = test_case.configuration
    rng = np.random.default_rng(trc.CONFIGURATION_TEST_SEED)

    program = test_case.build_quil_program()
    program += trc.build_cycle_program(configuration, test_case.readout_randomization)

    pauli_conjugates_map = rc.PAULI_CONJUGATES_MAPS["CZ"]
    memory_map, rc_seeds, readout_seeds = test_case.generate_seeds_and_memory_map(rng)

    translation_result = translate(program.out(), trc.TEST_SHOT_COUNT, quantum_processor_id, client_configuration)
    bitstrings, final_memory = _get_bitstrings_and_final_memory(
        quantum_processor_id,
        translation_result,
        memory_map,
        execution_options=execution_options,
        qcs_client=client_configuration,
    )

    configuration.verify_final_memory(
        final_memory,
        memory_map,
        trc.TEST_SHOT_COUNT,
        pauli_conjugates_map,
    )

    if test_case.readout_randomization is not None:
        if readout_seeds is None:
            raise ValueError("Readout seeds should not be None when readout randomization is provided.")
        pauli_pairs = test_case.configuration.get_final_pauli_pairs(
            trc.TEST_SHOT_COUNT, pauli_conjugates_map, rc_seeds, accumulate=False
        )
        test_case.readout_randomization.verify_final_memory(
            final_memory,
            readout_seeds,
            trc.TEST_SHOT_COUNT,
            test_case.configuration._cycle_count,
            pauli_pairs,
        )

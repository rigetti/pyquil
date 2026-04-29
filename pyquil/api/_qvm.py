##############################################################################
# Copyright 2016-2026 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
"""Quantum Virtual Machine backed by the quax density-matrix simulator.

The :class:`QVM` implements the :class:`~pyquil.api.QAM` interface so it
can be used as a drop-in replacement anywhere a ``QAM`` is expected
(e.g. inside :class:`~pyquil.api.QuantumComputer`).

Execution flow
--------------
1. The Quil program is simulated via
   :func:`~pyquil.simulation.density_matrix.compute_program_density_matrix`
   (with an optional :class:`~pyquil.noise.NoiseModel`).
2. Born-rule probabilities are extracted from the diagonal of the
   resulting density matrix.
3. Bitstrings are sampled from the probability distribution and
   packaged into a :class:`QAMExecutionResult`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import quax as qx
from qcs_sdk import ExecutionData, RegisterData, ResultData
from qcs_sdk.qvm import QVMResultData

from pyquil.api._qam import QAM, MemoryMap, QAMExecutionResult, QuantumExecutable
from pyquil.quil import Program

if TYPE_CHECKING:
    from pyquil.noise import NoiseModel


@dataclass
class QVMExecuteResponse:
    """Opaque handle returned by :meth:`QVM.execute`."""

    executable: Program
    memory: dict[str, np.ndarray]


class QVM(QAM[QVMExecuteResponse]):
    """A local quantum virtual machine backed by the quax density-matrix simulator.

    :param noise_model: An optional :class:`~pyquil.noise.NoiseModel`.  When
        ``None`` the simulation is noiseless (pure-state evolution via
        density matrix).
    :param random_seed: Seed for the random-number generator used when
        sampling bitstrings.  ``None`` means a fresh seed each time.
    """

    def __init__(
        self,
        noise_model: Optional[NoiseModel] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.noise_model = noise_model

        if random_seed is not None and (not isinstance(random_seed, int) or random_seed < 0):
            raise TypeError("random_seed should be None or a non-negative int")
        self.random_seed = random_seed

    # ── QAM interface ────────────────────────────────────────

    def execute(
        self,
        executable: QuantumExecutable,
        memory_map: Optional[MemoryMap] = None,
        **__: Any,
    ) -> QVMExecuteResponse:
        """Simulate the program and sample bitstrings."""
        if not isinstance(executable, Program):
            raise TypeError(f"`QVM#execute` argument must be a `Program`; got {type(executable)}")

        program: Program = executable
        trials = program.num_shots

        # Determine measured qubits from MEASURE instructions
        measured_qubits = sorted(program.get_qubit_indices())

        # Lazy import to avoid circular dependency
        from pyquil.simulation.density_matrix import compute_program_density_matrix

        # Simulate
        rho = compute_program_density_matrix(
            program,
            noise_model=self.noise_model,
            qubits=measured_qubits,
            memory_map=memory_map,
        )

        # Extract probabilities and sample
        probs = np.asarray(qx.probabilities(rho), dtype=np.float64)
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()  # renormalise for numerical safety

        rng = np.random.default_rng(self.random_seed)
        n_qubits = len(measured_qubits)
        indices = rng.choice(len(probs), size=trials, p=probs)

        # Convert flat indices to bitstrings (big-endian: qubit 0 is MSB)
        bitstrings = ((indices[:, None] >> np.arange(n_qubits - 1, -1, -1)) & 1).astype(np.int8)

        memory: dict[str, np.ndarray] = {"ro": bitstrings}
        return QVMExecuteResponse(executable=program, memory=memory)

    def execute_with_memory_map_batch(
        self,
        executable: QuantumExecutable,
        memory_maps: Iterable[MemoryMap],
        **__: Any,
    ) -> list[QVMExecuteResponse]:
        """Execute the program once per memory map."""
        return [self.execute(executable, memory_map) for memory_map in memory_maps]

    def get_result(self, execute_response: QVMExecuteResponse) -> QAMExecutionResult:
        """Package sampled bitstrings into a :class:`QAMExecutionResult`."""
        memory_map: dict[str, RegisterData] = {}
        for name, array in execute_response.memory.items():
            memory_map[name] = RegisterData(array.tolist())

        qvm_result = QVMResultData.from_memory_map(memory_map)
        result_data = ResultData(qvm_result)
        data = ExecutionData(result_data=result_data, duration=None)
        return QAMExecutionResult(executable=execute_response.executable, data=data)

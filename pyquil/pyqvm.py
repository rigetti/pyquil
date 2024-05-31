"""A pure Python implementation of the Quantum Virtual Machine (QVM)."""

##############################################################################
# Copyright 2018 Rigetti Computing
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
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Optional, Union

import numpy as np
from numpy.random.mtrand import RandomState
from qcs_sdk import ExecutionData, RegisterData, ResultData
from qcs_sdk.qvm import QVMResultData

from pyquil.api import QAM, MemoryMap, QAMExecutionResult, QuantumExecutable
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Label, LabelPlaceholder, MemoryReference
from pyquil.quilbase import (
    ArithmeticBinaryOp,
    ClassicalAdd,
    ClassicalAnd,
    ClassicalDiv,
    ClassicalExchange,
    ClassicalExclusiveOr,
    ClassicalInclusiveOr,
    ClassicalMove,
    ClassicalMul,
    ClassicalNeg,
    ClassicalNot,
    ClassicalSub,
    Declare,
    DefGate,
    DefGateByPaulis,
    DefPermutationGate,
    Gate,
    Halt,
    Jump,
    JumpTarget,
    JumpUnless,
    JumpWhen,
    LogicalBinaryOp,
    Measurement,
    Nop,
    Pragma,
    Reset,
    ResetQubit,
    UnaryClassicalInstruction,
    Wait,
)

log = logging.getLogger(__name__)

QUIL_TO_NUMPY_DTYPE = {"INT": np.int_, "REAL": np.float_, "BIT": np.int8, "OCTET": np.uint8}


class AbstractQuantumSimulator(ABC):
    """An abstract interface for a quantum simulator."""

    @abstractmethod
    def __init__(self, n_qubits: int, rs: Optional[RandomState]):
        """Initialize.

        :param n_qubits: Number of qubits to simulate.
        :param rs: a RandomState (shared with the owning :py:class:`PyQVM`) for
            doing anything stochastic.
        """

    @abstractmethod
    def do_gate(self, gate: Gate) -> "AbstractQuantumSimulator":
        """Perform a gate.

        :return: ``self`` to support method chaining.
        """

    @abstractmethod
    def do_gate_matrix(self, matrix: np.ndarray, qubits: Sequence[int]) -> "AbstractQuantumSimulator":
        """Apply an arbitrary unitary; not necessarily a named gate.

        :param matrix: The unitary matrix to apply. No checks are done
        :param qubits: A list of qubits to apply the unitary to.
        :return: ``self`` to support method chaining.
        """

    def do_program(self, program: Program) -> "AbstractQuantumSimulator":
        """Perform a sequence of gates contained within a program.

        :param program: The program
        :return: self
        """
        for gate in program:
            if not isinstance(gate, Gate):
                raise ValueError("Can only compute the simulate a program composed of `Gate`s")
            self.do_gate(gate)
        return self

    @abstractmethod
    def do_measurement(self, qubit: int) -> int:
        """Measure a qubit and collapse the wavefunction.

        :return: The measurement result. A 1 or a 0.
        """

    @abstractmethod
    def expectation(self, operator: Union[PauliTerm, PauliSum]) -> complex:
        """Compute the expectation of an operator.

        :param operator: The operator
        :return: The operator's expectation value
        """

    @abstractmethod
    def reset(self) -> "AbstractQuantumSimulator":
        """Reset the wavefunction to the ``|000...00>`` state.

        :return: ``self`` to support method chaining.
        """

    @abstractmethod
    def sample_bitstrings(self, n_samples: int) -> np.ndarray:
        """Sample bitstrings from the current state.

        :param n_samples: The number of bitstrings to sample
        :return: A numpy array of shape (n_samples, n_qubits)
        """

    @abstractmethod
    def do_post_gate_noise(self, noise_type: str, noise_prob: float, qubits: list[int]) -> "AbstractQuantumSimulator":
        """Apply noise that happens after each gate application.

        WARNING! This is experimental and the signature of this interface will likely change.

        :param noise_type: The name of the noise type
        :param noise_prob: The probability of that noise happening
        :param qubits: Apply noise to these qubits.
        :return: ``self`` to support method chaining
        """


class PyQVM(QAM["PyQVM"]):
    """A pure python implementation of the Quantum Virtual Machine."""

    def __init__(
        self,
        n_qubits: int,
        quantum_simulator_type: Optional[type[AbstractQuantumSimulator]] = None,
        seed: Optional[int] = None,
        post_gate_noise_probabilities: Optional[dict[str, float]] = None,
    ):
        """PyQuil's built-in Quil virtual machine.

        This class implements common control flow and plumbing and dispatches the "actual" work to
        quantum simulators like ReferenceWavefunctionSimulator, ReferenceDensitySimulator,
        and NumpyWavefunctionSimulator

        :param n_qubits: The number of qubits. Typically this results in the allocation of a large
            ndarray, so be judicious.
        :param quantum_simulator_type: A class that can be instantiated to handle the quantum
            aspects of this QVM. If not specified, the default will be either
            NumpyWavefunctionSimulator (no noise) or ReferenceDensitySimulator (noise)
        :param post_gate_noise_probabilities: A specification of noise model given by
            probabilities of certain types of noise. The dictionary keys are from "relaxation",
            "dephasing", "depolarizing", "phase_flip", "bit_flip", and "bitphase_flip".
            WARNING: experimental. This interface will likely change.
        :param seed: An optional random seed for performing stochastic aspects of the QVM.
        """
        if quantum_simulator_type is None:
            if post_gate_noise_probabilities is None:
                from pyquil.simulation._numpy import NumpyWavefunctionSimulator

                quantum_simulator_type = NumpyWavefunctionSimulator
            else:
                from pyquil.simulation._reference import ReferenceDensitySimulator

                log.info("Using ReferenceDensitySimulator as the backend for PyQVM")
                quantum_simulator_type = ReferenceDensitySimulator

        self.n_qubits = n_qubits
        self.ram: dict[str, list[Union[float, int]]] = {}

        if post_gate_noise_probabilities is None:
            post_gate_noise_probabilities = {}
        self.post_gate_noise_probabilities = post_gate_noise_probabilities

        self.program: Optional[Program] = None
        self.program_counter: int = 0
        self.defined_gates: dict[str, np.ndarray] = dict()

        # private implementation details
        self._qubit_to_ram: Optional[dict[int, int]] = None
        self._ro_size: Optional[int] = None
        self._memory_results = {}  # type: ignore

        self.rs = np.random.RandomState(seed=seed)
        self.wf_simulator = quantum_simulator_type(n_qubits=n_qubits, rs=self.rs)
        self._last_measure_program_loc = None

    def _extract_defined_gates(self) -> None:
        self.defined_gates = dict()
        if self.program is None:
            raise RuntimeError("No program loaded into PyQVM.")
        for dg in self.program.defined_gates:
            if dg.parameters is not None and len(dg.parameters) > 0:
                raise NotImplementedError("PyQVM does not support parameterized DEFGATEs")
            if isinstance(dg, DefPermutationGate) or isinstance(dg, DefGateByPaulis):
                raise NotImplementedError("PyQVM does not support DEFGATE ... AS MATRIX | PAULI-SUM.")
            self.defined_gates[dg.name] = dg.matrix

    def execute_with_memory_map_batch(
        self, executable: QuantumExecutable, memory_maps: Iterable[MemoryMap], **__: Any
    ) -> list["PyQVM"]:
        """Operation is not supported by PyQVM as the state of the instance is reset at the start of each execution."""
        raise NotImplementedError(
            "PyQVM does not support batch execution as the state of the instance is reset at the start of each execute."
        )

    def execute(self, executable: QuantumExecutable, memory_map: Optional[MemoryMap] = None, **__: Any) -> "PyQVM":
        """Execute a program on the PyQVM.

        Note that the state of the instance is reset on each call to ``execute``.

        :return: ``self`` to support method chaining.
        """
        if not isinstance(executable, Program):
            raise TypeError("`executable` argument must be a `Program`")

        self.program = executable
        self._memory_results = {}

        self.ram = {}
        if memory_map:
            self.ram.update(*memory_map)

        self.wf_simulator.reset()

        # grab the gate definitions for future use
        self._extract_defined_gates()

        self._memory_results = {}
        for _ in range(self.program.num_shots):
            self.wf_simulator.reset()
            self._execute_program()
            for name in self.ram.keys():
                self._memory_results.setdefault(name, list())
                self._memory_results[name].append(self.ram[name])

        self._memory_results = {k: np.asarray(v) for k, v in self._memory_results.items()}

        self._bitstrings = self._memory_results.get("ro")

        return self

    def get_result(self, execute_response: "PyQVM") -> QAMExecutionResult:
        """Return results from the PyQVM according to the common QAM API.

        Note that while the ``execute_response`` is not used, it's accepted in order to conform to that API; it's
        unused because the PyQVM, unlike other QAM's, is itself stateful.
        """
        if self.program is None:
            raise RuntimeError("No program loaded into PyQVM.")
        result_data = QVMResultData.from_memory_map(
            {key: RegisterData(matrix.tolist()) for key, matrix in self._memory_results.items()}
        )
        result_data = ResultData(result_data)
        data = ExecutionData(result_data=result_data, duration=None)
        return QAMExecutionResult(
            executable=self.program.copy(),
            data=data,
        )

    def read_memory(self, *, region_name: str) -> np.ndarray:
        """Read memory from the PyQVM."""
        if self._memory_results is None:
            raise ValueError("No memory results available.")
        return np.asarray(self._memory_results[region_name])

    def find_label(self, label: Union[Label, LabelPlaceholder]) -> int:
        """Iterate over the program and find a JumpTarget that has a Label matching the input label.

        :param label: Label object to search for in program
        :return: Program index where ``label`` is found
        """
        if self.program is None:
            raise RuntimeError("No program loaded into PyQVM.")
        for index, action in enumerate(self.program):
            if isinstance(action, JumpTarget):
                if label == action.label:
                    return index

        raise RuntimeError("Improper program - Jump Target not found in the input program!")

    def transition(self) -> bool:
        """Perform a QAM-like transition.

        This function assumes ``program`` and ``program_counter`` instance variables are set
        appropriately, and that the wavefunction simulator and classical memory ``ram`` instance
        variables are in the desired QAM input state.

        :return: whether the QAM should halt after this transition.
        """
        if self.program is None:
            raise RuntimeError("No program loaded into PyQVM.")
        instruction = self.program[self.program_counter]

        if isinstance(instruction, Gate):
            qubits = instruction.get_qubit_indices()
            if instruction.name in self.defined_gates:
                self.wf_simulator.do_gate_matrix(
                    matrix=self.defined_gates[instruction.name],
                    qubits=qubits,
                )
            else:
                self.wf_simulator.do_gate(gate=instruction)

            for noise_type, noise_prob in self.post_gate_noise_probabilities.items():
                self.wf_simulator.do_post_gate_noise(noise_type, noise_prob, qubits=qubits)

            self.program_counter += 1

        elif isinstance(instruction, Measurement):
            measured_val = self.wf_simulator.do_measurement(qubit=instruction.get_qubit_indices().pop())
            meas_reg: Optional[MemoryReference] = instruction.classical_reg
            if meas_reg is None:
                raise ValueError("Measurement instruction must have a classical register.")
            self.ram[meas_reg.name][meas_reg.offset] = measured_val
            self.program_counter += 1

        elif isinstance(instruction, Declare):
            if instruction.shared_region is not None:
                raise NotImplementedError("SHARING is not (yet) implemented.")

            self.ram[instruction.name] = list(
                np.zeros(instruction.memory_size, dtype=QUIL_TO_NUMPY_DTYPE[instruction.memory_type])
            )
            self.program_counter += 1

        elif isinstance(instruction, Pragma):
            # TODO: more stringent checks for what's being pragma'd and warnings
            self.program_counter += 1

        elif isinstance(instruction, Jump):
            # unconditional Jump; go directly to Label
            self.program_counter = self.find_label(instruction.target)

        elif isinstance(instruction, JumpTarget):
            # Label; pass straight over
            self.program_counter += 1

        elif isinstance(instruction, (JumpWhen, JumpUnless)):
            # JumpWhen/Unless; check classical reg
            jump_reg: Optional[MemoryReference] = instruction.condition
            if jump_reg is None:
                raise ValueError("JumpWhen/Unless instruction must have a classical register.")
            cond = self.ram[jump_reg.name][jump_reg.offset]
            if not isinstance(cond, (bool, np.bool_, np.int8, int)):
                raise ValueError(f"{type(instruction)} requires a data type of BIT; not {type(cond)}")
            dest_index = self.find_label(instruction.target)
            if isinstance(instruction, JumpWhen):
                jump_if_cond = True
            elif isinstance(instruction, JumpUnless):
                jump_if_cond = False
            else:
                raise TypeError(f"Invalid {type(instruction)}")

            if not (cond ^ jump_if_cond):
                # jumping: set prog counter to JumpTarget
                self.program_counter = dest_index
            else:
                # not jumping: hop over this instruction
                self.program_counter += 1

        elif isinstance(instruction, UnaryClassicalInstruction):
            # UnaryClassicalInstruction; set classical reg
            target = instruction.target
            old = self.ram[target.name][target.offset]
            if isinstance(instruction, ClassicalNeg):
                if not isinstance(old, (int, float, np.int_, np.float_)):
                    raise ValueError(f"NEG requires a data type of REAL or INTEGER; not {type(old)}")
                self.ram[target.name][target.offset] *= -1
            elif isinstance(instruction, ClassicalNot):
                if not isinstance(old, (bool, np.bool_)):
                    raise ValueError(f"NOT requires a data type of BIT; not {type(old)}")
                self.ram[target.name][target.offset] = not old
            else:
                raise TypeError("Invalid UnaryClassicalInstruction")

            self.program_counter += 1

        elif isinstance(instruction, (LogicalBinaryOp, ArithmeticBinaryOp, ClassicalMove)):
            left_ind = instruction.left
            left_val = self.ram[left_ind.name][left_ind.offset]
            if isinstance(instruction.right, MemoryReference):
                right_ind = instruction.right
                right_val = self.ram[right_ind.name][right_ind.offset]
            else:
                right_val = instruction.right

            if isinstance(instruction, ClassicalAnd):
                if not isinstance(left_val, int) or not isinstance(right_val, int):
                    raise ValueError("AND requires a data type of INTEGER; not {type(left_val)} and {type(right_val)}")
                new_val: Union[int, float] = left_val & right_val
            elif isinstance(instruction, ClassicalInclusiveOr):
                if not isinstance(left_val, int) or not isinstance(right_val, int):
                    raise ValueError("OR requires a data type of INTEGER; not {type(left_val)} and {type(right_val)}")
                new_val = left_val | right_val
            elif isinstance(instruction, ClassicalExclusiveOr):
                if not isinstance(left_val, int) or not isinstance(right_val, int):
                    raise ValueError("XOR requires a data type of INTEGER; not {type(left_val)} and {type(right_val)}")
                new_val = left_val ^ right_val
            elif isinstance(instruction, ClassicalAdd):
                new_val = left_val + right_val
            elif isinstance(instruction, ClassicalSub):
                new_val = left_val - right_val
            elif isinstance(instruction, ClassicalMul):
                new_val = left_val * right_val
            elif isinstance(instruction, ClassicalDiv):
                new_val = left_val / right_val
            elif isinstance(instruction, ClassicalMove):
                new_val = right_val
            else:
                raise ValueError(f"Unknown BinaryOp {type(instruction)}")
            self.ram[left_ind.name][left_ind.offset] = new_val
            self.program_counter += 1

        elif isinstance(instruction, ClassicalExchange):
            left_ind_ex = instruction.left
            right_ind_ex = instruction.right

            tmp = self.ram[left_ind_ex.name][left_ind_ex.offset]
            self.ram[left_ind_ex.name][left_ind_ex.offset] = self.ram[right_ind_ex.name][right_ind_ex.offset]
            self.ram[right_ind_ex.name][right_ind_ex.offset] = tmp
            self.program_counter += 1

        elif isinstance(instruction, Reset):
            self.wf_simulator.reset()
            self.program_counter += 1

        elif isinstance(instruction, ResetQubit):
            raise NotImplementedError("Need to implement in wf simulator")

        elif isinstance(instruction, Wait):
            self.program_counter += 1

        elif isinstance(instruction, Nop):
            # well that was easy
            self.program_counter += 1

        elif isinstance(instruction, DefGate):
            if instruction.parameters is not None and len(instruction.parameters) > 0:
                raise NotImplementedError("PyQVM does not support parameterized DEFGATEs")
            self.defined_gates[instruction.name] = instruction.matrix
            self.program_counter += 1

        elif isinstance(instruction, Halt):
            return True
        else:
            raise ValueError(f"Unsupported instruction type: {instruction}")

        # return HALTED (i.e. program_counter is end of program)
        if self.program is None:
            raise ValueError("No program loaded into PyQVM.")
        return self.program_counter == len(self.program)

    def _execute_program(self) -> "PyQVM":
        self.program_counter = 0

        if self.program is None:
            raise ValueError("No program loaded into PyQVM.")
        halted = len(self.program) == 0
        while not halted:
            halted = self.transition()

        return self

    def execute_once(self, program: Program) -> "PyQVM":
        """Execute one outer loop of a program on the PyQVM without re-initializing its state.

        Note that the PyQVM is stateful. Subsequent calls to :py:func:`execute_once` will not
        automatically reset the wavefunction or the classical RAM. If this is desired,
        consider starting your program with ``RESET``.

        :return: ``self`` to support method chaining.
        """
        self.program = program
        self._extract_defined_gates()
        return self._execute_program()

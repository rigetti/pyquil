"""A pure Python implementation of the Quantum Virtual Machine (QVM).

.. deprecated:: PENDING_DEPRECATION_RELEASE
    This private module holds the implementation of the deprecated :py:mod:`pyquil.pyqvm`, so that
    pyQuil can use it internally without triggering that module's import warning.
"""

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
from typing import Any

import numpy as np
from deprecated.sphinx import deprecated
from numpy.random.mtrand import RandomState
from qcs_sdk import ExecutionData, RegisterData
from qcs_sdk.qvm import QVMResultData

from pyquil._deprecation import PENDING_DEPRECATION_RELEASE, QUAX_REPLACEMENT_NOTE, PyQuilDeprecationWarning
from pyquil.api._abstract_compiler import QuantumExecutable
from pyquil.api._qam import QAM, MemoryMap, QAMExecutionResult
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import LabelPlaceholder, MemoryReference
from quil.instructions import ArithmeticOperand, BinaryOperand, Halt, Nop, Wait
from pyquil.quilbase import (
    Arithmetic,
    ArithmeticOperator,
    BinaryOperator,
    Exchange,
    Move,
    Declaration,
    Gate,
    GateDefinition,
    GateSpecification,
    Jump,
    Label,
    JumpUnless,
    JumpWhen,
    BinaryLogic,
    Measurement,
    Pragma,
    Reset,
    UnaryLogic,
    UnaryOperator,
    gate_definition_matrix,
)

log = logging.getLogger(__name__)

QUIL_TO_NUMPY_DTYPE = {"INT": np.int32, "REAL": np.float64, "BIT": np.int8, "OCTET": np.uint8}


def _gate_definition_matrix(dg: GateDefinition) -> np.ndarray:
    """Extract the unitary matrix from a ``GATE ... AS MATRIX`` :py:class:`GateDefinition`."""
    rows = gate_definition_matrix(dg)
    return np.array([[cell.evaluate({}, {}) for cell in row] for row in rows])


def _unwrap_operand(operand: "ArithmeticOperand | BinaryOperand") -> int | float | MemoryReference:
    """Unwrap an ``ArithmeticOperand``/``BinaryOperand`` enum variant to its underlying value."""
    match operand:
        case ArithmeticOperand.LiteralInteger(value) | BinaryOperand.LiteralInteger(value):
            return value
        case ArithmeticOperand.LiteralReal(value):
            return value
        case ArithmeticOperand.MemoryReference(value) | BinaryOperand.MemoryReference(value):
            return value
    raise TypeError(f"Unknown operand type: {type(operand)}")


# Not decorated: ``deprecated`` wraps ``__new__``, so every deprecated subclass (the NumPy and
# reference simulators) would also emit this class's warning. The module-level warning covers it.
class AbstractQuantumSimulator(ABC):
    """An abstract interface for a quantum simulator."""

    @abstractmethod
    def __init__(self, n_qubits: int, rs: RandomState | None):
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
    def expectation(self, operator: PauliTerm | PauliSum) -> complex:
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


@deprecated(
    version=PENDING_DEPRECATION_RELEASE,
    line_length=0,
    reason=f"To be removed in pyQuil v5, where TrajectorySimulator samples program outcomes instead. {QUAX_REPLACEMENT_NOTE}",
    category=PyQuilDeprecationWarning,
)
class PyQVM(QAM["PyQVM"]):
    """A pure python implementation of the Quantum Virtual Machine."""

    def __init__(
        self,
        n_qubits: int,
        quantum_simulator_type: type[AbstractQuantumSimulator] | None = None,
        seed: int | None = None,
        post_gate_noise_probabilities: dict[str, float] | None = None,
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
        self.ram: dict[str, list[float | int]] = {}

        if post_gate_noise_probabilities is None:
            post_gate_noise_probabilities = {}
        self.post_gate_noise_probabilities = post_gate_noise_probabilities

        self.program: Program | None = None
        self.program_counter: int = 0
        self.defined_gates: dict[str, np.ndarray] = dict()

        # private implementation details
        self._qubit_to_ram: dict[int, int] | None = None
        self._ro_size: int | None = None
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
            if not isinstance(dg.specification, GateSpecification.Matrix):
                raise NotImplementedError("PyQVM does not support DEFGATE ... AS PERMUTATION | PAULI-SUM.")
            self.defined_gates[dg.name] = _gate_definition_matrix(dg)

    def execute_with_memory_map_batch(
        self, executable: QuantumExecutable, memory_maps: Iterable[MemoryMap], **__: Any
    ) -> list["PyQVM"]:
        """Operation is not supported by PyQVM as the state of the instance is reset at the start of each execution."""
        raise NotImplementedError(
            "PyQVM does not support batch execution as the state of the instance is reset at the start of each execute."
        )

    def execute(self, executable: QuantumExecutable, memory_map: MemoryMap | None = None, **__: Any) -> "PyQVM":
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

    def find_label(self, label: "Target") -> int:
        """Iterate over the program and find a Label instruction matching the input label.

        :param label: Label object to search for in program
        :return: Program index where ``label`` is found
        """
        if self.program is None:
            raise RuntimeError("No program loaded into PyQVM.")
        for index, action in enumerate(self.program):
            if isinstance(action, Label):
                if label == action.target:
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
            meas_reg: MemoryReference | None = instruction.target
            if meas_reg is None:
                raise ValueError("Measurement instruction must have a classical register.")
            self.ram[meas_reg.name][meas_reg.offset] = measured_val
            self.program_counter += 1

        elif isinstance(instruction, Declaration):
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

        elif isinstance(instruction, Label):
            # Label; pass straight over
            self.program_counter += 1

        elif isinstance(instruction, (JumpWhen, JumpUnless)):
            # JumpWhen/Unless; check classical reg
            jump_reg: MemoryReference | None = instruction.condition
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
                # jumping: set prog counter to Label
                self.program_counter = dest_index
            else:
                # not jumping: hop over this instruction
                self.program_counter += 1

        elif isinstance(instruction, UnaryLogic):
            # UnaryLogic; set classical reg
            target = instruction.target
            old = self.ram[target.name][target.offset]
            if instruction.operator == UnaryOperator.NEG:
                if not isinstance(old, (int, float, np.int32, np.float64)):
                    raise ValueError(f"NEG requires a data type of REAL or INTEGER; not {type(old)}")
                self.ram[target.name][target.offset] *= -1
            elif instruction.operator == UnaryOperator.NOT:
                if not isinstance(old, (bool, np.bool_)):
                    raise ValueError(f"NOT requires a data type of BIT; not {type(old)}")
                self.ram[target.name][target.offset] = not old
            else:
                raise TypeError("Invalid UnaryLogic")

            self.program_counter += 1

        elif isinstance(instruction, (BinaryLogic, Arithmetic, Move)):
            left_ind = instruction.left
            left_val = self.ram[left_ind.name][left_ind.offset]
            right_operand = _unwrap_operand(instruction.right)
            if isinstance(right_operand, MemoryReference):
                right_ind = right_operand
                right_val = self.ram[right_ind.name][right_ind.offset]
            else:
                right_val = right_operand

            if isinstance(instruction, BinaryLogic) and instruction.operator == BinaryOperator.AND:
                if not isinstance(left_val, int) or not isinstance(right_val, int):
                    raise ValueError("AND requires a data type of INTEGER; not {type(left_val)} and {type(right_val)}")
                new_val: int | float = left_val & right_val
            elif isinstance(instruction, BinaryLogic) and instruction.operator == BinaryOperator.IOR:
                if not isinstance(left_val, int) or not isinstance(right_val, int):
                    raise ValueError("OR requires a data type of INTEGER; not {type(left_val)} and {type(right_val)}")
                new_val = left_val | right_val
            elif isinstance(instruction, BinaryLogic) and instruction.operator == BinaryOperator.XOR:
                if not isinstance(left_val, int) or not isinstance(right_val, int):
                    raise ValueError("XOR requires a data type of INTEGER; not {type(left_val)} and {type(right_val)}")
                new_val = left_val ^ right_val
            elif isinstance(instruction, Arithmetic) and instruction.operator == ArithmeticOperator.ADD:
                new_val = left_val + right_val
            elif isinstance(instruction, Arithmetic) and instruction.operator == ArithmeticOperator.SUBTRACT:
                new_val = left_val - right_val
            elif isinstance(instruction, Arithmetic) and instruction.operator == ArithmeticOperator.MULTIPLY:
                new_val = left_val * right_val
            elif isinstance(instruction, Arithmetic) and instruction.operator == ArithmeticOperator.DIVIDE:
                new_val = left_val / right_val
            elif isinstance(instruction, Move):
                new_val = right_val
            else:
                raise ValueError(f"Unknown BinaryOp {type(instruction)}")
            self.ram[left_ind.name][left_ind.offset] = new_val
            self.program_counter += 1

        elif isinstance(instruction, Exchange):
            left_ind_ex = instruction.left
            right_ind_ex = instruction.right

            tmp = self.ram[left_ind_ex.name][left_ind_ex.offset]
            self.ram[left_ind_ex.name][left_ind_ex.offset] = self.ram[right_ind_ex.name][right_ind_ex.offset]
            self.ram[right_ind_ex.name][right_ind_ex.offset] = tmp
            self.program_counter += 1

        elif isinstance(instruction, Reset) and instruction.qubit is not None:
            raise NotImplementedError("Need to implement in wf simulator")

        elif isinstance(instruction, Reset):
            self.wf_simulator.reset()
            self.program_counter += 1

        elif instruction is Wait:
            self.program_counter += 1

        elif instruction is Nop:
            # well that was easy
            self.program_counter += 1

        elif isinstance(instruction, GateDefinition):
            if instruction.parameters is not None and len(instruction.parameters) > 0:
                raise NotImplementedError("PyQVM does not support parameterized DEFGATEs")
            self.defined_gates[instruction.name] = _gate_definition_matrix(instruction)
            self.program_counter += 1

        elif instruction is Halt:
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

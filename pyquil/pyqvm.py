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
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Type, Dict, Tuple, Union, List, Sequence

import numpy as np
from numpy.random.mtrand import RandomState

from pyquil.api import QAM
from pyquil.api._compiler import _extract_program_from_pyquil_executable_response
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.quil import Program
from pyquil.quilbase import Gate, Measurement, ResetQubit, DefGate, JumpTarget, JumpConditional, \
    JumpWhen, JumpUnless, Halt, Wait, Reset, Nop, UnaryClassicalInstruction, ClassicalNeg, \
    ClassicalNot, LogicalBinaryOp, ClassicalAnd, ClassicalInclusiveOr, ClassicalExclusiveOr, \
    ArithmeticBinaryOp, ClassicalAdd, ClassicalSub, ClassicalMul, ClassicalDiv, ClassicalMove, \
    ClassicalExchange, ClassicalConvert, ClassicalLoad, ClassicalStore, ClassicalComparison, \
    ClassicalEqual, ClassicalLessThan, ClassicalLessEqual, ClassicalGreaterThan, \
    ClassicalGreaterEqual, Jump, Pragma, Declare, RawInstr
from pyquil.quilatom import Label, MemoryReference

import logging

from rpcq.messages import PyQuilExecutableResponse

log = logging.getLogger(__name__)

QUIL_TO_NUMPY_DTYPE = {
    'INT': np.int_,
    'REAL': np.float_,
    'BIT': np.int8,
    'OCTET': np.uint8,
}


class AbstractQuantumSimulator(ABC):
    @abstractmethod
    def __init__(self, n_qubits: int, rs: RandomState):
        """
        Initialize.

        :param n_qubits: Number of qubits to simulate.
        :param rs: a RandomState (shared with the owning :py:class:`PyQVM`) for
            doing anything stochastic.
        """

    @abstractmethod
    def do_gate(self, gate: Gate) -> 'AbstractQuantumSimulator':
        """
        Perform a gate.

        :return: ``self`` to support method chaining.
        """

    @abstractmethod
    def do_gate_matrix(self, matrix: np.ndarray,
                       qubits: Sequence[int]) -> 'AbstractQuantumSimulator':
        """
        Apply an arbitrary unitary; not necessarily a named gate.

        :param matrix: The unitary matrix to apply. No checks are done
        :param qubits: A list of qubits to apply the unitary to.
        :return: ``self`` to support method chaining.
        """

    def do_program(self, program: Program) -> 'AbstractQuantumSimulator':
        """
        Perform a sequence of gates contained within a program.

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
        """
        Measure a qubit and collapse the wavefunction

        :return: The measurement result. A 1 or a 0.
        """

    @abstractmethod
    def expectation(self, operator: Union[PauliTerm, PauliSum]) -> complex:
        """
        Compute the expectation of an operator.

        :param operator: The operator
        :return: The operator's expectation value
        """

    @abstractmethod
    def reset(self) -> 'AbstractQuantumSimulator':
        """
        Reset the wavefunction to the |000...00> state.

        :return: ``self`` to support method chaining.
        """

    @abstractmethod
    def sample_bitstrings(self, n_samples) -> np.ndarray:
        """
        Sample bitstrings from the current state.

        :param n_samples: The number of bitstrings to sample
        :return: A numpy array of shape (n_samples, n_qubits)
        """

    @abstractmethod
    def do_post_gate_noise(self, noise_type: str, noise_prob: float,
                           qubits: List[int]) -> 'AbstractQuantumSimulator':
        """
        Apply noise that happens after each gate application.

        WARNING! This is experimental and the signature of this interface will likely change.

        :param noise_type: The name of the noise type
        :param noise_prob: The probability of that noise happening
        :param qubits: Apply noise to these qubits.
        :return: ``self`` to support method chaining
        """


class NotRunAndMeasureProgramError(ValueError):
    pass


def _make_ram_program(program):
    """
    Check that this program is a series of quantum gates with terminal MEASURE instructions; pop
    MEASURE instructions.

    :param program: The program
    :return: A new program with MEASURE instructions removed.
    """
    new_prog = program.copy_everything_except_instructions()
    last_qubit_operation = {}
    times_qubit_measured = defaultdict(lambda: 0)
    ro_size = None
    qubit_to_ram = {}

    for instr in program:
        if isinstance(instr, Pragma):
            new_prog += instr
        elif isinstance(instr, Declare):
            if instr.name == 'ro':
                if instr.memory_type != 'BIT':
                    raise NotRunAndMeasureProgramError("The readout register `ro` "
                                                       "must be of type BIT")
                ro_size = instr.memory_size
            new_prog += instr
        elif isinstance(instr, Gate):
            for qubit in instr.qubits:
                last_qubit_operation[qubit.index] = 'gate'
            new_prog += instr
        elif isinstance(instr, Measurement):
            if instr.classical_reg is None:
                raise NotRunAndMeasureProgramError("No measure-for-effect allowed")
            if instr.classical_reg.name != 'ro':
                raise NotRunAndMeasureProgramError("The readout register must be named `ro`, "
                                                   "not {}".format(instr.classical_reg.name))
            last_qubit_operation[instr.qubit.index] = 'measure'
            times_qubit_measured[instr.qubit.index] += 1
            qubit_to_ram[instr.qubit.index] = instr.classical_reg.offset
        else:
            raise NotRunAndMeasureProgramError(f"Unsupported r_a_m instruction {instr}")

    for q, lqo in last_qubit_operation.items():
        if lqo != 'measure':
            raise NotRunAndMeasureProgramError(f"Qubit {q}'s last operation is a gate")

    for q, tqm in times_qubit_measured.items():
        if tqm > 1:
            raise NotRunAndMeasureProgramError(f"Qubit {q} is measured {tqm} times")

    if ro_size is None:
        raise NotRunAndMeasureProgramError("Please declare a readout register")

    return new_prog, qubit_to_ram, ro_size


class PyQVM(QAM):
    def __init__(self, n_qubits, quantum_simulator_type: Type[AbstractQuantumSimulator] = None,
                 seed=None,
                 post_gate_noise_probabilities: Dict[str, float] = None,
                 ):
        """
        PyQuil's built-in Quil virtual machine.

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
                from pyquil.numpy_simulator import NumpyWavefunctionSimulator
                quantum_simulator_type = NumpyWavefunctionSimulator
            else:
                from pyquil.reference_simulator import ReferenceDensitySimulator
                log.info("Using ReferenceDensitySimulator as the backend for PyQVM")
                quantum_simulator_type = ReferenceDensitySimulator

        self.n_qubits = n_qubits
        self.ram = {}

        if post_gate_noise_probabilities is None:
            post_gate_noise_probabilities = {}
        self.post_gate_noise_probabilities = post_gate_noise_probabilities

        self.program = None  # type: Program
        self.program_counter = None  # type: int
        self.defined_gates = dict()  # type: Dict[str, np.ndarray]

        # private implementation details
        self._qubit_to_ram = None  # type: Dict[int, int]
        self._ro_size = None  # type :int
        self._bitstrings = None  # type: np.ndarray

        self.rs = np.random.RandomState(seed=seed)
        self.wf_simulator = quantum_simulator_type(n_qubits=n_qubits, rs=self.rs)
        self._last_measure_program_loc = None

    def load(self, executable):
        if isinstance(executable, PyQuilExecutableResponse):
            program = _extract_program_from_pyquil_executable_response(executable)
        else:
            program = executable

        try:
            program, self._qubit_to_ram, self._ro_size = _make_ram_program(program)
        except NotRunAndMeasureProgramError as e:
            raise ValueError("PyQVM can only run run-and-measure style programs: {}"
                             .format(e))

        # initialize program counter
        self.program = program
        self.program_counter = 0
        self._bitstrings = None

        # clear RAM, although it's not strictly clear if this should happen here
        self.ram = {}

        self.status = 'loaded'
        return self

    def write_memory(self, *, region_name: str, offset: int = 0, value=None):
        assert self.status in ['loaded', 'done']
        assert region_name != 'ro'
        self.ram[region_name][offset] = value
        return self

    def run(self):
        self.status = 'running'
        assert self._qubit_to_ram is not None
        assert self._ro_size is not None

        # TODO: why are DEFGATEs not just included in the list of instructions?
        for dg in self.program.defined_gates:
            if dg.parameters is not None and len(dg.parameters) > 0:
                raise NotImplementedError("PyQVM does not support parameterized DEFGATEs")
            self.defined_gates[dg.name] = dg.matrix

        halted = len(self.program) == 0
        while not halted:
            halted = self.transition()

        bitstrings = self.wf_simulator.sample_bitstrings(self.program.num_shots)

        n_shots = self.program.num_shots
        self.ram['ro'] = np.zeros((n_shots, self._ro_size), dtype=int)
        for q in range(bitstrings.shape[1]):
            if q in self._qubit_to_ram:
                ram_offset = self._qubit_to_ram[q]
                self.ram['ro'][:, ram_offset] = bitstrings[:, q]

        # Finally, we RESET the system because it isn't mandated yet that programs
        # contain RESET instructions.
        self.wf_simulator.reset()
        return self

    def wait(self):
        assert self.status == 'running'
        self.status = 'done'
        return self

    def read_memory(self, *, region_name: str):
        return self.ram[region_name]

    def find_label(self, label: Label):
        """
        Helper function that iterates over the program and looks for a JumpTarget that has a
        Label matching the input label.

        :param label: Label object to search for in program
        :return: Program index where ``label`` is found
        """
        for index, action in enumerate(self.program):
            if isinstance(action, JumpTarget):
                if label == action.label:
                    return index

        raise RuntimeError("Improper program - Jump Target not found in the "
                           "input program!")

    def transition(self):
        """
        Implements a QAM-like transition.

        This function assumes ``program`` and ``program_counter`` instance variables are set
        appropriately, and that the wavefunction simulator and classical memory ``ram`` instance
        variables are in the desired QAM input state.

        :return: whether the QAM should halt after this transition.
        """
        instruction = self.program[self.program_counter]

        if isinstance(instruction, Gate):
            if instruction.name in self.defined_gates:
                self.wf_simulator.do_gate_matrix(matrix=self.defined_gates[instruction.name],
                                                 qubits=[q.index for q in instruction.qubits])
            else:
                self.wf_simulator.do_gate(gate=instruction)

            for noise_type, noise_prob in self.post_gate_noise_probabilities.items():
                self.wf_simulator.do_post_gate_noise(noise_type, noise_prob,
                                                     qubits=[q.index for q in instruction.qubits])

            self.program_counter += 1

        elif isinstance(instruction, Measurement):
            measured_val = self.wf_simulator.do_measurement(qubit=instruction.qubit.index)
            x = instruction.classical_reg  # type: MemoryReference
            self.ram[x.name][x.offset] = measured_val
            self.program_counter += 1

        elif isinstance(instruction, Declare):
            if instruction.shared_region is not None:
                raise NotImplementedError("SHARING is not (yet) implemented.")

            self.ram[instruction.name] = np.zeros(instruction.memory_size,
                                                  dtype=QUIL_TO_NUMPY_DTYPE[
                                                      instruction.memory_type])
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

        elif isinstance(instruction, JumpConditional):
            # JumpConditional; check classical reg
            x = instruction.condition  # type: MemoryReference
            cond = self.ram[x.name][x.offset]
            if not isinstance(cond, (bool, np.bool, np.int8)):
                raise ValueError("{} requires a data type of BIT; not {}"
                                 .format(instruction.op, type(cond)))
            dest_index = self.find_label(instruction.target)
            if isinstance(instruction, JumpWhen):
                jump_if_cond = True
            elif isinstance(instruction, JumpUnless):
                jump_if_cond = False
            else:
                raise TypeError("Invalid JumpConditional")

            if not (cond ^ jump_if_cond):
                # jumping: set prog counter to JumpTarget
                self.program_counter = dest_index
            else:
                # not jumping: hop over this JumpConditional
                self.program_counter += 1

        elif isinstance(instruction, UnaryClassicalInstruction):
            # UnaryClassicalInstruction; set classical reg
            target = instruction.target  # type:MemoryReference
            old = self.ram[target.name][target.offset]
            if isinstance(instruction, ClassicalNeg):
                if not isinstance(old, (int, float, np.int, np.float)):
                    raise ValueError("NEG requires a data type of REAL or INTEGER; not {}"
                                     .format(type(old)))
                self.ram[target.name][target.offset] *= -1
            elif isinstance(instruction, ClassicalNot):
                if not isinstance(old, (bool, np.bool)):
                    raise ValueError("NOT requires a data type of BIT; not {}"
                                     .format(type(old)))
                self.ram[target.name][target.offset] = not old
            else:
                raise TypeError("Invalid UnaryClassicalInstruction")

            self.program_counter += 1

        elif isinstance(instruction, (LogicalBinaryOp, ArithmeticBinaryOp, ClassicalMove)):
            left_ind = instruction.left  # type: MemoryReference
            left_val = self.ram[left_ind.name][left_ind.offset]
            if isinstance(instruction.right, MemoryReference):
                right_ind = instruction.right  # type: MemoryReference
                right_val = self.ram[right_ind.name][right_ind.offset]
            else:
                right_val = instruction.right

            if isinstance(instruction, ClassicalAnd):
                new_val = left_val & right_val
            elif isinstance(instruction, ClassicalInclusiveOr):
                new_val = left_val | right_val
            elif isinstance(instruction, ClassicalExclusiveOr):
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
                raise ValueError("Unknown BinaryOp {}".format(type(instruction)))
            self.ram[left_ind.name][left_ind.offset] = new_val
            self.program_counter += 1

        elif isinstance(instruction, ClassicalExchange):
            left_ind = instruction.left  # type: MemoryReference
            right_ind = instruction.right  # type: MemoryReference

            tmp = self.ram[left_ind.name][left_ind.offset]
            self.ram[left_ind.name][left_ind.offset] = self.ram[right_ind.name][right_ind.offset]
            self.ram[right_ind.name][right_ind.offset] = tmp
            self.program_counter += 1

        elif isinstance(instruction, Reset):
            self.wf_simulator.reset()
            self.program_counter += 1

        elif isinstance(instruction, ResetQubit):
            # TODO
            raise NotImplementedError("Need to implement in wf simulator")
            self.program_counter += 1

        elif isinstance(instruction, Wait):
            warnings.warn("WAIT does nothing for a noiseless simulator")
            self.program_counter += 1

        elif isinstance(instruction, Nop):
            # well that was easy
            self.program_counter += 1

        elif isinstance(instruction, DefGate):
            if instruction.parameters is not None and len(instruction.parameters) > 0:
                raise NotImplementedError("PyQVM does not support parameterized DEFGATEs")
            self.defined_gates[instruction.name] = instruction.name
            self.program_counter += 1

        elif isinstance(instruction, RawInstr):
            raise NotImplementedError("PyQVM does not support raw instructions. "
                                      "Parse your program")

        elif isinstance(instruction, Halt):
            return True
        else:
            raise ValueError("Unsupported instruction type: {}".format(instruction))

        # return HALTED (i.e. program_counter is end of program)
        return self.program_counter == len(self.program)

    def execute(self, program: Program):
        """
        Execute a program on the QVM.

        Note that the QAM is stateful. Subsequent calls to :py:func:`execute` will not
        automatically reset the wavefunction or the classical RAM. If this is desired,
        consider starting your program with ``RESET``.

        :return: ``self`` to support method chaining.
        """
        # TODO: why are DEFGATEs not just included in the list of instructions?
        for dg in program.defined_gates:
            if dg.parameters is not None:
                raise NotImplementedError("PyQVM does not support parameterized DEFGATEs")
            self.defined_gates[dg.name] = dg.matrix

        # initialize program counter
        self.program = program
        self.program_counter = 0

        halted = len(program) == 0
        while not halted:
            halted = self.transition()

        return self

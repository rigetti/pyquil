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
from typing import Type, Dict, Tuple

import numpy as np
from numpy.random.mtrand import RandomState

from pyquil.api import QAM
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
    def do_measurement(self, qubit: int) -> int:
        """
        Measure a qubit and collapse the wavefunction

        :return: The measurement result. A 1 or a 0.
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
    def do_post_gate_noise(self, noise_type: str, noise_prob: float) -> 'AbstractQuantumSimulator':
        """
        Apply noise that happens after each gate application

        :param noise_type: The name of the noise type
        :param noise_prob: The probability of that noise happening
        :return: ``self`` to support method chaining
        """


class NotRunAndMeasureProgramError(ValueError):
    pass


def _verify_ram_program(program):
    last_qubit_operation = {}
    times_qubit_measured = defaultdict(lambda: 0)
    last_measure_program_loc = 0

    for i, instr in enumerate(program):
        if isinstance(instr, Pragma):
            pass
        elif isinstance(instr, Declare):
            pass
        elif isinstance(instr, Gate):
            for qubit in instr.qubits:
                last_qubit_operation[qubit.index] = 'gate'
        elif isinstance(instr, Measurement):
            last_qubit_operation[instr.qubit.index] = 'measure'
            times_qubit_measured[instr.qubit.index] += 1
            last_measure_program_loc = i
        else:
            raise NotRunAndMeasureProgramError(f"Unsupported r_a_m instruction {instr}")

    for q, lqo in last_qubit_operation.items():
        if lqo != 'measure':
            raise NotRunAndMeasureProgramError(f"Qubit {q}'s last operation is a gate")

    for q, tqm in times_qubit_measured.items():
        if tqm > 1:
            raise NotRunAndMeasureProgramError(f"Qubit {q} is measured {tqm} times")

    return last_measure_program_loc


class PyQVM(QAM):
    def __init__(self, n_qubits, quantum_simulator_type: Type[AbstractQuantumSimulator] = None,
                 seed=None,
                 post_gate_noise_probabilities: Dict[str, float] = None,
                 pre_measure_noise_probabilities: Dict[str, float] = None,
                 ):
        """
        A quil virtual machine that implements common control flow and plumbing.

        This class farms out the "actual" work to quantum simulators like
        ReferenceWavefunctionSimulator, ReferenceDensitySimulator, and NumpyWavefunctionSimulator

        :param quantum_simulator_type: A class that can be instantiated to handle the quantum
            aspects of this QVM. If not specified, the default will be either
            NumpyWavefunctionSimulator (no noise) or ReferenceDensitySimulator (noise)
        :param post_gate_noise_probabilities: A specification of noise model given by
            probabilities of certain types of noise. The dictionary keys are from "relaxation",
            "dephasing", "depolarizing", "phase_flip", "bit_flip", and "bitphase_flip".
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

        if pre_measure_noise_probabilities is None:
            pre_measure_noise_probabilities = {}
        self.pre_measure_noise_probabilities = pre_measure_noise_probabilities

        self.program = None  # type: Program
        self.program_counter = None  # type: int
        self._ram_measure_mapping = None  # type: Dict[int, Tuple[str, int]]

        self.rs = np.random.RandomState(seed=seed)
        self.wf_simulator = quantum_simulator_type(n_qubits=n_qubits, rs=self.rs)
        self._last_measure_program_loc = None

    def load(self, program):
        if len(program.defined_gates) > 0:
            raise ValueError("PyQVM does not support defined gates")

        try:
            self._last_measure_program_loc = _verify_ram_program(program)
            self._ram_measure_mapping = {}
        except NotRunAndMeasureProgramError as e:
            raise ValueError("PyQVM can only run run-and-measure style programs: {}"
                             .format(e))

        # initialize program counter
        self.program = program
        self.program_counter = 0

        # clear RAM, although it's not strictly clear if this should happen here
        self.ram = {}

        self.status = 'loaded'
        return self

    def write_memory(self, *, region_name: str, offset: int = 0, value=None):
        assert self.status in ['loaded', 'done']
        self.ram[region_name][offset] = value
        return self

    def run(self):
        self.status = 'running'
        assert self._last_measure_program_loc is not None

        halted = len(self.program) == 0
        while not halted:
            halted = self.transition(run_and_measure=True)
        return self

    def wait(self):
        assert self.status == 'running'
        self.status = 'done'
        return self

    def read_from_memory_region(self, *, region_name: str, offsets=None):
        if offsets is not None:
            raise NotImplementedError("Can't handle offsets")

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

    def transition(self, run_and_measure=False):
        """
        Implements a QAM-like transition.

        This function assumes ``program`` and ``program_counter`` instance variables are set
        appropriately, and that the wavefunction simulator and classical memory ``ram`` instance
        variables are in the desired QAM input state.

        :param run_and_measure: A sneaky feature whereby you can certify that the loaded
            program follows the conventions for a "run and measure"-style program. The
            wavefunction will be prepared once, and bitstrings will be sampled from it.
            This requires self._last_measure_program_loc to be the program_counter index of the
            final measure instruction
        :return: whether the QAM should halt after this transition.
        """
        instruction = self.program[self.program_counter]

        if isinstance(instruction, Gate):
            self.wf_simulator.do_gate(gate=instruction)

            for noise_type, noise_prob in self.post_gate_noise_probabilities.items():
                self.wf_simulator.do_post_gate_noise(noise_type, noise_prob)

            self.program_counter += 1

        elif isinstance(instruction, Measurement):
            if not run_and_measure:
                measured_val = self.wf_simulator.do_measurement(qubit=instruction.qubit.index)
                x = instruction.classical_reg  # type: MemoryReference
                self.ram[x.name][x.offset] = measured_val
            else:
                # Hacky code to speed up run-and-measure programs
                # Don't actually do the measurement, just make a note of where in ram the bits
                # will go
                x = instruction.classical_reg  # type: MemoryReference
                self._ram_measure_mapping[instruction.qubit.index] = (x.name, x.offset)

                if self.program_counter == self._last_measure_program_loc:
                    # We've reached the last measure instruction. Time to sample from our
                    # wavefunction
                    bitstrings = self.wf_simulator.sample_bitstrings(self.program.num_shots)

                    # Quil2 doesn't support defining multidimensional arrays, and its typical
                    # to allocate a readout register of size n_bits and assume "the stack" will
                    # do the right thing and give you an array of shape (n_shots, n_bits) instead.
                    # Here we resize all of our readout registers to be this extended 2d array shape
                    ro_registers = sorted(set(ram_name for ram_name, ram_offset
                                              in self._ram_measure_mapping.values()))
                    for ro_register in ro_registers:
                        assert np.sum(self.ram[ro_register]) == 0, 'reading out into a parameter?'
                        prev_shape = self.ram[ro_register].shape
                        assert len(prev_shape) == 1, prev_shape
                        prev_shape = prev_shape[0]
                        prev_dtype = self.ram[ro_register].dtype
                        assert prev_dtype == QUIL_TO_NUMPY_DTYPE['BIT'], prev_dtype
                        self.ram[ro_register] = np.zeros((self.program.num_shots, prev_shape),
                                                         dtype=prev_dtype)

                    # Penultimately, we use our collected qubit-to-ram mappings to fill in our newly
                    # reshaped ram arrays
                    for q in range(bitstrings.shape[1]):
                        if q in self._ram_measure_mapping:
                            ram_name, ram_offset = self._ram_measure_mapping[q]
                            self.ram[ram_name][:, ram_offset] = bitstrings[:, q]

                    # Finally, we RESET the system because it isn't mandated yet that programs
                    # contain RESET instructions.
                    self.wf_simulator.reset()
            self.program_counter += 1

        elif isinstance(instruction, Declare):
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
            raise NotImplementedError("PyQVM does not support DEFGATE")

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
        if len(program.defined_gates) > 0:
            raise ValueError("PyQVM does not support defined gates")

        # initialize program counter
        self.program = program
        self.program_counter = 0

        halted = len(program) == 0
        while not halted:
            halted = self.transition()

        return self

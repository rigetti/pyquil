##############################################################################
# Copyright 2016-2018 Rigetti Computing
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
"""
Module for creating and defining Quil programs.
"""
import itertools
import types
import warnings
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import numpy as np
from rpcq.messages import ParameterAref

from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
    Label,
    LabelPlaceholder,
    MemoryReference,
    MemoryReferenceDesignator,
    Parameter,
    ParameterDesignator,
    Frame,
    Qubit,
    QubitDesignator,
    QubitPlaceholder,
    FormalArgument,
    format_parameter,
    unpack_classical_reg,
    unpack_qubit,
)
from pyquil.quilbase import (
    DefGate,
    Gate,
    Measurement,
    Pragma,
    Halt,
    AbstractInstruction,
    Jump,
    JumpConditional,
    JumpTarget,
    JumpUnless,
    JumpWhen,
    Declare,
    Reset,
    ResetQubit,
    DelayFrames,
    DelayQubits,
    Fence,
    FenceAll,
    Pulse,
    Capture,
    RawCapture,
    SetFrequency,
    ShiftFrequency,
    SetPhase,
    ShiftPhase,
    SwapPhase,
    SetScale,
    DefPermutationGate,
    DefCalibration,
    DefFrame,
    DefMeasureCalibration,
    DefWaveform,
)
from pyquil.quiltcalibrations import (
    CalibrationError,
    CalibrationMatch,
    expand_calibration,
    match_calibration,
)

from qcs_sdk.quil import Program as QuilProgram, Instruction as QuilInstruction

InstructionDesignator = Union[
    AbstractInstruction,
    "Program",
    List[Any],
    Tuple[Any, ...],
    str,  # required to be a pyquil program
    Generator[Any, Any, Any],
]


class Program:
    """
    A list of pyQuil instructions that comprise a quantum program.

    >>> from pyquil import Program
    >>> from pyquil.gates import H, CNOT
    >>> p = Program()
    >>> p += H(0)
    >>> p += CNOT(0, 1)
    """

    _memory: Memory
    """Contents of memory to be used as program parameters during execution"""

    def __init__(self, *instructions: InstructionDesignator):
        self._program = QuilProgram("")
        self.inst(*instructions)

        # default number of shots to loop through
        self.num_shots = 1

        # Will be moved to rust... later
        self._memory = Memory()

    @property
    def calibrations(self) -> List[Union[DefCalibration, DefMeasureCalibration]]:
        """A list of Quil-T calibration definitions."""
        return self._program.calibrations.calibrations + self._program.calibrations.measure_calibrations

    @property
    def measure_calibrations(self) -> List[DefMeasureCalibration]:
        """A list of measure calibrations"""
        return self._program.measure_calibrations

    @property
    def waveforms(self) -> Dict[str, DefWaveform]:
        """A mapping from waveform names to their corresponding definitions."""
        return self._program.waveforms

    @property
    def frames(self) -> Dict[Frame, DefFrame]:
        """A mapping from Quil-T frames to their definitions."""
        return self._program.frames.get_all_frames()

    @property
    def declarations(self) -> Dict[str, Declare]:
        """A mapping from declared region names to their declarations."""
        return self._program.declarations

    def copy_everything_except_instructions(self) -> "Program":
        """
        Copy all the members that live on a Program object.

        :return: a new Program
        """
        new_prog = Program(self._program.to_headers())
        new_prog.num_shots = self.num_shots
        new_prog._memory = self._memory.copy()
        return new_prog

    def copy(self) -> "Program":
        """
        Performs a deep copy of this program.

        :return: a new Program
        """
        return Program(self)

    @property
    def defined_gates(self) -> List[DefGate]:
        """
        A list of defined gates on the program.
        """
        return self._program.defined_gates

    @property
    def instructions(self) -> List[QuilInstruction]:
        """
        Fill in any placeholders and return a list of quil AbstractInstructions.
        """
        return self._program.to_instructions(False)

    def inst(self, *instructions: InstructionDesignator) -> "Program":
        """
        Mutates the Program object by appending new instructions.

        This function accepts a number of different valid forms, e.g.

            >>> p = Program()
            >>> p.inst(H(0)) # A single instruction
            >>> p.inst(H(0), H(1)) # Multiple instructions
            >>> p.inst([H(0), H(1)]) # A list of instructions
            >>> p.inst(H(i) for i in range(4)) # A generator of instructions
            >>> p.inst(("H", 1)) # A tuple representing an instruction
            >>> p.inst("H 0") # A string representing an instruction
            >>> q = Program()
            >>> p.inst(q) # Another program

        It can also be chained:
            >>> p = Program()
            >>> p.inst(H(0)).inst(H(1))

        :param instructions: A list of Instruction objects, e.g. Gates
        :return: self for method chaining
        """
        for instruction in instructions:
            if isinstance(instruction, list):
                self.inst(*instruction)
            elif isinstance(instruction, types.GeneratorType):
                self.inst(*instruction)
            elif isinstance(instruction, tuple):
                if len(instruction) == 0:
                    raise ValueError("tuple should have at least one element")
                else:
                    self.inst(" ".join(map(str, instruction)))
            elif isinstance(instruction, str):
                self.inst(QuilProgram(instruction.strip()))
            elif isinstance(instruction, AbstractInstruction):
                self.inst(QuilProgram(instruction.out()))
            elif isinstance(instruction, QuilInstruction):
                self._program.add_instruction(instruction)
            elif isinstance(instruction, QuilProgram):  # TODO: Add programs together in rs
                self._program += instruction
            elif isinstance(instruction, Program):
                self.inst(instruction._program)
            else:
                raise TypeError("Invalid instruction: {}".format(repr(instruction)))

        return self

    def gate(
        self,
        name: str,
        params: Iterable[ParameterDesignator],
        qubits: Iterable[Union[Qubit, QubitPlaceholder]],
    ) -> "Program":
        """
        Add a gate to the program.

        .. note::

            The matrix elements along each axis are ordered by bitstring. For two qubits the order
            is ``00, 01, 10, 11``, where the the bits **are ordered in reverse** by the qubit index,
            i.e., for qubits 0 and 1 the bitstring ``01`` indicates that qubit 0 is in the state 1.
            See also :ref:`the related docs in the WavefunctionSimulator Overview <basis_ordering>`.

        :param name: The name of the gate.
        :param params: Parameters to send to the gate.
        :param qubits: Qubits that the gate operates on.
        :return: The Program instance
        """
        return self.inst(Gate(name, params, [unpack_qubit(q) for q in qubits]))

    def defgate(
        self,
        name: str,
        matrix: Union[List[List[Any]], np.ndarray, np.matrix],
        parameters: Optional[List[Parameter]] = None,
    ) -> "Program":
        """
        Define a new static gate.

        .. note::

            The matrix elements along each axis are ordered by bitstring. For two qubits the order
            is ``00, 01, 10, 11``, where the the bits **are ordered in reverse** by the qubit index,
            i.e., for qubits 0 and 1 the bitstring ``01`` indicates that qubit 0 is in the state 1.
            See also :ref:`the related docs in the WavefunctionSimulator Overview <basis_ordering>`.


        :param name: The name of the gate.
        :param matrix: List of lists or Numpy 2d array.
        :param parameters: list of parameters that are used in this gate
        :return: The Program instance.
        """
        print("defgate", DefGate(name, matrix, parameters).out())
        return self.inst(DefGate(name, matrix, parameters))

    def define_noisy_gate(self, name: str, qubit_indices: Sequence[int], kraus_ops: Sequence[Any]) -> "Program":
        """
        Overload a static ideal gate with a noisy one defined in terms of a Kraus map.

        .. note::

            The matrix elements along each axis are ordered by bitstring. For two qubits the order
            is ``00, 01, 10, 11``, where the the bits **are ordered in reverse** by the qubit index,
            i.e., for qubits 0 and 1 the bitstring ``01`` indicates that qubit 0 is in the state 1.
            See also :ref:`the related docs in the WavefunctionSimulator Overview <basis_ordering>`.


        :param name: The name of the gate.
        :param qubit_indices: The qubits it acts on.
        :param kraus_ops: The Kraus operators.
        :return: The Program instance
        """
        kraus_ops = [np.asarray(k, dtype=np.complex128) for k in kraus_ops]
        _check_kraus_ops(len(qubit_indices), kraus_ops)
        return self.inst(_create_kraus_pragmas(name, tuple(qubit_indices), kraus_ops))

    def define_noisy_readout(self, qubit: Union[int, QubitPlaceholder], p00: float, p11: float) -> "Program":
        """
        For this program define a classical bit flip readout error channel parametrized by
        ``p00`` and ``p11``. This models the effect of thermal noise that corrupts the readout
        signal **after** it has interrogated the qubit.

        :param qubit: The qubit with noisy readout.
        :param p00: The probability of obtaining the measurement result 0 given that the qubit
          is in state 0.
        :param p11: The probability of obtaining the measurement result 1 given that the qubit
          is in state 1.
        :return: The Program with an appended READOUT-POVM Pragma.
        """
        if not 0.0 <= p00 <= 1.0:
            raise ValueError("p00 must be in the interval [0,1].")
        if not 0.0 <= p11 <= 1.0:
            raise ValueError("p11 must be in the interval [0,1].")
        if not (isinstance(qubit, int) or isinstance(qubit, QubitPlaceholder)):
            raise TypeError("qubit must be a non-negative integer, or QubitPlaceholder.")
        if isinstance(qubit, int) and qubit < 0:
            raise ValueError("qubit cannot be negative.")
        p00 = float(p00)
        p11 = float(p11)
        aprobs = [p00, 1.0 - p11, 1.0 - p00, p11]
        aprobs_str = "({})".format(" ".join(format_parameter(p) for p in aprobs))
        pragma = Pragma("READOUT-POVM", [qubit], aprobs_str)
        return self.inst(pragma)

    def no_noise(self) -> "Program":
        """
        Prevent a noisy gate definition from being applied to the immediately following Gate
        instruction.

        :return: Program
        """
        return self.inst(Pragma("NO-NOISE"))

    def measure(self, qubit: QubitDesignator, classical_reg: Optional[MemoryReferenceDesignator]) -> "Program":
        """
        Measures a qubit at qubit_index and puts the result in classical_reg

        :param qubit: The qubit to measure.
        :param classical_reg: The classical register to measure into, or None.
        :returns: The Quil Program with the appropriate measure instruction appended, e.g.
                  MEASURE 0 [1]
        """
        return self.inst(MEASURE(qubit, classical_reg))

    def reset(self, qubit_index: Optional[int] = None) -> "Program":
        """
        Reset all qubits or just a specific qubit at qubit_index.

        :param qubit_index: The address of the qubit to reset.
            If None, reset all qubits.
        :returns: The Quil Program with the appropriate reset instruction appended, e.g.
                  RESET 0
        """
        return self.inst(RESET(qubit_index))

    def measure_all(self, *qubit_reg_pairs: Tuple[QubitDesignator, Optional[MemoryReferenceDesignator]]) -> "Program":
        """
        Measures many qubits into their specified classical bits, in the order
        they were entered. If no qubit/register pairs are provided, measure all qubits present in
        the program into classical addresses of the same index.

        :param qubit_reg_pairs: Tuples of qubit indices paired with classical bits.
        :return: The Quil Program with the appropriate measure instructions appended, e.g.

        .. code::

                  MEASURE 0 [1]
                  MEASURE 1 [2]
                  MEASURE 2 [3]
        """
        if qubit_reg_pairs == ():
            qubits = self.get_qubits(indices=True)
            if len(qubits) == 0:
                return self
            if any(isinstance(q, QubitPlaceholder) for q in qubits):
                raise ValueError(
                    "Attempted to call measure_all on a Program that contains QubitPlaceholders. "
                    "You must either provide the qubit_reg_pairs argument to describe how to map "
                    "these QubitPlaceholders to memory registers, or else first call "
                    "pyquil.quil.address_qubits to instantiate the QubitPlaceholders."
                )
            if any(isinstance(q, FormalArgument) for q in qubits):
                raise ValueError("Cannot call measure_all on a Program that contains FormalArguments.")
            # Help mypy determine that qubits does not contain any QubitPlaceholders.
            qubit_inds = cast(List[int], qubits)
            ro = self.declare("ro", "BIT", max(qubit_inds) + 1)
            for qi in qubit_inds:
                self.inst(MEASURE(qi, ro[qi]))
        else:
            for qubit_index, classical_reg in qubit_reg_pairs:
                self.inst(MEASURE(qubit_index, classical_reg))
        return self

    def _set_parameter_values_at_runtime(self) -> "Program":
        """
        Store all parameter values directly within the Program using ``MOVE`` instructions. Mutates the receiver.
        """
        move_instructions = [
            MOVE(MemoryReference(name=k.name, offset=k.index), v) for k, v in self._memory.values.items()
        ]

        self.prepend_instructions(move_instructions)

        return self

    def write_memory(
        self,
        *,
        region_name: str,
        value: Union[int, float, Sequence[int], Sequence[float]],
        offset: Optional[int] = None,
    ) -> "Program":
        self._memory._write_value(parameter=ParameterAref(name=region_name, index=offset or 0), value=value)
        return self

    def prepend_instructions(self, instructions: Iterable[AbstractInstruction]) -> "Program":
        """
        Prepend instructions to the beginning of the program.
        """
        new_prog = Program(*instructions)
        return new_prog + self

    def while_do(self, classical_reg: MemoryReferenceDesignator, q_program: "Program") -> "Program":
        """
        While a classical register at index classical_reg is 1, loop q_program

        Equivalent to the following construction:

        .. code::

            WHILE [c]:
               instr...
            =>
              LABEL @START
              JUMP-UNLESS @END [c]
              instr...
              JUMP @START
              LABEL @END

        :param MemoryReferenceDesignator classical_reg: The classical register to check
        :param Program q_program: The Quil program to loop.
        :return: The Quil Program with the loop instructions added.
        """
        label_start = LabelPlaceholder("START")
        label_end = LabelPlaceholder("END")
        self.inst(JumpTarget(label_start))
        self.inst(JumpUnless(target=label_end, condition=unpack_classical_reg(classical_reg)))
        self.inst(q_program)
        self.inst(Jump(label_start))
        self.inst(JumpTarget(label_end))
        return self

    def if_then(
        self,
        classical_reg: MemoryReferenceDesignator,
        if_program: "Program",
        else_program: Optional["Program"] = None,
    ) -> "Program":
        """
        If the classical register at index classical reg is 1, run if_program, else run
        else_program.

        Equivalent to the following construction:

        .. code::

            IF [c]:
               instrA...
            ELSE:
               instrB...
            =>
              JUMP-WHEN @THEN [c]
              instrB...
              JUMP @END
              LABEL @THEN
              instrA...
              LABEL @END

        :param classical_reg: The classical register to check as the condition
        :param if_program: A Quil program to execute if classical_reg is 1
        :param else_program: A Quil program to execute if classical_reg is 0. This
            argument is optional and defaults to an empty Program.
        :returns: The Quil Program with the branching instructions added.
        """
        else_program = else_program if else_program is not None else Program()

        label_then = LabelPlaceholder("THEN")
        label_end = LabelPlaceholder("END")
        jump_when = JumpWhen(target=label_then, condition=unpack_classical_reg(classical_reg))
        print("jump_when", jump_when.out())
        self.inst(jump_when)
        self.inst(else_program)
        self.inst(Jump(label_end))
        self.inst(JumpTarget(label_then))
        self.inst(if_program)
        self.inst(JumpTarget(label_end))
        return self

    def declare(
        self,
        name: str,
        memory_type: str = "BIT",
        memory_size: int = 1,
        shared_region: Optional[str] = None,
        offsets: Optional[Iterable[Tuple[int, str]]] = None,
    ) -> MemoryReference:
        """DECLARE a quil variable

        This adds the declaration to the current program and returns a MemoryReference to the
        base (offset = 0) of the declared memory.

        .. note::
            This function returns a MemoryReference and cannot be chained like some
            of the other Program methods. Consider using ``inst(DECLARE(...))`` if you
            would like to chain methods, but please be aware that you must create your
            own MemoryReferences later on.

        :param name: Name of the declared variable
        :param memory_type: Type of the declared memory: 'BIT', 'REAL', 'OCTET' or 'INTEGER'
        :param memory_size: Number of array elements in the declared memory.
        :param shared_region: You can declare a variable that shares its underlying memory
            with another region. This allows aliasing. For example, you can interpret an array
            of measured bits as an integer.
        :param offsets: If you are using ``shared_region``, this allows you to share only
            a part of the parent region. The offset is given by an array type and the number
            of elements of that type. For example,
            ``DECLARE target-bit BIT SHARING real-region OFFSET 1 REAL 4 BIT`` will let you use
            target-bit to poke into the fourth bit of the second real from the leading edge of
            real-region.
        :return: a MemoryReference to the start of the declared memory region, ie a memory
            reference to ``name[0]``.
        """
        self.inst(
            Declare(
                name=name,
                memory_type=memory_type,
                memory_size=memory_size,
                shared_region=shared_region,
                offsets=offsets,
            )
        )
        return MemoryReference(name=name, declared_size=memory_size)

    def wrap_in_numshots_loop(self, shots: int) -> "Program":
        """
        Wraps a Quil program in a loop that re-runs the same program many times.

        Note: this function is a prototype of what will exist in the future when users will
        be responsible for writing this loop instead of having it happen automatically.

        :param shots: Number of iterations to loop through.
        """
        self.num_shots = shots
        return self

    def out(self, *, calibrations: Optional[bool] = True) -> str:
        """
        Serializes the Quil program to a string suitable for submitting to the QVM or QPU.
        """

        # TODO: program str without calibrations
        return str(self._program)

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        """
        Returns all of the qubit indices used in this program, including gate applications and
        allocated qubits. e.g.

            >>> p = Program()
            >>> p.inst(("H", 1))
            >>> p.get_qubits()
            {1}
            >>> q = QubitPlaceholder()
            >>> p.inst(H(q))
            >>> len(p.get_qubits())
            2

        :param indices: Return qubit indices as integers intead of the
            wrapping :py:class:`Qubit` object
        :return: A set of all the qubit indices used in this program
        """
        # TODO: what to do about indices flag?
        qubits = self._program.get_used_qubits()
        if indices:
            qubits = {q.as_fixed() for q in qubits}
        return qubits

    def match_calibrations(self, instr: AbstractInstruction) -> Optional[CalibrationMatch]:
        """
        Attempt to match a calibration to the provided instruction.

        Note: preference is given to later calibrations, i.e. in a program with

          DEFCAL X 0:
              <a>

          DEFCAL X 0:
             <b>

        the second calibration, with body <b>, would be the calibration matching `X 0`.

        :param instr: An instruction.
        :returns: a CalibrationMatch object, if one can be found.
        """
        if isinstance(instr, (Gate, Measurement)):
            for cal in reversed(self.calibrations):
                match = match_calibration(instr, cal)
                if match is not None:
                    return match

        return None

    def get_calibration(self, instr: AbstractInstruction) -> Optional[Union[DefCalibration, DefMeasureCalibration]]:
        """
        Get the calibration corresponding to the provided instruction.

        :param instr: An instruction.
        :returns: A matching Quil-T calibration definition, if one exists.
        """
        match = self.match_calibrations(instr)
        if match:
            return match.cal

        return None

    def calibrate(
        self,
        instruction: AbstractInstruction,
        previously_calibrated_instructions: Optional[Set[AbstractInstruction]] = None,
    ) -> List[AbstractInstruction]:
        """
        Expand an instruction into its calibrated definition.

        If a calibration definition matches the provided instruction, then the definition
        body is returned with appropriate substitutions made for parameters and qubit
        arguments. If no calibration definition matches, then the original instruction
        is returned. Calibrations are performed recursively, so that if a calibrated
        instruction produces an instruction that has a corresponding calibration, it
        will be expanded, and so on. If a cycle is encountered, a CalibrationError is
        raised.

        :param instruction: An instruction.
        :param previously_calibrated_instructions: A set of instructions that are the
            results of calibration expansions in the direct ancestry of `instruction`.
            Used to catch cyclic calibration expansions.
        :returns: A list of instructions, with the active calibrations expanded.
        """
        if previously_calibrated_instructions is None:
            previously_calibrated_instructions = set()
        elif instruction in previously_calibrated_instructions:
            raise CalibrationError(
                f"The instruction {instruction} appears in the set of "
                f"previously calibrated instructions {previously_calibrated_instructions}"
                " and would therefore result in a cyclic non-terminating expansion."
            )
        else:
            previously_calibrated_instructions = previously_calibrated_instructions.union({instruction})
        match = self.match_calibrations(instruction)
        if match is not None:
            return sum(
                [
                    self.calibrate(expansion, previously_calibrated_instructions)
                    for expansion in expand_calibration(match)
                ],
                [],
            )
        else:
            return [instruction]

    def is_protoquil(self, quilt: bool = False) -> bool:
        """
        Protoquil programs may only contain gates, Pragmas, and RESET. It may not contain
        classical instructions or jumps.

        :return: True if the Program is Protoquil, False otherwise

        .. deprecated:: 4.0
           The quilt flag will be removed. Use is_quilt instead.
        """
        try:
            if quilt:
                self._program.validate_quilt()
            else:
                self._program.validate_protoquil()
            return True
        except ValueError:
            return False

    def is_quilt(self) -> bool:
        """
        Returns true if the program is valid quil-t
        """
        try:
            self.validate_quilt()
            return True
        except ValueError:
            return False

    def validate_protoquil(self) -> None:
        """
        Raises a ValueError if the program isn't valid protoquil
        """
        self._program.validate_protoquil()

    def validate_quilt(self) -> None:
        """
        Raises a ValueError if the program isn't valid quil-t
        """
        self._program.validate_quilt()

    def is_supported_on_qpu(self) -> bool:
        """
        Whether the program can be compiled to the hardware to execute on a QPU. These Quil
        programs are more restricted than Protoquil: for instance, RESET must be before any
        gates or MEASUREs, and MEASURE on a qubit must be after any gates on that qubit.

        :return: True if the Program is supported Quil, False otherwise
        """
        try:
            validate_supported_quil(self)
            return True
        except ValueError:
            return False

    def _sort_declares_to_program_start(self) -> None:
        """
        Re-order DECLARE instructions within this program to the beginning, followed by
        all other instructions. Reordering is stable among DECLARE and non-DECLARE instructions.

        .. deprecated:: 4.0
           Sorting the program is managed by `qcs_sdk.quil`. This is a no-op.
        """
        pass

    # TODO: Why and where is this needed?
    # Ask Mark about it
    # If needed, should be in rs
    def dagger(self, inv_dict: Optional[Any] = None, suffix: str = "-INV") -> "Program":
        """
        Creates the conjugate transpose of the Quil program. The program must
        contain only gate applications.

        Note: the keyword arguments inv_dict and suffix are kept only
        for backwards compatibility and have no effect.

        :return: The Quil program's inverse
        """
        if any(not isinstance(instr, Gate) for instr in self.instructions):
            raise ValueError("Program to be daggered must contain only gate applications")

        # This is a bit hacky. Gate.dagger() mutates the gate object, rather than returning a fresh
        # (and daggered) copy. Also, mypy doesn't understand that we already asserted that every
        # instr in _instructions is a Gate, above, so help mypy out with a cast.
        surely_gate_instructions = cast(List[Gate], Program(self.out()).instructions)
        return Program([instr.dagger() for instr in reversed(surely_gate_instructions)])

    def pop(self) -> "Program":
        """
        Removes the last instruction from the program and return it

        .. deprecated:: 4.0
           This method will be removed in future versions in pyQuil
        """
        last = self.instructions[-1]
        instructions = self._program.to_headers() + self.instructions
        new_program = Program(instructions[:-1])
        self._program = new_program._program
        self._memory = new_program._memory
        return last

    # TODO: Probably deprecate
    # def _synthesize(self) -> "Program":
    #     """
    #     Assigns all placeholder labels to actual values.
    #
    #     Changed in 1.9: Either all qubits must be defined or all undefined. If qubits are
    #     undefined, this method will not help you. You must explicitly call `address_qubits`
    #     which will return a new Program.
    #
    #     Changed in 1.9: This function now returns ``self`` and updates
    #     ``self._synthesized_instructions``.
    #
    #     Changed in 2.0: This function will add an instruction to the top of the program
    #     to declare a register of bits called ``ro`` if and only if there are no other
    #     declarations in the program.
    #
    #     Changed in 3.0: Removed above change regarding implicit ``ro`` declaration.
    #
    #     :return: This object with the ``_synthesized_instructions`` member set.
    #     """
    #     self._synthesized_instructions = instantiate_labels(self._instructions)
    #     return self

    def __add__(self, other: InstructionDesignator) -> "Program":
        """
        Concatenate two programs together, returning a new one.

        :param other: Another program or instruction to concatenate to this one.
        :return: A newly concatenated program.
        """
        p = Program()
        p.inst(self)
        p.inst(other)
        return p

    def __iadd__(self, other: InstructionDesignator) -> "Program":
        """
        Concatenate two programs together using +=, appending the program on the right hand side to the one on the left
        hand side.

        :param other: Another program or instruction to concatenate to this one.
        :return: The updated program.
        """
        self.inst(other)
        return self

    def __getitem__(self, index: Union[slice, int]) -> Union[AbstractInstruction, "Program"]:
        """
        Allows indexing into the program to get an action.

        :param index: The action at the specified index.
        :return:
        """
        return Program(self.instructions[index]) if isinstance(index, slice) else self.instructions[index]

    def __iter__(self) -> Iterator[AbstractInstruction]:
        """
        Allow built in iteration through a program's instructions, e.g. [a for a in Program(X(0))]

        :return:
        """
        return self.instructions.__iter__()

    def __eq__(self, other: "Program") -> bool:
        return self._program == other._program

    def __ne__(self, other: "Program") -> bool:
        return not self.__eq__(other)

    def __len__(self) -> int:
        return len(self.instructions)

    def __str__(self) -> str:
        """
        A string representation of the Quil program for inspection.

        This may not be suitable for submission to a QPU or QVM for example if
        your program contains unaddressed QubitPlaceholders
        """
        return str(self._program)


def _what_type_of_qubit_does_it_use(
    program: Program,
) -> Tuple[bool, bool, List[Union[Qubit, QubitPlaceholder]]]:
    """Helper function to peruse through a program's qubits.

    This function will also enforce the condition that a Program uses either all placeholders
    or all instantiated qubits to avoid accidentally mixing the two. This function will warn
    if your program doesn't use any qubits.

    :return: tuple of (whether the program uses placeholder qubits, whether the program uses
        real qubits, a list of qubits ordered by their first appearance in the program)
    """
    has_placeholders = False
    has_real_qubits = False

    # We probably want to index qubits in the order they are encountered in the program
    # so an ordered set would be nice. Python doesn't *have* an ordered set. Use the keys
    # of an ordered dictionary instead
    qubits = {}

    for instr in program:
        if isinstance(instr, Gate):
            for q in instr.qubits:
                qubits[q] = 1
                if isinstance(q, QubitPlaceholder):
                    has_placeholders = True
                elif isinstance(q, Qubit):
                    has_real_qubits = True
                else:
                    raise ValueError("Unknown qubit type {}".format(q))
        elif isinstance(instr, Measurement):
            qubits[instr.qubit] = 1
            if isinstance(instr.qubit, QubitPlaceholder):
                has_placeholders = True
            elif isinstance(instr.qubit, Qubit):
                has_real_qubits = True
            else:
                raise ValueError("Unknown qubit type {}".format(instr.qubit))
        elif isinstance(instr, Pragma):
            for arg in instr.args:
                if isinstance(arg, QubitPlaceholder):
                    qubits[arg] = 1
                    has_placeholders = True
                elif isinstance(arg, Qubit):
                    qubits[arg] = 1
                    has_real_qubits = True
    if not (has_placeholders or has_real_qubits):
        warnings.warn("Your program doesn't use any qubits")

    if has_placeholders and has_real_qubits:
        raise ValueError("Your program mixes instantiated qubits with placeholders")

    # The isinstance checks above make sure that if any qubit is a
    # FormalArgument (which is permitted by Gate.qubits), then an
    # error should be raised. Unfortunately this doesn't help mypy
    # narrow down the return type, so gotta cast.
    return (
        has_placeholders,
        has_real_qubits,
        cast(List[Union[Qubit, QubitPlaceholder]], list(qubits.keys())),
    )


def get_default_qubit_mapping(program: Program) -> Dict[Union[Qubit, QubitPlaceholder], Qubit]:
    """
    Takes a program which contains qubit placeholders and provides a mapping to the integers
    0 through N-1.

    The output of this function is suitable for input to :py:func:`address_qubits`.

    :param program: A program containing qubit placeholders
    :return: A dictionary mapping qubit placeholder to an addressed qubit from 0 through N-1.
    """
    fake_qubits, real_qubits, qubits = _what_type_of_qubit_does_it_use(program)
    if real_qubits:
        warnings.warn("This program contains integer qubits, so getting a mapping doesn't make sense.")
        # _what_type_of_qubit_does_it_use ensures that if real_qubits is True, then qubits contains
        # only real Qubits, not QubitPlaceholders. Help mypy figure this out with cast.
        return {q: cast(Qubit, q) for q in qubits}
    return {qp: Qubit(i) for i, qp in enumerate(qubits)}


@no_type_check
def address_qubits(
    program: Program, qubit_mapping: Optional[Dict[QubitPlaceholder, Union[Qubit, int]]] = None
) -> Program:
    """
    Takes a program which contains placeholders and assigns them all defined values.

    Either all qubits must be defined or all undefined. If qubits are
    undefined, you may provide a qubit mapping to specify how placeholders get mapped
    to actual qubits. If a mapping is not provided, integers 0 through N are used.

    This function will also instantiate any label placeholders.

    :param program: The program.
    :param qubit_mapping: A dictionary-like object that maps from :py:class:`QubitPlaceholder`
        to :py:class:`Qubit` or ``int`` (but not both).
    :return: A new Program with all qubit and label placeholders assigned to real qubits and labels.
    """
    fake_qubits, real_qubits, qubits = _what_type_of_qubit_does_it_use(program)
    if real_qubits:
        if qubit_mapping is not None:
            warnings.warn("A qubit mapping was provided but the program does not " "contain any placeholders to map!")
        return program

    if qubit_mapping is None:
        qubit_mapping = {qp: Qubit(i) for i, qp in enumerate(qubits)}
    else:
        if all(isinstance(v, Qubit) for v in qubit_mapping.values()):
            pass  # we good
        elif all(isinstance(v, int) for v in qubit_mapping.values()):
            qubit_mapping = {k: Qubit(v) for k, v in qubit_mapping.items()}
        else:
            raise ValueError("Qubit mapping must map to type Qubit or int (but not both)")

    result: List[AbstractInstruction] = []
    for instr in program:
        # Remap qubits on Gate, Measurement, and ResetQubit instructions
        if isinstance(instr, Gate):
            remapped_qubits = [qubit_mapping[q] for q in instr.qubits]
            gate = Gate(instr.name, instr.params, remapped_qubits)
            gate.modifiers = instr.modifiers
            result.append(gate)
        elif isinstance(instr, Measurement):
            result.append(Measurement(qubit_mapping[instr.qubit], instr.classical_reg))
        elif isinstance(instr, ResetQubit):
            result.append(ResetQubit(qubit_mapping[instr.qubit]))
        elif isinstance(instr, Pragma):
            new_args: List[Union[Qubit, int, str]] = []
            for arg in instr.args:
                # Pragmas can have arguments that represent things besides qubits, so here we
                # make sure to just look up the QubitPlaceholders.
                if isinstance(arg, QubitPlaceholder):
                    new_args.append(qubit_mapping[arg])
                else:
                    new_args.append(arg)
            result.append(Pragma(instr.command, new_args, instr.freeform_string))
        # Otherwise simply add it to the result
        else:
            result.append(instr)

    new_program = program.copy()
    new_program._instructions = result

    return new_program


# TODO: consider deprecate
# def _get_label(
#     placeholder: LabelPlaceholder,
#     label_mapping: Dict[LabelPlaceholder, Label],
#     label_i: int,
# ) -> Tuple[Label, Dict[LabelPlaceholder, Label], int]:
#     """Helper function to either get the appropriate label for a given placeholder or generate
#     a new label and update the mapping.
#
#     See :py:func:`instantiate_labels` for usage.
#     """
#     if placeholder in label_mapping:
#         return label_mapping[placeholder], label_mapping, label_i
#
#     new_target = Label("{}{}".format(placeholder.prefix, label_i))
#     label_i += 1
#     label_mapping[placeholder] = new_target
#     return new_target, label_mapping, label_i
#
#
# def instantiate_labels(instructions: Iterable[AbstractInstruction]) -> List[AbstractInstruction]:
#     """
#     Takes an iterable of instructions which may contain label placeholders and assigns
#     them all defined values.
#
#     :return: list of instructions with all label placeholders assigned to real labels.
#     """
#     label_i = 1
#     result: List[AbstractInstruction] = []
#     label_mapping: Dict[LabelPlaceholder, Label] = dict()
#     for instr in instructions:
#         if isinstance(instr, Jump) and isinstance(instr.target, LabelPlaceholder):
#             new_target, label_mapping, label_i = _get_label(instr.target, label_mapping, label_i)
#             result.append(Jump(new_target))
#         elif isinstance(instr, JumpConditional) and isinstance(instr.target, LabelPlaceholder):
#             new_target, label_mapping, label_i = _get_label(instr.target, label_mapping, label_i)
#             cls = instr.__class__  # Make the correct subclass
#             result.append(cls(new_target, instr.condition))
#         elif isinstance(instr, JumpTarget) and isinstance(instr.label, LabelPlaceholder):
#             new_label, label_mapping, label_i = _get_label(instr.label, label_mapping, label_i)
#             result.append(JumpTarget(new_label))
#         else:
#             result.append(instr)
#
#     return result


def merge_with_pauli_noise(
    prog_list: Iterable[Program], probabilities: Sequence[float], qubits: Sequence[int]
) -> Program:
    """
    Insert pauli noise channels between each item in the list of programs.
    This noise channel is implemented as a single noisy identity gate acting on the provided qubits.
    This method does not rely on merge_programs and so avoids the inclusion of redundant Kraus
    Pragmas that would occur if merge_programs was called directly on programs with distinct noisy
    gate definitions.

    :param prog_list: an iterable such as a program or a list of programs.
        If a program is provided, a single noise gate will be applied after each gate in the
        program. If a list of programs is provided, the noise gate will be applied after each
        program.
    :param probabilities: The 4^num_qubits list of probabilities specifying the desired pauli
        channel. There should be either 4 or 16 probabilities specified in the order
        I, X, Y, Z or II, IX, IY, IZ, XI, XX, XY, etc respectively.
    :param qubits: a list of the qubits that the noisy gate should act on.
    :return: A single program with noisy gates inserted between each element of the program list.
    """
    p = Program()
    p.defgate("pauli_noise", np.eye(2 ** len(qubits)))
    p.define_noisy_gate("pauli_noise", qubits, pauli_kraus_map(probabilities))
    for elem in prog_list:
        p.inst(Program(elem))
        if isinstance(elem, Measurement):
            continue  # do not apply noise after measurement
        p.inst(("pauli_noise", *qubits))
    return p


# TODO: this needs to be deprecated or quil-rs needs to dedupe
def merge_programs(prog_list: Sequence[Program]) -> Program:
    """
    Merges a list of pyQuil programs into a single one by appending them in sequence.
    If multiple programs in the list contain the same gate and/or noisy gate definition
    with identical name, this definition will only be applied once. If different definitions
    with the same name appear multiple times in the program list, each will be applied once
    in the order of last occurrence.

    :param prog_list: A list of pyquil programs
    :return: a single pyQuil program
    """
    merged_program = Program()
    for prog in prog_list:
        merged_program += prog
    return merged_program


def get_classical_addresses_from_program(program: Program) -> Dict[str, List[int]]:
    """
    Returns a sorted list of classical addresses found in the MEASURE instructions in the program.

    :param program: The program from which to get the classical addresses.
    :return: A mapping from memory region names to lists of offsets appearing in the program.
    """
    addresses: Dict[str, List[int]] = defaultdict(list)
    flattened_addresses = {}

    # Required to use the `classical_reg.address` int attribute.
    # See https://github.com/rigetti/pyquil/issues/388.
    for instr in program:
        if isinstance(instr, Measurement) and instr.classical_reg:
            addresses[instr.classical_reg.name].append(instr.classical_reg.offset)

    # flatten duplicates
    for k, v in addresses.items():
        reduced_list = list(set(v))
        reduced_list.sort()
        flattened_addresses[k] = reduced_list

    return flattened_addresses


def percolate_declares(program: Program) -> Program:
    """
    Move all the DECLARE statements to the top of the program. Return a fresh object.

    :param program: Perhaps jumbled program.
    :return: Program with DECLAREs all at the top and otherwise the same sorted contents.
    """
    declare_program = Program()
    instrs_program = Program()

    for instr in program:
        if isinstance(instr, Declare):
            declare_program += instr
        else:
            instrs_program += instr

    p = declare_program + instrs_program
    p._defined_gates = program._defined_gates

    return p


def validate_protoquil(program: Program, quilt: bool = False) -> None:
    """
    Ensure that a program is valid ProtoQuil or Quil-T, otherwise raise a ValueError.
    Protoquil is a subset of Quil which excludes control flow and classical instructions.

    :param quilt: Validate the program as Quil-T.
    :param program: The Quil program to validate.
    """
    """
    Ensure that a program is valid ProtoQuil, otherwise raise a ValueError.
    Protoquil is a subset of Quil which excludes control flow and classical instructions.

    :param program: The Quil program to validate.

    .. deprecated:: 4.0
       This function will be removed, used the validate_protoquil and validate_quilt
       methods on the Program instead.
    """
    if quilt:
        program.validate_quilt()
    else:
        program.validate_protoquil()


def validate_supported_quil(program: Program) -> None:
    """
    Ensure that a program is supported Quil which can run on any QPU, otherwise raise a ValueError.
    We support a global RESET before any gates, and MEASUREs on each qubit after any gates
    on that qubit. PRAGMAs and DECLAREs are always allowed.

    :param program: The Quil program to validate.
    """
    gates_seen = False
    measured_qubits: Set[int] = set()
    for instr in program.instructions:
        if isinstance(instr, Pragma) or isinstance(instr, Declare):
            continue
        elif isinstance(instr, Gate):
            gates_seen = True
            if any(q.index in measured_qubits for q in instr.qubits):
                raise ValueError("Cannot apply gates to qubits that were already measured.")
        elif isinstance(instr, Reset):
            if gates_seen:
                raise ValueError("RESET can only be applied before any gate applications.")
        elif isinstance(instr, ResetQubit):
            raise ValueError("Only global RESETs are currently supported.")
        elif isinstance(instr, Measurement):
            if instr.qubit.index in measured_qubits:
                raise ValueError("Multiple measurements per qubit is not supported.")
            measured_qubits.add(instr.qubit.index)
        else:
            raise ValueError(f"Unhandled instruction type in supported Quil validation: {instr}")

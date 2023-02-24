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
import types
import warnings
from collections import defaultdict
from copy import deepcopy
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
    AbstractInstruction,
    Jump,
    JumpConditional,
    JumpTarget,
    JumpUnless,
    JumpWhen,
    Declare,
    ResetQubit,
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

from qcs_sdk.quil.program import Program as RSProgram
from qcs_sdk.quil.instructions import Instruction as RSInstruction, Gate as RSGate

InstructionDesignator = Union[
    AbstractInstruction,
    RSInstruction,
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
        self._program = RSProgram()
        self.inst(*instructions)

        # default number of shots to loop through
        self.num_shots = 1

        # Will be moved to rust... later
        self._memory = Memory()

    @property
    def calibrations(self) -> List[DefCalibration]:
        """A list of Quil-T calibration definitions."""
        return self._program.calibrations.calibrations

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
        Perform a deep copy of this program.

        :return: a new Program
        """
        return deepcopy(self)

    @property
    def defined_gates(self) -> List[DefGate]:
        """
        A list of defined gates on the program.
        """
        return self._program.defined_gates

    @property
    def instructions(self) -> List[AbstractInstruction]:
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
                self.inst(RSProgram.from_string(instruction.strip()))
            elif isinstance(instruction, AbstractInstruction):
                self.inst(RSProgram.from_string(instruction.out()))
            elif isinstance(instruction, Program):
                self.inst(instruction._program)
            elif isinstance(instruction, RSInstruction):
                self._program.add_instruction(instruction)
            elif isinstance(instruction, RSProgram):
                self._program += instruction
            else:
                try:
                    self.inst(str(instruction))
                except Exception:
                    raise TypeError("Invalid instruction: {}".format(instruction))

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

    # TODO: Implement control flow methods in quil-rs
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
        if calibrations:
            return str(self._program)
        else:
            return str(self._program.into_simplified())

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

        .. deprecated:: 4.0
           The indices flag will be removed. Use get_qubit_indices() instead
        """
        if indices:
            return self.get_qubit_indices()
        return self._program.get_used_qubits()

    def get_qubit_indices(self) -> Set[int]:
        """
        Returns the index of each qubit used in the program. Will raise an exception if any of the
        qubits are placeholders.
        """
        return {q.as_fixed() for q in self._program.get_used_qubits()}

    # TODO: Port calibrations logic from quil-rs
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

    # TODO: Port calibrations logic from quil-rs
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

    # TODO: Port calibrations logic from quil-rs
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
           This function always returns True and will be removed in future versions of pyQuil
        """
        return True

    def is_supported_on_qpu(self) -> bool:
        """
        Whether the program can be compiled to the hardware to execute on a QPU. These Quil
        programs are more restricted than Protoquil: for instance, RESET must be before any
        gates or MEASUREs, and MEASURE on a qubit must be after any gates on that qubit.

        :return: True if the Program is supported Quil, False otherwise

        .. deprecated:: 4.0
           This will always return True and will be removed in future versions of pyQuil.
           Attempting to run a program against on a QPU is the best way to validate if it is supported.
        """
        return True

    def dagger(self) -> "Program":
        """
        Creates the conjugate transpose of the Quil program. The program must
        contain only gate applications.

        :return: The Quil program's inverse
        """
        return Program(self._program.dagger())

    def pop(self) -> AbstractInstruction:
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

    def __iadd__(self, other) -> "Program":
        """
        Concatenate two programs together by appending them to the right-hand side to the left.

        :param other: Another program or instruction to concatenate to this one.
        :return: A newly concatenated program.
        """
        p = Program(other)
        return self._program.add_instructions(p._program.instructions)

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


# TODO: These should come from quil-rs. Requires Instruction::Measurement be ported
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

    .. deprecated:: 4.0
       Qubit placeholders are now managed internally by the Program. This is a no-op and will be
       removed in future versions of pyQuil
    """
    return program


def _get_label(
    placeholder: LabelPlaceholder,
    label_mapping: Dict[LabelPlaceholder, Label],
    label_i: int,
) -> Tuple[Label, Dict[LabelPlaceholder, Label], int]:
    """Helper function to either get the appropriate label for a given placeholder or generate
    a new label and update the mapping.
    See :py:func:`instantiate_labels` for usage.
    """
    if placeholder in label_mapping:
        return label_mapping[placeholder], label_mapping, label_i

    new_target = Label("{}{}".format(placeholder.prefix, label_i))
    label_i += 1
    label_mapping[placeholder] = new_target
    return new_target, label_mapping, label_i


def instantiate_labels(instructions: Iterable[AbstractInstruction]) -> List[AbstractInstruction]:
    """
    Takes an iterable of instructions which may contain label placeholders and assigns
    them all defined values.
    :return: list of instructions with all label placeholders assigned to real labels.
    """
    label_i = 1
    result: List[AbstractInstruction] = []
    label_mapping: Dict[LabelPlaceholder, Label] = dict()
    for instr in instructions:
        if isinstance(instr, Jump) and isinstance(instr.target, LabelPlaceholder):
            new_target, label_mapping, label_i = _get_label(instr.target, label_mapping, label_i)
            result.append(Jump(new_target))
        elif isinstance(instr, JumpConditional) and isinstance(instr.target, LabelPlaceholder):
            new_target, label_mapping, label_i = _get_label(instr.target, label_mapping, label_i)
            cls = instr.__class__  # Make the correct subclass
            result.append(cls(new_target, instr.condition))
        elif isinstance(instr, JumpTarget) and isinstance(instr.label, LabelPlaceholder):
            new_label, label_mapping, label_i = _get_label(instr.label, label_mapping, label_i)
            result.append(JumpTarget(new_label))
        else:
            result.append(instr)

    return result


def percolate_declares(program: Program) -> Program:
    """
    Move all the DECLARE statements to the top of the program. Return a fresh object.

    :param program: Perhaps jumbled program.
    :return: Program with DECLAREs all at the top and otherwise the same sorted contents.
    .. deprecated:: 4.0
       The Program class sorts instructions automatically. This is a no-op and will be removed
       in future versions of pyQuil.
    """
    return program


def merge_programs(prog_list: Sequence[Program]) -> Program:
    """
    Merges a list of pyQuil programs into a single one by appending them in sequence.

    :param prog_list: A list of pyquil programs
    :return: a single pyQuil program
    .. deprecated:: 4.0
       This function will be removed in future versions. Instead, use addition to combine
       programs together.
    """
    merged_program = Program()
    for prog in prog_list:
        merged_program += prog
    return merged_program


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
       This function is a no-op and will be removed in future versions of pyQuil
    """
    pass


def validate_supported_quil(program: Program) -> None:
    """
    Ensure that a program is supported Quil which can run on any QPU, otherwise raise a ValueError.
    We support a global RESET before any gates, and MEASUREs on each qubit after any gates
    on that qubit. PRAGMAs and DECLAREs are always allowed.

    :param program: The Quil program to validate.

    :deprecated: ..4.0
        This client side check is now a no-op and will be removed in future versions of pyQuil.
        Attempting to run a program against on a QPU is the best way to validate if it is supported.
    """
    pass

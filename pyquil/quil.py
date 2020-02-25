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
from rpcq.messages import NativeQuilMetadata

from pyquil._parser.PyQuilListener import run_parser
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
    Label,
    LabelPlaceholder,
    MemoryReference,
    MemoryReferenceDesignator,
    Parameter,
    ParameterDesignator,
    Qubit,
    QubitDesignator,
    QubitPlaceholder,
    format_parameter,
    unpack_classical_reg,
    unpack_qubit,
)
from pyquil.gates import MEASURE, RESET
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
    Reset,
    ResetQubit,
    DefPermutationGate,
)


InstructionDesignator = Union[
    AbstractInstruction,
    DefGate,
    "Program",
    List[Any],
    Tuple[Any, ...],
    str,  # required to be a pyquil program
    Generator[Any, Any, Any],
]


class Program(object):
    """A list of pyQuil instructions that comprise a quantum program.

    >>> from pyquil import Program
    >>> from pyquil.gates import H, CNOT
    >>> p = Program()
    >>> p += H(0)
    >>> p += CNOT(0, 1)
    """

    def __init__(self, *instructions: InstructionDesignator):
        self._defined_gates: List[DefGate] = []
        # Implementation note: the key difference between the private _instructions and
        # the public instructions property below is that the private _instructions list
        # may contain placeholder labels.
        self._instructions: List[AbstractInstruction] = []

        # Performance optimization: as stated above _instructions may contain placeholder
        # labels so the program must first be have its labels instantiated.
        # _synthesized_instructions is simply a cache on the result of the _synthesize()
        # method.  It is marked as None whenever new instructions are added.
        self._synthesized_instructions: Optional[List[AbstractInstruction]] = None

        self.inst(*instructions)

        # Filled in with quil_to_native_quil
        self.native_quil_metadata: Optional[NativeQuilMetadata] = None

        # default number of shots to loop through
        self.num_shots = 1

        # Note to developers: Have you changed this method? Have you changed the fields which
        # live on `Program`? Please update `Program.copy()`!

    def copy_everything_except_instructions(self) -> "Program":
        """
        Copy all the members that live on a Program object.

        :return: a new Program
        """
        new_prog = Program()
        new_prog._defined_gates = self._defined_gates.copy()
        if self.native_quil_metadata is not None:
            # TODO: remove this type: ignore once rpcq._base.Message gets type hints.
            new_prog.native_quil_metadata = self.native_quil_metadata.copy()  # type: ignore
        new_prog.num_shots = self.num_shots
        return new_prog

    def copy(self) -> "Program":
        """
        Perform a shallow copy of this program.

        QuilAtom and AbstractInstruction objects should be treated as immutable to avoid
        strange behavior when performing a copy.

        :return: a new Program
        """
        new_prog = self.copy_everything_except_instructions()
        new_prog._instructions = self._instructions.copy()
        return new_prog

    @property
    def defined_gates(self) -> List[DefGate]:
        """
        A list of defined gates on the program.
        """
        return self._defined_gates

    @property
    def instructions(self) -> List[AbstractInstruction]:
        """
        Fill in any placeholders and return a list of quil AbstractInstructions.
        """
        if self._synthesized_instructions is None:
            self._synthesize()
        assert self._synthesized_instructions is not None
        return self._synthesized_instructions

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
                elif len(instruction) == 1:
                    self.inst(instruction[0])
                else:
                    op = instruction[0]
                    if op == "MEASURE":
                        if len(instruction) == 2:
                            self.measure(instruction[1], None)
                        else:
                            self.measure(instruction[1], instruction[2])
                    else:
                        params: List[ParameterDesignator] = []
                        possible_params = instruction[1]
                        rest: Sequence[Any] = instruction[2:]
                        if isinstance(possible_params, list):
                            params = possible_params
                        else:
                            rest = [possible_params] + list(rest)
                        self.gate(op, params, rest)
            elif isinstance(instruction, str):
                self.inst(run_parser(instruction.strip()))
            elif isinstance(instruction, Program):
                if id(self) == id(instruction):
                    raise ValueError("Nesting a program inside itself is not supported")

                for defgate in instruction._defined_gates:
                    self.inst(defgate)
                for instr in instruction._instructions:
                    self.inst(instr)

            # Implementation note: these two base cases are the only ones which modify the program
            elif isinstance(instruction, DefGate):
                defined_gate_names = [gate.name for gate in self._defined_gates]
                if instruction.name in defined_gate_names:
                    warnings.warn(
                        "Gate {} has already been defined in this program".format(instruction.name)
                    )

                self._defined_gates.append(instruction)
            elif isinstance(instruction, AbstractInstruction):
                self._instructions.append(instruction)
                self._synthesized_instructions = None
            else:
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

    def define_noisy_gate(
        self, name: str, qubit_indices: Sequence[int], kraus_ops: Sequence[Any]
    ) -> "Program":
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

    def define_noisy_readout(
        self, qubit: Union[int, QubitPlaceholder], p00: float, p11: float
    ) -> "Program":
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

    def measure(
        self, qubit: QubitDesignator, classical_reg: Optional[MemoryReferenceDesignator]
    ) -> "Program":
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

    def measure_all(
        self, *qubit_reg_pairs: Tuple[QubitDesignator, Optional[MemoryReferenceDesignator]]
    ) -> "Program":
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
            # Help mypy determine that qubits does not contain any QubitPlaceholders.
            qubit_inds = cast(List[int], qubits)
            ro = self.declare("ro", "BIT", max(qubit_inds) + 1)
            for qi in qubit_inds:
                self.inst(MEASURE(qi, ro[qi]))
        else:
            for qubit_index, classical_reg in qubit_reg_pairs:
                self.inst(MEASURE(qubit_index, classical_reg))
        return self

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
        self.inst(JumpWhen(target=label_then, condition=unpack_classical_reg(classical_reg)))
        self.inst(else_program)
        self.inst(Jump(label_end))
        self.inst(JumpTarget(label_then))
        self.inst(if_program)
        self.inst(JumpTarget(label_end))
        return self

    def alloc(self) -> QubitPlaceholder:
        """
        Get a new qubit.

        :return: A qubit.
        """
        warnings.warn(
            "`alloc` is deprecated and will be removed in a future version of pyQuil. "
            "Please create a `QubitPlaceholder` directly",
            DeprecationWarning,
        )
        return QubitPlaceholder()

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

    def out(self) -> str:
        """
        Serializes the Quil program to a string suitable for submitting to the QVM or QPU.
        """
        return "\n".join(
            itertools.chain(
                (dg.out() for dg in self._defined_gates),
                (instr.out() for instr in self.instructions),
                [""],
            )
        )

    def get_qubits(self, indices: bool = True) -> Set[QubitDesignator]:
        """
        Returns all of the qubit indices used in this program, including gate applications and
        allocated qubits. e.g.

            >>> p = Program()
            >>> p.inst(("H", 1))
            >>> p.get_qubits()
            {1}
            >>> q = p.alloc()
            >>> p.inst(H(q))
            >>> len(p.get_qubits())
            2

        :param indices: Return qubit indices as integers intead of the
            wrapping :py:class:`Qubit` object
        :return: A set of all the qubit indices used in this program
        """
        qubits: Set[QubitDesignator] = set()
        for instr in self.instructions:
            if isinstance(instr, (Gate, Measurement)):
                qubits |= instr.get_qubits(indices=indices)
        return qubits

    def is_protoquil(self) -> bool:
        """
        Protoquil programs may only contain gates, Pragmas, and RESET. It may not contain
        classical instructions or jumps.

        :return: True if the Program is Protoquil, False otherwise
        """
        try:
            validate_protoquil(self)
            return True
        except ValueError:
            return False

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

    def pop(self) -> AbstractInstruction:
        """
        Pops off the last instruction.

        :return: The instruction that was popped.
        """
        res = self._instructions.pop()
        self._synthesized_instructions = None
        return res

    def dagger(self, inv_dict: Optional[Any] = None, suffix: str = "-INV") -> "Program":
        """
        Creates the conjugate transpose of the Quil program. The program must
        contain only gate applications.

        Note: the keyword arguments inv_dict and suffix are kept only
        for backwards compatibility and have no effect.

        :return: The Quil program's inverse
        """
        if any(not isinstance(instr, Gate) for instr in self._instructions):
            raise ValueError("Program to be daggered must contain only gate applications")

        # This is a bit hacky. Gate.dagger() mutates the gate object, rather than returning a fresh
        # (and daggered) copy. Also, mypy doesn't understand that we already asserted that every
        # instr in _instructions is a Gate, above, so help mypy out with a cast.
        surely_gate_instructions = cast(List[Gate], Program(self.out())._instructions)
        return Program([instr.dagger() for instr in reversed(surely_gate_instructions)])

    def _synthesize(self) -> "Program":
        """
        Assigns all placeholder labels to actual values and implicitly declares the ``ro``
        register for backwards compatibility.

        Changed in 1.9: Either all qubits must be defined or all undefined. If qubits are
        undefined, this method will not help you. You must explicitly call `address_qubits`
        which will return a new Program.

        Changed in 1.9: This function now returns ``self`` and updates
        ``self._synthesized_instructions``.

        Changed in 2.0: This function will add an instruction to the top of the program
        to declare a register of bits called ``ro`` if and only if there are no other
        declarations in the program.

        :return: This object with the ``_synthesized_instructions`` member set.
        """
        self._synthesized_instructions = instantiate_labels(self._instructions)
        self._synthesized_instructions = implicitly_declare_ro(self._synthesized_instructions)
        return self

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
        Concatenate two programs together using +=, returning a new one.

        :param other: Another program or instruction to concatenate to this one.
        :return: A newly concatenated program.
        """
        return self.inst(other)

    def __getitem__(self, index: Union[slice, int]) -> Union[AbstractInstruction, "Program"]:
        """
        Allows indexing into the program to get an action.

        :param index: The action at the specified index.
        :return:
        """
        return (
            Program(self.instructions[index])
            if isinstance(index, slice)
            else self.instructions[index]
        )

    def __iter__(self) -> Iterator[AbstractInstruction]:
        """
        Allow built in iteration through a program's instructions, e.g. [a for a in Program(X(0))]

        :return:
        """
        return self.instructions.__iter__()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.out() == other.out()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __len__(self) -> int:
        return len(self.instructions)

    def __str__(self) -> str:
        """
        A string representation of the Quil program for inspection.

        This may not be suitable for submission to a QPU or QVM for example if
        your program contains unaddressed QubitPlaceholders
        """
        return "\n".join(
            itertools.chain(
                (str(dg) for dg in self._defined_gates),
                (str(instr) for instr in self.instructions),
                [""],
            )
        )


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

    return has_placeholders, has_real_qubits, list(qubits.keys())


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
        warnings.warn(
            "This program contains integer qubits, so getting a mapping doesn't make sense."
        )
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
            warnings.warn(
                "A qubit mapping was provided but the program does not "
                "contain any placeholders to map!"
            )
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


def _get_label(
    placeholder: LabelPlaceholder, label_mapping: Dict[LabelPlaceholder, Label], label_i: int,
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


def implicitly_declare_ro(instructions: List[AbstractInstruction]) -> List[AbstractInstruction]:
    """
    Implicitly declare a register named ``ro`` for backwards compatibility with Quil 1.

    There used to be one un-named hunk of classical memory. Now there are variables with
    declarations. Instead of::

        MEASURE 0 [0]

    You must now measure into a named register, idiomatically::

        MEASURE 0 ro[0]

    The ``MEASURE`` instruction will emit this (with a deprecation warning) if you're still
    using bare integers for classical addresses. However, you must also declare memory in the
    new scheme::

        DECLARE ro BIT[8]
        MEASURE 0 ro[0]

    This method will determine if you are in "backwards compatibility mode" and will declare
    a read-out ``ro`` register for you. If you program contains any DECLARE commands or if it
    does not have any MEASURE x ro[x], this will not do anything.

    This behavior is included for backwards compatibility and will be removed in future releases
    of PyQuil. Please DECLARE all memory including ``ro``.
    """
    ro_addrs: List[int] = []
    for instr in instructions:
        if isinstance(instr, Declare):
            # The user has declared their own memory
            # so they are responsible for all declarations and memory references.
            return instructions

        if isinstance(instr, Measurement):
            if instr.classical_reg is None:
                continue

            if instr.classical_reg.name == "ro":
                ro_addrs += [instr.classical_reg.offset]
            else:
                # The user has used a classical register named something other than "ro"
                # so they are responsible for all declarations and memory references.
                return instructions

    if len(ro_addrs) == 0:
        return instructions

    warnings.warn(
        "Please DECLARE all memory. I'm adding a declaration for the `ro` register, "
        "but I won't do this for you in the future."
    )

    new_instr = instructions.copy()
    new_instr.insert(0, Declare(name="ro", memory_type="BIT", memory_size=max(ro_addrs) + 1))
    return new_instr


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
    definitions = [gate for prog in prog_list for gate in Program(prog).defined_gates]
    seen: Dict[str, List[DefGate]] = {}
    # Collect definitions in reverse order and reapply definitions in reverse
    # collected order to ensure that the last occurrence of a definition is applied last.
    for definition in reversed(definitions):
        name = definition.name
        if name in seen.keys():
            # Do not add truly identical definitions with the same name
            # If two different definitions share a name, we include each definition so as to provide
            # a waring to the user when the contradictory defgate is called.
            if definition not in seen[name]:
                seen[name].append(definition)
        else:
            seen[name] = [definition]
    new_definitions = [gate for key in seen.keys() for gate in reversed(seen[key])]

    # Combine programs without gate definitions; avoid call to _synthesize by using _instructions
    p = Program(*[prog._instructions for prog in prog_list])

    for definition in new_definitions:
        if isinstance(definition, DefPermutationGate):
            p.inst(DefPermutationGate(definition.name, list(definition.permutation)))
        else:
            p.defgate(definition.name, definition.matrix, definition.parameters)

    return p


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
    Move all the DECLARE statements to the top of the program. Return a fresh obejct.

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


def validate_protoquil(program: Program) -> None:
    """
    Ensure that a program is valid ProtoQuil, otherwise raise a ValueError.
    Protoquil is a subset of Quil which excludes control flow and classical instructions.

    :param program: The Quil program to validate.
    """
    valid_instruction_types = tuple([Pragma, Declare, Gate, Reset, ResetQubit, Measurement])
    for instr in program.instructions:
        if not isinstance(instr, valid_instruction_types):
            # Instructions like MOVE, NOT, JUMP, JUMP-UNLESS will fail here
            raise ValueError(f"ProtoQuil validation failed: {instr} is not allowed.")


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

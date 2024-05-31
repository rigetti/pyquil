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
"""Module for creating and defining Quil programs."""

import functools
import types
import warnings
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator, Sequence
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import quil.instructions as quil_rs
from deprecated.sphinx import deprecated
from qcs_sdk.compiler.quilc import NativeQuilMetadata
from quil.program import CalibrationSet
from quil.program import Program as RSProgram

from pyquil.control_flow_graph import ControlFlowGraph
from pyquil.gates import MEASURE, RESET
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
    FormalArgument,
    Frame,
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
from pyquil.quilbase import (
    AbstractInstruction,
    Declare,
    DefCalibration,
    DefFrame,
    DefGate,
    DefMeasureCalibration,
    DefWaveform,
    Gate,
    Jump,
    JumpTarget,
    JumpUnless,
    JumpWhen,
    Measurement,
    Pragma,
    _convert_to_py_instruction,
    _convert_to_py_instructions,
    _convert_to_py_qubits,
    _convert_to_rs_instruction,
    _convert_to_rs_instructions,
)
from pyquil.quiltcalibrations import (
    CalibrationMatch,
    _convert_to_calibration_match,
)

InstructionDesignator = Union[
    AbstractInstruction,
    quil_rs.Instruction,
    "Program",
    RSProgram,
    Sequence[Any],
    tuple[Any, ...],
    str,  # required to be a pyquil program
    Generator[Any, Any, Any],
]


RetType = TypeVar("RetType")


def _invalidates_cached_properties(func: Callable[..., RetType]) -> Callable[..., RetType]:
    """Check a class for any cached properties and clear them.

    This should be used on any `Program` method that changes the internal state of the program
    so that the next call to property rebuilds the cache.
    """

    @functools.wraps(func)
    def wrapper(self: "Program", *args: Any, **kwargs: Any) -> RetType:
        result = func(self, *args, **kwargs)
        cls = type(self)
        cached = {
            attr
            for attr in list(self.__dict__.keys())
            if (descriptor := getattr(cls, attr, None))
            if isinstance(descriptor, functools.cached_property)
        }
        for attr in cached:
            del self.__dict__[attr]
        return result

    return wrapper


class Program:
    """A list of pyQuil instructions that comprise a quantum program.

    >>> from pyquil import Program
    >>> from pyquil.gates import H, CNOT
    >>> p = Program()
    >>> p += H(0)
    >>> p += CNOT(0, 1)
    """

    def __init__(self, *instructions: InstructionDesignator):
        """Initialize a Program."""
        self._program: RSProgram = RSProgram()
        self.inst(*instructions)

        # default number of shots to loop through
        self.num_shots = 1

        self.native_quil_metadata: Optional[NativeQuilMetadata] = None

    # The following properties are cached on the first call and won't be re-built unless cleared.
    # Any method that mutates the state program should use the `@_invalidates_cached_properties`
    # decorator to clear these caches.

    @functools.cached_property
    def calibrations(self) -> list[DefCalibration]:
        """A list of Quil-T calibration definitions."""
        return [DefCalibration._from_rs_calibration(cal) for cal in self._program.calibrations.calibrations]

    @functools.cached_property
    def measure_calibrations(self) -> list[DefMeasureCalibration]:
        """A list of measure calibrations."""
        return [
            DefMeasureCalibration._from_rs_measure_calibration_definition(cal)
            for cal in self._program.calibrations.measure_calibrations
        ]

    @functools.cached_property
    def waveforms(self) -> dict[str, DefWaveform]:
        """A mapping from waveform names to their corresponding definitions."""
        return {
            name: DefWaveform._from_rs_waveform_definition(quil_rs.WaveformDefinition(name, waveform))
            for name, waveform in self._program.waveforms.items()
        }

    @functools.cached_property
    def frames(self) -> dict[Frame, DefFrame]:
        """A mapping from Quil-T frames to their definitions."""
        return {
            Frame._from_rs_frame_identifier(frame): DefFrame._from_rs_attribute_values(frame, attributes)
            for frame, attributes in self._program.frames.get_all_frames().items()
        }

    @functools.cached_property
    def declarations(self) -> dict[str, Declare]:
        """A mapping from declared region names to their declarations."""
        return {name: Declare._from_rs_declaration(inst) for name, inst in self._program.declarations.items()}

    def copy_everything_except_instructions(self) -> "Program":
        """Copy all the members that live on a Program object.

        :return: a new Program
        """
        new_prog = Program(
            list(self.frames.values()),
            list(self.waveforms.values()),
            self.calibrations,
            self.defined_gates,
            self.measure_calibrations,
        )
        if self.native_quil_metadata is not None:
            new_prog.native_quil_metadata = deepcopy(self.native_quil_metadata)
        new_prog.num_shots = self.num_shots
        return new_prog

    def copy(self) -> "Program":
        """Perform a deep copy of this program.

        :return: a new Program
        """
        new_program = Program(self._program.copy())
        new_program.native_quil_metadata = self.native_quil_metadata
        new_program.num_shots = self.num_shots
        return new_program

    @property
    def defined_gates(self) -> list[DefGate]:
        """A list of defined gates on the program."""
        return [DefGate._from_rs_gate_definition(gate) for gate in self._program.gate_definitions.values()]

    @property
    def instructions(self) -> list[AbstractInstruction]:
        """Fill in any placeholders and return a list of quil AbstractInstructions."""
        return list(self.declarations.values()) + _convert_to_py_instructions(self._program.body_instructions)

    @instructions.setter
    def instructions(self, instructions: list[AbstractInstruction]) -> None:
        new_program = self.copy_everything_except_instructions()
        new_program.inst(instructions)
        self._program = new_program._program

    @_invalidates_cached_properties
    def inst(self, *instructions: Union[InstructionDesignator, RSProgram]) -> "Program":
        """Mutates the Program object by appending new instructions.

        This function accepts a number of different valid forms, e.g.

            >>> from pyquil import Program
            >>> from pyquil.gates import H
            >>> p = Program()
            >>> p.inst(H(0))  # A single instruction
            Program { ... }
            >>> p.inst(H(0), H(1))  # Multiple instructions
            Program { ... }
            >>> p.inst([H(0), H(1)])  # A list of instructions
            Program { ... }
            >>> p.inst(H(i) for i in range(4))  # A generator of instructions
            Program { ... }
            >>> p.inst("H 0")  # A string representing an instruction
            Program { ... }
            >>> q = Program()
            >>> p.inst(q)  # Another program
            Program { ... }

        It can also be chained:
            >>> p = Program()
            >>> p.inst(H(0)).inst(H(1))
            Program { ... }

        :param instructions: A list of Instruction objects, e.g. Gates
        :return: self for method chaining
        """
        for instruction in instructions:
            if isinstance(instruction, list):
                self.inst(*instruction)
            elif isinstance(instruction, types.GeneratorType):
                self.inst(*instruction)
            elif isinstance(instruction, tuple):
                warnings.warn(
                    "Adding instructions to a program by specifying them as tuples is deprecated. Consider building "
                    "the instruction you need using classes from the `pyquil.gates` or `pyquil.quilbase` modules and "
                    "passing those to Program.inst() instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
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
                        params: list[ParameterDesignator] = []
                        possible_params = instruction[1]
                        rest: Sequence[Any] = instruction[2:]
                        if isinstance(possible_params, list):
                            params = possible_params
                        else:
                            rest = [possible_params] + list(rest)
                        self.gate(op, params, rest)
            elif isinstance(instruction, str):
                self.inst(RSProgram.parse(instruction.strip()))
            elif isinstance(instruction, Program):
                self.inst(instruction._program)
            elif isinstance(instruction, quil_rs.Instruction):
                self._add_instruction(instruction)
            elif isinstance(instruction, AbstractInstruction):
                self._add_instruction(_convert_to_rs_instruction(instruction))
            elif isinstance(instruction, RSProgram):
                self._program += instruction
            else:
                try:
                    instruction = quil_rs.Instruction(instruction)  # type: ignore
                    self.inst(instruction)
                except ValueError as e:
                    raise ValueError(f"Invalid instruction: {instruction}, type: {type(instruction)}") from e

        return self

    def control_flow_graph(self) -> ControlFlowGraph:
        """Return the :py:class:`~pyquil.control_flow_graph.ControlFlowGraph` of the program."""
        return ControlFlowGraph._from_rs(self._program.control_flow_graph())

    def with_loop(
        self,
        num_iterations: int,
        iteration_count_reference: MemoryReference,
        start_label: Union[Label, LabelPlaceholder],
        end_label: Union[Label, LabelPlaceholder],
    ) -> "Program":
        r"""Return a copy of the ``Program`` wrapped in a Quil loop that will execute ``num_iterations`` times.

        This loop is implemented with Quil and should not be confused with the ``num_shots`` property set by
        :py:meth:`~pyquil.quil.Program.wrap_in_numshots_loop`. The latter is metadata on the program that
        can tell an executor how many times to run the program. In comparison, this method adds Quil
        instructions to your program to specify a loop in the Quil program itself.

        The loop is constructed by wrapping the body of the program in classical Quil instructions.
        The given ``iteration_count_reference`` must refer to an INTEGER memory region. The value at the
        reference given will be set to ``num_iterations`` and decremented in the loop. The loop will
        terminate when the reference reaches 0. For this reason your program should not itself
        modify the value at the reference unless you intend to modify the remaining number of
        iterations (i.e. to break the loop).

        The given ``start_label`` and ``end_label`` will be used as the entry and exit points for
        the loop, respectively. You should provide unique ``JumpTarget``\\s that won't be used elsewhere
        in the program.

        If ``num_iterations`` is 1, then a copy of the program is returned without any changes. Raises a
        ``TypeError`` if ``num_iterations`` is negative.

        :param num_iterations: The number of times to execute the loop.
        :param loop_count_reference: The memory reference to use as the loop counter.
        :param start_label: The ``JumpTarget`` to use at the start of the loop.
        :param end_label: The ``JumpTarget`` to use to at the end of the loop.
        """
        looped_program = Program(
            self._program.wrap_in_loop(
                iteration_count_reference._to_rs_memory_reference(),
                start_label.target,
                end_label.target,
                num_iterations,
            )
        )
        return looped_program

    @_invalidates_cached_properties
    def resolve_placeholders(self) -> None:
        """Resolve all label and qubit placeholders in the program.

        A default resolver will generate a unique value for each placeholder within the scope of the program.
        """
        self._program.resolve_placeholders()

    def resolve_placeholders_with_custom_resolvers(
        self,
        *,
        label_resolver: Optional[Callable[[LabelPlaceholder], Optional[str]]] = None,
        qubit_resolver: Optional[Callable[[QubitPlaceholder], Optional[int]]] = None,
    ) -> None:
        r"""Resolve ``LabelPlaceholder``\\s and ``QubitPlaceholder``\\s within the program using a function.

        If you provide ``label_resolver`` and/or ``qubit_resolver``, they will be used to resolve those values
        respectively. If your resolver returns `None` for a particular placeholder, it will not be replaced but
        will be left as a placeholder.

        If you do not provide a resolver for a placeholder, a default resolver will be used which will generate a unique
        value for that placeholder within the scope of the program using an auto-incrementing value (for qubit) or
        suffix (for target) while ensuring that unique value is not already in use within the program.
        """
        rs_qubit_resolver = None
        if qubit_resolver is not None:

            def rs_qubit_resolver(placeholder: quil_rs.QubitPlaceholder) -> Optional[int]:
                return qubit_resolver(QubitPlaceholder(placeholder=placeholder))

        rs_label_resolver = None
        if label_resolver is not None:

            def rs_label_resolver(placeholder: quil_rs.TargetPlaceholder) -> Optional[str]:
                return label_resolver(LabelPlaceholder(placeholder=placeholder))

        self._program.resolve_placeholders_with_custom_resolvers(
            target_resolver=rs_label_resolver, qubit_resolver=rs_qubit_resolver
        )

    @_invalidates_cached_properties
    def resolve_qubit_placeholders(self) -> None:
        """Resolve all qubit placeholders in the program."""
        self._program.resolve_placeholders_with_custom_resolvers(target_resolver=lambda _: None)

    @_invalidates_cached_properties
    def resolve_qubit_placeholders_with_mapping(self, qubit_mapping: dict[QubitPlaceholder, int]) -> None:
        r"""Resolve all qubit placeholders using a mapping of ``QubitPlaceholder``\\s to the index they resolve to."""

        def qubit_resolver(placeholder: quil_rs.QubitPlaceholder) -> Optional[int]:
            return qubit_mapping.get(QubitPlaceholder(placeholder), None)

        def label_resolver(_: quil_rs.TargetPlaceholder) -> None:
            return None

        self._program.resolve_placeholders_with_custom_resolvers(
            qubit_resolver=qubit_resolver, target_resolver=label_resolver
        )

    @_invalidates_cached_properties
    def resolve_label_placeholders(self) -> None:
        """Resolve all label placeholders in the program."""
        self._program.resolve_placeholders_with_custom_resolvers(qubit_resolver=lambda _: None)

    def _add_instruction(self, instruction: quil_rs.Instruction) -> None:
        """Add an instruction to the Program after normalizing to a `quil_rs.Instruction`.

        For backwards compatibility, it also prevents duplicate calibration, measurement, and gate definitions from
        being added. Users of ``Program`` should use ``inst`` or ``Program`` addition instead.
        """
        if instruction.is_calibration_definition():
            defcal = instruction.to_calibration_definition()
            idx, existing_calibration = next(
                (
                    (i, existing_calibration)
                    for i, existing_calibration in enumerate(self._program.calibrations.calibrations)
                    if defcal.name == existing_calibration.name
                    and defcal.parameters == existing_calibration.parameters
                    and defcal.qubits == existing_calibration.qubits
                ),
                (0, None),
            )
            if existing_calibration is None:
                self._program.add_instruction(instruction)

            elif (
                existing_calibration.instructions != defcal.instructions
                or existing_calibration.modifiers != defcal.modifiers
            ):
                warnings.warn(f"Redefining calibration {defcal.name}", stacklevel=2)
                current_calibrations = self._program.calibrations
                new_calibrations = CalibrationSet(
                    current_calibrations.calibrations[:idx] + [defcal] + current_calibrations.calibrations[idx + 1 :],
                    current_calibrations.measure_calibrations,
                )
                self._program.calibrations = new_calibrations
        elif instruction.is_measure_calibration_definition():
            defmeasure = instruction.to_measure_calibration_definition()
            idx, existing_measure_calibration = next(
                (
                    (i, existing_measure_calibration)
                    for i, existing_measure_calibration in enumerate(self._program.calibrations.measure_calibrations)
                    if existing_measure_calibration.parameter == defmeasure.parameter
                    and existing_measure_calibration.qubit == defmeasure.qubit
                ),
                (0, None),
            )
            if existing_measure_calibration is None:
                self._program.add_instruction(instruction)

            else:
                warnings.warn(f"Redefining DefMeasureCalibration {instruction}", stacklevel=2)
                current_calibrations = self._program.calibrations
                new_calibrations = CalibrationSet(
                    current_calibrations.calibrations,
                    current_calibrations.measure_calibrations[:idx]
                    + [defmeasure]
                    + current_calibrations.measure_calibrations[idx + 1 :],
                )

                self._program.calibrations = new_calibrations
        else:
            self._program.add_instruction(instruction)

    def filter_instructions(self, predicate: Callable[[AbstractInstruction], bool]) -> "Program":
        """Return a new ``Program`` containing only the instructions for which ``predicate`` returns ``True``.

        :param predicate: A function that takes an instruction and returns ``True`` if the instruction should not be
            removed from the program, ``False`` otherwise.
        :return: A new ``Program`` object with the filtered instructions.
        """

        def rs_predicate(inst: quil_rs.Instruction) -> bool:
            return predicate(_convert_to_py_instruction(inst))

        filtered_program = Program(self._program.filter_instructions(rs_predicate))
        filtered_program.num_shots = self.num_shots
        return filtered_program

    def remove_quil_t_instructions(self) -> "Program":
        """Return a copy of the program with all Quil-T instructions removed."""
        filtered_program = Program(self._program.filter_instructions(lambda inst: not inst.is_quil_t()))
        filtered_program.num_shots = self.num_shots
        return filtered_program

    def gate(
        self,
        name: str,
        params: Sequence[ParameterDesignator],
        qubits: Sequence[Union[Qubit, QubitPlaceholder]],
    ) -> "Program":
        """Add a gate to the program.

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
        matrix: Union[list[list[Any]], np.ndarray, np.matrix],
        parameters: Optional[list[Parameter]] = None,
    ) -> "Program":
        """Define a new static gate.

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
        """Overload a static ideal gate with a noisy one defined in terms of a Kraus map.

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

    def define_noisy_readout(self, qubit: Union[int], p00: float, p11: float) -> "Program":
        """For this program define a classical bit flip readout error channel parametrized by ``p00`` and ``p11``.

        This models the effect of thermal noise that corrupts the readout signal **after** it has interrogated the
        qubit.

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
        """Prevent a noisy gate definition from being applied to the immediately following Gate instruction.

        :return: Program
        """
        return self.inst(Pragma("NO-NOISE"))

    def measure(self, qubit: QubitDesignator, classical_reg: Optional[MemoryReferenceDesignator]) -> "Program":
        """Measures a qubit at qubit_index and puts the result in classical_reg.

        :param qubit: The qubit to measure.
        :param classical_reg: The classical register to measure into, or None.
        :returns: The Quil Program with the appropriate measure instruction appended, e.g.
                  MEASURE 0 [1]
        """
        return self.inst(MEASURE(qubit, classical_reg))

    def reset(self, qubit_index: Optional[int] = None) -> "Program":
        """Reset all qubits or just a specific qubit at qubit_index.

        :param qubit_index: The address of the qubit to reset.
            If None, reset all qubits.
        :returns: The Quil Program with the appropriate reset instruction appended, e.g.
                  RESET 0
        """
        return self.inst(RESET(qubit_index))

    def measure_all(self, *qubit_reg_pairs: tuple[QubitDesignator, Optional[MemoryReferenceDesignator]]) -> "Program":
        """Measures many qubits into their specified classical bits, in the order they were entered.

        If no qubit/register pairs are provided, measure all qubits present in the program into classical addresses of
        the same index.

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
            qubit_inds = cast(list[int], qubits)
            ro = self.declare("ro", "BIT", max(qubit_inds) + 1)
            for qi in qubit_inds:
                self.inst(MEASURE(qi, ro[qi]))
        else:
            for qubit_index, classical_reg in qubit_reg_pairs:
                self.inst(MEASURE(qubit_index, classical_reg))
        return self

    def prepend_instructions(self, instructions: Iterable[AbstractInstruction]) -> "Program":
        """Prepend instructions to the beginning of the program."""
        new_prog = Program(*instructions)
        return new_prog + self

    def while_do(self, classical_reg: MemoryReferenceDesignator, q_program: "Program") -> "Program":
        """While a classical register at index classical_reg is 1, loop q_program.

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
        """If the classical register at index classical reg is 1, run if_program, else run else_program.

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
        offsets: Optional[Sequence[tuple[int, str]]] = None,
    ) -> MemoryReference:
        """DECLARE a quil variable.

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
        """Wrap a Quil program in a loop that re-runs the same program many times.

        Note: this function is a prototype of what will exist in the future when users will
        be responsible for writing this loop instead of having it happen automatically.

        :param shots: Number of iterations to loop through.
        """
        self.num_shots = shots
        return self

    def out(self, *, calibrations: Optional[bool] = True) -> str:
        """Serialize the Quil program to a string suitable for submitting to the QVM or QPU."""
        if calibrations:
            return self._program.to_quil()
        else:
            return self.filter_instructions(
                lambda inst: not isinstance(inst, (DefCalibration, DefMeasureCalibration))
            ).out()

    @deprecated(
        version="4.0",
        reason="The indices flag will be removed. Use get_qubit_indices() instead.",
    )
    def get_qubits(self, indices: bool = True) -> Union[set[QubitDesignator], set[int]]:
        """Return all of the qubit indices used in this program, including gate applications and allocated qubits.

        For example:

            >>> from pyquil.gates import H
            >>> p = Program()
            >>> p.inst(("H", 1))
            Program { ... }
            >>> p.get_qubits()
            {1}
            >>> q = QubitPlaceholder()
            >>> p.inst(H(q))
            Program { ... }
            >>> len(p.get_qubits(indices=False))
            2

        :param indices: Return qubit indices as integers instead of the
            wrapping :py:class:`Qubit` object
        :return: A set of all the qubit indices used in this program
        """
        if indices:
            return self.get_qubit_indices()
        return set(_convert_to_py_qubits(self._program.get_used_qubits()))

    def get_qubit_indices(self) -> set[int]:
        """Return the index of each qubit used in the program.

        Will raise an exception if any of the qubits are placeholders.
        """
        return {q.to_fixed() for q in self._program.get_used_qubits()}

    def match_calibrations(self, instr: AbstractInstruction) -> Optional[CalibrationMatch]:
        """Attempt to match a calibration to the provided instruction.

        Note: preference is given to later calibrations, i.e. in a program with

          DEFCAL X 0:
             <a>

          DEFCAL X 0:
             <b>

        the second calibration, with body <b>, would be the calibration matching `X 0`.

        :param instr: An instruction.
        :returns: a CalibrationMatch object, if one can be found.
        """
        if not isinstance(instr, (Gate, Measurement)):
            return None

        instruction = _convert_to_rs_instruction(instr)
        if instruction.is_gate():
            gate = instruction.to_gate()
            gate_match = self._program.calibrations.get_match_for_gate(gate)
            return _convert_to_calibration_match(gate, gate_match)

        if instruction.is_measurement():
            measurement = instruction.to_measurement()
            measure_match = self._program.calibrations.get_match_for_measurement(measurement)
            return _convert_to_calibration_match(measurement, measure_match)

        return None

    def get_calibration(self, instr: AbstractInstruction) -> Optional[Union[DefCalibration, DefMeasureCalibration]]:
        """Get the calibration corresponding to the provided instruction.

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
        previously_calibrated_instructions: Optional[set[AbstractInstruction]] = None,
    ) -> list[AbstractInstruction]:
        """Expand an instruction into its calibrated definition.

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

        calibrated_instructions = self._program.calibrations.expand(
            _convert_to_rs_instruction(instruction),
            _convert_to_rs_instructions(previously_calibrated_instructions),
        )

        return [instruction] if not calibrated_instructions else _convert_to_py_instructions(calibrated_instructions)

    @deprecated(
        version="4.0",
        reason="This function always returns True and will be removed.",
    )
    def is_protoquil(self, quilt: bool = False) -> bool:
        """Return True, this method is deprecated."""
        return True

    @deprecated(
        version="4.0",
        reason="This function always returns True and will be removed.",
    )
    def is_supported_on_qpu(self) -> bool:
        """Return True, this method is deprecated."""
        return True

    def dagger(self) -> "Program":
        """Create the conjugate transpose of the Quil program. The program must contain only gate applications.

        :return: The Quil program's inverse
        """
        return Program(self._program.dagger())

    def __add__(self, other: InstructionDesignator) -> "Program":
        """Concatenate two programs together, returning a new one.

        :param other: Another program or instruction to concatenate to this one.
        :return: A newly concatenated program.
        """
        p = Program()
        p.inst(self)
        p.inst(other)
        return p

    def __iadd__(self, other: InstructionDesignator) -> "Program":
        """Concatenate two programs together, by overwriting the left hand side."""
        self.inst(other)
        return self

    def __getitem__(self, index: Union[slice, int]) -> Union[AbstractInstruction, "Program"]:
        """Get the instruction at the given index, or a Program from a slice."""
        return Program(self.instructions[index]) if isinstance(index, slice) else self.instructions[index]

    def __iter__(self) -> Iterator[AbstractInstruction]:
        """Iterate through a program's instructions, e.g. [a for a in Program(X(0))]."""
        return self.instructions.__iter__()

    def __eq__(self, other: object) -> bool:
        """Check if two programs are equal."""
        if isinstance(other, Program):
            return self._program == other._program
        return False

    def __len__(self) -> int:
        """Get the number of instructions in the program."""
        return len(self.instructions)

    def __hash__(self) -> int:
        """Hash the program."""
        return hash(self.out())

    def __repr__(self) -> str:
        """Get a string representation of the Quil program for inspection.

        This may not be suitable for submission to a QPU or QVM for example if
        your program contains unaddressed QubitPlaceholders
        """
        return repr(self._program)

    def __str__(self) -> str:
        """Get a string representation of the Quil program for inspection.

        This may not be suitable for submission to a QPU or QVM for example if
        your program contains unaddressed QubitPlaceholders
        """
        return self._program.to_quil_or_debug()

    def get_all_instructions(self) -> list[AbstractInstruction]:
        """Get _all_ instructions that makeup the program."""
        return _convert_to_py_instructions(self._program.to_instructions())


def merge_with_pauli_noise(
    prog_list: Iterable[Program], probabilities: Sequence[float], qubits: Sequence[int]
) -> Program:
    """Insert pauli noise channels between each item in the list of programs.

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


def get_classical_addresses_from_program(program: Program) -> dict[str, list[int]]:
    """Return a sorted list of classical addresses found in the MEASURE instructions in the program.

    :param program: The program from which to get the classical addresses.
    :return: A mapping from memory region names to lists of offsets appearing in the program.
    """
    addresses: dict[str, list[int]] = defaultdict(list)
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


def address_qubits(program: Program, qubit_mapping: Optional[dict[QubitPlaceholder, int]] = None) -> Program:
    """Take a program which contains placeholders and assigns the all defined values.

    Either all qubits must be defined or all undefined. If qubits are
    undefined, you may provide a qubit mapping to specify how placeholders get mapped
    to actual qubits. If a mapping is not provided, integers 0 through N are used.
    This function will also instantiate any label placeholders.
    :param program: The program.
    :param qubit_mapping: A dictionary-like object that maps from :py:class:`QubitPlaceholder`
    to :py:class:`Qubit` or ``int`` (but not both).
    :return: A new Program with all qubit and label placeholders assigned to real qubits and labels.
    """
    new_program = program.copy()
    if qubit_mapping:
        new_program.resolve_qubit_placeholders_with_mapping(qubit_mapping)
    else:
        new_program.resolve_qubit_placeholders()
    return new_program


def _get_label(
    placeholder: LabelPlaceholder,
    label_mapping: dict[LabelPlaceholder, Label],
    label_i: int,
) -> tuple[Label, dict[LabelPlaceholder, Label], int]:
    """Get the appropriate label for a given placeholder or generate a new label and update the mapping.

    See :py:func:`instantiate_labels` for usage.
    """
    if placeholder in label_mapping:
        return label_mapping[placeholder], label_mapping, label_i

    new_target = Label(f"{placeholder.prefix}{label_i}")
    label_i += 1
    label_mapping[placeholder] = new_target
    return new_target, label_mapping, label_i


def instantiate_labels(instructions: Iterable[AbstractInstruction]) -> list[AbstractInstruction]:
    """Take an iterable of instructions which may contain label placeholders and assigns them all defined values.

    :return: list of instructions with all label placeholders assigned to real labels.
    """
    label_i = 1
    result: list[AbstractInstruction] = []
    label_mapping: dict[LabelPlaceholder, Label] = dict()
    for instr in instructions:
        if isinstance(instr, Jump) and isinstance(instr.target, LabelPlaceholder):
            new_target, label_mapping, label_i = _get_label(instr.target, label_mapping, label_i)
            result.append(Jump(new_target))
        elif isinstance(instr, (JumpWhen, JumpUnless)) and isinstance(instr.target, LabelPlaceholder):
            new_target, label_mapping, label_i = _get_label(instr.target, label_mapping, label_i)
            cls = instr.__class__  # Make the correct subclass
            result.append(cls(new_target, instr.condition))
        elif isinstance(instr, JumpTarget) and isinstance(instr.label, LabelPlaceholder):
            new_label, label_mapping, label_i = _get_label(instr.label, label_mapping, label_i)
            result.append(JumpTarget(new_label))
        else:
            result.append(instr)

    return result


@deprecated(
    version="4.0",
    reason="The Program class now sorts instructions automatically. This function will be removed.",
)
def percolate_declares(program: Program) -> Program:
    """As of pyQuil v4.0, the Program class does this automatically.

    This function is deprecated and just immediately returns the passed in program.

    :param program: A program.
    """
    return program


@deprecated(
    version="4.0",
    reason="This function will be removed. Use Program addition instead.",
)
def merge_programs(prog_list: Sequence[Program]) -> Program:
    """Merge a list of pyQuil programs into a single one by appending them in sequence."""
    merged_program = Program()
    for prog in prog_list:
        merged_program += prog
    return merged_program


@deprecated(
    version="4.0",
    reason="This is now a no-op and will be removed in future versions of pyQuil.",
)
def validate_protoquil(program: Program, quilt: bool = False) -> None:  # noqa: D103 - deprecated no-op
    pass


@deprecated(
    version="4.0",
    reason="This is now a no-op and will be removed in future versions of pyQuil.",
)
def validate_supported_quil(program: Program) -> None:  # noqa: D103 - deprecated no-op
    pass

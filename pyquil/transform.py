"""
transform module
----------------

Utility functions for Quil program manipulation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterator, List, Optional, Union

from pyquil.api import MemoryMap
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, substitute
from pyquil.quilbase import Declare, DefCircuit, Gate, Measurement, Reset, ResetQubit
from quil.instructions import CircuitDefinition
from quil.instructions import Instruction as QuilInstruction
from quil.program import Program as QuilProgram


def copy_everything_except_instructions(
    program: Program, include_defcircuits: bool = True, include_kraus: bool = True
) -> Program:
    """Create a new program with only the definitions of the input program.

    :param program: A pyQuil program.
    :param include_defcircuits: If True, include DEFCIRCUIT definitions.
    :param include_kraus: If True, include KRAUS definitions.
    """
    from pyquil.quilbase import Pragma

    p = QuilProgram()
    p.waveforms = program._program.waveforms
    p.calibrations = program._program.calibrations
    p.frames = program._program.frames
    p.gate_definitions = program._program.gate_definitions

    program_definitions = Program()
    program_definitions._program = p

    # Pragma externs are definitions
    program_definitions += (
        [QuilInstruction.from_pragma(pragma) for pragma in program._program.pragma_extern_map.values()],
    )

    if include_defcircuits is True:
        defcircuits = set()
        for inst in program._program.to_instructions():
            if isinstance(inst.inner(), CircuitDefinition) and str(inst) not in defcircuits:
                defcircuits.add(str(inst))
                program_definitions._program.add_instruction(inst)

    if include_kraus is True:
        for kraus_inst in program.instructions:
            if isinstance(kraus_inst, Pragma):
                try:
                    if kraus_inst.command == "ADD-KRAUS":
                        program_definitions._program.add_instruction(kraus_inst)  # type: ignore[arg-type]
                except Exception:
                    pass

    return program_definitions


def unparameterize(program: Program, memory_map: MemoryMap) -> Program:
    """Apply a memory map to a program, and evaluate any arithmetic.

    Memory declarations will be removed, except "ro".

    :param program: A pyquil program, possibly with parameters.
    :param memory_map: A memory map, with values for the parameters.
    """
    unparameterized_program = Program()
    unparameterized_program += copy_everything_except_instructions(program)
    instructions = program.instructions
    parameter_substitution_map = {}

    if memory_map is not None:
        parameter_substitution_map = {
            MemoryReference(name=name, offset=offset, declared_size=len(value))
            if isinstance(name, str)
            else name: value[offset]
            for name, value in memory_map.items()
            for offset in range(len(value))
        }

    for idx, inst in enumerate(instructions):
        if isinstance(inst, Declare):
            if inst.name == "ro":
                unparameterized_program += deepcopy(inst)
        elif isinstance(inst, Gate):
            if len(inst.params) > 0:
                unparameterized_program += Gate(
                    name=inst.name,
                    params=[substitute(p, parameter_substitution_map) for p in inst.params],  # type: ignore[arg-type]
                    qubits=inst.qubits,
                )
            else:
                unparameterized_program += inst

        else:
            unparameterized_program += inst

    unparameterized_program.wrap_in_numshots_loop(program.num_shots)

    return unparameterized_program


def expand_defcircuit_body(
    inst: Gate,
    defcircuit: DefCircuit,
    circuit_definitions: Dict[str, DefCircuit],
) -> Iterator[Union[Gate, Measurement, ResetQubit, Reset]]:
    """Yield concrete instructions from a DEFCIRCUIT invocation.

    Substitutes formal qubit/parameter arguments with the concrete values
    from ``inst``.  Handles nested DEFCIRCUITs via recursion.

    :param inst: The Gate that invokes the DEFCIRCUIT.
    :param defcircuit: The DefCircuit definition to expand.
    :param circuit_definitions: All known DEFCIRCUIT definitions (for nested expansion).
    :yields: Concrete instructions with physical qubits and resolved parameters.
    """
    qarg_to_arg_map = {qarg: q for q, qarg in zip(inst.qubits, defcircuit.qubit_variables)}
    parg_to_arg_map = {parg: param for param, parg in zip(inst.params, defcircuit.parameters)}

    for circuit_inst in defcircuit.instructions:
        if isinstance(circuit_inst, Gate):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubits = [qarg_to_arg_map[qarg] for qarg in circuit_inst.qubits]  # type: ignore[index,misc]
            if hasattr(circuit_inst, "params"):
                circuit_inst.params = [substitute(param, parg_to_arg_map) for param in circuit_inst.params]  # type: ignore[arg-type]
            if circuit_inst.name in circuit_definitions:
                yield from expand_defcircuit_body(
                    circuit_inst, circuit_definitions[circuit_inst.name], circuit_definitions
                )
            else:
                yield circuit_inst
        elif isinstance(circuit_inst, Measurement):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubit = qarg_to_arg_map[circuit_inst.qubit]  # type: ignore[index]
            yield circuit_inst
        elif isinstance(circuit_inst, ResetQubit):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubit = qarg_to_arg_map[circuit_inst.qubit]  # type: ignore[index]
            yield circuit_inst
        else:
            yield deepcopy(circuit_inst)  # type: ignore[misc]


def expand_defcircuits(
    program: Program,
    expand_if_defcal: bool = True,
    calibration_program: Optional[Program] = None,
    keep_defcircuits: bool = False,
) -> Program:
    """Expand DEFCIRCUITS into individual instructions.

    :param program: A Quil program, which may contain DefCircuits.
    :param expand_if_defcal: Expand the defcircuit even if it has a defcalibration.
    :param calibration_program: Calibrations to supplement those in ``program``. Existing
        calibrations in ``program`` take precedence.
    :param keep_defcircuits: If True, keep the DEFCIRCUIT definitions in the returned program.
    :return: A Quil program, with any Circuit instructions expanded to individual instructions.
    """
    instructions: List = []
    circuit_definitions: dict = {}
    for inst in program.instructions:
        if isinstance(inst, DefCircuit):
            circuit_definitions[inst.name] = inst
            if keep_defcircuits is True:
                instructions.append(inst)
        else:
            instructions.append(inst)

    holistic_calibration_program = Program()
    if calibration_program is not None:
        holistic_calibration_program += calibration_program
    holistic_calibration_program += copy_everything_except_instructions(program, include_defcircuits=False)

    expanded_program = Program()
    expanded_program += holistic_calibration_program

    if len(circuit_definitions) == 0 and len(instructions) == 0:
        return expanded_program

    def _should_expand(inst: Gate) -> bool:
        name = inst.name
        if name not in circuit_definitions:
            return False
        defcircuit = circuit_definitions[name]
        qubits = tuple(int(q) for q in inst.get_qubit_indices())
        if len(qubits) != len(defcircuit.qubit_variables) or len(inst.params) != len(defcircuit.parameters):
            return False
        if expand_if_defcal is False:
            if holistic_calibration_program.get_calibration(inst) is not None:
                return False
            if program.get_calibration(inst) is not None:
                return False
        return True

    expanded_instructions: List = []
    for inst in instructions:
        if isinstance(inst, Gate) and _should_expand(inst):
            expanded_instructions.extend(
                expand_defcircuit_body(inst, circuit_definitions[inst.name], circuit_definitions)
            )
        else:
            expanded_instructions.append(inst)

    expanded_program += expanded_instructions
    return expanded_program

"""
transform module
----------------

Utility functions for Quil program manipulation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

from pyquil.api import MemoryMap
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, substitute
from pyquil.quilbase import Declare, DefCircuit, Gate, Measurement, ResetQubit
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
        for inst in program.instructions:
            if isinstance(inst, Pragma):
                try:
                    if inst.command == "ADD-KRAUS":
                        program_definitions._program.add_instruction(inst)
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
                    params=[substitute(p, parameter_substitution_map) for p in inst.params],
                    qubits=inst.qubits,
                )
            else:
                unparameterized_program += inst

        else:
            unparameterized_program += inst

    unparameterized_program.wrap_in_numshots_loop(program.num_shots)

    return unparameterized_program


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

    def _expand_instruction(inst: Gate) -> List:
        instruction_name = inst.name
        expanded_instructions: List = []
        if expand_if_defcal is False:
            cal = holistic_calibration_program.get_calibration(inst)
            if cal is not None:
                return [inst]

        defcircuit = circuit_definitions[instruction_name]
        qubit_variables = defcircuit.qubit_variables
        qubits = inst.qubits

        qarg_to_arg_map = {qarg: q for q, qarg in zip(qubits, qubit_variables)}
        parg_to_arg_map = {parg: param for param, parg in zip(inst.params, defcircuit.parameters)}

        for circuit_inst in defcircuit.instructions:
            match circuit_inst:
                case Gate():
                    circuit_inst = deepcopy(circuit_inst)
                    circuit_inst.qubits = [qarg_to_arg_map[qarg] for qarg in circuit_inst.qubits]
                    if hasattr(circuit_inst, "params"):
                        circuit_inst.params = [substitute(param, parg_to_arg_map) for param in circuit_inst.params]
                    if circuit_inst.name in circuit_definitions:
                        expanded_instructions += _expand_instruction(circuit_inst)
                    else:
                        expanded_instructions.append(circuit_inst)
                case Measurement():
                    circuit_inst = deepcopy(circuit_inst)
                    circuit_inst.qubit = qarg_to_arg_map[circuit_inst.qubit]
                    expanded_instructions.append(circuit_inst)
                case ResetQubit():
                    circuit_inst = deepcopy(circuit_inst)
                    circuit_inst.qubit = qarg_to_arg_map[circuit_inst.qubit]
                    expanded_instructions.append(circuit_inst)
                case _:
                    expanded_instructions.append(deepcopy(circuit_inst))
        return expanded_instructions

    expanded_instructions: List = []
    for inst in instructions:
        if isinstance(inst, Gate):
            instruction_name = inst.name
            qubits = tuple(int(q) for q in inst.get_qubit_indices())
            if (
                (instruction_name in circuit_definitions)
                and len(qubits) == len(circuit_definitions[instruction_name].qubit_variables)
                and len(inst.params) == len(circuit_definitions[instruction_name].parameters)
            ):
                if expand_if_defcal is False:
                    cal = program.get_calibration(inst)
                    if cal is not None:
                        expanded_instructions.append(inst)
                        continue

                expanded_instructions += _expand_instruction(inst)
            else:
                expanded_instructions.append(inst)
        else:
            expanded_instructions.append(inst)

    expanded_program += expanded_instructions
    return expanded_program

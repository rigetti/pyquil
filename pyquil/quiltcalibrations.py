from copy import copy
from dataclasses import dataclass
from typing import Union, Dict, List, Any, Optional, no_type_check

from pyquil.quilatom import (
    substitute,
    Expression,
    FormalArgument,
    Frame,
    MemoryReference,
    Parameter,
    Qubit,
    TemplateWaveform,
    WaveformReference,
    _convert_to_py_parameter,
)
from pyquil.quilbase import (
    AbstractInstruction,
    Capture,
    Declare,
    DelayFrames,
    DelayQubits,
    DefCalibration,
    DefMeasureCalibration,
    Fence,
    FenceAll,
    Gate,
    Measurement,
    Pragma,
    Pulse,
    RawCapture,
    ResetQubit,
    SetFrequency,
    SetPhase,
    SetScale,
    ShiftFrequency,
    ShiftPhase,
    SwapPhase,
    _convert_to_rs_instruction,
    _convert_to_py_instruction,
    _convert_to_py_qubit,
)

from qcs_sdk.quil.program import CalibrationSet
import qcs_sdk.quil.instructions as quil_rs


class CalibrationError(Exception):
    pass


class CalibrationDoesntMatch(CalibrationError):
    pass


@dataclass
class CalibrationMatch:
    cal: Union[DefCalibration, DefMeasureCalibration]
    settings: Dict[Union[FormalArgument, Parameter], Any]


@no_type_check
def fill_placeholders(obj, placeholder_values: Dict[Union[FormalArgument, Parameter], Any]):
    """Update Parameter and FormalArgument references in objects with
    their corresponding definitions.

    It is an error if the object has a Parameter or FormalArgument reference without
    a corresponding definition in placeholder_values.

    :param obj: A Quil AST object.
    :param placeholder_values: A dictionary mapping placeholders to their values.
    :returns: The updated AST object.
    """
    try:
        if obj is None or isinstance(obj, (int, float, complex, Qubit, MemoryReference)):
            return obj
        elif isinstance(obj, Expression):
            # defer to the usual PyQuil substitution
            return substitute(obj, {k: v for k, v in placeholder_values.items() if isinstance(k, Parameter)})
        elif isinstance(obj, FormalArgument):
            return placeholder_values[obj]
        elif isinstance(obj, Frame):
            return Frame(fill_placeholders(obj.qubits, placeholder_values), obj.name)
        elif isinstance(obj, WaveformReference):
            return obj
        elif isinstance(obj, TemplateWaveform):
            return obj.__class__(**fill_placeholders(obj.__dict__, placeholder_values))
        elif isinstance(obj, list):
            return [fill_placeholders(elt, placeholder_values) for elt in obj]
        elif isinstance(obj, dict):
            return {k: fill_placeholders(v, placeholder_values) for (k, v) in obj.items()}
        elif isinstance(obj, tuple):
            return tuple([fill_placeholders(item, placeholder_values) for item in obj])
        elif isinstance(obj, Pragma) and obj.command == "LOAD-MEMORY":
            (source,) = obj.args
            arg = FormalArgument(obj.freeform_string)
            if arg in placeholder_values:
                return Pragma("LOAD-MEMORY", [source], str(placeholder_values[arg]))
            else:
                return obj
        else:
            specs = {
                Gate: ["params", "qubits"],
                Measurement: ["qubit", "classical_reg"],
                ResetQubit: ["qubit"],
                Pulse: ["frame", "waveform"],
                SetFrequency: ["frame", "freq"],
                ShiftFrequency: ["frame", "freq"],
                SetPhase: ["frame", "phase"],
                ShiftPhase: ["frame", "phase"],
                SwapPhase: ["frameA", "frameB"],
                SetScale: ["frame", "scale"],
                Capture: ["frame", "kernel", "memory_region"],
                RawCapture: ["frame", "duration", "memory_region"],
                DelayQubits: ["qubits", "duration"],
                DelayFrames: ["frames", "duration"],
                Fence: ["qubits"],
                FenceAll: [],
                Declare: [],
                Pragma: [],
            }
            if type(obj) in specs:
                attrs = specs[type(obj)]
                updated = copy(obj)
                for attr in attrs:
                    setattr(updated, attr, fill_placeholders(getattr(updated, attr), placeholder_values))
                return updated
            else:
                raise CalibrationError(f"Unable to fill placeholders in object {obj}.")
    except Exception as e:
        raise e
        # raise ValueError(f"Unable to fill placeholders in  object {obj}.")


def match_calibration(
    instr: AbstractInstruction, cal: Union[DefCalibration, DefMeasureCalibration]
) -> Optional[CalibrationMatch]:
    """Match a calibration definition to an instruction.

    On a successful match, return a (possibly empty) dictionary mapping calibration
    arguments and parameters to their values.

    On a failure, return None.
    """
    calibration = _convert_to_rs_instruction(cal)
    if calibration.is_calibration_definition():
        instruction = _convert_to_rs_instruction(instr)
        if not instruction.is_gate():
            return None
        calibration_set = CalibrationSet([calibration.to_calibration_definition()], [])
        gate = instruction.to_gate()
        matched_calibration = calibration_set.get_match_for_gate(
            gate.modifiers, gate.name, gate.parameters, gate.qubits
        )
        if matched_calibration is None:
            return None

        settings = {
            _convert_to_py_qubit(param): _convert_to_py_qubit(qubit)
            for param, qubit in zip(matched_calibration.qubits, gate.qubits)
            if param.is_variable()
        }
        settings.update(
            {
                _convert_to_py_parameter(param): _convert_to_py_parameter(value)
                for param, value in zip(matched_calibration.parameters, gate.parameters)
                if param.is_variable()
            }
        )
        return CalibrationMatch(DefCalibration._from_rs_calibration(matched_calibration), settings)

    if calibration.is_measure_calibration_definition():
        # TODO: CalibrationSet API needs measurement calibration
        instruction = _convert_to_rs_instruction(instr)
        calibration_set = CalibrationSet([], [calibration.to_measure_calibration_definition()])

    return None


def expand_calibration(match: CalibrationMatch) -> List[AbstractInstruction]:
    """ " Expand the body of a calibration from a match."""
    return [fill_placeholders(instr, match.settings) for instr in match.cal.instrs]

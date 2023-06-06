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
    SwapPhases,
)


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
                SwapPhases: ["frameA", "frameB"],
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
    settings: Dict[Any, Any] = {}

    @no_type_check
    def unpack_field(cal_field, instr_field):
        if isinstance(cal_field, (Parameter, FormalArgument)):
            if instr_field is None:
                raise CalibrationDoesntMatch()
            settings[cal_field] = instr_field
        elif cal_field != instr_field:
            raise CalibrationDoesntMatch()

    if isinstance(instr, Measurement) and isinstance(cal, DefMeasureCalibration):
        try:
            unpack_field(cal.qubit, instr.qubit)
            unpack_field(cal.memory_reference, instr.classical_reg)
            return CalibrationMatch(cal, settings)
        except CalibrationDoesntMatch:
            return None
    elif isinstance(instr, Gate) and isinstance(cal, DefCalibration):
        try:
            unpack_field(cal.name, instr.name)
            for cal_param, instr_param in zip(cal.parameters, instr.params):
                unpack_field(cal_param, instr_param)
            for cal_arg, instr_arg in zip(cal.qubits, instr.qubits):
                unpack_field(cal_arg, instr_arg)
            return CalibrationMatch(cal, settings)
        except CalibrationDoesntMatch:
            return None
    else:
        return None


def expand_calibration(match: CalibrationMatch) -> List[AbstractInstruction]:
    """ " Expand the body of a calibration from a match."""
    return [fill_placeholders(instr, match.settings) for instr in match.cal.instrs]

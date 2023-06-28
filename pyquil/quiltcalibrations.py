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
    _convert_to_py_expression,
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
    _convert_to_rs_instruction,
    _convert_to_py_instruction,
    _convert_to_py_qubit,
)

from quil.program import CalibrationSet
import quil.instructions as quil_rs


class CalibrationError(Exception):
    pass


class CalibrationDoesntMatch(CalibrationError):
    pass


@dataclass
class CalibrationMatch:
    cal: Union[DefCalibration, DefMeasureCalibration]
    settings: Dict[Union[FormalArgument, Parameter], Any]


def _convert_to_calibration_match(
    instruction: Union[quil_rs.Gate, quil_rs.Measurement],
    calibration: Union[quil_rs.Calibration, quil_rs.MeasureCalibrationDefinition],
) -> Optional[CalibrationMatch]:
    if isinstance(instruction, quil_rs.Gate) and isinstance(calibration, quil_rs.Calibration):
        target_qubits = instruction.qubits
        target_values = instruction.parameters
        parameter_qubits = calibration.qubits
        parameter_values = calibration.parameters
        py_calibration = DefCalibration._from_rs_calibration(calibration)
    elif isinstance(instruction, quil_rs.Measurement) and isinstance(calibration, quil_rs.MeasureCalibrationDefinition):
        target_qubits = [instruction.qubit]
        target_values = (
            [] if not instruction.target else [MemoryReference._from_rs_memory_reference(instruction.target)]
        )
        parameter_qubits = [] if not calibration.qubit else [calibration.qubit]
        parameter_values = [MemoryReference._from_parameter_str(calibration.parameter)]
        py_calibration = DefMeasureCalibration._from_rs_measure_calibration_definition(calibration)
    else:
        return None

    settings = {
        _convert_to_py_qubit(param): _convert_to_py_qubit(qubit)
        for param, qubit in zip(parameter_qubits, target_qubits)
        if isinstance(param, MemoryReference) or param.is_variable()
    }
    settings.update(
        {
            _convert_to_py_expression(param): _convert_to_py_expression(value)
            for param, value in zip(parameter_values, target_values)
            if isinstance(param, MemoryReference) or param.is_variable()
        }
    )

    return CalibrationMatch(py_calibration, settings)


def match_calibration(
    instr: AbstractInstruction, cal: Union[DefCalibration, DefMeasureCalibration]
) -> Optional[CalibrationMatch]:
    """Match a calibration definition to an instruction.

    On a successful match, return a (possibly empty) dictionary mapping calibration
    arguments and parameters to their values.

    On a failure, return None.
    """
    calibration = _convert_to_rs_instruction(cal)
    instruction = _convert_to_rs_instruction(instr)
    if calibration.is_calibration_definition() and instruction.is_gate():
        instruction = _convert_to_rs_instruction(instr)
        gate = instruction.to_gate()
        calibration_set = CalibrationSet([calibration.to_calibration_definition()], [])
        matched_calibration = calibration_set.get_match_for_gate(gate)
        return _convert_to_calibration_match(gate, matched_calibration)

    if calibration.is_measure_calibration_definition() and instruction.is_measurement():
        instruction = _convert_to_rs_instruction(instr)
        measurement = instruction.to_measurement()
        calibration_set = CalibrationSet([], [calibration.to_measure_calibration_definition()])
        matched_calibration = calibration_set.get_match_for_measurement(measurement)
        return _convert_to_calibration_match(measurement, matched_calibration)

    return None


def expand_calibration(match: CalibrationMatch) -> List[AbstractInstruction]:
    """ " Expand the body of a calibration from a match."""
    return [fill_placeholders(instr, match.settings) for instr in match.cal.instrs]

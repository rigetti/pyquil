"""A module containing utilities for working with Quil-T calibrations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from quil.expression import Expression, ExpressionDesignator, Parameter
from quil.instructions import (
    CalibrationDefinition,
    FormalArgument,
    Gate,
    Instruction,
    MeasureCalibrationDefinition,
    Measurement,
    MemoryReference,
    Qubit,
    QubitDesignator,
)
from quil.program import CalibrationSet


class CalibrationError(Exception):
    """Base class for calibration errors."""

    pass


class CalibrationDoesntMatch(CalibrationError):
    """Raised when a calibration doesn't match an instruction."""

    pass


@dataclass
class CalibrationMatch:
    """A match between a calibration definition and an instruction."""

    cal: CalibrationDefinition | MeasureCalibrationDefinition
    settings: dict[QubitDesignator | ExpressionDesignator, Any]


def _convert_to_calibration_match(
    instruction: Gate | Measurement,
    calibration: CalibrationDefinition | MeasureCalibrationDefinition | None,
) -> CalibrationMatch | None:
    if isinstance(instruction, Gate) and isinstance(calibration, CalibrationDefinition):
        target_qubits = instruction.qubits
        target_values: Sequence[Expression | MemoryReference] = instruction.parameters
        parameter_qubits = calibration.identifier.qubits
        parameter_values: Sequence[Expression | MemoryReference] = calibration.identifier.parameters
        py_calibration: CalibrationDefinition | MeasureCalibrationDefinition = calibration
    elif isinstance(instruction, Measurement) and isinstance(calibration, MeasureCalibrationDefinition):
        target_qubits = [instruction.qubit]
        target_values = [] if not instruction.target else [instruction.target]
        calibration_qubit = calibration.identifier.qubit
        parameter_qubits = [] + [calibration_qubit] if calibration_qubit else []
        parameter_values = [MemoryReference(calibration.identifier.target)]
        py_calibration = calibration
    else:
        return None

    settings: dict[MemoryReference | FormalArgument | Parameter, Expression | MemoryReference | Qubit] = {
        param: qubit
        for param, qubit in zip(parameter_qubits, target_qubits, strict=False)
        if isinstance(param, MemoryReference) or isinstance(param, FormalArgument)
    }
    settings.update(
        {
            param: value
            for param, value in zip(parameter_values, target_values, strict=False)
            if isinstance(param, MemoryReference) or isinstance(param, Parameter)
        }
    )

    return CalibrationMatch(py_calibration, settings)


def match_calibration(
    instr: Instruction, cal: CalibrationDefinition | MeasureCalibrationDefinition
) -> CalibrationMatch | None:
    """Match a calibration definition to an instruction.

    On a successful match, return a (possibly empty) dictionary mapping calibration
    arguments and parameters to their values.

    On a failure, return None.
    """
    if isinstance(cal, CalibrationDefinition) and isinstance(instr, Gate):
        calibration_set = CalibrationSet([cal], [])
        matched_calibration = calibration_set.get_match_for_gate(instr)
        return _convert_to_calibration_match(instr, matched_calibration)
    elif isinstance(cal, MeasureCalibrationDefinition) and isinstance(instr, Measurement):
        calibration_set = CalibrationSet([], [cal])
        matched_calibration = calibration_set.get_match_for_measurement(instr)
        return _convert_to_calibration_match(instr, matched_calibration)

    return None

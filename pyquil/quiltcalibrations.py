"""A module containing utilities for working with Quil-T calibrations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import quil.expression as quil_expr
import quil.instructions as quil_rs
from quil.program import CalibrationSet

from pyquil.quilatom import (
    ExpressionDesignator,
    MemoryReference,
    QubitDesignator,
    _convert_to_py_expression,
)
from pyquil.quilbase import (
    AbstractInstruction,
    DefCalibration,
    DefMeasureCalibration,
    _convert_to_py_qubit,
    _convert_to_rs_instruction,
)


class CalibrationError(Exception):
    """Base class for calibration errors."""

    pass


class CalibrationDoesntMatch(CalibrationError):
    """Raised when a calibration doesn't match an instruction."""

    pass


@dataclass
class CalibrationMatch:
    """A match between a calibration definition and an instruction."""

    cal: Union[DefCalibration, DefMeasureCalibration]
    settings: dict[Union[QubitDesignator, ExpressionDesignator], Any]


def _convert_to_calibration_match(
    instruction: Union[quil_rs.Gate, quil_rs.Measurement],
    calibration: Optional[Union[quil_rs.CalibrationDefinition, quil_rs.MeasureCalibrationDefinition]],
) -> Optional[CalibrationMatch]:
    if isinstance(instruction, quil_rs.Gate) and isinstance(calibration, quil_rs.CalibrationDefinition):
        target_qubits = instruction.qubits
        target_values: Sequence[Union[quil_expr.Expression, MemoryReference]] = instruction.parameters
        parameter_qubits = calibration.identifier.qubits
        parameter_values: Sequence[Union[quil_expr.Expression, MemoryReference]] = calibration.identifier.parameters
        py_calibration: Union[DefCalibration, DefMeasureCalibration] = DefCalibration._from_rs_calibration(calibration)
    elif isinstance(instruction, quil_rs.Measurement) and isinstance(calibration, quil_rs.MeasureCalibrationDefinition):
        target_qubits = [instruction.qubit]
        target_values = (
            [] if not instruction.target else [MemoryReference._from_rs_memory_reference(instruction.target)]
        )
        calibration_qubit = calibration.identifier.qubit
        parameter_qubits = [] + [calibration_qubit] if calibration_qubit else []
        parameter_values = [MemoryReference._from_parameter_str(calibration.identifier.target)]
        py_calibration = DefMeasureCalibration._from_rs_measure_calibration_definition(calibration)
    else:
        return None

    settings: dict[Union[QubitDesignator, ExpressionDesignator], Union[QubitDesignator, ExpressionDesignator]] = {
        _convert_to_py_qubit(param): _convert_to_py_qubit(qubit)
        for param, qubit in zip(parameter_qubits, target_qubits)
        if isinstance(param, MemoryReference) or isinstance(param, quil_rs.Qubit.Variable)
    }
    settings.update(
        {
            _convert_to_py_expression(param): _convert_to_py_expression(value)
            for param, value in zip(parameter_values, target_values)
            if isinstance(param, MemoryReference) or isinstance(param, quil_expr.Expression.Variable)
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

    match (calibration, instruction):
        case (quil_rs.Instruction.CalibrationDefinition(definition), quil_rs.Instruction.Gate(gate)):
            calibration_set = CalibrationSet([definition], [])
            matched_calibration = calibration_set.get_match_for_gate(gate)
            return _convert_to_calibration_match(gate, matched_calibration)

        case (
            quil_rs.Instruction.MeasureCalibrationDefinition(definition),
            quil_rs.Instruction.Measurement(measurement),
        ):
            calibration_set = CalibrationSet([], [definition])
            matched_calibration = calibration_set.get_match_for_measurement(measurement)
            return _convert_to_calibration_match(measurement, matched_calibration)

    return None

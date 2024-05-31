"""Tools for estimating the expectation value of operators on a quantum computer."""

import logging
from collections.abc import Generator, Mapping
from math import pi
from numbers import Complex
from typing import Callable, Optional, cast

import numpy as np

from pyquil.api import QuantumComputer

# import the full public API of the pyquil experiment module
from pyquil.experiment._group import (
    _max_weight_operator,
    _max_weight_state,
)
from pyquil.experiment._group import (
    group_settings as group_experiments,
)
from pyquil.experiment._group import (
    group_settings_clique_removal as group_experiments_clique_removal,
)
from pyquil.experiment._group import (
    group_settings_greedy as group_experiments_greedy,
)
from pyquil.experiment._main import (
    Experiment,
)
from pyquil.experiment._result import ExperimentResult, ratio_variance
from pyquil.experiment._setting import (
    ExperimentSetting,
    _OneQState,
)
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.gates import RESET, RX, RY, RZ, X
from pyquil.paulis import is_identity
from pyquil.quil import Program
from pyquil.quilatom import QubitDesignator

log = logging.getLogger(__name__)

__all__ = [
    "group_experiments",
    "group_experiments_clique_removal",
    "group_experiments_greedy",
    "measure_observables",
]


def _one_q_sic_prep(index: int, qubit: QubitDesignator) -> Program:
    """Prepare the index-th SIC basis state."""
    if index == 0:
        return Program()

    theta = 2 * np.arccos(1 / np.sqrt(3))
    zx_plane_rotation = Program([RX(-pi / 2, qubit), RZ(theta - pi, qubit), RX(-pi / 2, qubit)])

    if index == 1:
        return zx_plane_rotation

    elif index == 2:
        return zx_plane_rotation + RZ(-2 * pi / 3, qubit)

    elif index == 3:
        return zx_plane_rotation + RZ(2 * pi / 3, qubit)

    raise ValueError(f"Bad SIC index: {index}")


def _one_q_pauli_prep(label: str, index: int, qubit: QubitDesignator) -> Program:
    """Prepare the index-th eigenstate of the pauli operator given by label."""
    if index not in [0, 1]:
        raise ValueError(f"Bad Pauli index: {index}")

    if label == "X":
        if index == 0:
            return Program(RY(pi / 2, qubit))
        else:
            return Program(RY(-pi / 2, qubit))

    elif label == "Y":
        if index == 0:
            return Program(RX(-pi / 2, qubit))
        else:
            return Program(RX(pi / 2, qubit))

    elif label == "Z":
        if index == 0:
            return Program()
        else:
            return Program(RX(pi, qubit))

    raise ValueError(f"Bad Pauli label: {label}")


def _one_q_state_prep(oneq_state: _OneQState) -> Program:
    """Prepare a one qubit state.

    Either SIC[0-3], X[0-1], Y[0-1], or Z[0-1].
    """
    label = oneq_state.label
    if label == "SIC":
        return _one_q_sic_prep(oneq_state.index, oneq_state.qubit)
    elif label in ["X", "Y", "Z"]:
        return _one_q_pauli_prep(label, oneq_state.index, oneq_state.qubit)
    else:
        raise ValueError(f"Bad state label: {label}")


def _local_pauli_eig_meas(op: str, idx: QubitDesignator) -> Program:
    """Generate gate sequence to measure in a Pauli operator's eigenbasis using only Z eigenbasis measurements.

    Note: The unitary operations of this Program are essentially the Hermitian conjugates of those in
    :py:func:`_one_q_pauli_prep`
    """
    if op == "X":
        return Program(RY(-pi / 2, idx))
    elif op == "Y":
        return Program(RX(pi / 2, idx))
    elif op == "Z":
        return Program()
    raise ValueError(f"Unknown operation {op}")


def _generate_experiment_programs(
    tomo_experiment: Experiment, active_reset: bool = False
) -> tuple[list[Program], list[list[int]]]:
    """Generate the programs necessary to estimate the observables in a TomographyExperiment.

    Grouping of settings to be run in parallel, e.g. by a call to group_experiments, should be done before this function
    is called.

    .. CAUTION::
        One must be careful with compilation of the output programs before the appropriate MEASURE
        instructions are added, because compilation may re-index the qubits so that
        the output list of `measure_qubits` no longer accurately indexes the qubits that
        should be measured.

    :param tomo_experiment: a single TomographyExperiment to be translated to a series of programs
        that, when run serially, can be used to estimate each of its observables.
    :param active_reset: whether or not to begin the program by actively resetting. If true,
        execution of each of the returned programs in a loop on the QPU will generally be faster.
    :return: a list of programs along with a corresponding list of the groups of qubits that are
        measured by that program. The returned programs may be run on a qc after measurement
        instructions are added for the corresponding group of qubits in meas_qubits, or by a call
        to `qc.run_symmetrized_readout` -- see :func:`raw_estimate_observables` for possible usage.
    """
    # Outer loop over a collection of grouped settings for which we can simultaneously estimate.
    programs = []
    meas_qubits = []
    for settings in tomo_experiment:
        # Prepare a state according to the amalgam of all setting.in_state
        total_prog = Program()
        if active_reset:
            total_prog += RESET()
        max_weight_in_state = _max_weight_state(setting.in_state for setting in settings)
        if max_weight_in_state is None:
            raise ValueError(
                "Input states are not compatible. Re-group the experiment settings "
                "so that groups of parallel settings have compatible input states."
            )
        for oneq_state in max_weight_in_state.states:
            total_prog += _one_q_state_prep(oneq_state)

        # Add in the program
        total_prog += tomo_experiment.program

        # Prepare for measurement state according to setting.out_operator
        max_weight_out_op = _max_weight_operator(setting.out_operator for setting in settings)
        if max_weight_out_op is None:
            raise ValueError(
                "Observables not compatible. Re-group the experiment settings "
                "so that groups of parallel settings have compatible observables."
            )
        for qubit, op_str in max_weight_out_op:
            if not isinstance(qubit, int):
                raise TypeError("Qubit must be an integer.")
            total_prog += _local_pauli_eig_meas(op_str, qubit)

        programs.append(total_prog)

        meas_qubits.append(cast(list[int], max_weight_out_op.get_qubits()))
    return programs, meas_qubits


def measure_observables(
    qc: QuantumComputer,
    tomo_experiment: Experiment,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    calibrate_readout: Optional[str] = "plus-eig",
) -> Generator[ExperimentResult, None, None]:
    """Measure all the observables in a TomographyExperiment.

    :param qc: A QuantumComputer which can run quantum programs
    :param tomo_experiment: A suite of tomographic observables to measure
    :param progress_callback: If not None, this function is called each time a group of
        settings is run with arguments ``f(i, len(tomo_experiment)`` such that the progress
        is ``i / len(tomo_experiment)``.
    :param calibrate_readout: Method used to calibrate the readout results. Currently, the only
        method supported is normalizing against the operator's expectation value in its +1
        eigenstate, which can be specified by setting this variable to 'plus-eig' (default value).
        The preceding symmetrization and this step together yield a more accurate estimation of
        the observable. Set to `None` if no calibration is desired.
    """
    shots = tomo_experiment.shots
    symmetrization = tomo_experiment.symmetrization
    reset = tomo_experiment.reset

    # calibration readout only works with symmetrization turned on
    if calibrate_readout is not None and symmetrization != SymmetrizationLevel.EXHAUSTIVE:
        raise ValueError(
            "Readout calibration only currently works with exhaustive readout " "symmetrization turned on."
        )

    # generate programs for each group of simultaneous settings.
    programs, meas_qubits = _generate_experiment_programs(tomo_experiment, reset)

    for i, (prog, qubits, settings) in enumerate(zip(programs, meas_qubits, tomo_experiment)):
        log.info(f"Collecting bitstrings for the {len(settings)} settings: {settings}")

        # we don't need to do any actual measurement if the combined operator is simply the
        # identity, i.e. weight=0. We handle this specially below.
        if len(qubits) > 0:
            # obtain (optionally symmetrized) bitstring results for all of the qubits
            bitstrings = qc.run_symmetrized_readout(prog, shots, symmetrization, qubits)

        if progress_callback is not None:
            progress_callback(i, len(tomo_experiment))

        # Post-process
        # Inner loop over the grouped settings. They only differ in which qubits' measurements
        # we include in the post-processing. For example, if `settings` is Z1, Z2, Z1Z2 and we
        # measure (shots, n_qubits=2) obs_strings then the full operator value involves selecting
        # either the first column, second column, or both and multiplying along the row.
        for setting in settings:
            # Get the term's coefficient so we can multiply it in later.
            coeff = setting.out_operator.coefficient
            if not isinstance(coeff, Complex):
                raise TypeError("Coefficient must be a complex number.")
            if not np.isclose(coeff.imag, 0):
                raise ValueError(f"{setting}'s out_operator has a complex coefficient.")
            coeff = coeff.real

            # Special case for measuring the "identity" operator, which doesn't make much
            #     sense but should happen perfectly.
            if is_identity(setting.out_operator):
                yield ExperimentResult(setting=setting, expectation=coeff, std_err=0.0, total_counts=shots)
                continue

            # Obtain statistics from result of experiment
            obs_mean, obs_var = _stats_from_measurements(
                bitstrings, {q: idx for idx, q in enumerate(qubits)}, setting, shots, coeff
            )

            if calibrate_readout == "plus-eig":
                # Readout calibration
                # Obtain calibration program
                calibr_prog = _calibration_program(qc, tomo_experiment, setting)
                calibr_qubs = setting.out_operator.get_qubits()
                calibr_qub_dict = {cast(int, q): idx for idx, q in enumerate(calibr_qubs)}

                # Perform symmetrization on the calibration program
                calibr_results = qc.run_symmetrized_readout(
                    calibr_prog, shots, SymmetrizationLevel.EXHAUSTIVE, calibr_qubs
                )

                # Obtain statistics from the measurement process
                obs_calibr_mean, obs_calibr_var = _stats_from_measurements(
                    calibr_results, calibr_qub_dict, setting, shots
                )
                # Calibrate the readout results
                corrected_mean = obs_mean / obs_calibr_mean
                corrected_var = ratio_variance(obs_mean, obs_var, obs_calibr_mean, obs_calibr_var)

                yield ExperimentResult(
                    setting=setting,
                    expectation=corrected_mean.item(),
                    std_err=np.sqrt(corrected_var).item(),
                    total_counts=len(bitstrings),
                    raw_expectation=obs_mean.item(),
                    raw_std_err=np.sqrt(obs_var).item(),
                    calibration_expectation=obs_calibr_mean.item(),
                    calibration_std_err=np.sqrt(obs_calibr_var).item(),
                    calibration_counts=len(calibr_results),
                )

            elif calibrate_readout is None:
                # No calibration
                yield ExperimentResult(
                    setting=setting,
                    expectation=obs_mean.item(),
                    std_err=np.sqrt(obs_var).item(),
                    total_counts=len(bitstrings),
                )

            else:
                raise ValueError("Calibration readout method must be either 'plus-eig' or None")


def _ops_bool_to_prog(ops_bool: tuple[bool], qubits: list[int]) -> Program:
    """Specify a program based on the operations to be carried out on the qubits.

    :param ops_bool: tuple of booleans specifying the operation to be carried out on `qubits`
    :param qubits: list specifying the qubits to be carried operations on
    :return: Program with the operations specified in `ops_bool` on the qubits specified in
        `qubits`
    """
    if len(ops_bool) != len(qubits):
        raise ValueError("Length of ops_bool must match length of qubits.")
    prog = Program()
    for i, op_bool in enumerate(ops_bool):
        if op_bool == 0:
            continue
        elif op_bool == 1:
            prog += Program(X(qubits[i]))
        else:
            raise ValueError("ops_bool should only consist of 0s and/or 1s")
    return prog


def _stats_from_measurements(
    bs_results: np.ndarray,
    qubit_index_map: Mapping[int, int],
    setting: ExperimentSetting,
    n_shots: int,
    coeff: float = 1.0,
) -> tuple[np.number, np.number]:
    """Calculate statistics from the results of a measurement process.

    :param bs_results: results from running `qc.run`
    :param qubit_index_map: dict mapping qubit to classical register index
    :param setting: ExperimentSetting
    :param n_shots: number of shots in the measurement process
    :param coeff: coefficient of the operator being estimated
    :return: tuple specifying (mean, variance)
    """
    # Identify classical register indices to select
    idxs = [qubit_index_map[cast(int, q)] for q, _ in setting.out_operator]
    # Pick columns corresponding to qubits with a non-identity out_operation
    obs_strings = bs_results[:, idxs]
    # Transform bits to eigenvalues; ie (+1, -1)
    my_obs_strings = 1 - 2 * obs_strings
    # Multiply row-wise to get operator values. Do statistics. Return result.
    obs_vals = coeff * np.prod(my_obs_strings, axis=1)
    obs_mean = np.mean(obs_vals)
    obs_var = np.var(obs_vals) / n_shots

    return obs_mean, obs_var


def _calibration_program(qc: QuantumComputer, tomo_experiment: Experiment, setting: ExperimentSetting) -> Program:
    """Program required for calibration in a tomography-like experiment.

    :param tomo_experiment: A suite of tomographic observables
    :param ExperimentSetting: The particular tomographic observable to measure
    :param symmetrize_readout: Method used to symmetrize the readout errors (see docstring for
        `measure_observables` for more details)
    :param cablir_shots: number of shots to take in the measurement process
    :return: Program performing the calibration
    """
    # Inherit any noisy attributes from main Program, including gate definitions
    # and applications which can be handy in creating simulating noisy channels
    calibr_prog = Program()
    # Inherit readout error instructions from main Program
    readout_povm_instruction = [i for i in tomo_experiment.program.out().split("\n") if "PRAGMA READOUT-POVM" in i]
    calibr_prog += readout_povm_instruction
    # Inherit any definitions of noisy gates from main Program
    kraus_instructions = [i for i in tomo_experiment.program.out().split("\n") if "PRAGMA ADD-KRAUS" in i]
    calibr_prog += kraus_instructions
    # Prepare the +1 eigenstate for the out operator
    for q, op in setting.out_operator.operations_as_set():
        if not isinstance(q, int):
            raise TypeError("Qubit must be an integer.")
        calibr_prog += _one_q_pauli_prep(label=op, index=0, qubit=q)
    # Measure the out operator in this state
    for q, op in setting.out_operator.operations_as_set():
        if not isinstance(q, int):
            raise TypeError("Qubit must be an integer.")
        calibr_prog += _local_pauli_eig_meas(op, q)

    return calibr_prog

import functools
import itertools
import logging
import warnings
from math import pi
from operator import mul
from typing import Callable, Dict, List, Union, Iterable, Tuple, Optional

import networkx as nx
import numpy as np
from networkx.algorithms.approximation.clique import clique_removal

from pyquil import Program
from pyquil.api import QuantumComputer
# import the full public API of the pyquil experiment module
from pyquil.experiment import (_OneQState, _pauli_to_product_state, ExperimentResult,
                               ExperimentSetting, OperatorEncoder, SIC0, SIC1, SIC2, SIC3,
                               SymmetrizationLevel, TomographyExperiment, TensorProductState,
                               minusX, minusY, minusZ, plusX, plusY, plusZ, read_json, to_json,
                               zeros_state)
from pyquil.gates import RESET, RX, RY, RZ, X
from pyquil.paulis import PauliTerm, sI, is_identity


log = logging.getLogger(__name__)


def _one_q_sic_prep(index, qubit):
    """Prepare the index-th SIC basis state."""
    if index == 0:
        return Program()

    theta = 2 * np.arccos(1 / np.sqrt(3))
    zx_plane_rotation = Program([
        RX(-pi / 2, qubit),
        RZ(theta - pi, qubit),
        RX(-pi / 2, qubit),
    ])

    if index == 1:
        return zx_plane_rotation

    elif index == 2:
        return zx_plane_rotation + RZ(-2 * pi / 3, qubit)

    elif index == 3:
        return zx_plane_rotation + RZ(2 * pi / 3, qubit)

    raise ValueError(f'Bad SIC index: {index}')


def _one_q_pauli_prep(label, index, qubit):
    """Prepare the index-th eigenstate of the pauli operator given by label."""
    if index not in [0, 1]:
        raise ValueError(f'Bad Pauli index: {index}')

    if label == 'X':
        if index == 0:
            return Program(RY(pi / 2, qubit))
        else:
            return Program(RY(-pi / 2, qubit))

    elif label == 'Y':
        if index == 0:
            return Program(RX(-pi / 2, qubit))
        else:
            return Program(RX(pi / 2, qubit))

    elif label == 'Z':
        if index == 0:
            return Program()
        else:
            return Program(RX(pi, qubit))

    raise ValueError(f'Bad Pauli label: {label}')


def _one_q_state_prep(oneq_state: _OneQState):
    """Prepare a one qubit state.

    Either SIC[0-3], X[0-1], Y[0-1], or Z[0-1].
    """
    label = oneq_state.label
    if label == 'SIC':
        return _one_q_sic_prep(oneq_state.index, oneq_state.qubit)
    elif label in ['X', 'Y', 'Z']:
        return _one_q_pauli_prep(label, oneq_state.index, oneq_state.qubit)
    else:
        raise ValueError(f"Bad state label: {label}")


def _local_pauli_eig_meas(op, idx):
    """
    Generate gate sequence to measure in the eigenbasis of a Pauli operator, assuming
    we are only able to measure in the Z eigenbasis. (Note: The unitary operations of this
    Program are essentially the Hermitian conjugates of those in :py:func:`_one_q_pauli_prep`)

    """
    if op == 'X':
        return Program(RY(-pi / 2, idx))
    elif op == 'Y':
        return Program(RX(pi / 2, idx))
    elif op == 'Z':
        return Program()
    raise ValueError(f'Unknown operation {op}')


def construct_tpb_graph(experiments: TomographyExperiment):
    """
    Construct a graph where an edge signifies two experiments are diagonal in a TPB.
    """
    g = nx.Graph()
    for expt in experiments:
        assert len(expt) == 1, 'already grouped?'
        expt = expt[0]

        if expt not in g:
            g.add_node(expt, count=1)
        else:
            g.nodes[expt]['count'] += 1

    for expt1, expt2 in itertools.combinations(experiments, r=2):
        expt1 = expt1[0]
        expt2 = expt2[0]

        if expt1 == expt2:
            continue

        max_weight_in = _max_weight_state([expt1.in_state, expt2.in_state])
        max_weight_out = _max_weight_operator([expt1.out_operator, expt2.out_operator])
        if max_weight_in is not None and max_weight_out is not None:
            g.add_edge(expt1, expt2)

    return g


def group_experiments_clique_removal(experiments: TomographyExperiment) -> TomographyExperiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs, using a graph clique removal algorithm.

    :param experiments: a tomography experiment
    :return: a tomography experiment with all the same settings, just grouped according to shared
        TPBs.
    """
    g = construct_tpb_graph(experiments)
    _, cliqs = clique_removal(g)
    new_cliqs = []
    for cliq in cliqs:
        new_cliq = []
        for expt in cliq:
            # duplicate `count` times
            new_cliq += [expt] * g.nodes[expt]['count']

        new_cliqs += [new_cliq]

    return TomographyExperiment(new_cliqs, program=experiments.program)


def _max_weight_operator(ops: Iterable[PauliTerm]) -> Union[None, PauliTerm]:
    """Construct a PauliTerm operator by taking the non-identity single-qubit operator at each
    qubit position.

    This function will return ``None`` if the input operators do not share a natural tensor
    product basis.

    For example, the max_weight_operator of ["XI", "IZ"] is "XZ". Asking for the max weight
    operator of something like ["XI", "ZI"] will return None.
    """
    mapping = dict()  # type: Dict[int, str]
    for op in ops:
        for idx, op_str in op:
            if idx in mapping:
                if mapping[idx] != op_str:
                    return None
            else:
                mapping[idx] = op_str
    op = functools.reduce(mul, (PauliTerm(op, q) for q, op in mapping.items()), sI())
    return op


def _max_weight_state(states: Iterable[TensorProductState]) -> Union[None, TensorProductState]:
    """Construct a TensorProductState by taking the single-qubit state at each
    qubit position.

    This function will return ``None`` if the input states are not compatible

    For example, the max_weight_state of ["(+X, q0)", "(-Z, q1)"] is "(+X, q0; -Z q1)". Asking for
    the max weight state of something like ["(+X, q0)", "(+Z, q0)"] will return None.
    """
    mapping = dict()  # type: Dict[int, _OneQState]
    for state in states:
        for oneq_state in state.states:
            if oneq_state.qubit in mapping:
                if mapping[oneq_state.qubit] != oneq_state:
                    return None
            else:
                mapping[oneq_state.qubit] = oneq_state
    return TensorProductState(list(mapping.values()))


def _max_tpb_overlap(tomo_expt: TomographyExperiment):
    """
    Given an input TomographyExperiment, provide a dictionary indicating which ExperimentSettings
    share a tensor product basis

    :param tomo_expt: TomographyExperiment, from which to group ExperimentSettings that share a tpb
        and can be run together
    :return: dictionary keyed with ExperimentSetting (specifying a tpb), and with each value being a
            list of ExperimentSettings (diagonal in that tpb)
    """
    # initialize empty dictionary
    diagonal_sets = {}
    # loop through ExperimentSettings of the TomographyExperiment
    for expt_setting in tomo_expt:
        # no need to group already grouped TomographyExperiment
        assert len(expt_setting) == 1, 'already grouped?'
        expt_setting = expt_setting[0]
        # calculate max overlap of expt_setting with keys of diagonal_sets
        # keep track of whether a shared tpb was found
        found_tpb = False
        # loop through dict items
        for es, es_list in diagonal_sets.items():
            trial_es_list = es_list + [expt_setting]
            diag_in_term = _max_weight_state(expst.in_state for expst in trial_es_list)
            diag_out_term = _max_weight_operator(expst.out_operator for expst in trial_es_list)
            # max_weight_xxx returns None if the set of xxx's don't share a TPB, so the following
            # conditional is True if expt_setting can be inserted into the current es_list.
            if diag_in_term is not None and diag_out_term is not None:
                found_tpb = True
                assert len(diag_in_term) >= len(es.in_state), \
                    "Highest weight in-state can't be smaller than the given in-state"
                assert len(diag_out_term) >= len(es.out_operator), \
                    "Highest weight out-PauliTerm can't be smaller than the given out-PauliTerm"

                # update the diagonalizing basis (key of dict) if necessary
                if len(diag_in_term) > len(es.in_state) or len(diag_out_term) > len(es.out_operator):
                    del diagonal_sets[es]
                    new_es = ExperimentSetting(diag_in_term, diag_out_term)
                    diagonal_sets[new_es] = trial_es_list
                else:
                    diagonal_sets[es] = trial_es_list
                break

        if not found_tpb:
            # made it through entire dict without finding any ExperimentSetting with shared tpb,
            # so need to make a new item
            diagonal_sets[expt_setting] = [expt_setting]

    return diagonal_sets


def group_experiments_greedy(tomo_expt: TomographyExperiment):
    """
    Greedy method to group ExperimentSettings in a given TomographyExperiment

    :param tomo_expt: TomographyExperiment to group ExperimentSettings within
    :return: TomographyExperiment, with grouped ExperimentSettings according to whether
        it consists of PauliTerms diagonal in the same tensor product basis
    """
    diag_sets = _max_tpb_overlap(tomo_expt)
    grouped_expt_settings_list = list(diag_sets.values())
    grouped_tomo_expt = TomographyExperiment(grouped_expt_settings_list, program=tomo_expt.program)
    return grouped_tomo_expt


def group_experiments(experiments: TomographyExperiment,
                      method: str = 'greedy') -> TomographyExperiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs.

    .. rubric:: Background

    Given some PauliTerm operator, the 'natural' tensor product basis to
    diagonalize this term is the one which diagonalizes each Pauli operator in the
    product term-by-term.

    For example, X(1) * Z(0) would be diagonal in the 'natural' tensor product basis
    ``{(|0> +/- |1>)/Sqrt[2]} * {|0>, |1>}``, whereas Z(1) * X(0) would be diagonal
    in the 'natural' tpb ``{|0>, |1>} * {(|0> +/- |1>)/Sqrt[2]}``. The two operators
    commute but are not diagonal in each others 'natural' tpb (in fact, they are
    anti-diagonal in each others 'natural' tpb). This function tests whether two
    operators given as PauliTerms are both diagonal in each others 'natural' tpb.

    Note that for the given example of X(1) * Z(0) and Z(1) * X(0), we can construct
    the following basis which simultaneously diagonalizes both operators::

      -- |0>' = |0> (|+>) + |1> (|->)
      -- |1>' = |0> (|+>) - |1> (|->)
      -- |2>' = |0> (|->) + |1> (|+>)
      -- |3>' = |0> (-|->) + |1> (|+>)

    In this basis, X Z looks like diag(1, -1, 1, -1), and Z X looks like diag(1, 1, -1, -1).
    Notice however that this basis cannot be constructed with single-qubit operations, as each
    of the basis vectors are entangled states.


    .. rubric:: Methods

    The "greedy" method will keep a running set of 'buckets' into which grouped ExperimentSettings
    will be placed. Each new ExperimentSetting considered is assigned to the first applicable
    bucket and a new bucket is created if there are no applicable buckets.

    The "clique-removal" method maps the term grouping problem onto Max Clique graph problem.
    This method constructs a NetworkX graph where an edge exists between two settings that
    share an nTPB and then uses networkx's algorithm for clique removal. This method can give
    you marginally better groupings in certain circumstances, but constructing the
    graph is pretty slow so "greedy" is the default.

    :param experiments: a tomography experiment
    :param method: method used for grouping; the allowed methods are one of
        ['greedy', 'clique-removal']
    :return: a tomography experiment with all the same settings, just grouped according to shared
        TPBs.
    """
    allowed_methods = ['greedy', 'clique-removal']
    assert method in allowed_methods, f"'method' should be one of {allowed_methods}."
    if method == 'greedy':
        return group_experiments_greedy(experiments)
    elif method == 'clique-removal':
        return group_experiments_clique_removal(experiments)


def _generate_experiment_programs(
    tomo_experiment: TomographyExperiment,
    active_reset: bool = False,
) -> Tuple[List[Program], List[List[int]]]:
    """
    Generate the programs necessary to estimate the observables in a TomographyExperiment.
    Grouping of settings to be run in parallel, e.g. by a call to group_experiments, should be
    done before this function is called.

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
                'Input states are not compatible. Re-group the experiment settings '
                'so that groups of parallel settings have compatible input states.')
        for oneq_state in max_weight_in_state.states:
            total_prog += _one_q_state_prep(oneq_state)

        # Add in the program
        total_prog += tomo_experiment.program

        # Prepare for measurement state according to setting.out_operator
        max_weight_out_op = _max_weight_operator(setting.out_operator for setting in settings)
        if max_weight_out_op is None:
            raise ValueError('Observables not compatible. Re-group the experiment settings '
                             'so that groups of parallel settings have compatible observables.')
        for qubit, op_str in max_weight_out_op:
            total_prog += _local_pauli_eig_meas(op_str, qubit)

        programs.append(total_prog)

        meas_qubits.append(max_weight_out_op.get_qubits())
    return programs, meas_qubits


def measure_observables(qc: QuantumComputer,
                        tomo_experiment: TomographyExperiment,
                        n_shots: Optional[int] = None,
                        progress_callback: Optional[Callable[[int, int], None]] = None,
                        active_reset: Optional[bool] = None,
                        symmetrize_readout: Optional[Union[int, str]] = 'None',
                        calibrate_readout: Optional[str] = 'plus-eig',
                        readout_symmetrize: Optional[str] = None):
    """
    Measure all the observables in a TomographyExperiment.

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

    if n_shots is not None:
        warnings.warn("'n_shots' has been deprecated; if you want to set the number of shots "
                      "for this run of measure_observables please provide the number to "
                      "Program.wrap_in_numshots_loop() for the Quil program that you provide "
                      "when creating your TomographyExperiment object. For now, this value will "
                      "override that in the TomographyExperiment, but eventually this keyword "
                      "argument will be removed.",
                      FutureWarning)
        shots = n_shots
    else:
        if shots == 1:
            warnings.warn("'n_shots' has been deprecated; if you want to set the number of shots "
                          "for this run of measure_observables please provide the number to "
                          "Program.wrap_in_numshots_loop() for the Quil program that you provide "
                          "when creating your TomographyExperiment object. It looks like your "
                          "TomographyExperiment object has shots = 1, so for now we will change "
                          "that to 10000, which was the previous default value.",
                          FutureWarning)
            shots = 10000

    if active_reset is not None:
        warnings.warn("'active_reset' has been deprecated; if you want to enable active qubit "
                      "reset please provide a Quil program that has a RESET instruction in it when "
                      "creating your TomographyExperiment object. For now, this value will "
                      "override that in the TomographyExperiment, but eventually this keyword "
                      "argument will be removed.",
                      FutureWarning)
        reset = active_reset

    if readout_symmetrize is not None and symmetrize_readout != 'None':
        raise ValueError("'readout_symmetrize' and 'symmetrize_readout' are conflicting keyword "
                         "arguments -- please provide only one.")

    if readout_symmetrize is not None:
        warnings.warn("'readout_symmetrize' has been deprecated; please provide the symmetrization "
                      "level when creating your TomographyExperiment object. For now, this value "
                      "will override that in the TomographyExperiment, but eventually this keyword "
                      "argument will be removed.",
                      FutureWarning)
        symmetrization = SymmetrizationLevel(readout_symmetrize)

    if symmetrize_readout != 'None':
        warnings.warn("'symmetrize_readout' has been deprecated; please provide the symmetrization "
                      "level when creating your TomographyExperiment object. For now, this value "
                      "will override that in the TomographyExperiment, but eventually this keyword "
                      "argument will be removed.",
                      FutureWarning)
        if symmetrize_readout is None:
            symmetrize_readout = SymmetrizationLevel.NONE
        elif symmetrize_readout == 'exhaustive':
            symmetrize_readout = SymmetrizationLevel.EXHAUSTIVE
        symmetrization = SymmetrizationLevel(symmetrize_readout)

    # calibration readout only works with symmetrization turned on
    if calibrate_readout is not None and symmetrization != SymmetrizationLevel.EXHAUSTIVE:
        raise ValueError("Readout calibration only currently works with exhaustive readout "
                         "symmetrization turned on.")

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
            coeff = complex(setting.out_operator.coefficient)
            if not np.isclose(coeff.imag, 0):
                raise ValueError(f"{setting}'s out_operator has a complex coefficient.")
            coeff = coeff.real

            # Special case for measuring the "identity" operator, which doesn't make much
            #     sense but should happen perfectly.
            if is_identity(setting.out_operator):
                yield ExperimentResult(
                    setting=setting,
                    expectation=coeff,
                    std_err=0.0,
                    total_counts=shots,
                )
                continue

            # Obtain statistics from result of experiment
            obs_mean, obs_var = _stats_from_measurements(bitstrings,
                                                         {q: idx for idx, q in enumerate(qubits)},
                                                         setting, shots, coeff)

            if calibrate_readout == 'plus-eig':
                # Readout calibration
                # Obtain calibration program
                calibr_prog = _calibration_program(qc, tomo_experiment, setting)
                calibr_qubs = setting.out_operator.get_qubits()
                calibr_qub_dict = {q: idx for idx, q in enumerate(calibr_qubs)}

                # Perform symmetrization on the calibration program
                calibr_results = qc.run_symmetrized_readout(calibr_prog,
                                                            shots,
                                                            SymmetrizationLevel.EXHAUSTIVE,
                                                            calibr_qubs)

                # Obtain statistics from the measurement process
                obs_calibr_mean, obs_calibr_var = _stats_from_measurements(calibr_results,
                                                                           calibr_qub_dict,
                                                                           setting, shots)
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


def _ops_bool_to_prog(ops_bool: Tuple[bool], qubits: List[int]) -> Program:
    """
    :param ops_bool: tuple of booleans specifying the operation to be carried out on `qubits`
    :param qubits: list specifying the qubits to be carried operations on
    :return: Program with the operations specified in `ops_bool` on the qubits specified in
        `qubits`
    """
    assert len(ops_bool) == len(qubits), "Mismatch of qubits and operations"
    prog = Program()
    for i, op_bool in enumerate(ops_bool):
        if op_bool == 0:
            continue
        elif op_bool == 1:
            prog += Program(X(qubits[i]))
        else:
            raise ValueError("ops_bool should only consist of 0s and/or 1s")
    return prog


def _stats_from_measurements(bs_results: np.ndarray, qubit_index_map: Dict,
                             setting: ExperimentSetting, n_shots: int,
                             coeff: float = 1.0) -> Tuple[float, float]:
    """
    :param bs_results: results from running `qc.run`
    :param qubit_index_map: dict mapping qubit to classical register index
    :param setting: ExperimentSetting
    :param n_shots: number of shots in the measurement process
    :param coeff: coefficient of the operator being estimated
    :return: tuple specifying (mean, variance)
    """
    # Identify classical register indices to select
    idxs = [qubit_index_map[q] for q, _ in setting.out_operator]
    # Pick columns corresponding to qubits with a non-identity out_operation
    obs_strings = bs_results[:, idxs]
    # Transform bits to eigenvalues; ie (+1, -1)
    my_obs_strings = 1 - 2 * obs_strings
    # Multiply row-wise to get operator values. Do statistics. Return result.
    obs_vals = coeff * np.prod(my_obs_strings, axis=1)
    obs_mean = np.mean(obs_vals)
    obs_var = np.var(obs_vals) / n_shots

    return obs_mean, obs_var


def ratio_variance(a: Union[float, np.ndarray],
                   var_a: Union[float, np.ndarray],
                   b: Union[float, np.ndarray],
                   var_b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Given random variables 'A' and 'B', compute the variance on the ratio Y = A/B. Denote the
    mean of the random variables as a = E[A] and b = E[B] while the variances are var_a = Var[A]
    and var_b = Var[B] and the covariance as Cov[A,B]. The following expression approximates the
    variance of Y

    Var[Y] \approx (a/b) ^2 * ( var_a /a^2 + var_b / b^2 - 2 * Cov[A,B]/(a*b) )

    We assume the covariance of A and B is negligible, resting on the assumption that A and B
    are independently measured. The expression above rests on the assumption that B is non-zero,
    an assumption which we expect to hold true in most cases, but makes no such assumptions
    about A. If we allow E[A] = 0, then calculating the expression above via numpy would complain
    about dividing by zero. Instead, we can re-write the above expression as

    Var[Y] \approx var_a /b^2 + (a^2 * var_b) / b^4

    where we have dropped the covariance term as noted above.

    See the following for more details:
      - https://doi.org/10.1002/(SICI)1097-0320(20000401)39:4<300::AID-CYTO8>3.0.CO;2-O
      - http://www.stat.cmu.edu/~hseltman/files/ratio.pdf
      - https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables

    :param a: Mean of 'A', to be used as the numerator in a ratio.
    :param var_a: Variance in 'A'
    :param b: Mean of 'B', to be used as the numerator in a ratio.
    :param var_b: Variance in 'B'
    """
    return var_a / b**2 + (a**2 * var_b) / b**4


def _calibration_program(qc: QuantumComputer, tomo_experiment: TomographyExperiment,
                         setting: ExperimentSetting) -> Program:
    """
    Program required for calibration in a tomography-like experiment.

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
    # Inherit readout errro instructions from main Program
    readout_povm_instruction = [i for i in tomo_experiment.program.out().split('\n') if 'PRAGMA READOUT-POVM' in i]
    calibr_prog += readout_povm_instruction
    # Inherit any definitions of noisy gates from main Program
    kraus_instructions = [i for i in tomo_experiment.program.out().split('\n') if 'PRAGMA ADD-KRAUS' in i]
    calibr_prog += kraus_instructions
    # Prepare the +1 eigenstate for the out operator
    for q, op in setting.out_operator.operations_as_set():
        calibr_prog += _one_q_pauli_prep(label=op, index=0, qubit=q)
    # Measure the out operator in this state
    for q, op in setting.out_operator.operations_as_set():
        calibr_prog += _local_pauli_eig_meas(op, q)

    return calibr_prog

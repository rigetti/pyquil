##############################################################################
# Copyright 2021 Rigetti Computing
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
import itertools
import warnings
from math import log, pi
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast

import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration

from pyquil.api._compiler import AbstractCompiler, QVMCompiler
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import QuantumComputer as QuantumComputerV3
from pyquil.api._quantum_computer import get_qc as get_qc_v3, QuantumExecutable
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.gates import MEASURE, RX
from pyquil.noise import NoiseModel, decoherence_noise_with_asymmetric_ro
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.quil import Program, validate_supported_quil
from pyquil.quilatom import qubit_index
from ._qam import StatefulQAM


class QuantumComputer(QuantumComputerV3):
    compiler: AbstractCompiler
    qam: StatefulQAM[Any]

    def __init__(
        self,
        *,
        name: str,
        qam: QAM[Any],
        device: Any = None,
        compiler: AbstractCompiler,
        symmetrize_readout: bool = False,
    ) -> None:
        """
        An interface designed to ease migration from pyQuil v2 to v3, and compatible with most
        use cases for the pyQuil v2 QuantumComputer.

        A quantum computer for running quantum programs.

        A quantum computer has various characteristics like supported gates, qubits, qubit
        topologies, gate fidelities, and more. A quantum computer also has the ability to
        run quantum programs.

        A quantum computer can be a real Rigetti QPU that uses superconducting transmon
        qubits to run quantum programs, or it can be an emulator like the QVM with
        noise models and mimicked topologies.

        :param name: A string identifying this particular quantum computer.
        :param qam: A quantum abstract machine which handles executing quantum programs. This
            dispatches to a QVM or QPU.
        :param device: Ignored and accepted only for backwards compatibility.
        :param symmetrize_readout: Whether to apply readout error symmetrization. See
            :py:func:`run_symmetrized_readout` for a complete description.
        """
        self.name = name
        StatefulQAM.wrap(qam)
        self.qam = cast(StatefulQAM[Any], qam)
        self.compiler = compiler

        self.symmetrize_readout = symmetrize_readout

    def run(  # type: ignore
        self,
        executable: QuantumExecutable,
        memory_map: Optional[Mapping[str, Sequence[Union[int, float]]]] = None,
    ) -> np.ndarray:
        """
        Run a quil executable. If the executable contains declared parameters, then a memory
        map must be provided, which defines the runtime values of these parameters.

        :param executable: The program to run. You are responsible for compiling this first.
        :param memory_map: The mapping of declared parameters to their values. The values
            are a list of floats or integers.
        :return: A numpy array of shape (trials, len(ro-register)) that contains 0s and 1s.
        """
        self.qam.load(executable)
        if memory_map:
            for region_name, values_list in memory_map.items():
                self.qam.write_memory(region_name=region_name, value=values_list)
        result = self.qam.run().read_memory(region_name="ro")
        assert result is not None
        return result

    def calibrate(self, experiment: Experiment) -> List[ExperimentResult]:
        """
        Perform readout calibration on the various multi-qubit observables involved in the provided
        ``Experiment``.

        :param experiment: The ``Experiment`` to calibrate readout error for.
        :return: A list of ``ExperimentResult`` objects that contain the expectation values that
            correspond to the scale factors resulting from symmetric readout error.
        """
        calibration_experiment = experiment.generate_calibration_experiment()
        return self.experiment(calibration_experiment)

    def experiment(
        self,
        experiment: Experiment,
        memory_map: Optional[Mapping[str, Sequence[Union[int, float]]]] = None,
    ) -> List[ExperimentResult]:
        """
        Run an ``Experiment`` on a QVM or QPU backend. An ``Experiment`` is composed of:

            - A main ``Program`` body (or ansatz).
            - A collection of ``ExperimentSetting`` objects, each of which encodes a particular
              state preparation and measurement.
            - A ``SymmetrizationLevel`` for enacting different readout symmetrization strategies.
            - A number of shots to collect for each (unsymmetrized) ``ExperimentSetting``.

        Because the main ``Program`` is static from run to run of an ``Experiment``, we can leverage
        our platform's Parametric Compilation feature. This means that the ``Program`` can be
        compiled only once, and the various alterations due to state preparation, measurement,
        and symmetrization can all be realized at runtime by providing a ``memory_map``. Thus, the
        steps in the ``experiment`` method are as follows:

            1. Generate a parameterized program corresponding to the ``Experiment``
                (see the ``Experiment.generate_experiment_program()`` method for more
                details on how it changes the main body program to support state preparation,
                measurement, and symmetrization).
            2. Compile the parameterized program into a parametric (binary) executable, which
                   contains declared variables that can be assigned at runtime.

            3. For each ``ExperimentSetting`` in the ``Experiment``, we repeat the following:

                a. Build a collection of memory maps that correspond to the various state
                   preparation, measurement, and symmetrization specifications.
                b. Run the parametric executable on the QVM or QPU backend, providing the memory map
                   to assign variables at runtime.
                c. Extract the desired statistics from the classified bitstrings that are produced
                   by the QVM or QPU backend, and package them in an ``ExperimentResult`` object.

            4. Return the list of ``ExperimentResult`` objects.

        This method is extremely useful shorthand for running near-term applications and algorithms,
        which often have this ansatz + settings structure.

        :param experiment: The ``Experiment`` to run.
        :param memory_map: A dictionary mapping declared variables / parameters to their values.
            The values are a list of floats or integers. Each float or integer corresponds to
            a particular classical memory register. The memory map provided to the ``experiment``
            method corresponds to variables in the main body program that we would like to change
            at runtime (e.g. the variational parameters provided to the ansatz of the variational
            quantum eigensolver).
        :return: A list of ``ExperimentResult`` objects containing the statistics gathered
            according to the specifications of the ``Experiment``.
        """
        experiment_program = experiment.generate_experiment_program()
        executable = self.compile(experiment_program)

        if memory_map is None:
            memory_map = {}

        results = []
        for settings in experiment:
            if len(settings) > 1:
                raise ValueError("settings must be of length 1")
            setting = settings[0]

            qubits = cast(List[int], setting.out_operator.get_qubits())
            experiment_setting_memory_map = experiment.build_setting_memory_map(setting)
            symmetrization_memory_maps = experiment.build_symmetrization_memory_maps(qubits)
            merged_memory_maps = merge_memory_map_lists([experiment_setting_memory_map], symmetrization_memory_maps)

            all_bitstrings = []
            for merged_memory_map in merged_memory_maps:
                final_memory_map = {**memory_map, **merged_memory_map}
                self.qam.reset()
                bitstrings = self.run(executable, memory_map=final_memory_map)

                if "symmetrization" in final_memory_map:
                    bitmask = np.array(np.array(final_memory_map["symmetrization"]) / np.pi, dtype=int)
                    bitstrings = np.bitwise_xor(bitstrings, bitmask)
                all_bitstrings.append(bitstrings)
            symmetrized_bitstrings = np.concatenate(all_bitstrings)

            joint_expectations = [experiment.get_meas_registers(qubits)]
            if setting.additional_expectations:
                joint_expectations += setting.additional_expectations
            expectations = bitstrings_to_expectations(symmetrized_bitstrings, joint_expectations=joint_expectations)

            means = cast(np.ndarray, np.mean(expectations, axis=0))
            std_errs = np.std(expectations, axis=0, ddof=1) / np.sqrt(len(expectations))

            joint_results = []
            for qubit_subset, mean, std_err in zip(joint_expectations, means, std_errs):
                out_operator = PauliTerm.from_list([(setting.out_operator[i], i) for i in qubit_subset])
                s = ExperimentSetting(
                    in_state=setting.in_state,
                    out_operator=out_operator,
                    additional_expectations=None,
                )
                r = ExperimentResult(setting=s, expectation=mean, std_err=std_err, total_counts=len(expectations))
                joint_results.append(r)

            result = ExperimentResult(
                setting=setting,
                expectation=joint_results[0].expectation,
                std_err=joint_results[0].std_err,
                total_counts=joint_results[0].total_counts,
                additional_results=joint_results[1:],
            )
            results.append(result)

        return results

    def run_symmetrized_readout(
        self,
        program: Program,
        trials: int,
        symm_type: int = 3,
        meas_qubits: Optional[List[int]] = None,
    ) -> np.ndarray:
        r"""
        Run a quil program in such a way that the readout error is made symmetric. Enforcing
        symmetric readout error is useful in simplifying the assumptions in some near
        term error mitigation strategies, see ``measure_observables`` for more information.

        The simplest example is for one qubit. In a noisy device, the probability of accurately
        reading the 0 state might be higher than that of the 1 state; due to e.g. amplitude
        damping. This makes correcting for readout more difficult. In the simplest case, this
        function runs the program normally ``(trials//2)`` times. The other half of the time,
        it will insert an ``X`` gate prior to any ``MEASURE`` instruction and then flip the
        measured classical bit back. Overall this has the effect of symmetrizing the readout error.

        The details. Consider preparing the input bitstring ``|i>`` (in the computational basis) and
        measuring in the Z basis. Then the Confusion matrix for the readout error is specified by
        the probabilities

             p(j|i) := Pr(measured = j | prepared = i ).

        In the case of a single qubit i,j \in [0,1] then:
        there is no readout error if p(0|0) = p(1|1) = 1.
        the readout error is symmetric if p(0|0) = p(1|1) = 1 - epsilon.
        the readout error is asymmetric if p(0|0) != p(1|1).

        If your quantum computer has this kind of asymmetric readout error then
        ``qc.run_symmetrized_readout`` will symmetrize the readout error.

        The readout error above is only asymmetric on a single bit. In practice the confusion
        matrix on n bits need not be symmetric, e.g. for two qubits p(ij|ij) != 1 - epsilon for
        all i,j. In these situations a more sophisticated means of symmetrization is needed; and
        we use orthogonal arrays (OA) built from Hadamard matrices.

        The symmetrization types are specified by an int; the types available are:
        -1 -- exhaustive symmetrization uses every possible combination of flips
        0 -- trivial that is no symmetrization
        1 -- symmetrization using an OA with strength 1
        2 -- symmetrization using an OA with strength 2
        3 -- symmetrization using an OA with strength 3
        In the context of readout symmetrization the strength of the orthogonal array enforces
        the symmetry of the marginal confusion matrices.

        By default a strength 3 OA is used; this ensures expectations of the form
        ``<b_k . b_j . b_i>`` for bits any bits i,j,k will have symmetric readout errors. Here
        expectation of a random variable x as is denote ``<x> = sum_i Pr(i) x_i``. It turns out that
        a strength 3 OA is also a strength 2 and strength 1 OA it also ensures ``<b_j . b_i>`` and
        ``<b_i>`` have symmetric readout errors for any bits b_j and b_i.

        :param program: The program to run symmetrized readout on.
        :param trials: The minimum number of times to run the program; it is recommend that this
            number should be in the hundreds or thousands. This parameter will be mutated if
            necessary.
        :param symm_type: the type of symmetrization
        :param meas_qubits: An advanced feature. The groups of measurement qubits. Only these
            qubits will be symmetrized over, even if the program acts on other qubits.
        :return: A numpy array of shape (trials, len(ro-register)) that contains 0s and 1s.
        """
        if not isinstance(symm_type, int):
            raise ValueError(
                "Symmetrization options are indicated by an int. See " "the docstrings for more information."
            )

        if meas_qubits is None:
            meas_qubits = list(cast(Set[int], program.get_qubits()))

        # It is desirable to have hundreds or thousands of trials more than the minimum
        trials = _check_min_num_trials_for_symmetrized_readout(len(meas_qubits), trials, symm_type)

        sym_programs, flip_arrays = _symmetrization(program, meas_qubits, symm_type)

        # Floor division so e.g. 9 // 8 = 1 and 17 // 8 = 2.
        num_shots_per_prog = trials // len(sym_programs)

        if num_shots_per_prog * len(sym_programs) < trials:
            warnings.warn(
                f"The number of trials was modified from {trials} to "
                f"{num_shots_per_prog * len(sym_programs)}. To be consistent with the "
                f"number of trials required by the type of readout symmetrization "
                f"chosen."
            )

        results = _measure_bitstrings(self, sym_programs, meas_qubits, num_shots_per_prog)

        return _consolidate_symmetrization_outputs(results, flip_arrays)

    def run_and_measure(self, program: Program, trials: int) -> Dict[int, np.ndarray]:
        """
        Run the provided state preparation program and measure all qubits.

        The returned data is a dictionary keyed by qubit index because qubits for a given
        QuantumComputer may be non-contiguous and non-zero-indexed. To turn this dictionary
        into a 2d numpy array of bitstrings, consider::

            bitstrings = qc.run_and_measure(...)
            bitstring_array = np.vstack([bitstrings[q] for q in qc.qubits()]).T
            bitstring_array.shape  # (trials, len(qc.qubits()))

        .. note::

            If the target :py:class:`QuantumComputer` is a noiseless :py:class:`QVM` then
            only the qubits explicitly used in the program will be measured. Otherwise all
            qubits will be measured. In some circumstances this can exhaust the memory
            available to the simulator, and this may be manifested by the QVM failing to
            respond or timeout.

        .. note::

            In contrast to :py:class:`QVMConnection.run_and_measure`, this method simulates
            noise correctly for noisy QVMs. However, this method is slower for ``trials > 1``.
            For faster noise-free simulation, consider
            :py:class:`WavefunctionSimulator.run_and_measure`.

        :param program: The state preparation program to run and then measure.
        :param trials: The number of times to run the program.
        :return: A dictionary keyed by qubit index where the corresponding value is a 1D array of
            measured bits.
        """
        program = program.copy()
        validate_supported_quil(program)
        ro = program.declare("ro", "BIT", len(self.qubits()))
        measure_used = isinstance(self.qam, QVM) and self.qam.noise_model is None
        qubits_to_measure = set(map(qubit_index, program.get_qubits()) if measure_used else self.qubits())
        for i, q in enumerate(qubits_to_measure):
            program.inst(MEASURE(q, ro[i]))
        program.wrap_in_numshots_loop(trials)
        executable = self.compile(program)
        bitstring_array = self.run(executable=executable)
        bitstring_dict = {}
        for i, q in enumerate(qubits_to_measure):
            bitstring_dict[q] = bitstring_array[:, i]
        for q in set(self.qubits()) - set(qubits_to_measure):
            bitstring_dict[q] = np.zeros(trials)
        return bitstring_dict

    def compile(
        self,
        program: Program,
        to_native_gates: bool = True,
        optimize: bool = True,
        *,
        protoquil: Optional[bool] = None,
    ) -> QuantumExecutable:
        """
        A high-level interface to program compilation.

        Compilation currently consists of two stages. Please see the :py:class:`AbstractCompiler`
        docs for more information. This function does all stages of compilation.

        Right now both ``to_native_gates`` and ``optimize`` must be either both set or both
        unset. More modular compilation passes may be available in the future.

        Additionally, a call to compile also calls the ``reset`` method if one is running
        on the QPU. This is a bit of a sneaky hack to guard against stale compiler connections,
        but shouldn't result in any material hit to performance (especially when taking advantage
        of parametric compilation for hybrid applications).

        :param program: A Program
        :param to_native_gates: Whether to compile non-native gates to native gates.
        :param optimize: Whether to optimize the program to reduce the number of operations.
        :param protoquil: Whether to restrict the input program to and the compiled program
            to protoquil (executable on QPU). A value of ``None`` means defer to server.
        :return: An executable binary suitable for passing to :py:func:`QuantumComputer.run`.
        """

        if isinstance(self.qam, QPU):
            self.reset()

        flags = [to_native_gates, optimize]
        assert all(flags) or all(not f for f in flags), "Must turn quilc all on or all off"
        quilc = all(flags)

        if quilc:
            nq_program = self.compiler.quil_to_native_quil(program, protoquil=protoquil)
        else:
            nq_program = program
        binary = self.compiler.native_quil_to_executable(nq_program)
        return binary

    def reset(self) -> None:
        """
        Reset the QuantumComputer's QAM to its initial state.
        """
        self.qam.reset()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'QuantumComputer[name="{self.name}"]'


def get_qc(
    name: str,
    *,
    as_qvm: Optional[bool] = None,
    noisy: Optional[bool] = None,
    connection: Any = None,
    compiler_timeout: float = 10,
    client_configuration: Optional[QCSClientConfiguration] = None,
) -> QuantumComputer:
    """
    Compatibility layer to build a QuantumComputer supporting an API closely
    similar to that in pyQuil v2.
    """
    if connection is not None:
        raise ValueError(
            "`get_qc` no longer supports the `connection` parameter. "
            "Please update your code to use the current interface of `pyquil.get_qc`."
        )

    qc = get_qc_v3(
        name=name,
        as_qvm=as_qvm,
        noisy=noisy,
        compiler_timeout=compiler_timeout,
        client_configuration=client_configuration,
    )

    return QuantumComputer(
        name=qc.name, qam=qc.qam, device=None, compiler=qc.compiler, symmetrize_readout=qc.symmetrize_readout
    )


def _flip_array_to_prog(flip_array: Tuple[bool], qubits: List[int]) -> Program:
    """
    Generate a pre-measurement program that flips the qubit state according to the flip_array of
    bools.

    This is used, for example, in symmetrization to produce programs which flip a select subset
    of qubits immediately before measurement.

    :param flip_array: tuple of booleans specifying whether the qubit in the corresponding index
        should be flipped or not.
    :param qubits: list specifying the qubits in order corresponding to the flip_array
    :return: Program which flips each qubit (i.e. instructs RX(pi, q)) according to the flip_array.
    """
    assert len(flip_array) == len(qubits), "Mismatch of qubits and operations"
    prog = Program()
    for qubit, flip_output in zip(qubits, flip_array):
        if flip_output == 0:
            continue
        elif flip_output == 1:
            prog += Program(RX(pi, qubit))
        else:
            raise ValueError("flip_bools should only consist of 0s and/or 1s")
    return prog


def _symmetrization(
    program: Program, meas_qubits: List[int], symm_type: int = 3
) -> Tuple[List[Program], List[Tuple[bool]]]:
    """
    For the input program generate new programs which flip the measured qubits with an X gate in
    certain combinations in order to symmetrize readout.

    An expanded list of programs is returned along with a list of bools which indicates which
    qubits are flipped in each program.

    The symmetrization types are specified by an int; the types available are:

    * -1 -- exhaustive symmetrization uses every possible combination of flips
    *  0 -- trivial that is no symmetrization
    *  1 -- symmetrization using an OA with strength 1
    *  2 -- symmetrization using an OA with strength 2
    *  3 -- symmetrization using an OA with strength 3

    In the context of readout symmetrization the strength of the orthogonal array enforces the
    symmetry of the marginal confusion matrices.

    By default a strength 3 OA is used; this ensures expectations of the form <b_k * b_j * b_i>
    for bits any bits i,j,k will have symmetric readout errors. Here expectation of a random
    variable x as is denote <x> = sum_i Pr(i) x_i. It turns out that a strength 3 OA is also a
    strength 2 and strength 1 OA it also ensures <b_j * b_i> and <b_i> have symmetric readout
    errors for any bits b_j and b_i.

    :param programs: a program which will be symmetrized.
    :param meas_qubits: the groups of measurement qubits. Only these qubits will be symmetrized
        over, even if the program acts on other qubits.
    :param sym_type: an int determining the type of symmetrization performed.
    :return: a list of symmetrized programs, the corresponding array of bools indicating which
        qubits were flipped.
    """
    if symm_type < -1 or symm_type > 3:
        raise ValueError("symm_type must be one of the following ints [-1, 0, 1, 2, 3].")
    elif symm_type == -1:
        # exhaustive = all possible binary strings
        flip_matrix = np.asarray(list(itertools.product([0, 1], repeat=len(meas_qubits))))
    elif symm_type >= 0:
        flip_matrix = _construct_orthogonal_array(len(meas_qubits), symm_type)

    # The next part is not rigorous in the sense that we simply truncate to the desired
    # number of qubits. The problem is that orthogonal arrays of a certain strength for an
    # arbitrary number of qubits are not known to exist.
    flip_matrix = flip_matrix[:, : len(meas_qubits)]

    symm_programs = []
    flip_arrays = []
    for flip_array in flip_matrix:
        total_prog_symm = program.copy()
        prog_symm = _flip_array_to_prog(flip_array, meas_qubits)
        total_prog_symm += prog_symm
        symm_programs.append(total_prog_symm)
        flip_arrays.append(flip_array)

    return symm_programs, flip_arrays


def _consolidate_symmetrization_outputs(outputs: List[np.ndarray], flip_arrays: List[Tuple[bool]]) -> np.ndarray:
    """
    Given bitarray results from a series of symmetrization programs, appropriately flip output
    bits and consolidate results into new bitarrays.

    :param outputs: a list of the raw bitarrays resulting from running a list of symmetrized
        programs; for example, the results returned from _measure_bitstrings
    :param flip_arrays: a list of boolean arrays in one-to-one correspondence with the list of
        outputs indicating which qubits where flipped before each bitarray was measured.
    :return: an np.ndarray consisting of the consolidated bitarray outputs which can be treated as
        the symmetrized outputs of the original programs passed into a symmetrization method. See
        estimate_observables for example usage.
    """
    assert len(outputs) == len(flip_arrays)

    output = []
    for bitarray, flip_array in zip(outputs, flip_arrays):
        if len(flip_array) == 0:
            output.append(bitarray)
        else:
            output.append(bitarray ^ flip_array)

    return np.vstack(output)


def _measure_bitstrings(
    qc: QuantumComputer, programs: List[Program], meas_qubits: List[int], num_shots: int = 600
) -> List[np.ndarray]:
    """
    Wrapper for appending measure instructions onto each program, running the program,
    and accumulating the resulting bitarrays.

    :param qc: a quantum computer object on which to run each program
    :param programs: a list of programs to run
    :param meas_qubits: groups of qubits to measure for each program
    :param num_shots: the number of shots to run for each program
    :return: a len(programs) long list of num_shots by num_meas_qubits bit arrays of results for
        each program.
    """
    results = []
    for program in programs:
        # copy the program so the original is not mutated
        prog = program.copy()
        ro = prog.declare("ro", "BIT", len(meas_qubits))
        for idx, q in enumerate(meas_qubits):
            prog += MEASURE(q, ro[idx])

        prog.wrap_in_numshots_loop(num_shots)
        prog = qc.compiler.quil_to_native_quil(prog)
        executable = qc.compiler.native_quil_to_executable(prog)
        shots = qc.run(executable)
        results.append(shots)
    return results


def _construct_orthogonal_array(num_qubits: int, strength: int = 3) -> np.ndarray:
    """
    Given a strength and number of qubits this function returns an Orthogonal Array (OA)
    on 'n' or more qubits. Sometimes the size of the returned array is larger than num_qubits;
    typically the next power of two relative to num_qubits. This is corrected later in the code
    flow.

    :param num_qubits: the minimum number of qubits the OA should act on.
    :param strength: the statistical "strength" of the OA
    :return: a numpy array where the rows represent the different experiments
    """
    if strength < 0 or strength > 3:
        raise ValueError("'strength' must be one of the following ints [0, 1, 2, 3].")
    if strength == 0:
        # trivial flip matrix = an array of zeros
        flip_matrix = np.zeros((1, num_qubits)).astype(int)
    elif strength == 1:
        # orthogonal array with strength equal to 1. See Example 1.4 of [OATA], referenced in the
        # `construct_strength_two_orthogonal_array` docstrings, for more details.
        zero_array = np.zeros((1, num_qubits))
        one_array = np.ones((1, num_qubits))
        flip_matrix = np.concatenate((zero_array, one_array), axis=0).astype(int)
    elif strength == 2:
        flip_matrix = _construct_strength_two_orthogonal_array(num_qubits)
    elif strength == 3:
        flip_matrix = _construct_strength_three_orthogonal_array(num_qubits)

    return flip_matrix


def _next_power_of_2(x: int) -> int:
    return cast(int, 1 if x == 0 else 2 ** (x - 1).bit_length())


# The code below is directly copied from scipy see https://bit.ly/2RjAHJz, the docstrings have
# been modified.
def hadamard(n: int, dtype: np.dtype = int) -> np.ndarray:  # type: ignore
    """
    Construct a Hadamard matrix.
    Constructs an n-by-n Hadamard matrix, using Sylvester's
    construction.  `n` must be a power of 2.

    Parameters
    ----------
    n : int
        The order of the matrix.  `n` must be a power of 2.
    dtype : numpy dtype
        The data type of the array to be constructed.

    Returns
    -------
    H : (n, n) ndarray
        The Hadamard matrix.

    Notes
    -----
    .. versionadded:: 0.8.0

    Examples
    --------
    >>> hadamard(2, dtype=complex)
    array([[ 1.+0.j,  1.+0.j],
           [ 1.+0.j, -1.-0.j]])
    >>> hadamard(4)
    array([[ 1,  1,  1,  1],
           [ 1, -1,  1, -1],
           [ 1,  1, -1, -1],
           [ 1, -1, -1,  1]])
    """
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(log(n, 2))
    if 2 ** lg2 != n:
        raise ValueError("n must be an positive integer, and n must be a power of 2")

    H = np.array([[1]], dtype=dtype)

    # Sylvester's construction
    for _ in range(0, lg2):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    return H


def _construct_strength_three_orthogonal_array(num_qubits: int) -> np.ndarray:
    r"""
    Given a number of qubits this function returns an Orthogonal Array (OA)
    on 'n' qubits where n is the next power of two relative to num_qubits.

    Specifically it returns the OA(2n, n, 2, 3).

    The parameters of the OA(N, k, s, t) are interpreted as
    N: Number of rows, level combinations or runs
    k: Number of columns, constraints or factors
    s: Number of symbols or levels
    t: Strength

    See [OATA] for more details.

    [OATA] Orthogonal Arrays: theory and applications
           Hedayat, Sloane, Stufken
           Springer Science & Business Media, 2012.
           https://dx.doi.org/10.1007/978-1-4612-1478-6

    :param num_qubits: minimum number of qubits the OA should run on.
    :return: A numpy array representing the OA with shape N by k
    """
    num_qubits_power_of_2 = _next_power_of_2(num_qubits)
    H = hadamard(num_qubits_power_of_2)
    Hfold = np.concatenate((H, -H), axis=0)
    orthogonal_array = cast(np.ndarray, ((Hfold + 1) / 2).astype(int))
    return orthogonal_array


def _construct_strength_two_orthogonal_array(num_qubits: int) -> np.ndarray:
    r"""
    Given a number of qubits this function returns an Orthogonal Array (OA) on 'n-1' qubits
    where n-1 is the next integer lambda so that 4*lambda -1 is larger than num_qubits.

    Specifically it returns the OA(n, n − 1, 2, 2).

    The parameters of the OA(N, k, s, t) are interpreted as
    N: Number of rows, level combinations or runs
    k: Number of columns, constraints or factors
    s: Number of symbols or levels
    t: Strength

    See [OATA] for more details.

    [OATA] Orthogonal Arrays: theory and applications
           Hedayat, Sloane, Stufken
           Springer Science & Business Media, 2012.
           https://dx.doi.org/10.1007/978-1-4612-1478-6

    :param num_qubits: minimum number of qubits the OA should run on.
    :return: A numpy array representing the OA with shape N by k
    """
    # next line will break post denali at 275 qubits
    # valid_num_qubits = 4 * lambda - 1
    valid_numbers = [4 * lam - 1 for lam in range(1, 70)]
    # 4 * lambda
    four_lam = min(x for x in valid_numbers if x >= num_qubits) + 1
    H = hadamard(_next_power_of_2(four_lam))
    # The minus sign in front of H fixes the 0 <-> 1 inversion relative to the reference [OATA]
    orthogonal_array = cast(np.ndarray, ((-H[1:, :].T + 1) / 2).astype(int))
    return orthogonal_array


def _check_min_num_trials_for_symmetrized_readout(num_qubits: int, trials: int, symm_type: int) -> int:
    """
    This function sets the minimum number of trials; it is desirable to have hundreds or
    thousands of trials more than the minimum.

    :param num_qubits: number of qubits to symmetrize
    :param trials: number of trials
    :param symm_type: symmetrization type see
    :return: possibly modified number of trials
    """
    if symm_type < -1 or symm_type > 3:
        raise ValueError("symm_type must be one of the following ints [-1, 0, 1, 2, 3].")

    if symm_type == -1:
        min_num_trials = 2 ** num_qubits
    elif symm_type == 2:

        def _f(x: int) -> int:
            return 4 * x - 1

        min_num_trials = min(_f(x) for x in range(1, 1024) if _f(x) >= num_qubits) + 1
    elif symm_type == 3:
        min_num_trials = _next_power_of_2(2 * num_qubits)
    else:
        # symm_type == 0 or symm_type == 1 require one and two trials respectively; ensured by:
        min_num_trials = 2

    if trials < min_num_trials:
        trials = min_num_trials
        warnings.warn(f"Number of trials was too low, it is now {trials}.")
    return trials


def _get_qvm_or_pyqvm(
    *,
    client_configuration: QCSClientConfiguration,
    qvm_type: str,
    noise_model: Optional[NoiseModel],
    quantum_processor: Optional[AbstractQuantumProcessor],
    execution_timeout: float,
) -> Union[QVM, PyQVM]:
    if qvm_type == "qvm":
        return QVM(noise_model=noise_model, timeout=execution_timeout, client_configuration=client_configuration)
    elif qvm_type == "pyqvm":
        assert quantum_processor is not None
        return PyQVM(n_qubits=quantum_processor.qubit_topology().number_of_nodes())

    raise ValueError("Unknown qvm type {}".format(qvm_type))


def _get_qvm_qc(
    *,
    client_configuration: QCSClientConfiguration,
    name: str,
    qvm_type: str,
    quantum_processor: AbstractQuantumProcessor,
    compiler_timeout: float,
    execution_timeout: float,
    noise_model: Optional[NoiseModel],
) -> QuantumComputer:
    """Construct a QuantumComputer backed by a QVM.

    This is a minimal wrapper over the QuantumComputer, QVM, and QVMCompiler constructors.

    :param client_configuration: Client configuration.
    :param name: A string identifying this particular quantum computer.
    :param qvm_type: The type of QVM. Either qvm or pyqvm.
    :param quantum_processor: A quantum_processor following the AbstractQuantumProcessor interface.
    :param noise_model: An optional noise model
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param execution_timeout: Time limit for execution requests, in seconds.
    :return: A QuantumComputer backed by a QVM with the above options.
    """

    return QuantumComputer(
        name=name,
        qam=_get_qvm_or_pyqvm(
            client_configuration=client_configuration,
            qvm_type=qvm_type,
            noise_model=noise_model,
            quantum_processor=quantum_processor,
            execution_timeout=execution_timeout,
        ),
        compiler=QVMCompiler(
            quantum_processor=quantum_processor,
            timeout=compiler_timeout,
            client_configuration=client_configuration,
        ),
    )


def _get_qvm_with_topology(
    *,
    client_configuration: QCSClientConfiguration,
    name: str,
    topology: nx.Graph,
    noisy: bool,
    qvm_type: str,
    compiler_timeout: float,
    execution_timeout: float,
) -> QuantumComputer:
    """Construct a QVM with the provided topology.

    :param client_configuration: Client configuration.
    :param name: A name for your quantum computer. This field does not affect behavior of the
        constructed QuantumComputer.
    :param topology: A graph representing the desired qubit connectivity.
    :param noisy: Whether to include a generic noise model. If you want more control over
        the noise model, please construct your own :py:class:`NoiseModel` and use
        :py:func:`_get_qvm_qc` instead of this function.
    :param qvm_type: The type of QVM. Either 'qvm' or 'pyqvm'.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param execution_timeout: Time limit for execution requests, in seconds.
    :return: A pre-configured QuantumComputer
    """
    # Note to developers: consider making this function public and advertising it.
    quantum_processor = NxQuantumProcessor(topology=topology)
    if noisy:
        noise_model: Optional[NoiseModel] = decoherence_noise_with_asymmetric_ro(
            isa=quantum_processor.to_compiler_isa()
        )
    else:
        noise_model = None
    return _get_qvm_qc(
        client_configuration=client_configuration,
        name=name,
        qvm_type=qvm_type,
        quantum_processor=quantum_processor,
        noise_model=noise_model,
        compiler_timeout=compiler_timeout,
        execution_timeout=execution_timeout,
    )

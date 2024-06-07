##############################################################################
# Copyright 2018 Rigetti Computing
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
import re
import socket
import subprocess
import warnings
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from math import log, pi
from typing import (
    Any,
    Optional,
    Union,
    cast,
)

import networkx as nx
import numpy as np
from qcs_sdk import QCSClient
from qcs_sdk.compiler.quilc import QuilcClient
from qcs_sdk.qpu import list_quantum_processors
from qcs_sdk.qvm import QVMClient

from pyquil.api._abstract_compiler import AbstractCompiler, QuantumExecutable
from pyquil.api._compiler import QPUCompiler, QVMCompiler
from pyquil.api._qam import QAM, MemoryMap, QAMExecutionResult
from pyquil.api._qpu import QPU
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import MEASURE, RX
from pyquil.noise import NoiseModel, decoherence_noise_with_asymmetric_ro
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import (
    AbstractQuantumProcessor,
    NxQuantumProcessor,
    QCSQuantumProcessor,
    get_qcs_quantum_processor,
)
from pyquil.quil import Program
from pyquil.quilatom import QubitDesignator


class QuantumComputer:
    def __init__(
        self,
        *,
        name: str,
        qam: QAM[Any],
        compiler: AbstractCompiler,
        symmetrize_readout: bool = False,
    ) -> None:
        """Use a quantum computer to run quantum programs.

        A quantum computer has various characteristics like supported gates, qubits, qubit
        topologies, gate fidelities, and more. A quantum computer also has the ability to
        run quantum programs.

        A quantum computer can be a real Rigetti QPU that uses superconducting transmon
        qubits to run quantum programs, or it can be an emulator like the QVM with
        noise models and mimicked topologies.

        :param name: A string identifying this particular quantum computer.
        :param qam: A quantum abstract machine which handles executing quantum programs. This
            dispatches to a QVM or QPU.
        :param symmetrize_readout: Whether to apply readout error symmetrization. See
            :py:func:`run_symmetrized_readout` for a complete description.
        """
        self.name = name
        self.qam = qam
        self.compiler = compiler

        self.symmetrize_readout = symmetrize_readout

    @property
    def quantum_processor(self) -> AbstractQuantumProcessor:
        """The quantum processor associated with this quantum computer."""
        return self.compiler.quantum_processor

    def qubits(self) -> list[int]:
        """Return a sorted list of the quantum_processor's qubits.

        See :py:func:`AbstractQuantumProcessor.qubits` for more.
        """
        return self.compiler.quantum_processor.qubits()

    def qubit_topology(self) -> nx.graph:
        """Return a NetworkX graph representation of this QuantumComputer's quantum_processor's qubit connectivity.

        See :py:func:`AbstractQuantumProcessor.qubit_topology` for more.
        """
        return self.compiler.quantum_processor.qubit_topology()

    def to_compiler_isa(self) -> CompilerISA:
        """Return a ``CompilerISA`` for this QuantumComputer's quantum_processor.

        See :py:func:`AbstractQuantumProcessor.to_compiler_isa` for more.
        """
        return self.compiler.quantum_processor.to_compiler_isa()

    def run(
        self, executable: QuantumExecutable, memory_map: Optional[MemoryMap] = None, **kwargs: Any
    ) -> QAMExecutionResult:
        """Run a quil executable.

        :param executable: The program to run, previously compiled as needed for its target QAM.
        :param memory_map: A mapping of memory regions to a list containing the values to be written into that memory
            region for the run.
        :return: execution result including readout data.
        """
        return self.qam.run(executable, memory_map, **kwargs)

    def run_with_memory_map_batch(
        self, executable: QuantumExecutable, memory_maps: Iterable[MemoryMap], **kwargs: Any
    ) -> list[QAMExecutionResult]:
        """Run a QuantumExecutable with one or more memory_map.

        Returns a list of results corresponding to the length and order of the given MemoryMaps.

        How these programs are batched and executed is determined by the executor. See their respective documentation
        for details.

        Returns a list of ``QAMExecutionResult``, which can be used to fetch
        results in ``QAM#get_result``.
        """
        handles = self.qam.execute_with_memory_map_batch(executable, memory_maps, **kwargs)
        return [self.qam.get_result(handle) for handle in handles]

    def calibrate(self, experiment: Experiment) -> list[ExperimentResult]:
        """Perform readout calibration on the various multi-qubit observables involved in the provided ``Experiment``.

        :param experiment: The ``Experiment`` to calibrate readout error for.
        :return: A list of ``ExperimentResult`` objects that contain the expectation values that
            correspond to the scale factors resulting from symmetric readout error.
        """
        calibration_experiment = experiment.generate_calibration_experiment()
        return self.run_experiment(calibration_experiment)

    def run_experiment(
        self,
        experiment: Experiment,
        memory_map: Optional[MemoryMap] = None,
    ) -> list[ExperimentResult]:
        """Run an ``Experiment`` on a QVM or QPU backend.

        An ``Experiment`` is composed of:

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

            3. Return the list of ``ExperimentResult`` objects.

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
            # TODO: add support for grouped ExperimentSettings
            if len(settings) > 1:
                raise ValueError("We only support length-1 settings for now.")
            setting = settings[0]

            qubits = cast(list[int], setting.out_operator.get_qubits())
            experiment_setting_memory_map = experiment.build_setting_memory_map(setting)
            symmetrization_memory_maps = experiment.build_symmetrization_memory_maps(qubits)
            merged_memory_maps = merge_memory_map_lists([experiment_setting_memory_map], symmetrization_memory_maps)

            all_bitstrings = []
            for merged_memory_map in merged_memory_maps:
                final_memory_map = {**memory_map, **merged_memory_map}
                executable_copy = executable.copy()
                bitstrings = self.run(executable_copy, memory_map=final_memory_map).readout_data.get("ro")
                if bitstrings is None:
                    raise ValueError("No readout data returned.")

                if "symmetrization" in final_memory_map:
                    bitmask = np.array(np.array(final_memory_map["symmetrization"]) / np.pi, dtype=int)
                    bitstrings = np.bitwise_xor(bitstrings, bitmask)
                all_bitstrings.append(bitstrings)
            symmetrized_bitstrings = np.concatenate(all_bitstrings)

            joint_expectations = [experiment.get_meas_registers(qubits)]
            if setting.additional_expectations:
                joint_expectations += setting.additional_expectations

            expectations = bitstrings_to_expectations(symmetrized_bitstrings, joint_expectations=joint_expectations)

            means = np.mean(expectations, axis=0)
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
        meas_qubits: Optional[Sequence[QubitDesignator]] = None,
    ) -> np.ndarray:
        r"""Run a quil program in such a way that the readout error is made symmetric.

        Enforcing symmetric readout error is useful in simplifying the assumptions in some near term error mitigation
        strategies, see ``measure_observables`` for more information.

        The simplest example is for one qubit. In a noisy quantum_processor, the probability of accurately reading the
        0 state might be higher than that of the 1 state; due to e.g. amplitude damping. This makes correcting for
        readout more difficult. In the simplest case, this function runs the program normally ``(trials//2)`` times.
        The other half of the time, it will insert an ``X`` gate prior to any ``MEASURE`` instruction and then flip th
        measured classical bit back. Overall this has the effect of symmetrizing the readout error.

        The details. Consider preparing the input bitstring ``|i>`` (in the computational basis) and measuring in the
        Z basis. Then the Confusion matrix for the readout error is specified by the probabilities

             p(j|i) := Pr(measured = j | prepared = i ).

        In the case of a single qubit i,j \in [0,1] then:

        * there is no readout error if p(0|0) = p(1|1) = 1.
        * the readout error is symmetric if p(0|0) = p(1|1) = 1 - epsilon.
        * the readout error is asymmetric if p(0|0) != p(1|1).

        If your quantum computer has this kind of asymmetric readout error then ``qc.run_symmetrized_readout`` will
        symmetrize the readout error.

        The readout error above is only asymmetric on a single bit. In practice the confusion matrix on n bits need not
        be symmetric, e.g. for two qubits p(ij|ij) != 1 - epsilon for all i,j. In these situations a more sophisticated
        means of symmetrization is needed; and we use orthogonal arrays (OA) built from Hadamard matrices.

        The symmetrization types are specified by an int; the types available are:
        -1 -- exhaustive symmetrization uses every possible combination of flips
        0 -- trivial that is no symmetrization
        1 -- symmetrization using an OA with strength 1
        2 -- symmetrization using an OA with strength 2
        3 -- symmetrization using an OA with strength 3
        In the context of readout symmetrization the strength of the orthogonal array enforces
        the symmetry of the marginal confusion matrices.

        By default a strength 3 OA is used; this ensures expectations of the form ``<b_k . b_j . b_i>`` for bits any
        bits i,j,k will have symmetric readout errors. Here expectation of a random variable x as is denote
        ``<x> = sum_i Pr(i) x_i``. It turns out that a strength 3 OA is also a strength 2 and strength 1 OA it also
        ensures ``<b_j . b_i>`` and ``<b_i>`` have symmetric readout errors for any bits b_j and b_i.

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
            meas_qubits = list(program.get_qubit_indices())

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
                f"chosen.",
                stacklevel=2,
            )

        results = _measure_bitstrings(self, sym_programs, meas_qubits, num_shots_per_prog)

        return _consolidate_symmetrization_outputs(results, flip_arrays)

    def compile(
        self,
        program: Program,
        to_native_gates: bool = True,
        optimize: bool = True,
        *,
        protoquil: Optional[bool] = None,
    ) -> QuantumExecutable:
        """Provide a high-level interface for program compilation.

        Compilation currently consists of two stages. Please see the :py:class:`AbstractCompiler` docs for more
        information. This function does all stages of compilation.

        Right now both ``to_native_gates`` and ``optimize`` must be either both set or both unset. More modular
        compilation passes may be available in the future.

        Additionally, a call to compile also calls the ``reset`` method if one is running on the QPU. This is a bit of
        a sneaky hack to guard against stale compiler connections, but shouldn't result in any material hit to
        performance (especially when taking advantage of parametric compilation for hybrid applications).

        :param program: A Program
        :param to_native_gates: Whether to compile non-native gates to native gates.
        :param optimize: Whether to optimize the program to reduce the number of operations.
        :param protoquil: Whether to restrict the input program to and the compiled program
            to protoquil (executable on QPU). A value of ``None`` means defer to server.
        :return: An executable binary suitable for passing to :py:func:`QuantumComputer.run`.
        """
        flags = [to_native_gates, optimize]
        if any(flags) and not all(flags):
            raise ValueError("Must turn to_native_gates and optimize on or off together")

        quilc = all(flags)

        if quilc:
            nq_program = self.compiler.quil_to_native_quil(program, protoquil=protoquil)
        else:
            nq_program = program

        return self.compiler.native_quil_to_executable(nq_program)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'QuantumComputer[name="{self.name}"]'


def list_quantum_computers(
    qpus: bool = True,
    qvms: bool = True,
    timeout: float = 10.0,
    client_configuration: Optional[QCSClient] = None,
) -> list[str]:
    """List the names of available quantum computers.

    :param qpus: Whether to include QPUs in the list.
    :param qvms: Whether to include QVMs in the list.
    :param timeout: Time limit for request, in seconds.
    :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
    """
    client_configuration = client_configuration or QCSClient.load()
    qc_names: list[str] = []
    if qpus:
        qc_names += list_quantum_processors(client=client_configuration, timeout=timeout)

    if qvms:
        qc_names += ["9q-square-qvm", "9q-square-noisy-qvm"]

    return qc_names


def _parse_name(name: str, as_qvm: Optional[bool], noisy: Optional[bool]) -> tuple[str, Optional[str], bool]:
    """Try to figure out whether we're getting a (noisy) qvm, and the associated qpu name.

    See :py:func:`get_qc` for examples of valid names + flags.
    """
    qvm_type: Optional[str]
    parts = name.split("-")
    if len(parts) >= 2 and parts[-2] == "noisy" and parts[-1] in ["qvm", "pyqvm"]:
        if as_qvm is not None and (not as_qvm):
            raise ValueError(
                "The provided qc name indicates you are getting a noisy QVM, " "but you have specified `as_qvm=False`"
            )

        if noisy is not None and (not noisy):
            raise ValueError(
                "The provided qc name indicates you are getting a noisy QVM, " "but you have specified `noisy=False`"
            )

        qvm_type = parts[-1]
        noisy = True
        prefix = "-".join(parts[:-2])
        return prefix, qvm_type, noisy

    if len(parts) >= 1 and parts[-1] in ["qvm", "pyqvm"]:
        if as_qvm is not None and (not as_qvm):
            raise ValueError(
                "The provided qc name indicates you are getting a QVM, " "but you have specified `as_qvm=False`"
            )
        qvm_type = parts[-1]
        if noisy is None:
            noisy = False
        prefix = "-".join(parts[:-1])
        return prefix, qvm_type, noisy

    if as_qvm is not None and as_qvm:
        qvm_type = "qvm"
    else:
        qvm_type = None

    if noisy is None:
        noisy = False

    return name, qvm_type, noisy


def _canonicalize_name(prefix: str, qvm_type: Optional[str], noisy: bool) -> str:
    """Take the output of _parse_name to create a canonical name."""
    if noisy:
        noise_suffix = "-noisy"
    else:
        noise_suffix = ""

    if qvm_type is None:
        qvm_suffix = ""
    elif qvm_type == "qvm":
        qvm_suffix = "-qvm"
    elif qvm_type == "pyqvm":
        qvm_suffix = "-pyqvm"
    else:
        raise ValueError(f"Unknown qvm_type {qvm_type}")

    name = f"{prefix}{noise_suffix}{qvm_suffix}"
    return name


def _get_qvm_or_pyqvm(
    *,
    qvm_type: str,
    qvm_client: Optional[QVMClient],
    noise_model: Optional[NoiseModel],
    quantum_processor: Optional[AbstractQuantumProcessor],
    execution_timeout: float,
) -> Union[QVM, PyQVM]:
    if qvm_type == "qvm":
        return QVM(noise_model=noise_model, timeout=execution_timeout, client=qvm_client)
    elif qvm_type == "pyqvm":
        if quantum_processor is None:
            raise ValueError("Cannot construct a PyQVM without a quantum_processor.")
        return PyQVM(n_qubits=quantum_processor.qubit_topology().number_of_nodes())

    raise ValueError(f"Unknown qvm type {qvm_type}")


def _get_qvm_qc(
    *,
    client_configuration: QCSClient,
    name: str,
    qvm_type: str,
    quantum_processor: AbstractQuantumProcessor,
    compiler_timeout: float,
    execution_timeout: float,
    noise_model: Optional[NoiseModel],
    quilc_client: Optional[QuilcClient] = None,
    qvm_client: Optional[QVMClient] = None,
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
            qvm_type=qvm_type,
            noise_model=noise_model,
            quantum_processor=quantum_processor,
            execution_timeout=execution_timeout,
            qvm_client=qvm_client,
        ),
        compiler=QVMCompiler(
            quantum_processor=quantum_processor,
            timeout=compiler_timeout,
            client_configuration=client_configuration,
            quilc_client=quilc_client,
        ),
    )


def _get_qvm_with_topology(
    *,
    client_configuration: QCSClient,
    name: str,
    topology: nx.Graph,
    noisy: bool,
    qvm_type: str,
    compiler_timeout: float,
    execution_timeout: float,
    quilc_client: Optional[QuilcClient] = None,
    qvm_client: Optional[QVMClient] = None,
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
        quilc_client=quilc_client,
        qvm_client=qvm_client,
    )


def _get_9q_square_qvm(
    *,
    client_configuration: QCSClient,
    name: str,
    noisy: bool,
    qvm_type: str,
    compiler_timeout: float,
    execution_timeout: float,
    quilc_client: Optional[QuilcClient] = None,
    qvm_client: Optional[QVMClient] = None,
) -> QuantumComputer:
    """Nine-qubit 3x3 square lattice.

    This uses a "generic" lattice not tied to any specific quantum_processor. 9 qubits is large enough
    to do vaguely interesting algorithms and small enough to simulate quickly.

    :param client_configuration: Client configuration.
    :param name: The name of this QVM
    :param noisy: Whether to construct a noisy quantum computer
    :param qvm_type: The type of QVM. Either 'qvm' or 'pyqvm'.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param execution_timeout: Time limit for execution requests, in seconds.
    :return: A pre-configured QuantumComputer
    """
    topology = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    return _get_qvm_with_topology(
        client_configuration=client_configuration,
        name=name,
        topology=topology,
        noisy=noisy,
        qvm_type=qvm_type,
        compiler_timeout=compiler_timeout,
        execution_timeout=execution_timeout,
        quilc_client=quilc_client,
        qvm_client=qvm_client,
    )


def _get_unrestricted_qvm(
    *,
    client_configuration: QCSClient,
    name: str,
    noisy: bool,
    n_qubits: int,
    qvm_type: str,
    compiler_timeout: float,
    execution_timeout: float,
    quilc_client: Optional[QuilcClient] = None,
    qvm_client: Optional[QVMClient] = None,
) -> QuantumComputer:
    """QVM with a fully-connected topology.

    This is obviously the least realistic QVM, but who am I to tell users what they want.

    :param client_configuration: Client configuration.
    :param name: The name of this QVM
    :param noisy: Whether to construct a noisy quantum computer
    :param n_qubits: 34 qubits ought to be enough for anybody.
    :param qvm_type: The type of QVM. Either 'qvm' or 'pyqvm'.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param execution_timeout: Time limit for execution requests, in seconds.
    :return: A pre-configured QuantumComputer
    """
    topology = nx.complete_graph(n_qubits)
    return _get_qvm_with_topology(
        client_configuration=client_configuration,
        name=name,
        topology=topology,
        noisy=noisy,
        qvm_type=qvm_type,
        compiler_timeout=compiler_timeout,
        execution_timeout=execution_timeout,
        quilc_client=quilc_client,
        qvm_client=qvm_client,
    )


def _get_qvm_based_on_real_quantum_processor(
    *,
    client_configuration: QCSClient,
    name: str,
    quantum_processor: QCSQuantumProcessor,
    noisy: bool,
    qvm_type: str,
    compiler_timeout: float,
    execution_timeout: float,
    quilc_client: Optional[QuilcClient] = None,
    qvm_client: Optional[QVMClient] = None,
) -> QuantumComputer:
    """QVM based on a real quantum_processor.

    This is the most realistic QVM.

    :param client_configuration: Client configuration.
    :param name: The full name of this QVM
    :param quantum_processor: The quantum_processor from :py:func:`get_lattice`.
    :param noisy: Whether to construct a noisy quantum computer by using the quantum_processor's
        associated noise model.
    :param qvm_type: The type of QVM. Either 'qvm' or 'pyqvm'.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param execution_timeout: Time limit for execution requests, in seconds.
    :return: A pre-configured QuantumComputer based on the named quantum_processor.
    """
    if noisy:
        noise_model = quantum_processor.noise_model
    else:
        noise_model = None
    return _get_qvm_qc(
        client_configuration=client_configuration,
        name=name,
        quantum_processor=quantum_processor,
        noise_model=noise_model,
        qvm_type=qvm_type,
        compiler_timeout=compiler_timeout,
        execution_timeout=execution_timeout,
        quilc_client=quilc_client,
        qvm_client=qvm_client,
    )


def get_qc(
    name: str,
    *,
    as_qvm: Optional[bool] = None,
    noisy: Optional[bool] = None,
    compiler_timeout: float = 30.0,
    execution_timeout: float = 30.0,
    client_configuration: Optional[QCSClient] = None,
    endpoint_id: Optional[str] = None,
    quilc_client: Optional[QuilcClient] = None,
    qvm_client: Optional[QVMClient] = None,
) -> QuantumComputer:
    """Get a quantum computer.

    A quantum computer is an object of type :py:class:`QuantumComputer` and can be backed
    either by a QVM simulator ("Quantum/Quil Virtual Machine") or a physical Rigetti QPU ("Quantum
    Processing Unit") made of superconducting qubits.

    You can choose the quantum computer to target through a combination of its name and optional
    flags. There are multiple ways to get the same quantum computer. The following are equivalent::

        >>> qc = get_qc("Aspen-M-3-qvm")  # doctest: +SKIP
        >>> qc = get_qc("Aspen-M-3", as_qvm=True)  # doctest: +SKIP

    and will construct a simulator of an Aspen-M-3 lattice. We also provide a means for constructing
    generic quantum simulators that are not related to a given piece of Rigetti hardware::

        >>> qc = get_qc("9q-square-qvm")
        >>> qc = get_qc("9q-square", as_qvm=True)

    Finally, you can get request a QVM with "no" topology of a given number of qubits
    (technically, it's a fully connected graph among the given number of qubits) with::

        >>> qc = get_qc("5q-qvm") # or "6q-qvm", or "34q-qvm", ...

    These less-realistic, fully-connected QVMs will also be more lenient on what types of programs
    they will ``run``. Specifically, you do not need to do any compilation. For the other, realistic
    QVMs you must use :py:func:`qc.compile` or :py:func:`qc.compiler.native_quil_to_executable`
    prior to :py:func:`qc.run`.

    The Rigetti QVM must be downloaded from https://www.rigetti.com/forest and run as a server
    alongside your python program. To use pyQuil's built-in QVM, replace all ``"-qvm"`` suffixes
    with ``"-pyqvm"``::

        >>> qc = get_qc("5q-pyqvm")

    Redundant flags are acceptable, but conflicting flags will raise an exception::

        >>> qc = get_qc("9q-square-qvm") # qc is fully specified by its name
        >>> qc = get_qc("9q-square-qvm", as_qvm=True) # redundant, but ok
        >>> qc = get_qc("9q-square-qvm", as_qvm=False) # Error!
        Traceback (most recent call last):
        ValueError: The provided qc name indicates you are getting a QVM, but you have specified `as_qvm=False`

    Use :py:func:`list_quantum_computers` to retrieve a list of known qc names.

    This method is provided as a convenience to quickly construct and use QVM's and QPU's.
    Power users may wish to have more control over the specification of a quantum computer
    (e.g. custom noise models, bespoke topologies, etc.). This is possible by constructing
    a :py:class:`QuantumComputer` object by hand. Please refer to the documentation on
    :py:class:`QuantumComputer` for more information.

    :param name: The name of the desired quantum computer. This should correspond to a name
        returned by :py:func:`list_quantum_computers`. Names ending in "-qvm" will return
        a QVM. Names ending in "-pyqvm" will return a :py:class:`PyQVM`. Names ending in
        "-noisy-qvm" will return a QVM with a noise model. Otherwise, we will return a QPU with
        the given name.
    :param as_qvm: An optional flag to force construction of a QVM (instead of a QPU). If
        specified and set to ``True``, a QVM-backed quantum computer will be returned regardless
        of the name's suffix
    :param noisy: An optional flag to force inclusion of a noise model. If
        specified and set to ``True``, a quantum computer with a noise model will be returned
        regardless of the name's suffix. The generic QVM noise model is simple T1 and T2 noise
        plus readout error. See :py:func:`~pyquil.noise.decoherence_noise_with_asymmetric_ro`.
        Note, we currently do not support noise models based on QCS hardware; a value of
        `True`` will result in an error if the requested QPU is a QCS hardware QPU.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param execution_timeout: Time limit for execution requests, in seconds.
    :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        For more information on setting up QCS credentials, see documentation for using the QCS CLI:
        [https://docs.rigetti.com/qcs/guides/using-the-qcs-cli#configuring-credentials].
    :param endpoint_id: Optional quantum processor endpoint ID, as used in the `QCS API Docs`_.

    :return: A pre-configured QuantumComputer

    .. _QCS API Docs: https://docs.api.qcs.rigetti.com/#tag/endpoints
    """
    client_configuration = client_configuration or QCSClient.load()

    # 1. Parse name, check for redundant options, canonicalize names.
    prefix, qvm_type, noisy = _parse_name(name, as_qvm, noisy)
    del as_qvm  # do not use after _parse_name
    name = _canonicalize_name(prefix, qvm_type, noisy)

    # 2. Check for unrestricted {n}q-qvm
    ma = re.fullmatch(r"(\d+)q", prefix)
    if ma is not None:
        n_qubits = int(ma.group(1))
        if qvm_type is None:
            raise ValueError("Please name a valid quantum_processor or run as a QVM")
        return _get_unrestricted_qvm(
            client_configuration=client_configuration,
            name=name,
            noisy=noisy,
            n_qubits=n_qubits,
            qvm_type=qvm_type,
            compiler_timeout=compiler_timeout,
            execution_timeout=execution_timeout,
            quilc_client=quilc_client,
            qvm_client=qvm_client,
        )

    # 3. Check for "9q-square" qvm
    if prefix == "9q-square":
        if qvm_type is None:
            raise ValueError("The quantum_processor '9q-square' is only available as a QVM")
        return _get_9q_square_qvm(
            client_configuration=client_configuration,
            name=name,
            noisy=noisy,
            qvm_type=qvm_type,
            compiler_timeout=compiler_timeout,
            execution_timeout=execution_timeout,
            quilc_client=quilc_client,
            qvm_client=qvm_client,
        )

    if noisy:
        raise ValueError(
            "pyQuil currently does not support initializing a noisy QuantumComputer "
            "based on a QCSQuantumProcessor. Change noisy to False or specify the name of a QVM."
        )

    # 4. Not a special case, query the web for information about this quantum_processor.
    quantum_processor = get_qcs_quantum_processor(
        quantum_processor_id=prefix, client_configuration=client_configuration
    )
    if qvm_type is not None:
        # 4.1 QVM based on a real quantum_processor.
        return _get_qvm_based_on_real_quantum_processor(
            client_configuration=client_configuration,
            name=name,
            quantum_processor=quantum_processor,
            noisy=False,
            qvm_type=qvm_type,
            compiler_timeout=compiler_timeout,
            execution_timeout=execution_timeout,
            quilc_client=quilc_client,
            qvm_client=qvm_client,
        )
    else:
        qpu = QPU(
            quantum_processor_id=quantum_processor.quantum_processor_id,
            timeout=execution_timeout,
            client_configuration=client_configuration,
            endpoint_id=endpoint_id,
        )
        compiler = QPUCompiler(
            quantum_processor_id=prefix,
            quantum_processor=quantum_processor,
            timeout=compiler_timeout,
            client_configuration=client_configuration,
            quilc_client=quilc_client,
        )

        return QuantumComputer(name=name, qam=qpu, compiler=compiler)


def _port_used(host: str, port: int) -> bool:
    """Check if a (TCP) port is listening.

    :param host: Host address to check.
    :param port: TCP port to check.

    :returns: ``True`` if a process is listening on the specified host/port, ``False`` otherwise
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        return True
    except ConnectionRefusedError:
        return False
    finally:
        s.close()


@contextmanager
def local_forest_runtime(
    *,
    host: str = "127.0.0.1",
    qvm_port: int = 5000,
    quilc_port: int = 5555,
    use_protoquil: bool = False,
) -> Iterator[tuple[Optional[subprocess.Popen], Optional[subprocess.Popen]]]:
    """Context manager for local QVM and QUIL compiler.

    You must first have installed the `qvm` and `quilc` executables from
    the forest SDK. [https://www.rigetti.com/forest]

    This context manager will ensure that the designated ports are not used, start up `qvm` and
    `quilc` processes if possible and terminate them when the context is exited.
    If one of the ports is in use, a ``RuntimeWarning`` will be issued and the `qvm`/`quilc` process
    won't be started.

    .. note::
        Only processes started by this context manager will be terminated on exit, no external
        process will be touched.


    >>> from pyquil import get_qc, Program
    >>> from pyquil.gates import CNOT, Z
    >>> from pyquil.api import local_forest_runtime
    >>>
    >>> qvm = get_qc("9q-square-qvm")
    >>> prog = Program(Z(0), CNOT(0, 1))
    >>>
    >>> with local_forest_runtime():  # doctest: +SKIP
    >>>     results = qvm.run(prog)   # doctest: +SKIP

    :param host: Host on which `qvm` and `quilc` should listen on.
    :param qvm_port: Port which should be used by `qvm`.
    :param quilc_port: Port which should be used by `quilc`.
    :param use_protoquil: Restrict input/output to protoquil.

    .. warning::
        If ``use_protoquil`` is set to ``True`` language features you need
        may be disabled. Please use it with caution.

    :raises: FileNotFoundError: If either executable is not installed.

    :returns: The returned tuple contains two ``subprocess.Popen`` objects
        for the `qvm` and the `quilc` processes.  If one of the designated
        ports is in use, the process won't be started and the respective
        value in the tuple will be ``None``.
    """
    qvm: Optional[subprocess.Popen] = None
    quilc: Optional[subprocess.Popen] = None

    # If the host we should listen to is 0.0.0.0, we replace it
    # with 127.0.0.1 to use a valid IP when checking if the port is in use.
    if _port_used(host if host != "0.0.0.0" else "127.0.0.1", qvm_port):  # noqa: S104: prevents connection to 0.0.0.0
        warning_msg = f"Unable to start qvm server, since the specified port {qvm_port} is in use."
        warnings.warn(RuntimeWarning(warning_msg), stacklevel=2)
    else:
        qvm_cmd = ["qvm", "-S", "--host", host, "-p", str(qvm_port)]
        qvm = subprocess.Popen(qvm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603: input valid

    if _port_used(host if host != "0.0.0.0" else "127.0.0.1", quilc_port):  # noqa: S104: prevents connection to 0.0.0.0
        warning_msg = f"Unable to start quilc server, since the specified port {quilc_port} is in use."
        warnings.warn(RuntimeWarning(warning_msg), stacklevel=2)
    else:
        quilc_cmd = ["quilc", "--host", host, "-p", str(quilc_port), "-R"]

        if use_protoquil:
            quilc_cmd += ["-P"]

        quilc = subprocess.Popen(quilc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603

    # Return context
    try:
        yield (qvm, quilc)

    finally:
        # Exit. Release resource
        if qvm:
            qvm.terminate()
        if quilc:
            quilc.terminate()


def _flip_array_to_prog(flip_array: tuple[bool], qubits: Sequence[QubitDesignator]) -> Program:
    """Generate a pre-measurement program that flips the qubit state according to the flip_array of bools.

    This is used, for example, in symmetrization to produce programs which flip a select subset
    of qubits immediately before measurement.

    :param flip_array: tuple of booleans specifying whether the qubit in the corresponding index
        should be flipped or not.
    :param qubits: list specifying the qubits in order corresponding to the flip_array
    :return: Program which flips each qubit (i.e. instructs RX(pi, q)) according to the flip_array.
    """
    if len(flip_array) != len(qubits):
        raise ValueError("Mismatch of qubits and operations")

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
    program: Program, meas_qubits: Sequence[QubitDesignator], symm_type: int = 3
) -> tuple[list[Program], list[np.ndarray]]:
    """Generate new programs that flip the measured qubits with an X gate in order to symmetrize readout.

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
    else:
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


def _consolidate_symmetrization_outputs(outputs: list[np.ndarray], flip_arrays: list[np.ndarray]) -> np.ndarray:
    """Flip output bits and consolidate results into new bitarrays from symmetrization program bitarray results.

    :param outputs: a list of the raw bitarrays resulting from running a list of symmetrized
        programs; for example, the results returned from _measure_bitstrings
    :param flip_arrays: a list of boolean arrays in one-to-one correspondence with the list of
        outputs indicating which qubits where flipped before each bitarray was measured.
    :return: an np.ndarray consisting of the consolidated bitarray outputs which can be treated as
        the symmetrized outputs of the original programs passed into a symmetrization method. See
        estimate_observables for example usage.
    """
    if len(outputs) != len(flip_arrays):
        raise ValueError("The length of outputs must equal the length of flip_arrays")

    output = []
    for bitarray, flip_array in zip(outputs, flip_arrays):
        if len(flip_array) == 0:
            output.append(bitarray)
        else:
            output.append(bitarray ^ flip_array)

    return np.vstack(output)


def _measure_bitstrings(
    qc: QuantumComputer, programs: list[Program], meas_qubits: Sequence[QubitDesignator], num_shots: int = 600
) -> list[np.ndarray]:
    """Append measure instructions, run the program, and accumulate the resulting bitarrays.

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
        result = qc.run(executable)
        shot_values = result.readout_data.get("ro")
        if shot_values is None:
            raise RuntimeError("Got no readout data from the quantum computer.")
        results.append(shot_values)
    return results


def _construct_orthogonal_array(num_qubits: int, strength: int = 3) -> np.ndarray:
    """Return an Orthogonal Array (OA) based on the given strength and number of qubits.

    Given a strength and number of qubits, this function returns an Orthogonal Array (OA) on 'n' or more qubits.
    Sometimes, the size of the returned array is larger than the number of qubits; typically, it is the next power of
    two relative to the number of qubits. This is corrected later in the code flow.

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
    else:  # strength == 3
        flip_matrix = _construct_strength_three_orthogonal_array(num_qubits)

    return flip_matrix


def _next_power_of_2(x: int) -> int:
    return cast(int, 1 if x == 0 else 2 ** (x - 1).bit_length())


# The code below is directly copied from scipy see https://bit.ly/2RjAHJz, the docstrings have
# been modified.
def hadamard(n: int, dtype: np.dtype = int) -> np.ndarray:  # type: ignore
    """Construct a Hadamard matrix.

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
    if 2**lg2 != n:
        raise ValueError("n must be an positive integer, and n must be a power of 2")

    H = np.array([[1]], dtype=dtype)

    # Sylvester's construction
    for _ in range(0, lg2):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    return H


def _construct_strength_three_orthogonal_array(num_qubits: int) -> np.ndarray:
    r"""Return an Orthogonal Array (OA) on 'n' qubits where n is the next power of two relative to num_qubits.

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
    orthogonal_array: np.ndarray = ((Hfold + 1) / 2).astype(int)
    return orthogonal_array


def _construct_strength_two_orthogonal_array(num_qubits: int) -> np.ndarray:
    r"""Return an Orthogonal Array (OA) on 'n-1' qubits where 4*(n-1) - 1 exceeds num_qubits.

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
    orthogonal_array: np.ndarray = ((-H[1:, :].T + 1) / 2).astype(int)
    return orthogonal_array


def _check_min_num_trials_for_symmetrized_readout(num_qubits: int, trials: int, symm_type: int) -> int:
    """Set the minimum number of trials; it is desirable to have hundreds or thousands of trials more than the minimum.

    :param num_qubits: number of qubits to symmetrize
    :param trials: number of trials
    :param symm_type: symmetrization type see
    :return: possibly modified number of trials
    """
    if symm_type < -1 or symm_type > 3:
        raise ValueError("symm_type must be one of the following ints [-1, 0, 1, 2, 3].")

    if symm_type == -1:
        min_num_trials = 2**num_qubits
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
        warnings.warn(f"Number of trials was too low, it is now {trials}.", stacklevel=2)
    return trials

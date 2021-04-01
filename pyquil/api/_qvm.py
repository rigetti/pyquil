##############################################################################
# Copyright 2016-2018 Rigetti Computing
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
import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union, cast, Tuple

import numpy as np
from httpx import RequestError
from qcs_api_client.client import QCSClientConfiguration

from pyquil._version import pyquil_version
from pyquil.api import QuantumExecutable
from pyquil.api._compiler import QVMCompiler
from pyquil.api._error_reporting import _record_call
from pyquil.api._qam import QAM
from pyquil.api._qvm_client import (
    QVMClient,
    MeasureExpectationRequest,
    GetWavefunctionRequest,
    RunAndMeasureProgramRequest,
    RunProgramRequest,
)
from pyquil.gates import MOVE
from pyquil.noise import NoiseModel, apply_noise_model
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quantum_processor import QCSQuantumProcessor
from pyquil.quil import Program, get_classical_addresses_from_program, percolate_declares
from pyquil.quilatom import MemoryReference
from pyquil.wavefunction import Wavefunction


class QVMVersionMismatch(Exception):
    pass


class QVMNotRunning(Exception):
    pass


def check_qvm_version(version: str) -> None:
    """
    Verify that there is no mismatch between pyquil and QVM versions.

    :param version: The version of the QVM
    """
    major, minor, patch = map(int, version.split("."))
    if major == 1 and minor < 8:
        raise QVMVersionMismatch(
            "Must use QVM >= 1.8.0 with pyquil >= 2.8.0, but you " f"have QVM {version} and pyquil {pyquil_version}"
        )


class QVMConnection(object):
    """
    Represents a connection to the QVM.
    """

    @_record_call
    def __init__(
        self,
        quantum_processor: Optional[QCSQuantumProcessor] = None,
        gate_noise: Optional[Tuple[float, float, float]] = None,
        measurement_noise: Optional[Tuple[float, float, float]] = None,
        random_seed: Optional[int] = None,
        timeout: float = 5.0,
        client_configuration: Optional[QCSClientConfiguration] = None,
    ):
        """
        Constructor for QVMConnection. Sets up any necessary security, and establishes the noise
        model to use.

        :param quantum_processor: The optional quantum_processor, from which noise will be added by default to all
            programs run on this instance.
        :param gate_noise: A tuple of three numbers [Px, Py, Pz] indicating the probability of an X,
            Y, or Z gate getting applied to each qubit after a gate application or reset.
        :param measurement_noise: A tuple of three numbers [Px, Py, Pz] indicating the probability
            of an X, Y, or Z gate getting applied before a a measurement.
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
            automatically generated seed) or a non-negative integer.
        :param timeout: Time limit for requests, in seconds.
        :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        """
        client_configuration = client_configuration or QCSClientConfiguration.load()
        self._qvm_client = QVMClient(client_configuration=client_configuration, request_timeout=timeout)

        if (quantum_processor is not None and quantum_processor.noise_model is not None) and (
            gate_noise is not None or measurement_noise is not None
        ):
            raise ValueError(
                """
You have attempted to supply the QVM with both a quantum_processor noise model
(by having supplied a quantum_processor argument), as well as either gate_noise
or measurement_noise. At this time, only one may be supplied.

To read more about supplying noise to the QVM, see
http://pyquil.readthedocs.io/en/latest/noise_models.html#support-for-noisy-gates-on-the-rigetti-qvm.
"""
            )

        if quantum_processor is not None and quantum_processor.noise_model is None:
            warnings.warn(
                """
You have supplied the QVM with a quantum_processor that does not have a noise model. No noise will be added to
programs run on this QVM.
"""
            )

        self.noise_model = quantum_processor.noise_model if quantum_processor else None
        self.compiler = (
            QVMCompiler(
                quantum_processor=quantum_processor,
                timeout=timeout,
                client_configuration=client_configuration,
            )
            if quantum_processor
            else None
        )

        validate_noise_probabilities(gate_noise)
        validate_noise_probabilities(measurement_noise)
        self.gate_noise = gate_noise
        self.measurement_noise = measurement_noise

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, int) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int")

        self.connect()

    def connect(self) -> None:
        try:
            version_dict = self.get_version_info()
            check_qvm_version(version_dict)
        except RequestError:
            raise QVMNotRunning(f"No QVM server running at {self._qvm_client.base_url}")

    @_record_call
    def get_version_info(self) -> str:
        """
        Return version information for the QVM.

        :return: String with version information
        """
        return self._qvm_client.get_version()

    @_record_call
    def run(
        self,
        quil_program: Program,
        classical_addresses: Optional[Sequence[int]] = None,
        trials: int = 1,
    ) -> List[List[int]]:
        """
        Run a Quil program multiple times, accumulating the values deposited in
        a list of classical addresses.

        :param quil_program: A Quil program.
        :param classical_addresses: The classical memory to retrieve. Specified as a list of
            integers that index into a readout register named ``ro``. This function--and
            particularly this argument--are included for backwards compatibility and will
            be removed in the future.
        :param trials: Number of shots to collect.
        :return: A list of dictionaries of bits. Each dictionary corresponds to the values in
            `classical_addresses`.
        """
        if classical_addresses is None:
            caddresses: Mapping[str, Sequence[int]] = get_classical_addresses_from_program(quil_program)

        else:
            caddresses = {"ro": classical_addresses}

        request = qvm_run_request(
            quil_program,
            caddresses,
            trials,
            self.measurement_noise,
            self.gate_noise,
            self.random_seed,
        )
        response = self._qvm_client.run_program(request)
        buffers = {key: np.array(val) for key, val in response.results.items()}

        if len(buffers) == 0:
            return []
        if "ro" in buffers:
            return cast(List[List[int]], buffers["ro"].tolist())

        raise ValueError(
            "You are using QVMConnection.run with multiple readout registers not "
            "named `ro`. Please use the new `QuantumComputer` abstraction."
        )

    @_record_call
    def run_and_measure(self, quil_program: Program, qubits: Sequence[int], trials: int = 1) -> List[List[int]]:
        """
        Run a Quil program once to determine the final wavefunction, and measure multiple times.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the outcomes sampled from *different* ``run_and_measure`` calls *generally sample
            different bitstring distributions*.

        :param quil_program: A Quil program.
        :param qubits: A list of qubits.
        :param trials: Number of shots to collect.
        :return: A list of a list of bits.
        """
        request = self._run_and_measure_request(quil_program, qubits, trials)
        return self._qvm_client.run_and_measure_program(request).results

    @_record_call
    def _run_and_measure_request(
        self,
        quil_program: Program,
        qubits: Sequence[int],
        trials: int,
    ) -> RunAndMeasureProgramRequest:
        if not quil_program:
            raise ValueError(
                "You have attempted to run an empty program."
                " Please provide gates or measure instructions to your program."
            )

        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")
        qubits = validate_qubit_list(qubits)
        if not isinstance(trials, int):
            raise TypeError("trials must be an integer")

        if self.noise_model is not None:
            assert self.compiler is not None
            compiled_program = self.compiler.quil_to_native_quil(quil_program)
            quil_program = apply_noise_model(compiled_program, self.noise_model)

        return RunAndMeasureProgramRequest(
            program=quil_program.out(calibrations=False),
            qubits=list(qubits),
            trials=trials,
            measurement_noise=self.measurement_noise,
            gate_noise=self.gate_noise,
            seed=self.random_seed,
        )

    @_record_call
    def wavefunction(self, quil_program: Program) -> Wavefunction:
        """
        Simulate a Quil program and get the wavefunction back.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            returned bitstrings are sampled itself only represents a stochastically generated sample
            and the wavefunctions returned by *different* ``wavefunction`` calls *will generally be
            different*.

        :param quil_program: A Quil program.
        :return: A Wavefunction object representing the state of the QVM.
        """
        request = self._wavefunction_request(quil_program)
        response = self._qvm_client.get_wavefunction(request)
        return Wavefunction.from_bit_packed_string(response.wavefunction)

    @_record_call
    def _wavefunction_request(self, quil_program: Program) -> GetWavefunctionRequest:
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")

        return GetWavefunctionRequest(
            program=quil_program.out(calibrations=False),
            measurement_noise=self.measurement_noise,
            gate_noise=self.gate_noise,
            seed=self.random_seed,
        )

    @_record_call
    def expectation(self, prep_prog: Program, operator_programs: Optional[Iterable[Program]] = None) -> List[float]:
        """
        Calculate the expectation value of operators given a state prepared by
        prep_program.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            expectation values are computed itself only represents a stochastically generated
            sample. The expectations returned from *different* ``expectation`` calls *will then
            generally be different*.

        To measure the expectation of a PauliSum, you probably want to
        do something like this::

                progs, coefs = hamiltonian.get_programs()
                expect_coeffs = np.array(cxn.expectation(prep_program, operator_programs=progs))
                return np.real_if_close(np.dot(coefs, expect_coeffs))

        :param prep_prog: Quil program for state preparation.
        :param operator_programs: A list of Programs, each specifying an operator whose
            expectation to compute. Default is a list containing only the empty Program.
        :return: Expectation values of the operators.
        """

        if isinstance(operator_programs, Program):
            warnings.warn(
                "You have provided a Program rather than a list of Programs. The results from "
                "expectation will be line-wise expectation values of the operator_programs.",
                SyntaxWarning,
            )

        request = self._expectation_request(prep_prog, operator_programs)
        response = self._qvm_client.measure_expectation(request)
        return response.expectations

    @_record_call
    def pauli_expectation(
        self,
        prep_prog: Program,
        pauli_terms: Union[Sequence[PauliTerm], PauliSum],
    ) -> Union[float, List[float]]:
        """
        Calculate the expectation value of Pauli operators given a state prepared by prep_program.

        If ``pauli_terms`` is a ``PauliSum`` then the returned value is a single ``float``,
        otherwise the returned value is a list of ``float``s, one for each ``PauliTerm`` in the
        list.

        :note: If the execution of ``quil_program`` is **non-deterministic**, i.e., if it includes
            measurements and/or noisy quantum gates, then the final wavefunction from which the
            expectation values are computed itself only represents a stochastically generated
            sample. The expectations returned from *different* ``expectation`` calls *will then
            generally be different*.

        :param Program prep_prog: Quil program for state preparation.
        :param Sequence[PauliTerm]|PauliSum pauli_terms: A list of PauliTerms or a PauliSum.
        :return: If ``pauli_terms`` is a PauliSum return its expectation value. Otherwise return
          a list of expectation values.
        :rtype: float|List[float]
        """

        is_pauli_sum = False
        if isinstance(pauli_terms, PauliSum):
            progs, coeffs = pauli_terms.get_programs()
            is_pauli_sum = True
        else:
            coeffs = np.array([pt.coefficient for pt in pauli_terms])
            progs = [pt.program for pt in pauli_terms]

        bare_results = self.expectation(prep_prog, progs)
        results = [c * r for c, r in zip(coeffs, bare_results)]
        if is_pauli_sum:
            return sum(results)
        return results

    def _expectation_request(
        self,
        prep_prog: Program,
        operator_programs: Optional[Iterable[Program]],
    ) -> MeasureExpectationRequest:
        if operator_programs is None:
            operator_programs = [Program()]

        if not isinstance(prep_prog, Program):
            raise TypeError("prep_prog variable must be a Quil program object")

        return MeasureExpectationRequest(
            prep_program=prep_prog.out(calibrations=False),
            pauli_operators=[x.out(calibrations=False) for x in operator_programs],
            seed=self.random_seed,
        )


class QVM(QAM):
    @_record_call
    def __init__(
        self,
        noise_model: Optional[NoiseModel] = None,
        gate_noise: Optional[Tuple[float, float, float]] = None,
        measurement_noise: Optional[Tuple[float, float, float]] = None,
        random_seed: Optional[int] = None,
        timeout: float = 5.0,
        client_configuration: Optional[QCSClientConfiguration] = None,
    ) -> None:
        """
        A virtual machine that classically emulates the execution of Quil programs.

        :param noise_model: A noise model that describes noise to apply when emulating a program's
            execution.
        :param gate_noise: A tuple of three numbers [Px, Py, Pz] indicating the probability of an X,
           Y, or Z gate getting applied to each qubit after a gate application or reset. The
           default value of None indicates no noise.
        :param measurement_noise: A tuple of three numbers [Px, Py, Pz] indicating the probability
            of an X, Y, or Z gate getting applied before a measurement. The default value of
            None indicates no noise.
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
            automatically generated seed) or a non-negative integer.
        :param timeout: Time limit for requests, in seconds.
        :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        """
        super().__init__()

        if (noise_model is not None) and (gate_noise is not None or measurement_noise is not None):
            raise ValueError(
                """
You have attempted to supply the QVM with both a Kraus noise model
(by supplying a `noise_model` argument), as well as either `gate_noise`
or `measurement_noise`. At this time, only one may be supplied.

To read more about supplying noise to the QVM, see
http://pyquil.readthedocs.io/en/latest/noise_models.html#support-for-noisy-gates-on-the-rigetti-qvm.
"""
            )

        self.noise_model = noise_model

        validate_noise_probabilities(gate_noise)
        validate_noise_probabilities(measurement_noise)
        self.gate_noise = gate_noise
        self.measurement_noise = measurement_noise

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, int) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int")

        client_configuration = client_configuration or QCSClientConfiguration.load()
        self._qvm_client = QVMClient(client_configuration=client_configuration, request_timeout=timeout)
        self.connect()

    def connect(self) -> None:
        try:
            version = self.get_version_info()
            check_qvm_version(version)
        except ConnectionError:
            raise QVMNotRunning(f"No QVM server running at {self._qvm_client.base_url}")

    @_record_call
    def get_version_info(self) -> str:
        """
        Return version information for the QVM.

        :return: String with version information
        """
        return self._qvm_client.get_version()

    @_record_call
    def load(self, executable: QuantumExecutable) -> "QVM":
        """
        Initialize a QAM and load a program to be executed with a call to :py:func:`run`.

        :param executable: A compiled executable.
        """
        if not isinstance(executable, Program):
            raise TypeError("`executable` argument must be a `Program`.")

        super().load(executable)
        for region in executable.declarations.keys():
            self._memory_results[region] = np.ndarray((executable.num_shots, 0), dtype=np.int64)
        return self

    @_record_call
    def run(self) -> "QVM":
        """
        Run a Quil program on the QVM multiple times and return the values stored in the
        classical registers designated by the classical_addresses parameter.

        :return: An array of bitstrings of shape ``(trials, len(classical_addresses))``
        """
        super().run()
        assert isinstance(self.executable, Program)

        quil_program = self.executable
        trials = quil_program.num_shots
        classical_addresses = get_classical_addresses_from_program(quil_program)

        if self.noise_model is not None:
            quil_program = apply_noise_model(quil_program, self.noise_model)

        quil_program = self.augment_program_with_memory_values(quil_program)

        request = qvm_run_request(
            quil_program,
            classical_addresses,
            trials,
            self.measurement_noise,
            self.gate_noise,
            self.random_seed,
        )
        response = self._qvm_client.run_program(request)
        ram = {key: np.array(val) for key, val in response.results.items()}
        self._memory_results.update(ram)

        return self

    def augment_program_with_memory_values(self, quil_program: Program) -> Program:
        p = Program()

        for k, v in self._variables_shim.items():
            p += MOVE(MemoryReference(name=k.name, offset=k.index), v)

        p += quil_program

        return percolate_declares(p)


def validate_noise_probabilities(noise_parameter: Optional[Tuple[float, float, float]]) -> None:
    """
    Is noise_parameter a valid specification of noise probabilities for depolarizing noise?

    :param tuple noise_parameter: Tuple of 3 noise parameter values to be validated.
    """
    if not noise_parameter:
        return
    if not isinstance(noise_parameter, tuple):
        raise TypeError("noise_parameter must be a tuple")
    if any([not isinstance(value, float) for value in noise_parameter]):
        raise TypeError("noise_parameter values should all be floats")
    if len(noise_parameter) != 3:
        raise ValueError("noise_parameter tuple must be of length 3")
    if sum(noise_parameter) > 1 or sum(noise_parameter) < 0:
        raise ValueError("sum of entries in noise_parameter must be between 0 and 1 (inclusive)")
    if any([value < 0 for value in noise_parameter]):
        raise ValueError("noise_parameter values should all be non-negative")


def validate_qubit_list(qubit_list: Sequence[int]) -> Sequence[int]:
    """
    Check the validity of qubits for the payload.

    :param qubit_list: List of qubits to be validated.
    """
    if not isinstance(qubit_list, Sequence):
        raise TypeError("'qubit_list' must be of type 'Sequence'")
    if any(not isinstance(i, int) or i < 0 for i in qubit_list):
        raise TypeError("'qubit_list' must contain positive integer values")
    return qubit_list


def prepare_register_list(
    register_dict: Mapping[str, Union[bool, Sequence[int]]]
) -> Dict[str, Union[bool, Sequence[int]]]:
    """
    Canonicalize classical addresses for the payload and ready MemoryReference instances
    for serialization.

    This function will cast keys that are iterables of int-likes to a list of Python
    ints. This is to support specifying the register offsets as ``range()`` or numpy
    arrays. This mutates ``register_dict``.

    :param register_dict: The classical memory to retrieve. Specified as a dictionary:
        the keys are the names of memory regions, and the values are either (1) a list of
        integers for reading out specific entries in that memory region, or (2) True, for
        reading out the entire memory region.
    """
    if not isinstance(register_dict, dict):
        raise TypeError("register_dict must be a dict but got " + repr(register_dict))

    for k, v in register_dict.items():
        if isinstance(v, bool):
            assert v  # If boolean v must be True
            continue

        indices = [int(x) for x in v]  # support ranges, numpy, ...

        if not all(x >= 0 for x in indices):
            raise TypeError("Negative indices into classical arrays are not allowed.")
        register_dict[k] = indices

    return register_dict


def qvm_run_request(
    quil_program: Program,
    classical_addresses: Mapping[str, Union[bool, Sequence[int]]],
    trials: int,
    measurement_noise: Optional[Tuple[float, float, float]],
    gate_noise: Optional[Tuple[float, float, float]],
    random_seed: Optional[int],
) -> RunProgramRequest:
    if not quil_program:
        raise ValueError(
            "You have attempted to run an empty program."
            " Please provide gates or measure instructions to your program."
        )
    if not isinstance(quil_program, Program):
        raise TypeError("quil_program must be a Quil program object")
    classical_addresses = prepare_register_list(classical_addresses)
    if not isinstance(trials, int):
        raise TypeError("trials must be an integer")

    return RunProgramRequest(
        program=quil_program.out(calibrations=False),
        addresses=classical_addresses,  # type: ignore
        trials=trials,
        measurement_noise=measurement_noise,
        gate_noise=gate_noise,
        seed=random_seed,
    )

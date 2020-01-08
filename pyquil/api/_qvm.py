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
import numpy as np
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union, cast

from requests.exceptions import ConnectionError
from rpcq.messages import PyQuilExecutableResponse

from pyquil.api._base_connection import (
    validate_qubit_list,
    validate_noise_probabilities,
    TYPE_MULTISHOT_MEASURE,
    TYPE_WAVEFUNCTION,
    TYPE_EXPECTATION,
    post_json,
    ForestConnection,
)
from pyquil.api._compiler import QVMCompiler, _extract_program_from_pyquil_executable_response
from pyquil.api._config import PyquilConfig
from pyquil.api._error_reporting import _record_call
from pyquil.api._qam import QAM
from pyquil.device._main import Device
from pyquil.gates import MOVE
from pyquil.noise import NoiseModel, apply_noise_model
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program, get_classical_addresses_from_program, percolate_declares
from pyquil.quilatom import MemoryReference
from pyquil.version import __version__
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
            "Must use QVM >= 1.8.0 with pyquil >= 2.8.0, but you "
            f"have QVM {version} and pyquil {__version__}"
        )


class QVMConnection(object):
    """
    Represents a connection to the QVM.
    """

    @_record_call
    def __init__(
        self,
        device: Optional[Device] = None,
        endpoint: Optional[str] = None,
        gate_noise: Optional[List[float]] = None,
        measurement_noise: Optional[List[float]] = None,
        random_seed: Optional[int] = None,
        compiler_endpoint: Optional[str] = None,
    ):
        """
        Constructor for QVMConnection. Sets up any necessary security, and establishes the noise
        model to use.

        :param device: The optional device, from which noise will be added by default to all
            programs run on this instance.
        :param endpoint: The endpoint of the server for running small jobs
        :param gate_noise: A list of three numbers [Px, Py, Pz] indicating the probability of an X,
            Y, or Z gate getting applied to each qubit after a gate application or reset.
        :param measurement_noise: A list of three numbers [Px, Py, Pz] indicating the probability of
            an X, Y, or Z gate getting applied before a a measurement.
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
            automatically generated seed) or a non-negative integer.
        """
        if endpoint is None:
            pyquil_config = PyquilConfig()
            endpoint = pyquil_config.qvm_url

        if compiler_endpoint is None:
            pyquil_config = PyquilConfig()
            compiler_endpoint = pyquil_config.quilc_url

        if (device is not None and device.noise_model is not None) and (
            gate_noise is not None or measurement_noise is not None
        ):
            raise ValueError(
                """
You have attempted to supply the QVM with both a device noise model
(by having supplied a device argument), as well as either gate_noise
or measurement_noise. At this time, only one may be supplied.

To read more about supplying noise to the QVM, see
http://pyquil.readthedocs.io/en/latest/noise_models.html#support-for-noisy-gates-on-the-rigetti-qvm.
"""
            )

        if device is not None and device.noise_model is None:
            warnings.warn(
                """
You have supplied the QVM with a device that does not have a noise model. No noise will be added to
programs run on this QVM.
"""
            )

        self.noise_model = device.noise_model if device else None
        self.compiler = QVMCompiler(endpoint=compiler_endpoint, device=device) if device else None

        self.sync_endpoint = endpoint

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

        self._connection = ForestConnection(sync_endpoint=endpoint)
        self.session = self._connection.session  # backwards compatibility
        self.connect()

    def connect(self) -> None:
        try:
            version_dict = self.get_version_info()
            check_qvm_version(version_dict)
        except ConnectionError:
            raise QVMNotRunning(f"No QVM server running at {self._connection.sync_endpoint}")

    @_record_call
    def get_version_info(self) -> str:
        """
        Return version information for the QVM.

        :return: String with version information
        """
        return cast(str, self._connection._qvm_get_version_info())

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
            caddresses: Mapping[str, Sequence[int]] = get_classical_addresses_from_program(
                quil_program
            )

        else:
            caddresses = {"ro": classical_addresses}

        buffers = self._connection._qvm_run(
            quil_program,
            caddresses,
            trials,
            self.measurement_noise,
            self.gate_noise,
            self.random_seed,
        )

        if len(buffers) == 0:
            return []
        if "ro" in buffers:
            return cast(List[List[int]], buffers["ro"].tolist())

        raise ValueError(
            "You are using QVMConnection.run with multiple readout registers not "
            "named `ro`. Please use the new `QuantumComputer` abstraction."
        )

    @_record_call
    def run_and_measure(
        self, quil_program: Program, qubits: Sequence[int], trials: int = 1
    ) -> List[List[int]]:
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
        # Developer note: This code is for backwards compatibility. It can't be replaced with
        # ForestConnection._run_and_measure because we've turned off the ability to set
        # `needs_compilation` (that usually indicates the user is doing something iffy like
        # using a noise model with this function)

        payload = self._run_and_measure_payload(quil_program, qubits, trials)
        assert self.sync_endpoint is not None
        response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
        return cast(List[List[int]], response.json())

    @_record_call
    def _run_and_measure_payload(
        self, quil_program: Program, qubits: Sequence[int], trials: int
    ) -> Dict[str, Any]:
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

        payload = {
            "type": TYPE_MULTISHOT_MEASURE,
            "qubits": list(qubits),
            "trials": trials,
            "compiled-quil": quil_program.out(),
        }

        self._maybe_add_noise_to_payload(payload)
        self._add_rng_seed_to_payload(payload)

        return payload

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
        # Developer note: This code is for backwards compatibility. It can't be replaced with
        # ForestConnection._wavefunction because we've turned off the ability to set
        # `needs_compilation` (that usually indicates the user is doing something iffy like
        # using a noise model with this function)

        payload = self._wavefunction_payload(quil_program)
        assert self.sync_endpoint is not None
        response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
        return Wavefunction.from_bit_packed_string(response.content)

    @_record_call
    def _wavefunction_payload(self, quil_program: Program) -> Dict[str, Any]:
        # Developer note: This code is for backwards compatibility. It can't be replaced with
        # _base_connection._wavefunction_payload because we've turned off the ability to set
        # `needs_compilation` (that usually indicates the user is doing something iffy like
        # using a noise model with this function)
        if not isinstance(quil_program, Program):
            raise TypeError("quil_program must be a Quil program object")

        payload = {"type": TYPE_WAVEFUNCTION, "compiled-quil": quil_program.out()}

        self._maybe_add_noise_to_payload(payload)
        self._add_rng_seed_to_payload(payload)

        return payload

    @_record_call
    def expectation(
        self, prep_prog: Program, operator_programs: Optional[Iterable[Program]] = None
    ) -> List[float]:
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
        # Developer note: This code is for backwards compatibility. It can't be replaced with
        # ForestConnection._expectation because we've turned off the ability to set
        # `needs_compilation` (that usually indicates the user is doing something iffy like
        # using a noise model with this function)

        if isinstance(operator_programs, Program):
            warnings.warn(
                "You have provided a Program rather than a list of Programs. The results from "
                "expectation will be line-wise expectation values of the operator_programs.",
                SyntaxWarning,
            )

        payload = self._expectation_payload(prep_prog, operator_programs)
        assert self.sync_endpoint is not None
        response = post_json(self.session, self.sync_endpoint + "/qvm", payload)
        return cast(List[float], response.json())

    @_record_call
    def pauli_expectation(
        self, prep_prog: Program, pauli_terms: Union[Sequence[PauliTerm], PauliSum]
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
            coeffs = [pt.coefficient for pt in pauli_terms]
            progs = [pt.program for pt in pauli_terms]

        bare_results = self.expectation(prep_prog, progs)
        results = [c * r for c, r in zip(coeffs, bare_results)]
        if is_pauli_sum:
            return sum(results)
        return results

    def _expectation_payload(
        self, prep_prog: Program, operator_programs: Optional[Iterable[Program]]
    ) -> Dict[str, Any]:
        if operator_programs is None:
            operator_programs = [Program()]

        if not isinstance(prep_prog, Program):
            raise TypeError("prep_prog variable must be a Quil program object")

        payload = {
            "type": TYPE_EXPECTATION,
            "state-preparation": prep_prog.out(),
            "operators": [x.out() for x in operator_programs],
        }

        self._add_rng_seed_to_payload(payload)

        return payload

    def _maybe_add_noise_to_payload(self, payload: Dict[str, Any]) -> None:
        """
        Set the gate noise and measurement noise of a payload.
        """
        if self.measurement_noise is not None:
            payload["measurement-noise"] = self.measurement_noise
        if self.gate_noise is not None:
            payload["gate-noise"] = self.gate_noise

    def _add_rng_seed_to_payload(self, payload: Dict[str, Any]) -> None:
        """
        Add a random seed to the payload.
        """
        if self.random_seed is not None:
            payload["rng-seed"] = self.random_seed


class QVM(QAM):
    @_record_call
    def __init__(
        self,
        connection: ForestConnection,
        noise_model: Optional[NoiseModel] = None,
        gate_noise: Optional[List[float]] = None,
        measurement_noise: Optional[List[float]] = None,
        random_seed: Optional[int] = None,
        requires_executable: bool = False,
    ) -> None:
        """
        A virtual machine that classically emulates the execution of Quil programs.

        :param connection: A connection to the Forest web API.
        :param noise_model: A noise model that describes noise to apply when emulating a program's
            execution.
        :param gate_noise: A list of three numbers [Px, Py, Pz] indicating the probability of an X,
           Y, or Z gate getting applied to each qubit after a gate application or reset. The
           default value of None indicates no noise.
        :param measurement_noise: A list of three numbers [Px, Py, Pz] indicating the probability
            of an X, Y, or Z gate getting applied before a measurement. The default value of
            None indicates no noise.
        :param random_seed: A seed for the QVM's random number generators. Either None (for an
            automatically generated seed) or a non-negative integer.
        :param requires_executable: Whether this QVM will refuse to run a :py:class:`Program` and
            only accept the result of :py:func:`compiler.native_quil_to_executable`. Setting this
            to True better emulates the behavior of a QPU.
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
        self.connection = connection

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

        self.requires_executable = requires_executable
        self.connect()

    def connect(self) -> None:
        try:
            version_dict = self.get_version_info()
            check_qvm_version(version_dict)
        except ConnectionError:
            raise QVMNotRunning(f"No QVM server running at {self.connection.sync_endpoint}")

    @_record_call
    def get_version_info(self) -> str:
        """
        Return version information for the QVM.

        :return: String with version information
        """
        return cast(str, self.connection._qvm_get_version_info())

    @_record_call
    def load(self, executable: Union[Program, PyQuilExecutableResponse]) -> "QVM":
        """
        Initialize a QAM and load a program to be executed with a call to :py:func:`run`.

        If ``QVM.requires_executable`` is set to ``True``, this function will only load
        :py:class:`PyQuilExecutableResponse` executables. This more closely follows the behavior
        of :py:class:`QPU`. However, the quantum simulator doesn't *actually* need a compiled
        binary executable, so if this flag is set to ``False`` we also accept :py:class:`Program`
        objects.

        :param executable: An executable. See the above note for acceptable types.
        """
        if self.requires_executable:
            if isinstance(executable, PyQuilExecutableResponse):
                executable = _extract_program_from_pyquil_executable_response(executable)
            else:
                raise TypeError(
                    "`executable` argument must be a `PyQuilExecutableResponse`. Make "
                    "sure you have explicitly compiled your program via `qc.compile` "
                    "or `qc.compiler.native_quil_to_executable(...)` for more "
                    "fine-grained control. This explicit step is required for running "
                    "on a QPU."
                )
        else:
            if isinstance(executable, PyQuilExecutableResponse):
                executable = _extract_program_from_pyquil_executable_response(executable)
            elif isinstance(executable, Program):
                pass
            else:
                raise TypeError(
                    "`executable` argument must be a `PyQuilExecutableResponse` or a "
                    "`Program`. You provided {}".format(type(executable))
                )

        return cast("QVM", super().load(executable))

    @_record_call
    def run(self) -> "QVM":
        """
        Run a Quil program on the QVM multiple times and return the values stored in the
        classical registers designated by the classical_addresses parameter.

        :return: An array of bitstrings of shape ``(trials, len(classical_addresses))``
        """

        super().run()

        if not isinstance(self._executable, Program):
            # This should really never happen
            # unless a user monkeys with `self.status` and `self._executable`.
            raise ValueError("Please `load` an appropriate executable.")

        quil_program = self._executable
        trials = quil_program.num_shots
        classical_addresses = get_classical_addresses_from_program(quil_program)

        if self.noise_model is not None:
            quil_program = apply_noise_model(quil_program, self.noise_model)

        quil_program = self.augment_program_with_memory_values(quil_program)

        self._memory_results = self.connection._qvm_run(
            quil_program=quil_program,
            classical_addresses=classical_addresses,
            trials=trials,
            measurement_noise=self.measurement_noise,
            gate_noise=self.gate_noise,
            random_seed=self.random_seed,
        )

        if "ro" not in self._memory_results or len(self._memory_results["ro"]) == 0:
            self._memory_results["ro"] = np.zeros((trials, 0), dtype=np.int64)

        return self

    def augment_program_with_memory_values(self, quil_program: Program) -> Program:
        p = Program()

        for k, v in self._variables_shim.items():
            p += MOVE(MemoryReference(name=k.name, offset=k.index), v)

        p += quil_program

        return percolate_declares(p)

    @_record_call
    def reset(self) -> None:
        """
        Reset the state of the underlying QAM, and the QVM connection information.
        """
        super().reset()
        forest_connection = ForestConnection()
        self.connection = forest_connection

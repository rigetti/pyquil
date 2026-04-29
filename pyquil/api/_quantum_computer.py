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
from __future__ import annotations

import re
import socket
import subprocess
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

import networkx as nx
from qcs_sdk import QCSClient
from qcs_sdk.compiler.quilc import QuilcClient
from qcs_sdk.qpu import list_quantum_processors

from pyquil.api._abstract_compiler import AbstractCompiler, QuantumExecutable
from pyquil.api._compiler import QPUCompiler, QVMCompiler
from pyquil.api._qam import QAM, MemoryMap, QAMExecutionResult
from pyquil.api._qpu import QPU
from pyquil.api._qvm import QVM
from pyquil.external.rpcq import CompilerISA
from pyquil.quantum_processor import (
    AbstractQuantumProcessor,
    NxQuantumProcessor,
    QCSQuantumProcessor,
    get_qcs_quantum_processor,
)
from pyquil.quil import Program

if TYPE_CHECKING:
    from pyquil.noise import NoiseModel


class QuantumComputer:
    def __init__(
        self,
        *,
        name: str,
        qam: QAM[Any],
        compiler: AbstractCompiler,
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
        """
        self.name = name
        self.qam = qam
        self.compiler = compiler

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
    if len(parts) >= 2 and parts[-2] == "noisy" and parts[-1] == "qvm":
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

    if len(parts) >= 1 and parts[-1] == "qvm":
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
    else:
        raise ValueError(f"Unknown qvm_type {qvm_type}")

    name = f"{prefix}{noise_suffix}{qvm_suffix}"
    return name


def _get_qvm_or_pyqvm(
    *,
    qvm_type: str,
    noise_model: Optional[NoiseModel] = None,
    random_seed: Optional[int] = None,
) -> QVM:
    if qvm_type == "qvm":
        return QVM(noise_model=noise_model, random_seed=random_seed)

    raise ValueError(f"Unknown qvm type {qvm_type}")


def _get_qvm_qc(
    *,
    client_configuration: QCSClient,
    name: str,
    qvm_type: str,
    quantum_processor: AbstractQuantumProcessor,
    compiler_timeout: float,
    noise_model: Optional[NoiseModel] = None,
    random_seed: Optional[int] = None,
    quilc_client: Optional[QuilcClient] = None,
) -> QuantumComputer:
    """Construct a QuantumComputer backed by a QVM.

    This is a minimal wrapper over the QuantumComputer, QVM, and QVMCompiler constructors.

    :param client_configuration: Client configuration.
    :param name: A string identifying this particular quantum computer.
    :param qvm_type: The type of QVM.
    :param quantum_processor: A quantum_processor following the AbstractQuantumProcessor interface.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param noise_model: An optional noise model for noisy simulation.
    :param random_seed: An optional random seed for reproducible simulation.
    :return: A QuantumComputer backed by a QVM with the above options.
    """
    return QuantumComputer(
        name=name,
        qam=_get_qvm_or_pyqvm(
            qvm_type=qvm_type,
            noise_model=noise_model,
            random_seed=random_seed,
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
    noise_model: Optional[NoiseModel] = None,
    random_seed: Optional[int] = None,
    quilc_client: Optional[QuilcClient] = None,
) -> QuantumComputer:
    """Construct a QVM with the provided topology.

    :param client_configuration: Client configuration.
    :param name: A name for your quantum computer. This field does not affect behavior of the
        constructed QuantumComputer.
    :param topology: A graph representing the desired qubit connectivity.
    :param noisy: Whether to include a generic noise model. If you want more control over
        the noise model, please construct your own :py:class:`NoiseModel` and pass it
        via the ``noise_model`` parameter.
    :param qvm_type: The type of QVM.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param noise_model: An optional noise model for noisy simulation.
    :param random_seed: An optional random seed for reproducible simulation.
    :return: A pre-configured QuantumComputer
    """
    quantum_processor = NxQuantumProcessor(topology=topology)
    if noisy and noise_model is None:
        from pyquil.noise import NoiseModel as _NoiseModel

        noise_model = _NoiseModel.from_isa(quantum_processor.to_compiler_isa())
    return _get_qvm_qc(
        client_configuration=client_configuration,
        name=name,
        qvm_type=qvm_type,
        quantum_processor=quantum_processor,
        compiler_timeout=compiler_timeout,
        noise_model=noise_model,
        random_seed=random_seed,
        quilc_client=quilc_client,
    )


def _get_9q_square_qvm(
    *,
    client_configuration: QCSClient,
    name: str,
    noisy: bool,
    qvm_type: str,
    compiler_timeout: float,
    noise_model: Optional[NoiseModel] = None,
    random_seed: Optional[int] = None,
    quilc_client: Optional[QuilcClient] = None,
) -> QuantumComputer:
    """Nine-qubit 3x3 square lattice.

    This uses a "generic" lattice not tied to any specific quantum_processor. 9 qubits is large enough
    to do vaguely interesting algorithms and small enough to simulate quickly.

    :param client_configuration: Client configuration.
    :param name: The name of this QVM
    :param noisy: Whether to construct a noisy quantum computer
    :param qvm_type: The type of QVM.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param noise_model: An optional noise model for noisy simulation.
    :param random_seed: An optional random seed for reproducible simulation.
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
        noise_model=noise_model,
        random_seed=random_seed,
        quilc_client=quilc_client,
    )


def _get_unrestricted_qvm(
    *,
    client_configuration: QCSClient,
    name: str,
    noisy: bool,
    n_qubits: int,
    qvm_type: str,
    compiler_timeout: float,
    noise_model: Optional[NoiseModel] = None,
    random_seed: Optional[int] = None,
    quilc_client: Optional[QuilcClient] = None,
) -> QuantumComputer:
    """QVM with a fully-connected topology.

    :param client_configuration: Client configuration.
    :param name: The name of this QVM
    :param noisy: Whether to construct a noisy quantum computer
    :param n_qubits: 34 qubits ought to be enough for anybody.
    :param qvm_type: The type of QVM.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param noise_model: An optional noise model for noisy simulation.
    :param random_seed: An optional random seed for reproducible simulation.
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
        noise_model=noise_model,
        random_seed=random_seed,
        quilc_client=quilc_client,
    )


def _get_qvm_based_on_real_quantum_processor(
    *,
    client_configuration: QCSClient,
    name: str,
    quantum_processor: QCSQuantumProcessor,
    noisy: bool,
    qvm_type: str,
    compiler_timeout: float,
    noise_model: Optional[NoiseModel] = None,
    random_seed: Optional[int] = None,
    quilc_client: Optional[QuilcClient] = None,
) -> QuantumComputer:
    """QVM based on a real quantum_processor.

    This is the most realistic QVM.

    :param client_configuration: Client configuration.
    :param name: The full name of this QVM
    :param quantum_processor: The quantum_processor from :py:func:`get_lattice`.
    :param noisy: Whether to construct a noisy quantum computer by using the quantum_processor's
        associated noise model.
    :param qvm_type: The type of QVM.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param noise_model: An optional noise model for noisy simulation.
    :param random_seed: An optional random seed for reproducible simulation.
    :return: A pre-configured QuantumComputer based on the named quantum_processor.
    """
    if noisy and noise_model is None:
        from pyquil.noise import NoiseModel as _NoiseModel

        noise_model = _NoiseModel.from_isa(quantum_processor.to_compiler_isa())
    return _get_qvm_qc(
        client_configuration=client_configuration,
        name=name,
        quantum_processor=quantum_processor,
        qvm_type=qvm_type,
        compiler_timeout=compiler_timeout,
        noise_model=noise_model,
        random_seed=random_seed,
        quilc_client=quilc_client,
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
    noise_model: Optional[NoiseModel] = None,
    random_seed: Optional[int] = None,
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
        a QVM. Names ending in "-noisy-qvm" will return a QVM with a noise model. Otherwise,
        we will return a QPU with the given name.
    :param as_qvm: An optional flag to force construction of a QVM (instead of a QPU). If
        specified and set to ``True``, a QVM-backed quantum computer will be returned regardless
        of the name's suffix
    :param noisy: An optional flag to force inclusion of a noise model. If
        specified and set to ``True``, a quantum computer with a noise model will be returned
        regardless of the name's suffix.
    :param compiler_timeout: Time limit for compilation requests, in seconds.
    :param execution_timeout: Time limit for execution requests, in seconds.
    :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        For more information on setting up QCS credentials, see documentation for using the QCS CLI:
        [https://docs.rigetti.com/qcs/guides/using-the-qcs-cli#configuring-credentials].
    :param endpoint_id: Optional quantum processor endpoint ID, as used in the `QCS API Docs`_.
    :param noise_model: An optional custom noise model for QVM simulation. If provided, this
        takes precedence over auto-generated noise models from the ``noisy`` flag.
    :param random_seed: An optional random seed for reproducible QVM simulation.

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
            noise_model=noise_model,
            random_seed=random_seed,
            quilc_client=quilc_client,
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
            noise_model=noise_model,
            random_seed=random_seed,
            quilc_client=quilc_client,
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
            noisy=noisy,
            qvm_type=qvm_type,
            compiler_timeout=compiler_timeout,
            noise_model=noise_model,
            random_seed=random_seed,
            quilc_client=quilc_client,
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


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
import re
import warnings
from math import pi
from typing import List, Dict, Tuple, Iterator, Union
import subprocess
from contextlib import contextmanager

import networkx as nx
import numpy as np
from rpcq.messages import BinaryExecutableResponse, Message, PyQuilExecutableResponse

from pyquil.api._compiler import QVMCompiler, QPUCompiler, LocalQVMCompiler
from pyquil.api._config import PyquilConfig
from pyquil.api._devices import get_lattice, list_lattices
from pyquil.api._error_reporting import _record_call
from pyquil.api._qac import AbstractCompiler
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._qvm import ForestConnection, QVM
from pyquil.device import AbstractDevice, NxDevice, gates_in_isa, ISA
from pyquil.gates import RX, MEASURE
from pyquil.noise import decoherence_noise_with_asymmetric_ro
from pyquil.quil import Program
from pyquil.quilbase import Measurement, Pragma, Gate, Reset

pyquil_config = PyquilConfig()

Executable = Union[BinaryExecutableResponse, PyQuilExecutableResponse]


def _get_flipped_protoquil_program(program: Program) -> Program:
    """For symmetrization, generate a program where X gates are added before measurement.

    Forest 1.3 is really picky about where the measure instructions happen. It has to be
    at the end!
    """
    program = program.copy()
    to_measure = []
    while len(program) > 0:
        inst = program.instructions[-1]
        if isinstance(inst, Measurement):
            program.pop()
            to_measure.append((inst.qubit, inst.classical_reg))
        else:
            break

    program += Pragma('PRESERVE_BLOCK')
    for qu, addr in to_measure[::-1]:
        program += RX(pi, qu)
    program += Pragma('END_PRESERVE_BLOCK')

    for qu, addr in to_measure[::-1]:
        program += Measurement(qubit=qu, classical_reg=addr)

    return program


def _validate_run_and_measure_program(program: Program) -> Program:
    for instr in program.instructions:
        if not isinstance(instr, Gate) and not isinstance(instr, Reset):
            raise ValueError("run_and_measure programs must consist only of quantum gates.")

    # TODO: more logic here?
    return program


class QuantumComputer:
    def __init__(self, *,
                 name: str,
                 qam: QAM,
                 device: AbstractDevice,
                 compiler: AbstractCompiler,
                 symmetrize_readout: bool = False) -> None:
        """
        A quantum computer for running quantum programs.

        A quantum computer has various characteristics like supported gates, qubits, qubit
        topologies, gate fidelities, and more. A quantum computer also has the ability to
        run quantum programs.

        A quantum computer can be a real Rigetti QPU that uses superconducting transmon
        qubits to run quantum programs, or it can be an emulator like the Rigetti QVM with
        noise models and mimicked topologies.

        :param name: A string identifying this particular quantum computer.
        :param qam: A quantum abstract machine which handles executing quantum programs. This
            dispatches to a QVM or QPU.
        :param device: A collection of connected qubits and associated specs and topology.
        :param symmetrize_readout: Whether to apply readout error symmetrization. See
            :py:func:`run_symmetrized_readout` for a complete description.
        """
        self.name = name
        self.qam = qam
        self.device = device
        self.compiler = compiler

        self.symmetrize_readout = symmetrize_readout

    def qubits(self) -> List[int]:
        """
        Return a sorted list of this QuantumComputer's device's qubits

        See :py:func:`AbstractDevice.qubits` for more.
        """
        return self.device.qubits()

    def qubit_topology(self) -> nx.graph:
        """
        Return a NetworkX graph representation of this QuantumComputer's device's qubit
        connectivity.

        See :py:func:`AbstractDevice.qubit_topology` for more.
        """
        return self.device.qubit_topology()

    def get_isa(self, oneq_type: str = 'Xhalves',
                twoq_type: str = 'CZ') -> ISA:
        """
        Return a target ISA for this QuantumComputer's device.

        See :py:func:`AbstractDevice.get_isa` for more.

        :param oneq_type: The family of one-qubit gates to target
        :param twoq_type: The family of two-qubit gates to target
        """
        return self.device.get_isa(oneq_type=oneq_type, twoq_type=twoq_type)

    @_record_call
    def run(self, executable: Executable,
            memory_map: Dict[str, List[Union[int, float]]] = None) -> np.ndarray:
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
                for offset, value in enumerate(values_list):
                    # TODO gh-658: have write_memory take a list rather than value + offset
                    self.qam.write_memory(region_name=region_name, offset=offset, value=value)
        return self.qam.run() \
            .wait() \
            .read_memory(region_name='ro')

    @_record_call
    def run_symmetrized_readout(self, program: Program, trials: int) -> np.ndarray:
        """
        Run a quil program in such a way that the readout error is made collectively symmetric

        This means the probability of a bitstring ``b`` being mistaken for a bitstring ``c`` is
        the same as the probability of ``not(b)`` being mistaken for ``not(c)``

        A more general symmetrization would guarantee that the probability of ``b`` being
        mistaken for ``c`` depends only on which bit of ``c`` are different from ``b``. This
        would require choosing random subsets of bits to flip.

        In a noisy device, the probability of accurately reading the 0 state might be higher
        than that of the 1 state. This makes correcting for readout more difficult. This
        function runs the program normally ``(trials//2)`` times. The other half of the time,
        it will insert an ``X`` gate prior to any ``MEASURE`` instruction and then flip the
        measured classical bit back.

        See :py:func:`run` for this function's parameter descriptions.
        """
        flipped_program = _get_flipped_protoquil_program(program)
        if trials % 2 != 0:
            raise ValueError("Using symmetrized measurement functionality requires that you "
                             "take an even number of trials.")
        half_trials = trials // 2
        flipped_program = flipped_program.wrap_in_numshots_loop(shots=half_trials)
        flipped_executable = self.compile(flipped_program)

        executable = self.compile(program.wrap_in_numshots_loop(half_trials))
        samples = self.run(executable)
        flipped_samples = self.run(flipped_executable)
        double_flipped_samples = np.logical_not(flipped_samples).astype(int)
        results = np.concatenate((samples, double_flipped_samples), axis=0)
        np.random.shuffle(results)
        return results

    @_record_call
    def run_and_measure(self, program: Program, trials: int) -> Dict[int, np.ndarray]:
        """
        Run the provided state preparation program and measure all qubits.

        This will measure all the qubits on this QuantumComputer, not just qubits
        that are used in the program.

        The returned data is a dictionary keyed by qubit index because qubits for a given
        QuantumComputer may be non-contiguous and non-zero-indexed. To turn this dictionary
        into a 2d numpy array of bitstrings, consider::

            bitstrings = qc.run_and_measure(...)
            bitstring_array = np.vstack(bitstrings[q] for q in qc.qubits()).T
            bitstring_array.shape  # (trials, len(qc.qubits()))

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
        program = _validate_run_and_measure_program(program)
        ro = program.declare('ro', 'BIT', len(self.qubits()))
        for i, q in enumerate(self.qubits()):
            program.inst(MEASURE(q, ro[i]))
        program.wrap_in_numshots_loop(trials)
        executable = self.compile(program)
        bitstring_array = self.run(executable=executable)
        bitstring_dict = {}
        for i, q in enumerate(self.qubits()):
            bitstring_dict[q] = bitstring_array[:, i]
        return bitstring_dict

    @_record_call
    def compile(self, program: Program,
                to_native_gates: bool = True,
                optimize: bool = True) -> Message:
        """
        A high-level interface to program compilation.

        Compilation currently consists of two stages. Please see the :py:class:`AbstractCompiler`
        docs for more information. This function does all stages of compilation.

        Right now both ``to_native_gates`` and ``optimize`` must be either both set or both
        unset. More modular compilation passes may be available in the future.

        :param program: A Program
        :param to_native_gates: Whether to compile non-native gates to native gates.
        :param optimize: Whether to optimize programs to reduce the number of operations.
        :return: An executable binary suitable for passing to :py:func:`QuantumComputer.run`.
        """
        flags = [to_native_gates, optimize]
        assert all(flags) or all(not f for f in flags), "Must turn quilc all on or all off"
        quilc = all(flags)

        if quilc:
            nq_program = self.compiler.quil_to_native_quil(program)
        else:
            nq_program = program
        binary = self.compiler.native_quil_to_executable(nq_program)
        return binary

    def reset(self):
        """
        Reset the QuantumComputer's QAM to its initial state.
        """
        self.qam.reset()

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return f'QuantumComputer[name="{self.name}"]'


@_record_call
def list_quantum_computers(connection: ForestConnection = None,
                           qpus: bool = True,
                           qvms: bool = True) -> List[str]:
    """
    List the names of available quantum computers

    :param connection: An optional :py:class:ForestConnection` object. If not specified,
        the default values for URL endpoints will be used, and your API key
        will be read from ~/.pyquil_config. If you deign to change any
        of these parameters, pass your own :py:class:`ForestConnection` object.
    :param qpus: Whether to include QPU's in the list.
    :param qvms: Whether to include QVM's in the list.
    """
    if connection is None:
        connection = ForestConnection()

    qc_names: List[str] = []
    if qpus:
        qc_names += list(list_lattices(connection=connection).keys())

    if qvms:
        qc_names += ['9q-square-qvm', '9q-square-noisy-qvm']

    return qc_names


def _parse_name(name: str, as_qvm: bool, noisy: bool) -> Tuple[str, bool, bool]:
    """
    Try to figure out whether we're getting a (noisy) qvm, and the associated qpu name.

    See :py:func:`get_qc` for examples of valid names + flags.
    """
    if name.endswith('noisy-qvm'):
        if as_qvm is not None and (not as_qvm):
            raise ValueError("The provided qc name indicates you are getting a noisy QVM, "
                             "but you have specified `as_qvm=False`")

        if noisy is not None and (not noisy):
            raise ValueError("The provided qc name indicates you are getting a noisy QVM, "
                             "but you have specified `noisy=False`")

        as_qvm = True
        noisy = True
        prefix = name[:-len('-noisy-qvm')]
        return prefix, as_qvm, noisy

    if name.endswith('qvm'):
        if as_qvm is not None and (not as_qvm):
            raise ValueError("The provided qc name indicates you are getting a QVM, "
                             "but you have specified `as_qvm=False`")
        as_qvm = True
        if noisy is not None:
            noisy = False
        prefix = name[:-len('-qvm')]
        return prefix, as_qvm, noisy

    if as_qvm is None:
        as_qvm = False

    if noisy is None:
        noisy = False

    return name, as_qvm, noisy


def _get_qvm_compiler_based_on_endpoint(endpoint: str = None,
                                        device: NxDevice = None) \
        -> AbstractCompiler:
    if endpoint.startswith("http"):
        return LocalQVMCompiler(endpoint=endpoint, device=device)
    elif endpoint.startswith("tcp"):
        return QVMCompiler(endpoint=endpoint, device=device)
    else:
        raise ValueError("Protocol for QVM compiler endpoints must be HTTP or TCP.")


def _get_9q_square_qvm(connection: ForestConnection, noisy: bool) -> QuantumComputer:
    """
    A nine-qubit 3x3 square lattice.

    This uses a "generic" lattice not tied to any specific device. 9 qubits is large enough
    to do vaguely interesting algorithms and small enough to simulate quickly.

    Users interested in building their own QuantumComputer from parts may wish to look
    to this function for inspiration, but should not use this private function directly.

    :param connection: The connection to use to talk to external services
    :param noisy: Whether to construct a noisy quantum computer
    :return: A pre-configured QuantumComputer
    """
    nineq_square = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    nineq_device = NxDevice(topology=nineq_square)
    if noisy:
        noise_model = decoherence_noise_with_asymmetric_ro(
            gates=gates_in_isa(nineq_device.get_isa()))
    else:
        noise_model = None

    name = '9q-square-noisy-qvm' if noisy else '9q-square-qvm'
    return QuantumComputer(name=name,
                           qam=QVM(connection=connection,
                                   noise_model=noise_model,
                                   requires_executable=True),
                           device=nineq_device,
                           compiler=_get_qvm_compiler_based_on_endpoint(
                               device=nineq_device,
                               endpoint=connection.compiler_endpoint))


def _get_unrestricted_qvm(connection: ForestConnection, noisy: bool,
                          n_qubits: int = 34) -> QuantumComputer:
    """
    A qvm with a fully-connected topology.

    This is obviously the least realistic QVM, but who am I to tell users what they want.

    Users interested in building their own QuantumComputer from parts may wish to look
    to this function for inspiration, but should not use this private function directly.

    :param connection: The connection to use to talk to external services
    :param noisy: Whether to construct a noisy quantum computer
    :param n_qubits: 34 qubits ought to be enough for anybody.
    :return: A pre-configured QuantumComputer
    """
    fully_connected_device = NxDevice(topology=nx.complete_graph(n_qubits))
    if noisy:
        # note to developers: the noise model specifies noise for each possible gate. In a fully
        # connected topology, there are a lot.
        noise_model = decoherence_noise_with_asymmetric_ro(
            gates=gates_in_isa(fully_connected_device.get_isa()))
    else:
        noise_model = None

    name = f'{n_qubits}q-noisy-qvm' if noisy else f'{n_qubits}q-qvm'
    return QuantumComputer(name=name,
                           qam=QVM(connection=connection, noise_model=noise_model),
                           device=fully_connected_device,
                           compiler=_get_qvm_compiler_based_on_endpoint(
                               device=fully_connected_device,
                               endpoint=connection.compiler_endpoint))


@_record_call
def get_qc(name: str, *, as_qvm: bool = None, noisy: bool = None,
           connection: ForestConnection = None) -> QuantumComputer:
    """
    Get a quantum computer.

    A quantum computer is an object of type :py:class:`QuantumComputer` and can be backed
    either by a QVM simulator ("Quantum/Quil Virtual Machine") or a physical Rigetti QPU ("Quantum
    Processing Unit") made of superconducting qubits.

    You can choose the quantum computer to target through a combination of its name and optional
    flags. There are multiple ways to get the same quantum computer. The following are equivalent::

        >>> qc = get_qc("Aspen-1-16Q-A-noisy-qvm")
        >>> qc = get_qc("Aspen-1-16Q-A", as_qvm=True, noisy=True)

    and will construct a simulator of an Aspen-1 lattice with a noise model based on device
    characteristics. We also provide a means for constructing generic quantum simulators that
    are not related to a given piece of Rigetti hardware::

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
        regardless of the name's suffix. The noise model for QVMs based on a real QPU
        is an empirically parameterized model based on real device noise characteristics.
        The generic QVM noise model is simple T1 and T2 noise plus readout error. See
        :py:func:`~pyquil.noise.decoherence_noise_with_asymmetric_ro`.
    :param connection: An optional :py:class:`ForestConnection` object. If not specified,
        the default values for URL endpoints, ping time, and status time will be used. Your
        user id and API key will be read from ~/.pyquil_config. If you deign to change any
        of these parameters, pass your own :py:class:`ForestConnection` object.
    :return: A pre-configured QuantumComputer
    """
    if connection is None:
        connection = ForestConnection()

    full_name = name
    name, as_qvm, noisy = _parse_name(name, as_qvm, noisy)

    ma = re.fullmatch(r'(\d+)q', name)
    if ma is not None:
        n_qubits = int(ma.group(1))
        if not as_qvm:
            raise ValueError("Please name a valid device or run as a QVM")
        return _get_unrestricted_qvm(connection=connection, noisy=noisy, n_qubits=n_qubits)

    if name == '9q-generic' or name == '9q-square':
        if name == '9q-generic':
            warnings.warn("Please prefer '9q-square' instead of '9q-generic'", DeprecationWarning)

        if not as_qvm:
            raise ValueError("The device '9q-square' is only available as a QVM")
        return _get_9q_square_qvm(connection=connection, noisy=noisy)

    device = get_lattice(name)
    if not as_qvm:
        if noisy is not None and noisy:
            warnings.warn("You have specified `noisy=True`, but you're getting a QPU. This flag "
                          "is meant for controlling noise models on QVMs.")
        return QuantumComputer(name=full_name,
                               qam=QPU(endpoint=pyquil_config.qpu_url, user=pyquil_config.user_id),
                               device=device,
                               compiler=QPUCompiler(endpoint=pyquil_config.compiler_url,
                                                    device=device))

    if noisy:
        noise_model = device.noise_model
    else:
        noise_model = None

    return QuantumComputer(name=full_name,
                           qam=QVM(connection=connection,
                                   noise_model=noise_model,
                                   requires_executable=True),
                           device=device,
                           compiler=_get_qvm_compiler_based_on_endpoint(
                               device=device,
                               endpoint=connection.compiler_endpoint))


@contextmanager
def local_qvm() -> Iterator[Tuple[subprocess.Popen, subprocess.Popen]]:
    """A context manager for the Rigetti local QVM and QUIL compiler.

    You must first have installed the `qvm` and `quilc` executables from
    the forest SDK. [https://www.rigetti.com/forest]

    This context manager will start up external processes for both the
    compiler and virtual machine, and then terminate them when the context
    is exited.

    If `qvm` (or `quilc`) is already running, then the existing process will
    be used, and will not terminated at exit.

    >>> from pyquil import get_qc, Program
    >>> from pyquil.gates import CNOT, Z
    >>> from pyquil.api import local_qvm
    >>>
    >>> qvm = get_qc('9q-square-qvm')
    >>> prog = Program(Z(0), CNOT(0, 1))
    >>>
    >>> with local_qvm():
    >>>     results = qvm.run_and_measure(prog, trials=10)

    :raises: FileNotFoundError: If either executable is not installed.
    """
    # Enter. Acquire resource
    qvm = subprocess.Popen(['qvm', '-S'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)

    quilc = subprocess.Popen(['quilc', '-S'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

    # Return context
    try:
        yield (qvm, quilc)

    finally:
        # Exit. Release resource
        qvm.terminate()
        quilc.terminate()

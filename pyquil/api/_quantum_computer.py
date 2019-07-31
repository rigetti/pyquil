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
from rpcq.messages import BinaryExecutableResponse, PyQuilExecutableResponse

from pyquil.api._compiler import QPUCompiler, QVMCompiler
from pyquil.api._config import PyquilConfig
from pyquil.api._devices import get_lattice, list_lattices
from pyquil.api._error_reporting import _record_call
from pyquil.api._qac import AbstractCompiler
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._qvm import ForestConnection, QVM
from pyquil.device import AbstractDevice, NxDevice, gates_in_isa, ISA, Device
from pyquil.gates import RX, MEASURE
from pyquil.noise import decoherence_noise_with_asymmetric_ro, NoiseModel
from pyquil.pyqvm import PyQVM
from pyquil.quil import Program, validate_supported_quil
from pyquil.quilbase import Measurement, Pragma

pyquil_config = PyquilConfig()

Executable = Union[BinaryExecutableResponse, PyQuilExecutableResponse]


def _get_flipped_protoquil_program(program: Program) -> Program:
    """For symmetrization, generate a program where X gates are added before measurement.

    Forest is picky about where the measure instructions happen. It has to be at the end!
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
        validate_supported_quil(program)
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
                optimize: bool = True,
                protoquil: bool = None) -> Union[BinaryExecutableResponse, PyQuilExecutableResponse]:
        """
        A high-level interface to program compilation.

        Compilation currently consists of two stages. Please see the :py:class:`AbstractCompiler`
        docs for more information. This function does all stages of compilation.

        Right now both ``to_native_gates`` and ``optimize`` must be either both set or both
        unset. More modular compilation passes may be available in the future.

        :param program: A Program
        :param to_native_gates: Whether to compile non-native gates to native gates.
        :param optimize: Whether to optimize the program to reduce the number of operations.
        :param protoquil: Whether to restrict the input program to and the compiled program
            to protoquil (executable on QPU). A value of ``None`` means defer to server.
        :return: An executable binary suitable for passing to :py:func:`QuantumComputer.run`.
        """
        flags = [to_native_gates, optimize]
        assert all(flags) or all(not f for f in flags), "Must turn quilc all on or all off"
        quilc = all(flags)

        if quilc:
            nq_program = self.compiler.quil_to_native_quil(program, protoquil=protoquil)
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


def _parse_name(name: str, as_qvm: bool, noisy: bool) -> Tuple[str, str, bool]:
    """
    Try to figure out whether we're getting a (noisy) qvm, and the associated qpu name.

    See :py:func:`get_qc` for examples of valid names + flags.
    """
    parts = name.split('-')
    if len(parts) >= 2 and parts[-2] == 'noisy' and parts[-1] in ['qvm', 'pyqvm']:
        if as_qvm is not None and (not as_qvm):
            raise ValueError("The provided qc name indicates you are getting a noisy QVM, "
                             "but you have specified `as_qvm=False`")

        if noisy is not None and (not noisy):
            raise ValueError("The provided qc name indicates you are getting a noisy QVM, "
                             "but you have specified `noisy=False`")

        qvm_type = parts[-1]
        noisy = True
        prefix = '-'.join(parts[:-2])
        return prefix, qvm_type, noisy

    if len(parts) >= 1 and parts[-1] in ['qvm', 'pyqvm']:
        if as_qvm is not None and (not as_qvm):
            raise ValueError("The provided qc name indicates you are getting a QVM, "
                             "but you have specified `as_qvm=False`")
        qvm_type = parts[-1]
        if noisy is None:
            noisy = False
        prefix = '-'.join(parts[:-1])
        return prefix, qvm_type, noisy

    if as_qvm is not None and as_qvm:
        qvm_type = 'qvm'
    else:
        qvm_type = None

    if noisy is None:
        noisy = False

    return name, qvm_type, noisy


def _canonicalize_name(prefix, qvm_type, noisy):
    """Take the output of _parse_name to create a canonical name.
    """
    if noisy:
        noise_suffix = '-noisy'
    else:
        noise_suffix = ''

    if qvm_type is None:
        qvm_suffix = ''
    elif qvm_type == 'qvm':
        qvm_suffix = '-qvm'
    elif qvm_type == 'pyqvm':
        qvm_suffix = '-pyqvm'
    else:
        raise ValueError(f"Unknown qvm_type {qvm_type}")

    name = f'{prefix}{noise_suffix}{qvm_suffix}'
    return name


def _get_qvm_or_pyqvm(qvm_type, connection, noise_model=None, device=None,
                      requires_executable=False):
    if qvm_type == 'qvm':
        return QVM(connection=connection, noise_model=noise_model,
                   requires_executable=requires_executable)
    elif qvm_type == 'pyqvm':
        return PyQVM(n_qubits=device.qubit_topology().number_of_nodes())

    raise ValueError("Unknown qvm type {}".format(qvm_type))


def _get_qvm_qc(name: str, qvm_type: str, device: AbstractDevice, noise_model: NoiseModel = None,
                requires_executable: bool = False,
                connection: ForestConnection = None) -> QuantumComputer:
    """Construct a QuantumComputer backed by a QVM.

    This is a minimal wrapper over the QuantumComputer, QVM, and QVMCompiler constructors.

    :param name: A string identifying this particular quantum computer.
    :param qvm_type: The type of QVM. Either qvm or pyqvm.
    :param device: A device following the AbstractDevice interface.
    :param noise_model: An optional noise model
    :param requires_executable: Whether this QVM will refuse to run a :py:class:`Program` and
        only accept the result of :py:func:`compiler.native_quil_to_executable`. Setting this
        to True better emulates the behavior of a QPU.
    :param connection: An optional :py:class:`ForestConnection` object. If not specified,
        the default values for URL endpoints will be used.
    :return: A QuantumComputer backed by a QVM with the above options.
    """
    if connection is None:
        connection = ForestConnection()

    return QuantumComputer(name=name,
                           qam=_get_qvm_or_pyqvm(
                               qvm_type=qvm_type,
                               connection=connection,
                               noise_model=noise_model,
                               device=device,
                               requires_executable=requires_executable),
                           device=device,
                           compiler=QVMCompiler(
                               device=device,
                               endpoint=connection.compiler_endpoint))


def _get_qvm_with_topology(name: str, topology: nx.Graph,
                           noisy: bool = False,
                           requires_executable: bool = True,
                           connection: ForestConnection = None,
                           qvm_type: str = 'qvm') -> QuantumComputer:
    """Construct a QVM with the provided topology.

    :param name: A name for your quantum computer. This field does not affect behavior of the
        constructed QuantumComputer.
    :param topology: A graph representing the desired qubit connectivity.
    :param noisy: Whether to include a generic noise model. If you want more control over
        the noise model, please construct your own :py:class:`NoiseModel` and use
        :py:func:`_get_qvm_qc` instead of this function.
    :param requires_executable: Whether this QVM will refuse to run a :py:class:`Program` and
        only accept the result of :py:func:`compiler.native_quil_to_executable`. Setting this
        to True better emulates the behavior of a QPU.
    :param connection: An optional :py:class:`ForestConnection` object. If not specified,
        the default values for URL endpoints will be used.
    :param qvm_type: The type of QVM. Either 'qvm' or 'pyqvm'.
    :return: A pre-configured QuantumComputer
    """
    # Note to developers: consider making this function public and advertising it.
    device = NxDevice(topology=topology)
    if noisy:
        noise_model = decoherence_noise_with_asymmetric_ro(gates=gates_in_isa(device.get_isa()))
    else:
        noise_model = None
    return _get_qvm_qc(name=name, qvm_type=qvm_type, connection=connection, device=device,
                       noise_model=noise_model, requires_executable=requires_executable)


def _get_9q_square_qvm(name: str, noisy: bool,
                       connection: ForestConnection = None,
                       qvm_type: str = 'qvm') -> QuantumComputer:
    """
    A nine-qubit 3x3 square lattice.

    This uses a "generic" lattice not tied to any specific device. 9 qubits is large enough
    to do vaguely interesting algorithms and small enough to simulate quickly.

    :param name: The name of this QVM
    :param connection: The connection to use to talk to external services
    :param noisy: Whether to construct a noisy quantum computer
    :param qvm_type: The type of QVM. Either 'qvm' or 'pyqvm'.
    :return: A pre-configured QuantumComputer
    """
    topology = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    return _get_qvm_with_topology(name=name, connection=connection,
                                  topology=topology,
                                  noisy=noisy,
                                  requires_executable=True,
                                  qvm_type=qvm_type)


def _get_unrestricted_qvm(name: str, noisy: bool,
                          n_qubits: int = 34,
                          connection: ForestConnection = None,
                          qvm_type: str = 'qvm') -> QuantumComputer:
    """
    A qvm with a fully-connected topology.

    This is obviously the least realistic QVM, but who am I to tell users what they want.

    :param name: The name of this QVM
    :param noisy: Whether to construct a noisy quantum computer
    :param n_qubits: 34 qubits ought to be enough for anybody.
    :param connection: The connection to use to talk to external services
    :param qvm_type: The type of QVM. Either 'qvm' or 'pyqvm'.
    :return: A pre-configured QuantumComputer
    """
    topology = nx.complete_graph(n_qubits)
    return _get_qvm_with_topology(name=name, connection=connection,
                                  topology=topology,
                                  noisy=noisy,
                                  requires_executable=False,
                                  qvm_type=qvm_type)


def _get_qvm_based_on_real_device(name: str, device: Device,
                                  noisy: bool, connection: ForestConnection = None,
                                  qvm_type: str = 'qvm'):
    """
    A qvm with a based on a real device.

    This is the most realistic QVM.

    :param name: The full name of this QVM
    :param device: The device from :py:func:`get_lattice`.
    :param noisy: Whether to construct a noisy quantum computer by using the device's
        associated noise model.
    :param connection: An optional :py:class:`ForestConnection` object. If not specified,
        the default values for URL endpoints will be used.
    :return: A pre-configured QuantumComputer based on the named device.
    """
    if noisy:
        noise_model = device.noise_model
    else:
        noise_model = None
    return _get_qvm_qc(name=name, connection=connection, device=device,
                       noise_model=noise_model, requires_executable=True,
                       qvm_type=qvm_type)


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

    The Rigetti QVM must be downloaded from https://www.rigetti.com/forest and run as a server
    alongside your python program. To use pyQuil's built-in QVM, replace all ``"-qvm"`` suffixes
    with ``"-pyqvm"``::

        >>> qc = get_qc("5q-pyqvm")

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
        a QVM. Names ending in "-pyqvm" will return a :py:class:`PyQVM`. Names ending in
        "-noisy-qvm" will return a QVM with a noise model. Otherwise, we will return a QPU with
        the given name.
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
        the default values for URL endpoints will be used. If you deign to change any
        of these parameters, pass your own :py:class:`ForestConnection` object.
    :return: A pre-configured QuantumComputer
    """
    # 1. Parse name, check for redundant options, canonicalize names.
    prefix, qvm_type, noisy = _parse_name(name, as_qvm, noisy)
    del as_qvm  # do not use after _parse_name
    name = _canonicalize_name(prefix, qvm_type, noisy)

    # 2. Check for unrestricted {n}q-qvm
    ma = re.fullmatch(r'(\d+)q', prefix)
    if ma is not None:
        n_qubits = int(ma.group(1))
        if qvm_type is None:
            raise ValueError("Please name a valid device or run as a QVM")
        return _get_unrestricted_qvm(name=name, connection=connection,
                                     noisy=noisy, n_qubits=n_qubits, qvm_type=qvm_type)

    # 3. Check for "9q-square" qvm
    if prefix == '9q-generic' or prefix == '9q-square':
        if prefix == '9q-generic':
            warnings.warn("Please prefer '9q-square' instead of '9q-generic'", DeprecationWarning)

        if qvm_type is None:
            raise ValueError("The device '9q-square' is only available as a QVM")
        return _get_9q_square_qvm(name=name, connection=connection, noisy=noisy, qvm_type=qvm_type)

    # 4. Not a special case, query the web for information about this device.
    device = get_lattice(prefix)
    if qvm_type is not None:
        # 4.1 QVM based on a real device.
        return _get_qvm_based_on_real_device(name=name, device=device,
                                             noisy=noisy, connection=connection, qvm_type=qvm_type)
    else:
        # 4.2 A real device
        if noisy is not None and noisy:
            warnings.warn("You have specified `noisy=True`, but you're getting a QPU. This flag "
                          "is meant for controlling noise models on QVMs.")
        return QuantumComputer(name=name,
                               qam=QPU(
                                   endpoint=pyquil_config.qpu_url,
                                   user=pyquil_config.user_id),
                               device=device,
                               compiler=QPUCompiler(
                                   quilc_endpoint=pyquil_config.quilc_url,
                                   qpu_compiler_endpoint=pyquil_config.qpu_compiler_url,
                                   device=device,
                                   name=prefix))


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

    quilc = subprocess.Popen(['quilc', '-RP'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

    # Return context
    try:
        yield (qvm, quilc)

    finally:
        # Exit. Release resource
        qvm.terminate()
        quilc.terminate()

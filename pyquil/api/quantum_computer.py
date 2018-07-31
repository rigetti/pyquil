import warnings
from math import pi
from typing import List

import networkx as nx
import numpy as np

from pyquil.api import get_devices, QPU, ForestConnection, QVM
from pyquil.api._qam import QAM
from pyquil.device import AbstractDevice, NxDevice
from pyquil.gates import MEASURE, RX
from pyquil.noise import decoherance_noise_with_asymettric_ro
from pyquil.quil import Program, get_classical_addresses_from_program
from pyquil.quilbase import Measurement, Pragma


def _get_flipped_protoquil_program(program: Program):
    """For symmetrization, generate a program where X gates are added before measurement.

    Forest 1.3 is really picky about where the measure instructions happen. It has to be
    at the end!
    """
    program = Program(program.instructions)  # Copy
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
    def __init__(self, *, name: str, qam: QAM, device: AbstractDevice, symmetrize_readout=False):
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

        self.symmetrize_readout = symmetrize_readout

    def qubit_topology(self):
        return self.device.qubit_topology()

    def get_isa(self, oneq_type='Xhalves', twoq_type='CZ'):
        return self.device.get_isa(oneq_type=oneq_type, twoq_type=twoq_type)

    def run(self, program, classical_addresses, trials, symmetrize_readout=None) -> np.ndarray:
        """
        Run a quil program.

        :param program: The program to run. You probably want to put MEASURE instructions
            in your program somewhere (like at the end) because qubits are not automatically
            measured
        :param classical_addresses: The addresses of the classical bits to return. These don't
            necessarily correspond to qubit indices; rather they are the second argument to
            any MEASURE instructions you've added to your program
        :param trials: The number of times to run the program.
        :param symmetrize_readout: Whether to apply readout error symmetrization. If not
            specified, the instance attribute ``symmetrize_readout`` will be used. See
            :py:func:`run_symmetrized_readout` for a complete description.
        :return: A numpy array of shape (trials, len(classical_addresses)) that contains 0s and 1s
        """
        if symmetrize_readout is None:
            symmetrize_readout = self.symmetrize_readout

        if not classical_addresses:
            classical_addresses = get_classical_addresses_from_program(program)

        if symmetrize_readout:
            return self.run_symmetrized_readout(program, classical_addresses, trials)

        return self.qam.run(program, classical_addresses, trials)

    def run_async(self, program, classical_addresses, trials, symmetrize_readout=None) -> str:
        """
        Queue a quil program for running, but return immediately with a job id.

        Use :py:func:`QuantumComputer.wait_for_job` to get the actual job results, probably
        after queueing up a whole batch of jobs.

        See :py:func:`run` for this function's parameter descriptions.

        :returns: a job id
        """
        if symmetrize_readout is None:
            symmetrize_readout = self.symmetrize_readout

        if not classical_addresses:
            classical_addresses = get_classical_addresses_from_program(program)

        if symmetrize_readout:
            raise NotImplementedError("Async symmetrized readout isn't supported")

        return self.qam.run_async(program, classical_addresses, trials)

    def run_symmetrized_readout(self, program, classical_addresses, trials):
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

        samples = self.run(program, classical_addresses, half_trials, symmetrize_readout=False)
        flipped_samples = self.run(flipped_program, classical_addresses, half_trials,
                                   symmetrize_readout=False)
        double_flipped_samples = np.logical_not(flipped_samples).astype(int)
        results = np.concatenate((samples, double_flipped_samples), axis=0)
        np.random.shuffle(results)
        return results

    def run_and_measure(self, program: Program, qubits: List[int], trials: int,
                        symmetrize_readout=None):
        """
        Run the provided state preparation program and measure all qubits contained in the program.

        .. note::

            In contrast to :py:class:`QVMConnection.run_and_measure`, this method simulates
            noise correctly for noisy QVMs. However, this method is slower for ``trials > 1``.
            For faster noise-free simulation, consider
            :py:class:`WavefunctionSimulator.run_and_measure`.

        :param program: The state preparation program to run and then measure.
        :param qubits: Qubit indices to measure.
        :param trials: The number of times to run the program.
        :param symmetrize_readout: Whether to apply readout error symmetrization. If not specified,
            the class attribute ``symmetrize_readout`` will be used. See
            :py:func:`run_symmetrized_readout` for a complete description.
        :return: A numpy array of shape (trials, len(qubits)) that contains 0s and 1s
        """
        new_prog = Program().inst(program)  # make a copy?

        classical_addrs = list(range(len(qubits)))
        for q, i in zip(qubits, classical_addrs):
            new_prog += MEASURE(q, i)

        return self.run(program=new_prog, classical_addresses=classical_addrs,
                        trials=trials, symmetrize_readout=symmetrize_readout)

    def wait_for_job(self, job_id):
        return self.qam.wait_for_job(job_id)

    def __str__(self):
        return self.name


def list_quantum_computers(connection: ForestConnection = None, qpus=True, qvms=True) -> List[str]:
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
        # TODO: Use this to list devices?
        connection = ForestConnection()

    qc_names = []
    if qpus:
        for qpu_name in ['8Q-Agave', '19Q-Acorn']:
            qc_names += [qpu_name]
            if qvms:
                qc_names += ["{}-qvm".format(qpu_name), "{}-noisy-qvm".format(qpu_name)]

    if qvms:
        qc_names += ['9q-generic-qvm', '9q-generic-noisy-qvm']

    return qc_names


def _parse_name(name, as_qvm, noisy):
    """
    Try to figure out whether we're getting a (noisy) qvm, and the associated qpu name.

    See :py:func:`get_qc` for examples of valid names + flags.
    """
    if name.endswith('-noisy-qvm'):
        if as_qvm is not None and (not as_qvm):
            raise ValueError("The provided qc name indicates you are getting a noisy QVM, "
                             "but you have specified `as_qvm=False`")

        if noisy is not None and (not noisy):
            raise ValueError("The provided qc name indicates you are getting a noisy QVM, "
                             "but you have specified `noisy=False`")

        as_qvm = True
        noisy = True
        prefix = name.strip('-noisy-qvm')
        return prefix, as_qvm, noisy

    if name.endswith('-qvm'):
        if as_qvm is not None and (not as_qvm):
            raise ValueError("The provided qc name indicates you are getting a QVM, "
                             "but you have specified `as_qvm=False`")
        as_qvm = True
        if noisy is not None:
            noisy = False
        prefix = name.strip('-qvm')
        return prefix, as_qvm, noisy

    if as_qvm is None:
        as_qvm = False

    if noisy is None:
        noisy = False

    return name, as_qvm, noisy


def get_qc(name: str, *, as_qvm: bool = None, noisy: bool = None,
           connection: ForestConnection = None):
    """
    Get a quantum computer.

    A quantum computer is an object of type :py:class:`QuantumComputer` and can be backed
    either by a QVM simulator ("Quantum/Quil Virtual Machine") or a physical Rigetti QPU ("Quantum
    Processing Unit") made of superconducting qubits.

    You can choose the quantum computer to target through a combination of its name and optional
    flags. There are multiple ways to get the same quantum computer. The following are equivalent::

        >>> qc = get_qc("8Q-Agave-noisy-qvm")
        >>> qc = get_qc("8Q-Agave", as_qvm=True, noisy=True)

    and will construct a simulator of the 8q-agave chip with a noise model based on device
    characteristics. We also provide a means for constructing generic quantum simulators that
    are not related to a given piece of Rigetti hardware::

        >>> qc = get_qc("9q-generic-qvm")
        >>> qc = get_qc("9q-generic", as_qvm=True)

    Redundant flags are acceptable, but conflicting flags will raise an exception::

        >>> qc = get_qc("9q-generic-qvm") # qc is fully specified by its name
        >>> qc = get_qc("9q-generic-qvm", as_qvm=True) # redundant, but ok
        >>> qc = get_qc("9q-generic-qvm", as_qvm=False) # Error!

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
        regardless of the name's suffix
    :param connection: An optional :py:class:ForestConnection` object. If not specified,
        the default values for URL endpoints, ping time, and status time will be used. Your
        user id and API key will be read from ~/.pyquil_config. If you deign to change any
        of these parameters, pass your own :py:class:`ForestConnection` object.
    :return:
    """
    if connection is None:
        connection = ForestConnection()

    name, as_qvm, noisy = _parse_name(name, as_qvm, noisy)

    if name == '9q-generic':
        if not as_qvm:
            raise ValueError("The device '9q-generic' is only available as a QVM")

        nineq_square = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3,3))
        nineq_device = NxDevice(topology=nineq_square)
        if noisy:
            noise_model = decoherance_noise_with_asymettric_ro(nineq_device.get_isa())
        else:
            noise_model = None

        return QuantumComputer(name='9q-generic-qvm',
                               qam=QVM(connection=connection, noise_model=noise_model),
                               device=nineq_device)

    # At least based off a real device.
    device = get_devices(as_dict=True)[name]

    if not as_qvm:
        if noisy is not None and noisy:
            warnings.warn("You have specified `noisy=True`, but you're getting a QPU. This flag "
                          "is meant for controling noise models on QVMs.")
        return QuantumComputer(name=name,
                               qam=QPU(device_name=name, connection=connection),
                               device=device)

    if noisy:
        noise_model = device.noise_model
        name = "{name}-noisy-qvm".format(name=name)
    else:
        noise_model = None
        name = "{name}-qvm".format(name=name)

    return QuantumComputer(name=name,
                           qam=QVM(connection=connection, noise_model=noise_model),
                           device=device)

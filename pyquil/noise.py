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
"""
Module for creating and verifying noisy gate and readout definitions.
"""
from __future__ import print_function
from collections import namedtuple
from typing import Sequence

import numpy as np
import sys

from pyquil.gates import I, MEASURE, X
from pyquil.parameters import format_parameter
from pyquil.quilbase import Pragma, Gate

INFINITY = float("inf")
"Used for infinite coherence times."


_KrausModel = namedtuple("_KrausModel", ["gate", "params", "targets", "kraus_ops", "fidelity"])


class KrausModel(_KrausModel):
    """
    Encapsulate a single gate's noise model.

    :ivar str gate: The name of the gate.
    :ivar Sequence[float] params: Optional parameters for the gate.
    :ivar Sequence[int] targets: The target qubit ids.
    :ivar Sequence[np.array] kraus_ops: The Kraus operators (must be square complex numpy arrays).
    :ivar float fidelity: The average gate fidelity associated with the Kraus map relative to the
        ideal operation.
    """

    @staticmethod
    def unpack_kraus_matrix(m):
        """
        Helper to optionally unpack a JSON compatible representation of a complex Kraus matrix.

        :param Union[list,np.array] m: The representation of a Kraus operator. Either a complex
            square matrix (as numpy array or nested lists) or a JSON-able pair of real matrices
            (as nested lists) representing the element-wise real and imaginary part of m.
        :return: A complex square numpy array representing the Kraus operator.
        :rtype: np.array
        """
        m = np.asarray(m, dtype=complex)
        if m.ndim == 3:
            m = m[0] + 1j * m[1]
        if not m.ndim == 2:  # pragma no coverage
            raise ValueError("Need 2d array.")
        if not m.shape[0] == m.shape[1]:  # pragma no coverage
            raise ValueError("Need square matrix.")
        return m

    def to_dict(self):
        """
        Create a dictionary representation of a KrausModel.

        For example::

            {
                "gate": "RX",
                "params": np.pi,
                "targets": [0],
                "kraus_ops": [            # In this example single Kraus op = ideal RX(pi) gate
                    [[[0,   0],           # element-wise real part of matrix
                      [0,   0]],
                      [[0, -1],           # element-wise imaginary part of matrix
                      [-1, 0]]]
                ],
                "fidelity": 1.0
            }

        :return: A JSON compatible dictionary representation.
        :rtype: Dict[str,Any]
        """
        res = self._asdict()
        res['kraus_ops'] = [[k.real.tolist(), k.imag.tolist()] for k in self.kraus_ops]
        return res

    @staticmethod
    def from_dict(d):
        """
        Recreate a KrausModel from the dictionary representation.

        :param dict d: The dictionary representing the KrausModel. See `to_dict` for an
            example.
        :return: The deserialized KrausModel.
        :rtype: KrausModel
        """
        kraus_ops = [KrausModel.unpack_kraus_matrix(k) for k in d['kraus_ops']]
        return KrausModel(d['gate'], d['params'], d['targets'], kraus_ops, d['fidelity'])

    def __eq__(self, other):
        return isinstance(other, KrausModel) and self.to_dict() == other.to_dict()

    def __neq__(self, other):
        return not self.__eq__(other)


_NoiseModel = namedtuple("_NoiseModel", ["gates", "assignment_probs"])


class NoiseModel(_NoiseModel):
    """
    Encapsulate the QPU noise model containing information about the noisy gates.

    :ivar Sequence[KrausModel] gates: The tomographic estimates of all gates.
    :ivar Dict[int,np.array] assignment_probs: The single qubit readout assignment
        probability matrices keyed by qubit id.
    """

    def to_dict(self):
        """
        Create a JSON serializable representation of the noise model.

        For example::

            {
                "gates": [
                    # list of embedded dictionary representations of KrausModels here [...]
                ]
                "assignment_probs": {
                    "0": [[.8, .1],
                          [.2, .9]],
                    "1": [[.9, .4],
                          [.1, .6]],
                }
            }

        :return: A dictionary representation of self.
        :rtype: Dict[str,Any]
        """
        return {
            "gates": [km.to_dict() for km in self.gates],
            "assignment_probs": {str(qid): a.tolist() for qid, a in self.assignment_probs.items()},
        }

    @staticmethod
    def from_dict(d):
        """
        Re-create the noise model from a dictionary representation.

        :param Dict[str,Any] d: The dictionary representation.
        :return: The restored noise model.
        :rtype: NoiseModel
        """
        return NoiseModel(
            gates=[KrausModel.from_dict(t) for t in d["gates"]],
            assignment_probs={int(qid): np.array(a) for qid, a in d["assignment_probs"].items()},
        )

    def gates_by_name(self, name):
        """
        Return all defined noisy gates of a particular gate name.

        :param str name: The gate name.
        :return: A list of noise models representing that gate.
        :rtype: Sequence[KrausModel]
        """
        return [g for g in self.gates if g.gate == name]

    def __eq__(self, other):
        return isinstance(other, NoiseModel) and self.to_dict() == other.to_dict()

    def __neq__(self, other):
        return not self.__eq__(other)


def _check_kraus_ops(n, kraus_ops):
    """
    Verify that the Kraus operators are of the correct shape and satisfy the correct normalization.

    :param int n: Number of qubits
    :param list|tuple kraus_ops: The Kraus operators as numpy.ndarrays.
    """
    for k in kraus_ops:
        if not np.shape(k) == (2 ** n, 2 ** n):
            raise ValueError(
                "Kraus operators for {0} qubits must have shape {1}x{1}: {2}".format(n, 2 ** n, k))

    kdk_sum = sum(np.transpose(k).conjugate().dot(k) for k in kraus_ops)
    if not np.allclose(kdk_sum, np.eye(2 ** n), atol=1e-3):
        raise ValueError(
            "Kraus operator not correctly normalized: sum_j K_j^*K_j == {}".format(kdk_sum))


def _create_kraus_pragmas(name, qubit_indices, kraus_ops):
    """
    Generate the pragmas to define a Kraus map for a specific gate on some qubits.

    :param str name: The name of the gate.
    :param list|tuple qubit_indices: The qubits
    :param list|tuple kraus_ops: The Kraus operators as matrices.
    :return: A QUIL string with PRAGMA ADD-KRAUS ... statements.
    :rtype: str
    """

    pragmas = [Pragma("ADD-KRAUS",
                      [name] + list(qubit_indices),
                      "({})".format(" ".join(map(format_parameter, np.ravel(k)))))
               for k in kraus_ops]
    return pragmas


def append_kraus_to_gate(kraus_ops, gate_matrix):
    """
    Follow a gate ``gate_matrix`` by a Kraus map described by ``kraus_ops``.

    :param list kraus_ops: The Kraus operators.
    :param numpy.ndarray gate_matrix: The unitary gate.
    :return: A list of transformed Kraus operators.
    """
    return [kj.dot(gate_matrix) for kj in kraus_ops]


def pauli_kraus_map(probabilities):
    r"""
    Generate the Kraus operators corresponding to a pauli channel.

    :params list|floats probabilities: The 4^num_qubits list of probabilities specifying the desired pauli channel.
    There should be either 4 or 16 probabilities specified in the order I, X, Y, Z for 1 qubit
    or II, IX, IY, IZ, XI, XX, XY, etc for 2 qubits.

            For example::

                The d-dimensional depolarizing channel \Delta parameterized as
                \Delta(\rho) = p \rho + [(1-p)/d] I
                is specified by the list of probabilities
                [p + (1-p)/d, (1-p)/d,  (1-p)/d), ... , (1-p)/d)]

    :return: A list of the 4^num_qubits Kraus operators that parametrize the map.
    :rtype: list
    """
    if len(probabilities) not in [4, 16]:
        raise ValueError("Currently we only support one or two qubits, "
                         "so the provided list of probabilities must have length 4 or 16.")
    if not np.allclose(sum(probabilities), 1.0, atol=1e-3):
        raise ValueError("Probabilities must sum to one.")

    paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]

    if len(probabilities) == 4:
        operators = paulis
    else:
        operators = np.kron(paulis, paulis)

    return [coeff * op for coeff, op in zip(np.sqrt(probabilities), operators)]


def damping_kraus_map(p=0.10):
    """
    Generate the Kraus operators corresponding to an amplitude damping
    noise channel.

    :param float p: The one-step damping probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    damping_op = np.sqrt(p) * np.array([[0, 1],
                                        [0, 0]])

    residual_kraus = np.diag([1, np.sqrt(1 - p)])
    return [residual_kraus, damping_op]


def dephasing_kraus_map(p=0.10):
    """
    Generate the Kraus operators corresponding to a dephasing channel.

    :params float p: The one-step dephasing probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    return [np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * np.diag([1, -1])]


def tensor_kraus_maps(k1, k2):
    """
    Generate the Kraus map corresponding to the composition
    of two maps on different qubits.

    :param list k1: The Kraus operators for the first qubit.
    :param list k2: The Kraus operators for the second qubit.
    :return: A list of tensored Kraus operators.
    """
    return [np.kron(k1j, k2l) for k1j in k1 for k2l in k2]


def combine_kraus_maps(k1, k2):
    """
    Generate the Kraus map corresponding to the composition
    of two maps on the same qubits with k1 being applied to the state
    after k2.

    :param list k1: The list of Kraus operators that are applied second.
    :param list k2: The list of Kraus operators that are applied first.
    :return: A combinatorially generated list of composed Kraus operators.
    """
    return [np.dot(k1j, k2l) for k1j in k1 for k2l in k2]


def damping_after_dephasing(T1, T2, gate_time):
    """
    Generate the Kraus map corresponding to the composition
    of a dephasing channel followed by an amplitude damping channel.

    :param float T1: The amplitude damping time
    :param float T2: The dephasing time
    :param float gate_time: The gate duration.
    :return: A list of Kraus operators.
    """
    damping = damping_kraus_map(p=1 - np.exp(-float(gate_time) / float(T1))) \
        if T1 != INFINITY else [np.eye(2)]
    dephasing = dephasing_kraus_map(p=.5 * (1 - np.exp(-2 * gate_time / float(T2)))) \
        if T2 != INFINITY else [np.eye(2)]
    return combine_kraus_maps(damping, dephasing)


# You can only apply gate-noise to non-parametrized gates or parametrized gates at fixed parameters.
NO_NOISE = ["RZ"]
ANGLE_TOLERANCE = 1e-10


class NoisyGateUndefined(Exception):
    """Raise when user attempts to use noisy gate outside of currently supported set."""
    pass


def get_noisy_gate(gate_name, params):
    """
    Look up the numerical gate representation and a proposed 'noisy' name.

    :param str gate_name: The Quil gate name
    :param Tuple[float] params: The gate parameters.
    :return: A tuple (matrix, noisy_name) with the representation of the ideal gate matrix
        and a proposed name for the noisy version.
    :rtype: Tuple[np.array, str]
    """
    params = tuple(params)
    if gate_name == "I":
        assert params == ()
        return np.eye(2), "NOISY-I"
    if gate_name == "RX":
        angle, = params
        if np.isclose(angle, np.pi / 2, atol=ANGLE_TOLERANCE):
            return (np.array([[1, -1j],
                              [-1j, 1]]) / np.sqrt(2),
                    "NOISY-RX-PLUS-90")
        elif np.isclose(angle, -np.pi / 2, atol=ANGLE_TOLERANCE):
            return (np.array([[1, 1j],
                              [1j, 1]]) / np.sqrt(2),
                    "NOISY-RX-MINUS-90")
        elif np.isclose(angle, np.pi, atol=ANGLE_TOLERANCE):
            return (np.array([[0, -1j],
                              [-1j, 0]]),
                    "NOISY-RX-PLUS-180")
        elif np.isclose(angle, -np.pi, atol=ANGLE_TOLERANCE):
            return (np.array([[0, 1j],
                              [1j, 0]]),
                    "NOISY-RX-MINUS-180")
    elif gate_name == "CZ":
        assert params == ()
        return np.diag([1, 1, 1, -1]), "NOISY-CZ"
    raise NoisyGateUndefined("Undefined gate and params: {}{}\n"
                             "Please restrict yourself to I, RX(+/-pi), RX(+/-pi/2), CZ"
                             .format(gate_name, params))


def _get_program_gates(prog):
    """
    Get all gate applications appearing in prog.

    :param Program prog: The program
    :return: A list of all Gates in prog (without duplicates).
    :rtype: List[Gate]
    """
    return sorted({i for i in prog if isinstance(i, Gate)}, key=lambda g: g.out())


def _decoherence_noise_model(gates, T1=30e-6, T2=30e-6, gate_time_1q=50e-9,
                             gate_time_2q=150e-09, ro_fidelity=0.95):
    """
    The default noise parameters

    - T1 = 30 us
    - T2 = 30 us
    - 1q gate time = 50 ns
    - 2q gate time = 150 ns

    are currently typical for near-term devices.

    This function will define new gates and add Kraus noise to these gates. It will translate
    the input program to use the noisy version of the gates.

    :param Sequence[Gate] gates: The gates to provide the noise model for.
    :param Union[Dict[int,float],float] T1: The T1 amplitude damping time either globally or in a
        dictionary indexed by qubit id. By default, this is 30 us.
    :param Union[Dict[int,float],float] T2: The T2 dephasing time either globally or in a
        dictionary indexed by qubit id. By default, this is also 30 us.
    :param float gate_time_1q: The duration of the one-qubit gates, namely RX(+pi/2) and RX(-pi/2).
        By default, this is 50 ns.
    :param float gate_time_2q: The duration of the two-qubit gates, namely CZ.
        By default, this is 150 ns.
    :param Union[Dict[int,float],float] ro_fidelity: The readout assignment fidelity
        :math:`F = (p(0|0) + p(1|1))/2` either globally or in a dictionary indexed by qubit id.
    :return: A NoiseModel with the appropriate Kraus operators defined.
    """
    all_qubits = set(sum(([t.index for t in g.qubits] for g in gates), []))
    if isinstance(T1, dict):
        all_qubits.update(T1.keys())
    if isinstance(T2, dict):
        all_qubits.update(T2.keys())
    if isinstance(ro_fidelity, dict):
        all_qubits.update(ro_fidelity.keys())

    if not isinstance(T1, dict):
        T1 = {q: T1 for q in all_qubits}

    if not isinstance(T2, dict):
        T2 = {q: T2 for q in all_qubits}

    if not isinstance(ro_fidelity, dict):
        ro_fidelity = {q: ro_fidelity for q in all_qubits}

    noisy_identities_1q = {
        q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time_1q)
        for q in all_qubits
    }
    noisy_identities_2q = {
        q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time_2q)
        for q in all_qubits
    }
    kraus_maps = []
    for g in gates:
        targets = tuple(t.index for t in g.qubits)
        key = (g.name, tuple(g.params))
        if g.name in NO_NOISE:
            continue
        matrix, _ = get_noisy_gate(g.name, g.params)

        if len(targets) == 1:
            noisy_I = noisy_identities_1q[targets[0]]
        else:
            if len(targets) != 2:
                raise ValueError("Noisy gates on more than 2Q not currently supported")

            # note this ordering of the tensor factors is necessary due to how the QVM orders
            # the wavefunction basis
            noisy_I = tensor_kraus_maps(noisy_identities_2q[targets[1]],
                                        noisy_identities_2q[targets[0]])
        kraus_maps.append(KrausModel(g.name, tuple(g.params), targets,
                                     combine_kraus_maps(noisy_I, [matrix]),
                                     # FIXME (Nik): compute actual avg gate fidelity for this simple
                                     # noise model
                                     1.0))
    aprobs = {}
    for q, f_ro in ro_fidelity.items():
        aprobs[q] = np.array([[f_ro, 1. - f_ro],
                              [1. - f_ro, f_ro]])

    return NoiseModel(kraus_maps, aprobs)


def decoherence_noise_with_asymmetric_ro(gates: Sequence[Gate], p00=0.975, p11=0.911):
    """Similar to :py:func:`_decoherence_noise_model`, but with asymmetric readout.

    For simplicity, we use the default values for T1, T2, gate times, et al. and only allow
    the specification of readout fidelities.
    """
    noise_model = _decoherence_noise_model(gates)
    aprobs = np.array([[p00, 1 - p00],
                       [1 - p11, p11]])
    aprobs = {q: aprobs for q in noise_model.assignment_probs.keys()}
    return NoiseModel(noise_model.gates, aprobs)


def _noise_model_program_header(noise_model):
    """
    Generate the header for a pyquil Program that uses ``noise_model`` to overload noisy gates.
    The program header consists of 3 sections:

        - The ``DEFGATE`` statements that define the meaning of the newly introduced "noisy" gate
          names.
        - The ``PRAGMA ADD-KRAUS`` statements to overload these noisy gates on specific qubit
          targets with their noisy implementation.
        - THe ``PRAGMA READOUT-POVM`` statements that define the noisy readout per qubit.

    :param NoiseModel noise_model: The assumed noise model.
    :return: A quil Program with the noise pragmas.
    :rtype: pyquil.quil.Program
    """
    from pyquil.quil import Program
    p = Program()
    defgates = set()
    for k in noise_model.gates:

        # obtain ideal gate matrix and new, noisy name by looking it up in the NOISY_GATES dict
        try:
            ideal_gate, new_name = get_noisy_gate(k.gate, tuple(k.params))

            # if ideal version of gate has not yet been DEFGATE'd, do this
            if new_name not in defgates:
                p.defgate(new_name, ideal_gate)
                defgates.add(new_name)
        except NoisyGateUndefined:
            print("WARNING: Could not find ideal gate definition for gate {}".format(k.gate),
                  file=sys.stderr)
            new_name = k.gate

        # define noisy version of gate on specific targets
        p.define_noisy_gate(new_name, k.targets, k.kraus_ops)

    # define noisy readouts
    for q, ap in noise_model.assignment_probs.items():
        p.define_noisy_readout(q, p00=ap[0, 0], p11=ap[1, 1])
    return p


def apply_noise_model(prog, noise_model):
    """
    Apply a noise model to a program and generated a 'noisy-fied' version of the program.

    :param Program prog: A Quil Program object.
    :param NoiseModel noise_model: A NoiseModel, either generated from an ISA or
        from a simple decoherence model.
    :return: A new program translated to a noisy gateset and with noisy readout as described by the
        noisemodel.
    :rtype: Program
    """
    new_prog = _noise_model_program_header(noise_model)
    for i in prog:
        if isinstance(i, Gate):
            try:
                _, new_name = get_noisy_gate(i.name, tuple(i.params))
                new_prog += Gate(new_name, [], i.qubits)
            except NoisyGateUndefined:
                new_prog += i
        else:
            new_prog += i
    return new_prog


def add_decoherence_noise(prog, T1=30e-6, T2=30e-6, gate_time_1q=50e-9, gate_time_2q=150e-09,
                          ro_fidelity=0.95):
    """
    Add generic damping and dephasing noise to a program.

    This high-level function is provided as a convenience to investigate the effects of a
    generic noise model on a program. For more fine-grained control, please investigate
    the other methods available in the ``pyquil.noise`` module.

    In an attempt to closely model the QPU, noisy versions of RX(+-pi/2) and CZ are provided;
    I and parametric RZ are noiseless, and other gates are not allowed. To use this function,
    you need to compile your program to this native gate set.

    The default noise parameters

    - T1 = 30 us
    - T2 = 30 us
    - 1q gate time = 50 ns
    - 2q gate time = 150 ns

    are currently typical for near-term devices.

    This function will define new gates and add Kraus noise to these gates. It will translate
    the input program to use the noisy version of the gates.

    :param prog: A pyquil program consisting of I, RZ, CZ, and RX(+-pi/2) instructions
    :param Union[Dict[int,float],float] T1: The T1 amplitude damping time either globally or in a
        dictionary indexed by qubit id. By default, this is 30 us.
    :param Union[Dict[int,float],float] T2: The T2 dephasing time either globally or in a
        dictionary indexed by qubit id. By default, this is also 30 us.
    :param float gate_time_1q: The duration of the one-qubit gates, namely RX(+pi/2) and RX(-pi/2).
        By default, this is 50 ns.
    :param float gate_time_2q: The duration of the two-qubit gates, namely CZ.
        By default, this is 150 ns.
    :param Union[Dict[int,float],float] ro_fidelity: The readout assignment fidelity
        :math:`F = (p(0|0) + p(1|1))/2` either globally or in a dictionary indexed by qubit id.
    :return: A new program with noisy operators.
    """
    gates = _get_program_gates(prog)
    noise_model = _decoherence_noise_model(
        gates,
        T1=T1,
        T2=T2,
        gate_time_1q=gate_time_1q,
        gate_time_2q=gate_time_2q,
        ro_fidelity=ro_fidelity
    )
    return apply_noise_model(prog, noise_model)


def _bitstring_probs_by_qubit(p):
    """
    Ensure that an array ``p`` with bitstring probabilities has a separate axis for each qubit such
    that ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.

    This should not allocate much memory if ``p`` is already in ``C``-contiguous order (row-major).

    :param np.array p: An array that enumerates bitstring probabilities. When flattened out
        ``p = [p_00...0, p_00...1, ...,p_11...1]``. The total number of elements must therefore be a
        power of 2.
    :return: A reshaped view of ``p`` with a separate length-2 axis for each bit.
    """
    p = np.asarray(p, order="C")
    num_qubits = int(round(np.log2(p.size)))
    return p.reshape((2,) * num_qubits)


def estimate_bitstring_probs(results):
    """
    Given an array of single shot results estimate the probability distribution over all bitstrings.

    :param np.array results: A 2d array where the outer axis iterates over shots
        and the inner axis over bits.
    :return: An array with as many axes as there are qubit and normalized such that it sums to one.
        ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.
    :rtype: np.array
    """
    nshots, nq = np.shape(results)
    outcomes = np.array([int("".join(map(str, r)), 2) for r in results])
    probs = np.histogram(outcomes, bins=np.arange(-.5, 2 ** nq, 1))[0] / float(nshots)
    return _bitstring_probs_by_qubit(probs)


_CHARS = 'klmnopqrstuvwxyzabcdefgh0123456789'


def _apply_local_transforms(p, ts):
    """
    Given a 2d array of single shot results (outer axis iterates over shots, inner axis over bits)
    and a list of assignment probability matrices (one for each bit in the readout, ordered like
    the inner axis of results) apply local 2x2 matrices to each bit index.

    :param np.array p: An array that enumerates a function indexed by bitstrings::

            f(ijk...) = p[i,j,k,...]

    :param Sequence[np.array] ts: A sequence of 2x2 transform-matrices, one for each bit.
    :return: ``p_transformed`` an array with as many dimensions as there are bits with the result of
        contracting p along each axis by the corresponding bit transformation.

            p_transformed[ijk...] = f'(ijk...) = sum_lmn... ts[0][il] ts[1][jm] ts[2][kn] f(lmn...)

    :rtype: np.array
    """
    p_corrected = _bitstring_probs_by_qubit(p)
    nq = p_corrected.ndim
    for idx, trafo_idx in enumerate(ts):

        # this contraction pattern looks like
        # 'ij,abcd...jklm...->abcd...iklm...' so it properly applies a "local"
        # transformation to a single tensor-index without changing the order of
        # indices
        einsum_pat = ('ij,' + _CHARS[:idx] + 'j' + _CHARS[idx:nq - 1]
                      + '->' + _CHARS[:idx] + 'i' + _CHARS[idx:nq - 1])
        p_corrected = np.einsum(einsum_pat, trafo_idx, p_corrected)

    return p_corrected


def corrupt_bitstring_probs(p, assignment_probabilities):
    """
    Given a 2d array of true bitstring probabilities (outer axis iterates over shots, inner axis
    over bits) and a list of assignment probability matrices (one for each bit in the readout,
    ordered like the inner axis of results) compute the corrupted probabilities.

    :param np.array p: An array that enumerates bitstring probabilities. When
        flattened out ``p = [p_00...0, p_00...1, ...,p_11...1]``. The total number of elements must
        therefore be a power of 2. The canonical shape has a separate axis for each qubit, such that
        ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.
    :param List[np.array] assignment_probabilities: A list of assignment probability matrices
        per qubit. Each assignment probability matrix is expected to be of the form::

            [[p00 p01]
             [p10 p11]]

    :return: ``p_corrected`` an array with as many dimensions as there are qubits that contains
        the noisy-readout-corrected estimated probabilities for each measured bitstring, i.e.,
        ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.
    :rtype: np.array
    """
    return _apply_local_transforms(p, assignment_probabilities)


def correct_bitstring_probs(p, assignment_probabilities):
    """
    Given a 2d array of corrupted bitstring probabilities (outer axis iterates over shots, inner
    axis over bits) and a list of assignment probability matrices (one for each bit in the readout)
    compute the corrected probabilities.

    :param np.array p: An array that enumerates bitstring probabilities. When
        flattened out ``p = [p_00...0, p_00...1, ...,p_11...1]``. The total number of elements must
        therefore be a power of 2. The canonical shape has a separate axis for each qubit, such that
        ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.
    :param List[np.array] assignment_probabilities: A list of assignment probability matrices
        per qubit. Each assignment probability matrix is expected to be of the form::

            [[p00 p01]
             [p10 p11]]

    :return: ``p_corrected`` an array with as many dimensions as there are qubits that contains
        the noisy-readout-corrected estimated probabilities for each measured bitstring, i.e.,
        ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.
    :rtype: np.array
    """
    return _apply_local_transforms(p, (np.linalg.inv(ap) for ap in assignment_probabilities))


def bitstring_probs_to_z_moments(p):
    """
    Convert between bitstring probabilities and joint Z moment expectations.

    :param np.array p: An array that enumerates bitstring probabilities. When
        flattened out ``p = [p_00...0, p_00...1, ...,p_11...1]``. The total number of elements must
        therefore be a power of 2. The canonical shape has a separate axis for each qubit, such that
        ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.
    :return: ``z_moments``, an np.array with one length-2 axis per qubit which contains the
        expectations of all monomials in ``{I, Z_0, Z_1, ..., Z_{n-1}}``. The expectations of each
        monomial can be accessed via::

            <Z_0^j_0 Z_1^j_1 ... Z_m^j_m> = z_moments[j_0,j_1,...,j_m]

    :rtype: np.array
    """
    zmat = np.array([[1, 1],
                     [1, -1]])
    return _apply_local_transforms(p, (zmat for _ in range(p.ndim)))


def estimate_assignment_probs(q, trials, cxn, p0=None):
    """
    Estimate the readout assignment probabilities for a given qubit ``q``.
    The returned matrix is of the form::

            [[p00 p01]
             [p10 p11]]

    :param int q: The index of the qubit.
    :param int trials: The number of samples for each state preparation.
    :param Union[QVMConnection,QPUConnection] cxn: The quantum abstract machine to sample from.
    :param Program p0: A header program to prepend to the state preparation programs.
    :return: The assignment probability matrix
    :rtype: np.array
    """
    from pyquil.quil import Program
    if p0 is None:  # pragma no coverage
        p0 = Program()
    results_i = np.sum(cxn.run(p0 + Program(I(q), MEASURE(q, 0)), [0], trials))
    results_x = np.sum(cxn.run(p0 + Program(X(q), MEASURE(q, 0)), [0], trials))

    p00 = 1. - results_i / float(trials)
    p11 = results_x / float(trials)
    return np.array([[p00, 1 - p11],
                     [1 - p00, p11]])

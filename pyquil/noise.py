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
import warnings
from collections import namedtuple

import numpy as np

from pyquil.parameters import format_parameter
from pyquil.quilbase import Pragma, Gate, Qubit as QuilQubit


INFTY = float("inf")



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
            square matrix (as numpy array or nested lists) or a pair of real matrices (as numpy
            arrays or nested lists) representing the element-wise real and imaginary part of m.
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

    # def gate(self):
    #     """
    #     Generate the corresponding Quil Gate application that the Kraus map implements.
    #
    #     :return: A pyquil.quilbase.Gate object.
    #     :rtype: Gate
    #     """
    #     return Gate(self.gate, self.params, [QuilQubit(q) for q in self.targets])

    def __eq__(self, other):
        return isinstance(other, KrausModel) and self.to_dict() == other.to_dict()

    def __neq__(self, other):
        return not self.__eq__(other)


_NoiseModel = namedtuple("_NoiseModel", ["isa_name", "gates", "assignment_probs"])


class NoiseModel(_NoiseModel):
    """
    Encapsulate the QPU noise model containing information about the noisy gates.

    :ivar str isa_name: The name of the instruction set architecture for the QPU.
    :ivar Sequence[KrausModel] gates: The tomographic estimates of all gates.
    :ivar Dict[int,np.array] assignment_probs: The single qubit readout assignment
        probability matrices keyed by qubit id.
    """

    def to_dict(self):
        """
        Create a JSON serializable representation of the noise model.

        For example::

            {
                "isa_name": "example_qpu",
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
            "isa_name": self.isa_name,
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
            isa_name=d["isa_name"],
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
    if not np.allclose(kdk_sum, np.eye(2 ** n), atol=1e-5):
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
    damping = damping_kraus_map(p=gate_time / float(T1)) if T1 != INFTY else [np.eye(2)]
    dephasing = dephasing_kraus_map(p=gate_time / float(T2)) if T2 != INFTY else [np.eye(2)]
    return combine_kraus_maps(damping, dephasing)


# You can only apply gate-noise to non-parametrized gates or parametrized gates at fixed parameters.
NO_NOISE = ["RZ"]
SINGLE_Q = {
    ("I", ()): (np.eye(2), "NOISY-I"),
    ("RX", (np.pi/2,)): (np.array([[1, -1j],
                                   [-1j, 1]]) / np.sqrt(2), "NOISY-RX-PLUS-90"),
    ("RX", (-np.pi/2,)): (np.array([[1, 1j],
                                    [1j, 1]]) / np.sqrt(2), "NOISY-RX-MINUS-90"),
}
TWO_Q = {
    ("CZ", ()): (np.diag([1, 1, 1, -1]), "NOISY-CZ"),
}


def _get_program_gates(prog):
    """
    Get all gate applications appearing in prog.

    :param Program prog: The program
    :return: A list of all Gates in prog (without duplicates).
    :rtype: List[Gate]
    """
    return sorted({i for i in prog if isinstance(i, Gate)}, key=lambda g: g.out())


def _get_noisy_names(gates):
    """
    Generate new gate names for noisy gates.

    :param Sequence[Gate] gates: A list of Gate objects.
    :return: A dictionary with keys given by the input ``gates`` and values given by new gate name
        strings.
    :rtype: Dict[Gate,str]
    """
    ret = {}
    for g in gates:
        key = g.name, tuple(g.params)
        if g.name in NO_NOISE:
            ret[g] = g.name
        elif key in SINGLE_Q:
            ret[g] = SINGLE_Q[key][1]
        elif key in TWO_Q:
            ret[g] = TWO_Q[key][1]
        else:
            raise ValueError("Noise model for {} not yet supported".format(g))
    return ret


def decoherence_noise_model(gates, T1=30e-6, T2=None, gate_time_1q=50e-9,
                             gate_time_2q=150e-09, ro_fidelity=0.95):
    """
    The default noise parameters

    - T1 = 30 us
    - T2 = T1 / 2
    - 1q gate time = 50 ns
    - 2q gate time = 150 ns

    are currently typical for near-term devices.

    This function will define new gates and add Kraus noise to these gates. It will translate
    the input program to use the noisy version of the gates.

    :param Sequence[Gate] gates: The gates to provide the noise model for.
    :param Union[Dict[int,float],float] T1: The T1 amplitude damping time either globally or in a
        dictionary indexed by qubit id. By default, this is 30 us.
    :param Optional[Union[Dict[int,float],float]] T2: The T2 dephasing time either globally or in a
        dictionary indexed by qubit id. If None, this defaults to one-half of the T1 time.
        T2 is also constrained to not exceed T1/2.
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

    if T2 is None:
        T2 = {q: T1 / 2. for q, T1 in T1.items()}
    elif not isinstance(T2, dict):
        T2 = {q: T2 for q in all_qubits}

    # T2 can be at most T1/2
    T2 = {q: min(qT2, T1.get(q, INFTY) / 2.) for q, qT2 in T2.items()}

    if not isinstance(ro_fidelity, dict):
        ro_fidelity = {q: ro_fidelity for q in all_qubits}

    noisy_identities_1q = {
        q: damping_after_dephasing(T1.get(q, INFTY), T2.get(q, INFTY), gate_time_1q)
        for q in all_qubits
    }
    noisy_identities_2q = {
        q: damping_after_dephasing(T1.get(q, INFTY), T2.get(q, INFTY), gate_time_2q)
        for q in all_qubits
    }
    kraus_maps = []
    for g in gates:
        targets = tuple(t.index for t in g.qubits)
        key = (g.name, tuple(g.params))
        if g.name in NO_NOISE:
            continue
        if key in SINGLE_Q:
            matrix, _ = SINGLE_Q[key]
            noisy_I = noisy_identities_1q[targets[0]]
        elif key in TWO_Q:
            matrix, _ = TWO_Q[key]
            # note this ordering of the tensor factors is necessary due to how the QVM orders
            # the wavefunction basis
            noisy_I = tensor_kraus_maps(noisy_identities_2q[targets[1]],
                                        noisy_identities_2q[targets[0]])
        else:
            raise ValueError("Cannot create noisy version of {}. ".format(g) +
                             "Please restrict yourself to CZ, RX(+/-pi/2), I, RZ(theta)")
        kraus_maps.append(KrausModel(g.name, tuple(g.params), targets,
                                     combine_kraus_maps(noisy_I, [matrix]),
                                     # FIXME (Nik): compute actual avg gate fidelity for this simple
                                     # noise model
                                     1.0))
    aprobs = {}
    for q, f_ro in ro_fidelity.items():
        aprobs[q] = np.array([[f_ro, 1.-f_ro],
                              [1.-f_ro, f_ro]])

    # FIXME (Nik): decide on whether isa_name is set to something more useful
    return NoiseModel("DECOHERENCE_ISA", kraus_maps, aprobs)


def _noise_model_program_header(noise_model, name_translator):
    """
    Generate the header for a pyquil Program that uses ``noise_model`` to overload noisy gates.

    :param NoiseModel noise_model: The assumed noise model.
    :return: A quil Program with the noise pragmas.
    :rtype: pyquil.quil.Program
    """
    from pyquil.quil import Program
    p = Program()
    for k in noise_model.gates:
        name = name_translator(k.gate)
        p.define_noisy_gate(name, k.targets, k.kraus_ops)
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
    gates = _get_program_gates(prog)
    noisy_names = _get_noisy_names(gates)
    new_prog = _noise_model_program_header(noise_model, lambda g: noisy_names.get(g, g))
    for i in prog:
        if isinstance(i, Gate):
            new_prog += Gate(noisy_names[i], i.params, i.qubits)
        else:
            new_prog += i
    return new_prog


def add_noise_to_program(prog, T1=30e-6, T2=None, gate_time_1q=50e-9, gate_time_2q=150e-09,
                         ro_fidelity=0.95):
    """
    Add generic damping and dephasing noise to a program.

    .. warning::

        This function is deprecated. Please use :py:func:`add_decoherence_noise` instead.

    :param prog: A pyquil program consisting of I, RZ, CZ, and RX(+-pi/2) instructions
    :param Union[Dict[int,float],float] T1: The T1 amplitude damping time either globally or in a
        dictionary indexed by qubit id. By default, this is 30 us.
    :param Optional[Union[Dict[int,float],float]] T2: The T2 dephasing time either globally or in a
        dictionary indexed by qubit id. By default, this is one-half of the T1 time.
    :param float gate_time_1q: The duration of the one-qubit gates, namely RX(+pi/2) and RX(-pi/2).
        By default, this is 50 ns.
    :param float gate_time_2q: The duration of the two-qubit gates, namely CZ.
        By default, this is 150 ns.
    :param Union[Dict[int,float],float] ro_fidelity: The readout assignment fidelity
        :math:`F = (p(0|0) + p(1|1))/2` either globally or in a dictionary indexed by qubit id.
    :return: A new program with noisy operators.
    """
    warnings.warn("pyquil.noise.add_noise_to_program is deprecated, please use "
                  "pyquil.noise.add_decoherence_noise instead.",
                  DeprecationWarning)
    return add_decoherence_noise(prog, T1=T1, T2=T2, gate_time_1q=gate_time_1q,
                                 gate_time_2q=gate_time_2q, ro_fidelity=ro_fidelity)


def add_decoherence_noise(prog, T1=30e-6, T2=None, gate_time_1q=50e-9, gate_time_2q=150e-09,
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
    - T2 = T1 / 2
    - 1q gate time = 50 ns
    - 2q gate time = 150 ns

    are currently typical for near-term devices.

    This function will define new gates and add Kraus noise to these gates. It will translate
    the input program to use the noisy version of the gates.

    :param prog: A pyquil program consisting of I, RZ, CZ, and RX(+-pi/2) instructions
    :param Union[Dict[int,float],float] T1: The T1 amplitude damping time either globally or in a
        dictionary indexed by qubit id. By default, this is 30 us.
    :param Optional[Union[Dict[int,float],float]] T2: The T2 dephasing time either globally or in a
        dictionary indexed by qubit id. By default, this is one-half of the T1 time.
    :param float gate_time_1q: The duration of the one-qubit gates, namely RX(+pi/2) and RX(-pi/2).
        By default, this is 50 ns.
    :param float gate_time_2q: The duration of the two-qubit gates, namely CZ.
        By default, this is 150 ns.
    :param Union[Dict[int,float],float] ro_fidelity: The readout assignment fidelity
        :math:`F = (p(0|0) + p(1|1))/2` either globally or in a dictionary indexed by qubit id.
    :return: A new program with noisy operators.
    """
    gates = _get_program_gates(prog)
    noise_model = decoherence_noise_model(
        gates,
        T1=T1,
        T2=T2,
        gate_time_1q=gate_time_1q,
        gate_time_2q=gate_time_2q,
        ro_fidelity=ro_fidelity
    )
    return apply_noise_model(prog, noise_model)

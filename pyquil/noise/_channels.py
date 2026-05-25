##############################################################################
# Copyright 2016-2026 Rigetti Computing
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
Noise channel classes and gate-resolution utilities.

This module defines ``Channel``, ``MeasurementChannel``, ``ResetChannel``, and
``CycleChannel`` dataclasses for representing noise in quantum circuits, along
with helper functions for resolving gate unitaries and extracting custom gate
definitions from Quil programs.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, replace
from functools import cached_property, reduce
from itertools import product
from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp
import numpy as np
import quax as qx
from jax import Array
from plotly.graph_objs import Figure
from quil.program import Program as RSProgram
from scipy.linalg import logm as scipy_logm

from pyquil.quilatom import Expression, FormalArgument, Parameter, substitute
from pyquil.quilbase import DefCircuit, DefGate, Gate, Measurement, Reset
from quil.expression import Expression as QuilExpression

if TYPE_CHECKING:
    from pyquil import Program

logger = logging.getLogger(__name__)

# Type alias for the custom-gate lookup map used throughout the Channel constructors.
CustomGateMap = dict[str, qx.Unitary | Callable[..., qx.Unitary]]


def _parse_quil_instruction(quil_str: str) -> Gate | Measurement | Reset:
    """Parse a single Quil instruction string into a pyquil instruction object.

    Uses the ``quil`` Rust parser directly, avoiding a dependency on ``pyquil.Program``.
    """
    rs_inst = RSProgram.parse(quil_str).body_instructions[0]
    if rs_inst.is_gate():
        return Gate._from_rs_gate(rs_inst.to_gate())
    elif rs_inst.is_measurement():
        return Measurement._from_rs_measurement(rs_inst.to_measurement())
    elif rs_inst.is_reset():
        return Reset._from_rs_reset(rs_inst.to_reset())
    raise ValueError(f"Unsupported instruction type in: {quil_str}")


def _resolve_params(params: list) -> list[float]:
    """
    Resolve gate parameters to concrete float values.

    :param params: The gate parameters (may include symbolic Parameters or Expressions).
    :return: A list of concrete float values.
    :raises ValueError: If any parameter is symbolic and cannot be evaluated to a number.
    """
    fixed_params = []
    for p in params:
        if isinstance(p, (Parameter, Expression)):
            evaluated = p._evaluate()
            if isinstance(evaluated, (Parameter, Expression)):
                raise ValueError(
                    f"Cannot resolve symbolic parameter {p}. Provide a gate with concrete numeric parameters."
                )
            fixed_params.append(float(evaluated))
        elif isinstance(p, QuilExpression):
            result = p.evaluate({}, {})
            fixed_params.append(float(result.to_number()) if hasattr(result, "to_number") else float(result))  # type: ignore[arg-type]
        else:
            fixed_params.append(float(p.real))
    return fixed_params


def get_custom_gates_from_program(program: Program) -> CustomGateMap:
    """
    Extract custom gate definitions from a Quil program.

    Returns a dictionary mapping gate names to unitary matrices (for fixed gates) or callables
    (for parametric gates). Does not include the standard gate set — use this to augment
    the standard ``qx.gates.QUANTUM_GATES`` when resolving instructions with custom gates.

    :param program: A Quil program containing DefGate definitions.
    :return: A dictionary of custom gate names to unitary matrices or callables.
    """
    custom_gates: CustomGateMap = {}
    for defgate in program.defined_gates:
        if defgate.parameters:

            def parametric_gate(*args: float, defgate: DefGate = defgate) -> qx.Unitary:
                parameter_map = {Parameter(p.name): arg for p, arg in zip(defgate.parameters, args)}
                matrix = jnp.asarray(
                    [[substitute(element, parameter_map) for element in row] for row in defgate.matrix],  # type: ignore[arg-type]
                    dtype=complex,
                )
                num_qubits = int(jnp.round(jnp.log2(matrix.shape[0])))
                return qx.Unitary.from_matrix(matrix, ((2,) * num_qubits, (2,) * num_qubits))

            custom_gates[defgate.name] = parametric_gate
        else:
            matrix = jnp.asarray(defgate.matrix, dtype=complex)
            num_qubits = int(jnp.round(jnp.log2(matrix.shape[0])))
            custom_gates[defgate.name] = qx.Unitary.from_matrix(matrix, ((2,) * num_qubits, (2,) * num_qubits))
    return custom_gates


def get_instruction_unitary(
    inst: Gate,
    custom_gates: CustomGateMap | None = None,
) -> qx.Unitary:
    """
    Get the unitary matrix associated with a gate instruction.

    Looks up the gate by name — first in ``custom_gates`` (if provided), then in the
    standard quax gate table ``qx.gates.QUANTUM_GATES``. Parametric gates are supported
    provided all parameters are concrete numeric values.

    :param inst: The gate instruction.
    :param custom_gates: Optional dictionary of additional gate definitions (e.g. from
        :func:`get_custom_gates_from_program`). Takes precedence over the standard gate set.
    :return: The unitary matrix.
    :raises ValueError: If any gate parameter is symbolic.
    :raises KeyError: If the gate name is not found in either the custom or standard gate set.
    """
    name = inst.name

    # Look up gate definition: custom gates take precedence
    if custom_gates is not None and name in custom_gates:
        gate_def = custom_gates[name]
    elif name in qx.gates.QUANTUM_GATES:
        gate_def = qx.gates.QUANTUM_GATES[name]
    else:
        raise KeyError(f"Unknown gate '{name}'. Provide it via custom_gates (e.g. custom_gates={{'{name}': matrix}}).")

    if inst.params:
        fixed_params = _resolve_params(list(inst.params))
        if not callable(gate_def):
            raise ValueError(f"Gate '{name}' is not parametric but parameters were provided.")
        result = gate_def(*fixed_params)
    else:
        if callable(gate_def):
            result = gate_def()
        else:
            result = gate_def

    # quax parametric gates may return Operator instead of Unitary; wrap if needed
    if not isinstance(result, qx.Unitary):
        result = qx.Unitary.from_matrix(result.matrix, result.dims)  # type: ignore[union-attr]
    return result


@dataclass(frozen=True)
class Channel:
    """
    A noise channel attaches a superoperator to a specific gate.

    The superoperator *includes* the gate unitary, so the channel replaces the gate
    rather than being applied after it.

    The ``process`` field is a ``qx.SuperOp`` which can be converted to alternative
    representations (Choi, Kraus, Pauli-Liouville) via ``quax``.

    Fidelity metrics are computed relative to the ideal gate unitary stored in
    ``target_unitary``. For standard gates use the class methods (e.g.
    :meth:`from_gate_fidelity`) which resolve the unitary automatically.
    """

    inst: Gate
    """Quil gate to which the channel applies."""

    process: qx.SuperOp
    """The noisy process (superoperator) for the gate, including the gate unitary."""

    target_unitary: qx.Unitary
    """The noiseless unitary of the gate."""

    @cached_property
    def unitary(self) -> qx.Unitary:
        """The noiseless unitary of the gate."""
        return self.target_unitary

    @cached_property
    def qubits(self) -> list[int]:
        """The qubits which the channel applies to."""
        return self.inst.get_qubit_indices()

    @cached_property
    def num_qubits(self) -> int:
        """The number of qubits the channel acts on."""
        return len(self.qubits)

    # ──────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────

    @classmethod
    def from_gate_fidelity(
        cls: type[Channel],
        inst: Gate,
        fidelity: float,
        custom_gates: CustomGateMap | None = None,
    ) -> "Channel":
        """
        Create a depolarizing noise channel from an average gate fidelity.

        The resulting channel is the composition of the ideal gate unitary with a
        depolarizing channel calibrated to the specified fidelity:
        :math:`\\mathcal{E} = \\mathcal{D}_p \\circ \\mathcal{U}`

        :param inst: The gate to which the channel applies.
        :param fidelity: The average gate fidelity, :math:`F_{\\mathrm{avg}} \\in [0, 1]`.
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        p = qx.average_fidelity_to_depolarizing_constant(fidelity, unitary.dims[0])
        return cls.from_depolarizing_constant(inst, p, custom_gates)

    @classmethod
    def from_pauli_fidelity(
        cls: type[Channel],
        inst: Gate,
        pauli_fidelity: float,
        custom_gates: CustomGateMap | None = None,
    ) -> "Channel":
        """
        Create a depolarizing noise channel from a process (Pauli) fidelity.

        The process fidelity :math:`F_e` is related to the average gate fidelity by
        :math:`F_{\\mathrm{avg}} = (d \\cdot F_e + 1) / (d + 1)`.

        :param inst: The gate to which the channel applies.
        :param pauli_fidelity: The process fidelity (entanglement fidelity), :math:`F_e \\in [0, 1]`.
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        p = qx.process_fidelity_to_depolarizing_constant(pauli_fidelity, unitary.dims[0])
        return cls.from_depolarizing_constant(inst, p, custom_gates)

    @classmethod
    def from_depolarizing_constant(
        cls: type[Channel],
        inst: Gate,
        depolarizing_constant: float,
        custom_gates: CustomGateMap | None = None,
    ) -> "Channel":
        """
        Create a depolarizing noise channel from a depolarization constant.

        The depolarizing constant :math:`p` parameterizes the channel as
        :math:`\\mathcal{D}_p(\\rho) = p \\, \\rho + (1-p) \\, I/d`.

        :param inst: The gate to which the channel applies.
        :param depolarizing_constant: The depolarization constant, e.g. 0.98 for 2% depolarization.
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        depolarizing_superop = qx.depolarizing_channel_superoperator(1 - depolarizing_constant, unitary.dims[0])
        combined_superop = depolarizing_superop @ unitary
        return cls(inst=inst, process=qx.to_superop(combined_superop), target_unitary=unitary)

    @classmethod
    def from_pauli_noise(
        cls: type[Channel],
        inst: Gate,
        pauli_noise: dict[str, float],
        custom_gates: CustomGateMap | None = None,
    ) -> "Channel":
        """
        Create a stochastic Pauli noise channel from Pauli error rates.

        The noise is specified as a dictionary mapping Pauli strings to error probabilities,
        e.g. ``{"XX": 0.03, "ZI": 0.001}``. The probabilities must sum to at most 1.0;
        any remainder is assigned to the identity (no-error) term.

        :param inst: The gate to which the channel applies.
        :param pauli_noise: Pauli error rates, e.g. ``{"IX": 0.01, "ZZ": 0.02}``.
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        num_qubits = len(unitary.dims[0])

        for pauli in pauli_noise:
            if len(pauli) != num_qubits:
                raise ValueError(f"Pauli term '{pauli}' has length {len(pauli)}, expected {num_qubits}.")

        all_pauli_terms = tuple("".join(term) for term in product("IXYZ", repeat=num_qubits))

        pauli_error_rates = []
        for term in reversed(all_pauli_terms):
            if term in pauli_noise:
                error_rate = pauli_noise[term]
            elif all(p == "I" for p in term):
                error_rate = 1 - sum(pauli_error_rates)
            else:
                error_rate = 0
            pauli_error_rates.append(error_rate)
        assert jnp.isclose(1.0, sum(pauli_error_rates))
        pauli_error_rates = list(reversed(pauli_error_rates))

        # Build Pauli Kraus operators using quax ensembles
        single_paulis = qx.ensembles.PAULIS  # ensemble of (I, X, Y, Z)
        if num_qubits == 1:
            pauli_ops = single_paulis
        else:
            pauli_ops = reduce(lambda a, b: a | b, [single_paulis for _ in range(num_qubits)])

        # Scale each Pauli by sqrt(probability) to form Kraus operators
        coeffs = jnp.sqrt(jnp.array(pauli_error_rates, dtype=float))
        kraus_matrices = coeffs[:, None, None] * pauli_ops.matrix
        kraus_map = qx.KrausMap.from_matrix(kraus_matrices, unitary.dims)

        process_superop = qx.to_superop(kraus_map @ unitary)
        return cls(inst=inst, process=process_superop, target_unitary=unitary)

    @classmethod
    def from_random_coherent_error(
        cls: type[Channel],
        inst: Gate,
        process_fidelity: float,
        rng: np.random.Generator | None = None,
        custom_gates: CustomGateMap | None = None,
    ) -> "Channel":
        """
        Create a channel with a random coherent (unitary) error at the specified process fidelity.

        A random unitary close to identity is generated with the given process fidelity,
        then composed with the ideal gate.

        :param inst: The gate to which the channel applies.
        :param process_fidelity: The process fidelity of the coherent error, :math:`F_e \\in [0, 1]`.
        :param rng: NumPy random number generator for reproducibility.
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        if rng is None:
            rng = np.random.default_rng()

        ideal = get_instruction_unitary(inst, custom_gates)
        num_qubits = len(ideal.dims[0])
        d = 2**num_qubits

        # Generate a random unitary error with the specified process fidelity
        # using Pauli generator decomposition
        angle = jnp.arccos(2 * process_fidelity - 1) / (2 * jnp.pi)
        id_coeff = 1 - float(angle)
        coeffs = rng.random(4**num_qubits - 1)
        coeffs = (1 - id_coeff) / np.sqrt(np.sum(np.square(coeffs))) * coeffs

        # Build Pauli generator sum using quax Pauli matrices
        pauli_matrices = qx.ensembles.PAULIS.matrix  # shape (4, 2, 2)
        pauli_sum = jnp.eye(d, dtype=complex) * id_coeff
        pauli_products = list(itertools.product(pauli_matrices, repeat=num_qubits))[1:]
        for paulis, coefficient in zip(pauli_products, coeffs):
            pauli_sum = pauli_sum + reduce(jnp.kron, paulis) * coefficient

        from jax.scipy.linalg import expm as jax_expm

        error_unitary = jax_expm(-1j * jnp.pi * pauli_sum)
        # Fix global phase
        phase = jnp.exp(-1j * jnp.angle(error_unitary[0, 0]))
        error_unitary = error_unitary * phase

        error_u = qx.Unitary.from_matrix(error_unitary, ideal.dims)
        noisy_superop = qx.to_superop(error_u @ ideal)
        return cls(inst=inst, process=noisy_superop, target_unitary=ideal)

    @classmethod
    def from_mixture(
        cls: type[Channel],
        inst: Gate,
        constituents: list[qx.Unitary],
        probabilities: list[float],
        custom_gates: CustomGateMap | None = None,
    ) -> "Channel":
        """
        Create a mixture channel from a set of unitary errors with given probabilities.

        The channel is :math:`\\mathcal{E}(\\rho) = (1-\\sum p_i) U\\rho U^\\dagger + \\sum p_i V_i U \\rho U^\\dagger V_i^\\dagger`
        where :math:`U` is the ideal gate and :math:`V_i` are the error unitaries.

        :param inst: The gate to which the channel applies.
        :param constituents: Unitary error operators to mix.
        :param probabilities: Probability of each unitary error. Must sum to at most 1.0.
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        ideal = get_instruction_unitary(inst, custom_gates)

        if len(constituents) != len(probabilities):
            raise ValueError("The number of constituents and probabilities must match.")
        error_prob = sum(probabilities)
        if error_prob > 1.0:
            raise ValueError(f"The sum of probabilities ({error_prob}) must be at most 1.0.")

        # Build the mixture superop: (1-p_total) S(U) + sum p_i S(V_i @ U)
        p0 = 1.0 - error_prob
        noisy_superop_matrix = p0 * qx.to_superop(ideal).matrix
        for p, v in zip(probabilities, constituents):
            composed = v @ ideal
            noisy_superop_matrix = noisy_superop_matrix + p * qx.to_superop(composed).matrix
        noisy_superop = qx.SuperOp.from_matrix(noisy_superop_matrix, ideal.dims)
        return cls(inst=inst, process=noisy_superop, target_unitary=ideal)

    @classmethod
    def from_coherence_times(
        cls: type[Channel],
        inst: Gate,
        gate_duration: float,
        t1s: list[float],
        t2s: list[float] | None = None,
        custom_gates: CustomGateMap | None = None,
    ) -> "Channel":
        """
        Create a decoherence Channel based on the coherence times.

        In this construction, decoherence is applied _after_ the ideal gate unitary.

        :param inst: The target instruction.
        :param gate_duration: The duration of the gate.
        :param t1s: The t1 time(s) of the qubits
        :param t2s: The t2 time(s) of the qubits. Default to 2*t1.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        qubits = inst.get_qubit_indices()
        num_sys = len(qubits)
        assert num_sys == len(t1s)
        if t2s is None:
            t2s = [2 * t1 for t1 in t1s]
        else:
            assert num_sys == len(t2s)

        t1_array = jnp.asarray(t1s)
        tphi_array = 1 / (1 / jnp.asarray(t2s) - 1 / t1_array)

        choi = qx.thermal_relaxation_choi(t1s=t1_array, tphis=tphi_array, duration=gate_duration)
        process = qx.to_superop(choi @ unitary)
        return cls(
            inst=inst,
            process=process,
            target_unitary=unitary,
        )

    @classmethod
    def from_superoperator(
        cls: type[Channel],
        inst: Gate,
        process: qx.SuperOp,
        target_unitary: qx.Unitary | None = None,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        """
        Create a Channel from a pre-built superoperator.

        If ``target_unitary`` is not provided it is inferred from the gate
        instruction using the standard gate set (and ``custom_gates`` if given).

        :param inst: The gate to which the channel applies.
        :param process: The noisy process superoperator (includes the gate unitary).
        :param target_unitary: The ideal gate unitary.  Resolved automatically
            when omitted.
        :param custom_gates: Optional dictionary of custom gate definitions,
            used only when ``target_unitary`` is ``None``.
        :return: A Channel instance.
        """
        if target_unitary is None:
            target_unitary = get_instruction_unitary(inst, custom_gates)
        return cls(inst=inst, process=process, target_unitary=target_unitary)

    # ──────────────────────────────────────────────
    # Cached representation conversions
    # ──────────────────────────────────────────────

    @cached_property
    def noise_process(self) -> qx.SuperOp:
        """
        The noise-only channel with the ideal gate unitary factored out.

        If the full channel is :math:`\\mathcal{E} = \\Lambda \\circ \\mathcal{U}`, this
        returns :math:`\\Lambda`.
        """
        return qx.to_superop(self.process @ self.unitary.h)

    # ──────────────────────────────────────────────
    # Fidelity properties
    # ──────────────────────────────────────────────

    @cached_property
    def fidelity(self) -> float:
        """Average gate fidelity :math:`F_{\\mathrm{avg}}` of the channel relative to the ideal gate."""
        return float(qx.process_fidelity_to_average_fidelity(self.pauli_fidelity, dims=self.unitary.dims[0]))

    @cached_property
    def infidelity(self) -> float:
        """Average gate infidelity :math:`1 - F_{\\mathrm{avg}}`."""
        return 1.0 - self.fidelity

    @cached_property
    def pauli_fidelity(self) -> float:
        """Process fidelity (entanglement fidelity) :math:`F_e` relative to the ideal gate."""
        process, unitary = qx.promote_hilbert_space(self.process, qx.to_superop(self.unitary))
        return float(qx.process_fidelity(process, unitary))

    @cached_property
    def pauli_infidelity(self) -> float:
        """Process infidelity :math:`1 - F_e`."""
        return 1.0 - self.pauli_fidelity

    @cached_property
    def stochastic_infidelity(self) -> float:
        """Stochastic (incoherent) component of the process infidelity."""
        return float(qx.stochastic_infidelity(self.noise_process))

    @cached_property
    def stochastic_fidelity(self) -> float:
        """Stochastic fidelity :math:`1 - e_S`."""
        return 1.0 - self.stochastic_infidelity

    @cached_property
    def coherent_infidelity(self) -> float:
        """Coherent component of the process infidelity: :math:`e_C = e - e_S`."""
        return self.pauli_infidelity - self.stochastic_infidelity

    @cached_property
    def coherent_fidelity(self) -> float:
        """Coherent fidelity :math:`1 - e_C`."""
        return 1.0 - self.coherent_infidelity

    @cached_property
    def unitarity(self) -> float:
        """Unitarity of the channel."""
        return float(qx.unitarity(self.noise_process))

    # ──────────────────────────────────────────────
    # Channel analysis methods
    # ──────────────────────────────────────────────

    def pauli_twirl(self) -> "Channel":
        """
        Return a Pauli-twirled version of this channel.

        Pauli twirling projects the channel onto the Pauli diagonal, eliminating
        off-diagonal coherences in the Pauli-Liouville representation. The
        resulting channel is a stochastic Pauli channel with the same diagonal
        error rates.
        """
        ptm = qx.to_pauli_liouville(self.process)
        # Keep only the diagonal of the PTM
        twirled_ptm_matrix = jnp.diag(jnp.diag(ptm.matrix))
        twirled_superop = qx.to_superop(qx.PauliLiouville.from_matrix(twirled_ptm_matrix, self.process.dims))
        return replace(self, process=twirled_superop)

    @cached_property
    def _unitary_error_component(self) -> Array:
        """
        Extract the dominant unitary from the noise-only channel.

        Uses eigendecomposition + SVD polar decomposition to find the closest
        unitary to the noise channel.
        """
        choi_matrix = qx.to_choi(self.noise_process).matrix
        d = 2**self.num_qubits

        # Dominant eigenvector of the Choi matrix
        eigenvalues, eigenvectors = jnp.linalg.eigh(choi_matrix)
        dominant_eigenvector = eigenvectors[:, jnp.argmax(jnp.abs(eigenvalues))]

        # SVD polar decomposition to extract the closest unitary
        u, _, vh = jnp.linalg.svd(dominant_eigenvector.reshape(d, d).T)
        return u @ vh

    def to_coherent_channel(self) -> "Channel":
        """
        Isolate the coherent (unitary) component of the noise.

        Extracts the dominant unitary from the noise Choi matrix via polar
        decomposition and returns a channel consisting of that unitary error
        composed with the ideal gate.
        """
        u_error = self._unitary_error_component
        u_error_qx = qx.Unitary.from_matrix(u_error, self.process.dims)
        coherent_superop = qx.to_superop(u_error_qx @ self.unitary)
        return replace(self, process=coherent_superop)

    def to_stochastic_channel(self) -> "Channel":
        """
        Isolate the stochastic (incoherent) component of the noise.

        The full channel decomposes as
        :math:`\\mathcal{E} = \\mathcal{S} \\circ \\mathcal{U}_{\\mathrm{err}} \\circ \\mathcal{U}_{\\mathrm{gate}}`.
        This method factors out the coherent unitary error and returns
        :math:`\\mathcal{S} \\circ \\mathcal{U}_{\\mathrm{gate}}`.
        """
        u_error = self._unitary_error_component
        # Get the noise-only superoperator and compose with U_err†
        noise_superop = self.noise_process.matrix
        u_err_inv_superop = jnp.kron(u_error.conj(), u_error.conj().T)
        stochastic_noise_superop = noise_superop @ u_err_inv_superop
        # Recompose with the ideal gate
        ideal_superop = jnp.kron(self.unitary.matrix, self.unitary.matrix.conj())
        stochastic_superop = stochastic_noise_superop @ ideal_superop
        return replace(self, process=qx.SuperOp.from_matrix(stochastic_superop, self.process.dims))

    def is_pauli(self) -> bool:
        """
        Check if the noise channel is a Pauli (stochastic Pauli) channel.

        A Pauli channel has a diagonal Pauli transfer matrix (noise-only part).
        """
        ptm = qx.to_pauli_liouville(self.noise_process).matrix
        mask = ~jnp.eye(ptm.shape[0], dtype=bool)
        return bool(jnp.allclose(ptm[mask], 0))

    def to_pauli_vector(self) -> Array:
        """
        Convert the noise channel to a Pauli error probability vector.

        Returns the vector of probabilities for each Pauli error in lexicographic
        order (II, IX, IY, IZ, XI, XX, ...). The vector sums to 1.0.
        """
        noise_superop = self.noise_process.matrix
        num_qubits = self.num_qubits
        dim = noise_superop.shape[0]

        # Build all Pauli operators and their superoperators
        pauli_matrices = qx.ensembles.PAULIS.matrix  # (4, 2, 2): I, X, Y, Z
        all_pauli_products = list(product(pauli_matrices, repeat=num_qubits))
        pauli_error_rates = []
        for pauli_tuple in all_pauli_products:
            pauli_op = reduce(jnp.kron, pauli_tuple)
            pauli_superop = jnp.kron(pauli_op, pauli_op.conj())
            rate = float(jnp.abs(jnp.trace(noise_superop @ pauli_superop) / dim))
            pauli_error_rates.append(rate)

        return jnp.array(pauli_error_rates, dtype=float)

    @cached_property
    def pauli_vector(self) -> Array:
        """The Pauli error probability vector of the noise channel."""
        return self.to_pauli_vector()

    # ──────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────

    def plot(self, only_noise: bool = True, show_identity: bool = False) -> Figure:
        """
        Plot the Pauli transfer matrix of the channel.

        :param only_noise: If True, plot the noise-only channel (gate unitary factored out).
            If False, plot the full channel including the gate unitary.
        :param show_identity: If True, include the identity component in the noise-only plot.
            If False (default), visualize the generator of the noise channel via the matrix
            logarithm of the PTM.  For near-identity noise this approximates PTM - I, but
            correctly captures the Lie-algebraic structure of the channel.
            Only applies when ``only_noise=True``.
        :return: A Plotly Figure.
        """
        if only_noise:
            channel = self.noise_process
            if not show_identity:
                ptm = qx.to_pauli_liouville(channel)
                log_ptm = scipy_logm(np.asarray(ptm.matrix))
                channel = qx.PauliLiouville.from_matrix(jnp.array(log_ptm), channel.dims)
            title_prefix = "Noise Channel"
        else:
            channel = self.process
            title_prefix = "Full Channel"

        fig = qx.plot(channel)
        fig.update_layout(
            title=(
                f"{title_prefix} for {self.inst.out()}<br>"
                f"𝜀={self.pauli_infidelity * 100:.2f}%, "
                f"𝜀<sub>u</sub>={self.coherent_infidelity * 100:.2f}%, "
                f"𝜀<sub>s</sub>={self.stochastic_infidelity * 100:.2f}%"
            )
        )
        return fig

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """
        Serialize Channel to a JSON string.

        :return: JSON string representation.
        """
        superop_array = np.asarray(self.process.matrix)
        flat_data = [[float(val.real), float(val.imag)] for val in superop_array.flat]

        data = {
            "inst": self.inst.out(),
            "superop": {"_complex_array": flat_data, "shape": list(superop_array.shape)},
        }

        u_array = np.asarray(self.target_unitary.matrix)
        u_flat = [[float(val.real), float(val.imag)] for val in u_array.flat]
        data["target_unitary"] = {"_complex_array": u_flat, "shape": list(u_array.shape)}

        return json.dumps(data)

    @classmethod
    def from_json(cls: type[Channel], json_str: str) -> "Channel":
        """
        Deserialize a Channel from a JSON string.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: Channel instance.
        """
        data = json.loads(json_str)
        inst = _parse_quil_instruction(data["inst"])
        assert isinstance(inst, Gate)

        superop_data = data["superop"]
        flat = superop_data["_complex_array"]
        shape = tuple(superop_data["shape"])
        superop_array = jnp.array([complex(pair[0], pair[1]) for pair in flat], dtype=complex).reshape(shape)
        # Infer dims from matrix shape: (d^2, d^2) -> d qubits each of dim 2
        d = int(jnp.sqrt(shape[0]))
        num_qubits = int(jnp.round(jnp.log2(d)))
        dims = ((2,) * num_qubits, (2,) * num_qubits)
        superop = qx.SuperOp.from_matrix(superop_array, dims)

        if "target_unitary" in data:
            u_data = data["target_unitary"]
            u_flat = u_data["_complex_array"]
            u_shape = tuple(u_data["shape"])
            u_array = jnp.array([complex(pair[0], pair[1]) for pair in u_flat], dtype=complex).reshape(u_shape)
            u_num_qubits = int(jnp.round(jnp.log2(u_shape[0])))
            u_dims = ((2,) * u_num_qubits, (2,) * u_num_qubits)
            target_unitary = qx.Unitary.from_matrix(u_array, u_dims)
        else:
            target_unitary = get_instruction_unitary(inst)

        return cls(inst=inst, process=superop, target_unitary=target_unitary)

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        """Return a simplified string representation showing the gate and process fidelity."""
        return f"<{self.inst.out()} ~ ({100 * self.pauli_fidelity:.2f}%)>"

    def __eq__(self, other: object) -> bool:
        """Check equality based on instruction and process fidelity."""
        if not isinstance(other, Channel):
            return False
        if self.inst != other.inst:
            return False
        return bool(jnp.isclose(float(qx.process_fidelity(self.process, other.process)), 1.0, atol=1e-9))

    __hash__ = None

    def __matmul__(self, other: "Channel") -> "Channel":
        """
        Compose two channels: ``channel_B @ channel_A``.

        Both channels share the same gate instruction. The composition factors
        out one copy of the gate unitary so the result represents the sequential
        application of the two noisy processes:

        :math:`\\mathcal{E}_B \\circ \\mathcal{U}^\\dagger \\circ \\mathcal{E}_A`

        This is the natural composition: if ``channel_A`` already includes the
        gate, applying ``channel_B`` after it should not double-count the gate.
        """
        if not isinstance(other, Channel):
            return NotImplemented
        if self.inst != other.inst:
            raise ValueError(f"Cannot compose channels for different gates: {self.inst.out()} vs {other.inst.out()}")
        # E_B @ U† @ E_A  (factor out one gate unitary between the two channels)
        u_dag_superop = qx.to_superop(self.unitary.h)
        composed_superop = qx.to_superop(self.process @ u_dag_superop @ other.process)
        return replace(self, process=composed_superop)

    def __or__(self, other: "Channel | MeasurementChannel") -> "CycleChannel":
        """
        Tensor product of two channels on disjoint qubits, producing a CycleChannel.

        The result represents a cycle containing both operations acting in parallel
        on disjoint qubits. The DefCircuit encodes the parallel operations as
        formal instructions.

        :param other: Another Channel or MeasurementChannel on disjoint qubits.
        :return: A CycleChannel representing the tensor product.
        """
        if not isinstance(other, (Channel, MeasurementChannel)):
            return NotImplemented

        # Validate disjoint qubits
        self_qubits = set(self.qubits)
        other_qubits = set(other.qubits)
        if self_qubits & other_qubits:
            raise ValueError(f"Cannot tensor channels with overlapping qubits: {self_qubits & other_qubits}")

        return _build_cycle_channel([self, other])


@dataclass(frozen=True)
class MeasurementChannel:
    """
    A measurement noise channel attaches a quantum instrument to a specific measurement operation.

    The ``process`` field is a ``qx.QuantumInstrument`` which models both classification
    errors and post-measurement back-action.
    """

    inst: Measurement
    """The measurement operation to which the channel applies."""

    process: qx.QuantumInstrument
    """A quantum instrument representation of the noisy measurement."""

    @cached_property
    def qubits(self) -> list[int]:
        """The qubits which the measurement applies to."""
        qubit = self.inst.qubit
        return [qubit.index if hasattr(qubit, "index") else int(qubit)]  # type: ignore[union-attr,arg-type]

    # ──────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────

    @classmethod
    def from_readout_fidelity(
        cls: type[MeasurementChannel],
        inst: Measurement,
        fidelity: float,
        asymmetry: float = 0.0,
        dim: int = 2,
    ) -> "MeasurementChannel":
        """
        Create a readout quantum instrument with optional asymmetry.

        Produces a perfectly QND measurement with the given classification fidelity.
        Error is distributed only between adjacent levels: P(j+1|j) and P(j|j+1).
        Non-adjacent confusion is zero.

        :param inst: The measurement instruction.
        :param fidelity: The average readout fidelity.
        :param asymmetry: Value between -1 and +1. Zero is symmetric.
            Positive biases toward upward confusion P(j+1|j), negative toward downward P(j|j+1).
        :param dim: The dimension of the measured system (2 for qubits, 3 for qutrits, etc.).
        :return: A MeasurementChannel instance.
        """
        # Compute per-pair error factor so that the average diagonal equals fidelity.
        # Each adjacent pair (j, j+1) contributes error_factor*(1+a) + error_factor*(1-a)
        # = 2*error_factor to total off-diagonal sum. With (dim-1) pairs, the average
        # column error is 2*(dim-1)*error_factor/dim, which we set equal to (1-fidelity).
        error_factor = dim * (1 - fidelity) / (2 * (dim - 1))

        confusion = jnp.zeros((dim, dim))
        for j in range(dim - 1):
            confusion = confusion.at[j + 1, j].set(error_factor * (1 + asymmetry))
            confusion = confusion.at[j, j + 1].set(error_factor * (1 - asymmetry))
        # Set diagonal so each column sums to 1
        col_sums = confusion.sum(axis=0)
        confusion = confusion + jnp.diag(1 - col_sums)

        transition = jnp.eye(dim)
        instrument = qx.instrument_from_confusion_and_transition(
            confusion_matrix=confusion,
            transition_matrix=transition,
            dims=(dim,),
            measured_qudits=(0,),
        )
        return cls(inst=inst, process=instrument)

    @classmethod
    def from_confusion_and_transition(
        cls: type[MeasurementChannel],
        inst: Measurement,
        confusion_matrix: Array,
        transition_matrix: Array,
    ) -> "MeasurementChannel":
        """
        Create a MeasurementChannel from a confusion matrix and a transition matrix.

        Provides independent control over measurement classification accuracy
        and post-measurement quantum state evolution.

        **Matrix Conventions (column-stochastic):**

        - ``confusion_matrix[i, j]``: P(outcome i | prepared j)
        - ``transition_matrix[k, j]``: P(ending in k | input j)
        - Columns sum to 1.0

        :param inst: The measurement instruction.
        :param confusion_matrix: A (d, d) classification matrix.
        :param transition_matrix: A (d, d) post-measurement transition matrix.
        :return: A MeasurementChannel instance.
        """
        confusion = jnp.asarray(confusion_matrix)
        dim = confusion.shape[0]
        instrument = qx.instrument_from_confusion_and_transition(
            confusion_matrix=confusion,
            transition_matrix=jnp.asarray(transition_matrix),
            dims=(dim,),
            measured_qudits=(0,),
        )
        return cls(inst=inst, process=instrument)

    @classmethod
    def from_axis(
        cls: type[MeasurementChannel],
        inst: Measurement,
        theta: float = 0.0,
        phi: float = 0.0,
        sharpness: float = 1.0,
    ) -> "MeasurementChannel":
        """
        Create a MeasurementChannel from a Bloch sphere measurement axis.

        The angles refer to the standard Bloch sphere notation.
        Theta=0, phi=0 is the Z axis (computational basis measurement).

        :param inst: The measurement instruction.
        :param theta: The colatitude with respect to the z-axis.
        :param phi: The longitude with respect to the x-axis.
        :param sharpness: The sharpness of the measurement. 1.0 is projective,
            0.0 is no measurement. 0 < s < 1 is a weak measurement.
        :return: A MeasurementChannel instance.
        """
        instrument = qx.instrument_from_axis(
            theta=theta,
            phi=phi,
            sharpness=sharpness,
        )
        return cls(inst=inst, process=instrument)

    @classmethod
    def from_binary_discriminator(
        cls: type[MeasurementChannel],
        inst: Measurement,
        dim: int,
        threshold: int,
        fidelity: float = 1.0,
    ) -> "MeasurementChannel":
        """
        Create a MeasurementChannel for a binary discriminator.

        Models a measurement that confuses each state at or above ``threshold`` with
        the state one level below it. This is useful for measurements calibrated as
        binary discriminators between groups of energy levels.

        For example, ``threshold=2, dim=3`` always confuses state 2 for state 1
        (discriminates ``{0, 1}`` vs ``{2}``). ``threshold=1, dim=3`` confuses
        state 1 for state 0 and state 2 for state 1 (discriminates ``{0}`` vs ``{1, 2}``).

        An optional ``fidelity`` parameter degrades the ideal discriminator with
        uniform classification noise.

        :param inst: The measurement instruction.
        :param dim: The dimension of the measured system.
        :param threshold: States at or above this level are confused with the level below.
            Must satisfy ``1 <= threshold < dim``.
        :param fidelity: Additional classification fidelity applied on top of the
            discrimination (1.0 = perfect discriminator).
        :return: A MeasurementChannel instance.
        """
        if not (1 <= threshold < dim):
            raise ValueError(f"threshold must satisfy 1 <= threshold < dim, got threshold={threshold}, dim={dim}")

        # Build the ideal binary discriminator confusion matrix:
        # states below threshold are classified correctly,
        # states at or above threshold are classified as the state one below.
        confusion = jnp.zeros((dim, dim))
        for j in range(dim):
            if j < threshold:
                confusion = confusion.at[j, j].set(1.0)
            else:
                confusion = confusion.at[j - 1, j].set(1.0)

        # Optionally degrade with uniform noise
        if fidelity < 1.0:
            confusion = fidelity * confusion + (1 - fidelity) * jnp.ones((dim, dim)) / dim

        transition = jnp.eye(dim)
        instrument = qx.instrument_from_confusion_and_transition(
            confusion_matrix=confusion,
            transition_matrix=transition,
            dims=(dim,),
            measured_qudits=(0,),
        )
        return cls(inst=inst, process=instrument)

    # ──────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────

    @cached_property
    def confusion_matrix(self) -> Array:
        """The confusion matrix of the measurement.

        Shape ``(num_outcomes, d_measured)``.
        Entry ``[i, j]`` is P(outcome i | prepared j).
        """
        return self.process.confusion_matrix

    @cached_property
    def transition_matrix(self) -> Array:
        """The post-measurement transition matrix.

        Shape ``(d, d)``. Entry ``[k, j]`` is P(ending in k | input j),
        marginalized over all measurement outcomes.
        """
        return self.process.transition_matrix

    @cached_property
    def non_demolition_fidelity(self) -> float:
        """Quantum non-demolition (QND) fidelity.

        Measures how well the measurement preserves computational basis states,
        averaged over outcomes and input states.
        """
        return float(qx.non_demolition_fidelity(self.process))

    @cached_property
    def instrument_fidelity(self) -> float:
        """Overall instrument fidelity w.r.t. ideal QND measurement.

        Accounts for both classification errors and post-measurement state disturbance.
        """
        return float(qx.instrument_fidelity(self.process))

    @cached_property
    def classification_fidelity(self) -> float:
        """Classification fidelity: average probability of correctly identifying the measurement outcome."""
        return float(qx.classification_fidelity(self.process))

    # ──────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────

    def plot(self) -> Figure:
        """
        Plot the quantum instrument using the quax visualization.

        Shows per-outcome superoperator matrices and the total CPTP channel.

        :return: A Plotly Figure.
        """
        fig = qx.plot(self.process)
        fig.update_layout(
            title=(
                f"Quantum Instrument MEASURE {self.qubits[0]}<br>"
                f"<sub>Cls: {100 * self.classification_fidelity:.2f}%, "
                f"QND: {100 * self.non_demolition_fidelity:.2f}%, "
                f"Instrument: {100 * self.instrument_fidelity:.2f}%</sub>"
            )
        )
        return fig

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """
        Serialize MeasurementChannel to a JSON string.

        :return: JSON string representation.
        """
        # Store per-outcome Choi matrices
        instrument_data = []
        for i in range(self.process.num_outcomes):
            choi_i, _ = self.process.outcome_choi(i)
            choi_array = np.asarray(choi_i.matrix)
            flat = [[float(val.real), float(val.imag)] for val in choi_array.flat]
            instrument_data.append({"_complex_array": flat, "shape": list(choi_array.shape)})

        data = {
            "inst": self.inst.out(),
            "instruments": instrument_data,
            "measured_qudits": list(self.process.measured_qudits),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls: type[MeasurementChannel], json_str: str) -> "MeasurementChannel":
        """
        Deserialize a MeasurementChannel from a JSON string.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: MeasurementChannel instance.
        """
        data = json.loads(json_str)
        inst = _parse_quil_instruction(data["inst"])
        assert isinstance(inst, Measurement)
        measured_qudits = tuple(data["measured_qudits"])

        choi_list = []
        for inst_data in data["instruments"]:
            flat = inst_data["_complex_array"]
            shape = tuple(inst_data["shape"])
            arr = jnp.array([complex(pair[0], pair[1]) for pair in flat], dtype=complex).reshape(shape)
            d = int(jnp.sqrt(shape[0]))
            n_qubits = int(jnp.round(jnp.log2(d)))
            choi_dims = ((2,) * n_qubits, (2,) * n_qubits)
            choi_list.append(qx.Choi.from_matrix(arr, choi_dims))

        instrument = qx.QuantumInstrument.from_choi(choi_list, measured_qudits)
        return cls(inst=inst, process=instrument)

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        """Return a simplified string representation."""
        return f"<MEASURE({self.classification_fidelity:.2f}) {self.qubits[0]} ~ QND({100 * self.non_demolition_fidelity:.2f}%)>"

    def __eq__(self, other: object) -> bool:
        """Check equality based on instruction and operator."""
        if not isinstance(other, MeasurementChannel):
            return False
        if self.inst != other.inst:
            return False
        return bool(jnp.allclose(self.process.matrix, other.process.matrix, atol=1e-9))

    __hash__ = None

    def __matmul__(self, other: "MeasurementChannel") -> "MeasurementChannel":
        """
        Compose two measurement channels on the same qubit.

        Models sequential application: ``channel_B @ channel_A`` means
        apply ``channel_A`` first, then ``channel_B``.
        """
        if not isinstance(other, MeasurementChannel):
            return NotImplemented
        if self.inst != other.inst:
            raise ValueError(
                f"Cannot compose measurement channels for different qubits: {self.inst.out()} vs {other.inst.out()}"
            )
        composed = self.process @ other.process
        return replace(self, process=composed)

    def __or__(self, other: "Channel | MeasurementChannel") -> "CycleChannel":
        """
        Tensor product of two channels on disjoint qubits, producing a CycleChannel.

        :param other: Another Channel or MeasurementChannel on disjoint qubits.
        :return: A CycleChannel representing the tensor product.
        """
        if not isinstance(other, (Channel, MeasurementChannel)):
            return NotImplemented

        self_qubits = set(self.qubits)
        other_qubits = set(other.qubits)
        if self_qubits & other_qubits:
            raise ValueError(f"Cannot tensor channels with overlapping qubits: {self_qubits & other_qubits}")

        return _build_cycle_channel([self, other])


@dataclass(frozen=True)
class ResetChannel:
    """
    A reset noise channel attaches a superoperator to a specific reset operation.

    The ``process`` field is a ``qx.SuperOp`` which *includes* the ideal reset, so the channel
    replaces the reset instruction rather than being applied after it.
    """

    inst: Reset
    """The reset operation to which the channel applies."""

    process: qx.SuperOp
    """A superoperator representation of the noisy reset (including ideal reset)."""

    # ──────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────

    @classmethod
    def from_reset_fidelity(
        cls: type[ResetChannel],
        inst: Reset,
        fidelity: float,
        dim: int = 2,
    ) -> "ResetChannel":
        """
        Create a ResetChannel with depolarizing noise scaled to the given process fidelity.

        The ideal reset channel maps every state to :math:`|0\\rangle\\langle 0|`.  Noise is
        modelled as a depolarising channel applied after the ideal reset.

        :param inst: The reset instruction.
        :param fidelity: Process fidelity of the reset channel, :math:`F \\in [0, 1]`.
            1.0 yields an ideal reset; values below 1 introduce depolarising noise.
        :param dim: Hilbert-space dimension (2 for qubits).
        :return: A ResetChannel instance.
        """
        ideal_superop = qx.gates.RESET(dim=dim)
        p = 1.0 - fidelity
        d2 = dim * dim
        # Depolarising channel in superop form: (1-p)*S_ideal + p*(I/d) for all inputs
        # The completely depolarising superop maps everything to I/d:
        # its rows are all zero except the diagonal entries corresponding to
        # the trace extraction (maps vec(ρ) → vec(I/d) = vec(I)/d).
        depol_superop_matrix = jnp.zeros((d2, d2), dtype=complex)
        # vec(I/d) has value 1/d at positions 0, d+1, 2(d+1), ... i.e. diagonal entries
        vec_identity_over_d = jnp.zeros(d2, dtype=complex)
        for i in range(dim):
            vec_identity_over_d = vec_identity_over_d.at[i * dim + i].set(1.0 / dim)
        # The trace functional extracts sum of diagonal: positions 0, d+1, ...
        trace_row = jnp.zeros(d2, dtype=complex)
        for i in range(dim):
            trace_row = trace_row.at[i * dim + i].set(1.0)
        # Depolarising superop: each row of output is vec(I/d) * Tr(ρ)
        depol_superop_matrix = jnp.outer(vec_identity_over_d, trace_row)
        noisy_superop_matrix = (1.0 - p) * ideal_superop.matrix + p * depol_superop_matrix
        noisy_superop = qx.SuperOp.from_matrix(noisy_superop_matrix, ideal_superop.dims)
        return cls(inst=inst, process=noisy_superop)

    # ──────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────

    @cached_property
    def qubits(self) -> list[int]:
        """The qubit(s) that the reset applies to."""
        qubit = self.inst.qubit
        if qubit is None:
            return []
        return [qubit.index if hasattr(qubit, "index") else int(qubit)]  # type: ignore[union-attr,arg-type]

    @cached_property
    def fidelity(self) -> float:
        """Process fidelity of the reset channel relative to the ideal reset.

        Defined as :math:`F = \\mathrm{Tr}[\\Lambda_{\\mathrm{ideal}}^\\dagger \\Lambda] / d^2`
        where :math:`\\Lambda` is the Choi matrix of the noisy channel and
        :math:`\\Lambda_{\\mathrm{ideal}}` is the ideal-reset Choi.
        """
        dim = self.process.dims[0][0]
        ideal_choi = qx.to_choi(qx.gates.RESET(dim=dim))
        noisy_choi = qx.to_choi(self.process)
        # Process fidelity = Tr[ideal_choi† @ noisy_choi] / d^2
        d2 = float(dim * dim)
        return float(jnp.real(jnp.trace(ideal_choi.matrix.conj().T @ noisy_choi.matrix)) / d2)

    @cached_property
    def noise_process(self) -> qx.SuperOp:
        """The noise-only channel (ideal reset factored out).

        For a reset channel the noise framing is less natural than for unitary gates;
        this property returns the full process superoperator.
        """
        return self.process

    # ──────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────

    def plot(self) -> Figure:
        """
        Plot the Pauli transfer matrix of the reset channel.

        :return: A Plotly Figure.
        """
        fig = qx.plot(self.process)
        qubit_str = str(self.qubits[0]) if self.qubits else "?"
        fig.update_layout(title=(f"Reset Channel RESET {qubit_str}<br><sub>F_\u03c7={self.fidelity * 100:.2f}%</sub>"))
        return fig

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """
        Serialize ResetChannel to a JSON string.

        :return: JSON string representation.
        """
        superop_array = np.asarray(self.process.matrix)
        flat = [[float(v.real), float(v.imag)] for v in superop_array.flat]
        data = {
            "inst": self.inst.out(),
            "superop": {"_complex_array": flat, "shape": list(superop_array.shape)},
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls: type[ResetChannel], json_str: str) -> "ResetChannel":
        """
        Deserialize a ResetChannel from a JSON string.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: ResetChannel instance.
        """
        data = json.loads(json_str)
        inst = _parse_quil_instruction(data["inst"])
        assert isinstance(inst, Reset)
        superop_data = data["superop"]
        flat = superop_data["_complex_array"]
        shape = tuple(superop_data["shape"])
        arr = jnp.array([complex(pair[0], pair[1]) for pair in flat], dtype=complex).reshape(shape)
        d = int(jnp.sqrt(shape[0]))
        num_qubits = int(jnp.round(jnp.log2(d)))
        dims = ((2,) * num_qubits, (2,) * num_qubits)
        process = qx.SuperOp.from_matrix(arr, dims)
        return cls(inst=inst, process=process)

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        """Return a simplified string representation."""
        qubit_str = str(self.qubits[0]) if self.qubits else "?"
        return f"<RESET({self.fidelity:.2f}) {qubit_str}>"

    def __eq__(self, other: object) -> bool:
        """Check equality based on instruction and process matrix."""
        if not isinstance(other, ResetChannel):
            return False
        if self.inst != other.inst:
            return False
        return bool(jnp.allclose(self.process.matrix, other.process.matrix, atol=1e-9))

    __hash__ = None


@dataclass(frozen=True)
class CycleChannel:
    """
    A cycle noise channel attaches superoperators to a specific cycle.

    Cycles can include gates and measurements. The constituent channels are stored
    directly, allowing fidelity metrics and serialization to be derived from them.
    """

    inst: Gate
    """The cycle to which the channel applies."""

    defcircuit: DefCircuit
    """The DefCircuit representing the logical cycle to which instruction represents."""

    channels: tuple["Channel | MeasurementChannel", ...]
    """Constituent channels (one per operation in the cycle) on disjoint qubits."""

    # ──────────────────────────────────────────────
    # Derived properties
    # ──────────────────────────────────────────────

    @cached_property
    def operator(self) -> tuple[qx.SuperOp | qx.QuantumInstrument, ...]:
        """Tuple of process superoperators, one per constituent channel."""
        return tuple(ch.process for ch in self.channels)

    @cached_property
    def qubits(self) -> list[int]:
        """All qubits in the cycle, derived from the instruction."""
        return self.inst.get_qubit_indices()

    @cached_property
    def pauli_fidelity(self) -> float:
        """Product of process (Pauli) fidelities over all gate channels in the cycle.

        Measurement channels do not contribute a gate fidelity and are skipped.
        For near-ideal noise the product approximation is exact since constituent
        channels act on disjoint subsystems.
        """
        f = 1.0
        for ch in self.channels:
            if isinstance(ch, Channel):
                f *= ch.pauli_fidelity
        return f

    @cached_property
    def fidelity(self) -> float:
        """Product of average gate fidelities over all gate channels in the cycle.

        Measurement channels do not contribute a gate fidelity and are skipped.
        """
        f = 1.0
        for ch in self.channels:
            if isinstance(ch, Channel):
                f *= ch.fidelity
        return f

    @cached_property
    def infidelity(self) -> float:
        """``1 - fidelity``."""
        return 1.0 - self.fidelity

    @cached_property
    def pauli_infidelity(self) -> float:
        """``1 - pauli_fidelity``."""
        return 1.0 - self.pauli_fidelity

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """
        Serialize CycleChannel to a JSON string.

        :return: JSON string representation.
        """
        ch_data = []
        for ch in self.channels:
            ch_data.append({"type": type(ch).__name__, "data": ch.to_json()})
        data = {
            "channels": ch_data,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls: type[CycleChannel], json_str: str) -> "CycleChannel":
        """
        Deserialize a CycleChannel from a JSON string.

        The ``inst`` and ``defcircuit`` fields are reconstructed from the constituent
        channels, consistent with how :func:`_build_cycle_channel` builds them.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: CycleChannel instance.
        """
        data = json.loads(json_str)
        _type_map: dict[str, type[Channel | MeasurementChannel]] = {
            "Channel": Channel,
            "MeasurementChannel": MeasurementChannel,
        }
        constituent_channels: list["Channel | MeasurementChannel"] = [
            _type_map[ch_data["type"]].from_json(ch_data["data"])  # type: ignore[index]
            for ch_data in data["channels"]
        ]
        return _build_cycle_channel(constituent_channels)

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        """Return a simplified string representation showing the gate and process fidelity."""
        return f"<{self.inst.out()} ~ ({100 * self.pauli_fidelity:.2f}%)>"

    def __eq__(self, other: object) -> bool:
        """Check equality based on instruction and constituent channels."""
        if not isinstance(other, CycleChannel):
            return False
        if self.inst != other.inst:
            return False
        return self.channels == other.channels

    __hash__ = None


def _channel_to_formal_inst(channel: Channel | MeasurementChannel) -> Gate | Measurement:
    """Convert a channel's instruction to use formal arguments for DefCircuit."""
    if isinstance(channel, Channel):
        inst = channel.inst
        return Gate(
            name=inst.name,
            params=inst.params,
            qubits=[FormalArgument(f"q{q}") for q in inst.get_qubit_indices()],
            modifiers=inst.modifiers,  # type: ignore[arg-type]
        )
    elif isinstance(channel, MeasurementChannel):
        qubit_idx = channel.qubits[0]
        return Measurement(
            qubit=FormalArgument(f"q{qubit_idx}"),
            classical_reg=None,
        )
    raise TypeError(f"Unsupported channel type: {type(channel)}")


def _build_cycle_channel(
    channels: list["Channel | MeasurementChannel"],
) -> "CycleChannel":
    """Build a CycleChannel from a list of Channel/MeasurementChannel on disjoint qubits."""
    all_qubits = sorted(q for ch in channels for q in ch.qubits)
    cycle_name = "CYCLE"
    formal_insts = [_channel_to_formal_inst(ch) for ch in channels]

    defcircuit = DefCircuit(
        name=cycle_name,
        parameters=[],
        qubits=[FormalArgument(f"q{q}") for q in all_qubits],
        instructions=list(formal_insts),  # type: ignore[arg-type]
    )
    inst = Gate(name=cycle_name, params=[], qubits=all_qubits)
    return CycleChannel(inst=inst, defcircuit=defcircuit, channels=tuple(channels))

##############################################################################
# Copyright 2026 Rigetti Computing
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
"""Noise channel classes and gate-resolution utilities.

This module defines ``SuperopChannel``, ``MeasurementChannel``, ``SuperopResetChannel``, and
``CycleChannel`` dataclasses for representing noise in quantum circuits, along
with helper functions for resolving gate unitaries and extracting custom gate
definitions from Quil programs.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property, reduce
from itertools import product
from typing import TYPE_CHECKING, Any, Protocol, SupportsFloat, runtime_checkable

import jax.numpy as jnp
import numpy as np
import quax as qx
from jax import Array
from jax.scipy.linalg import expm as jax_expm
from quil.expression import Expression as QuilExpression
from quil.instructions import Instruction as RSInstruction
from scipy.linalg import logm as scipy_logm

from pyquil.quilatom import Expression, FormalArgument, MemoryReference, Parameter, substitute
from pyquil.quilbase import DefCircuit, DefGate, Gate, Measurement, Reset, ResetQubit, _integer_base_and_exponent

if TYPE_CHECKING:
    from plotly.graph_objs import Figure

    from pyquil import Program

logger = logging.getLogger(__name__)

# Type alias for the custom-gate lookup map used throughout the SuperopChannel constructors.
CustomGateMap = dict[str, qx.Unitary | Callable[..., qx.Unitary]]


@runtime_checkable
class SupportsReal(Protocol):
    """A value exposing a ``real`` attribute convertible to ``float`` (e.g. ``complex``)."""

    @property
    def real(self) -> SupportsFloat: ...


# A gate parameter that :func:`_resolve_params` can reduce to a concrete float.
ResolvableParam = Parameter | Expression | QuilExpression | SupportsReal


def _parse_quil_instruction(quil_str: str) -> Gate | Measurement | Reset:
    """Parse a single Quil instruction string into a pyquil instruction object.

    Uses the ``quil`` Rust parser directly, avoiding a dependency on ``pyquil.Program``.
    """
    rs_inst = RSInstruction.parse(quil_str)
    if rs_inst.is_gate():
        return Gate._from_rs_gate(rs_inst.to_gate())
    elif rs_inst.is_measurement():
        return Measurement._from_rs_measurement(rs_inst.to_measurement())
    elif rs_inst.is_reset():
        reset = rs_inst.to_reset()
        if reset.qubit is None:
            return Reset._from_rs_reset(reset)
        return ResetQubit._from_rs_reset(reset)
    raise ValueError(f"Unsupported instruction type in: {quil_str}")


def _pack_complex_array(array: Array | np.ndarray) -> dict[str, Any]:
    """Pack a complex array into JSON-compatible real/imaginary pairs."""
    np_array = np.asarray(array)
    return {
        "_complex_array": [[float(value.real), float(value.imag)] for value in np_array.flat],
        "shape": list(np_array.shape),
    }


def _unpack_complex_array(data: dict[str, Any]) -> Array:
    """Unpack a complex array from :func:`_pack_complex_array` data."""
    shape = tuple(data["shape"])
    return jnp.array([complex(*pair) for pair in data["_complex_array"]], dtype=complex).reshape(shape)


def _pack_dims(dims: tuple[tuple[int, ...], tuple[int, ...]]) -> list[list[int]]:
    """Pack quax operator dims into JSON-compatible lists."""
    return [list(dims[0]), list(dims[1])]


def _unpack_dims(data: list[list[int]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Unpack quax operator dims from JSON-compatible lists."""
    if len(data) != 2:
        raise ValueError(f"Serialized operator dims must contain output and input dims, got {data}.")
    return (tuple(int(dim) for dim in data[0]), tuple(int(dim) for dim in data[1]))


def _pack_operator(operator: qx.SuperOp | qx.Unitary | qx.Choi | qx.Observable | qx.Operator) -> dict[str, Any]:
    """Pack a quax operator matrix with explicit dimension metadata."""
    data = _pack_complex_array(operator.matrix)
    data["dims"] = _pack_dims(operator.dims)
    return data


def _resolve_params(params: list[ResolvableParam]) -> list[float]:
    """Resolve gate parameters to concrete float values.

    Gate parameters are expected to be real. If a parameter evaluates to a complex number with a
    non-negligible imaginary part (detected via :func:`numpy.real_if_close`), the imaginary part is
    discarded with a warning.

    :param params: The gate parameters (may include symbolic Parameters or Expressions).
    :return: A list of concrete float values.
    :raises ValueError: If any parameter is symbolic and cannot be evaluated to a number.
    :raises quil.EvaluationError: If a ``quil`` expression cannot be evaluated to a single number.
    """
    fixed_params = []
    for p in params:
        if isinstance(p, (Parameter, Expression)):
            value: Any = p._evaluate()
            if isinstance(value, (Parameter, Expression)):
                raise ValueError(
                    f"Cannot resolve symbolic parameter {p}. Provide a gate with concrete numeric parameters."
                )
        elif isinstance(p, QuilExpression):
            # QuilExpression.evaluate returns a plain complex when fully resolved.
            value = p.evaluate({}, {})
        else:
            value = p

        resolved = np.real_if_close(value)
        if np.iscomplexobj(resolved):
            logger.warning("Gate parameter %r has a non-negligible imaginary part; using its real part.", p)
        fixed_params.append(float(np.real(resolved)))
    return fixed_params


def _qudit_dims(dimension: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Infer square operator dims ``((d,)*k, (d,)*k)`` from a matrix dimension ``d**k``.

    Supports qudits: the base ``d`` is the qudit dimension (2 for qubits, 3 for qutrits, ...)
    and ``k`` the number of qudits. Raises if ``dimension`` is not a prime power.
    """
    decomposition = _integer_base_and_exponent(dimension)
    if decomposition is None:
        raise ValueError(f"Matrix dimension {dimension} is not a prime power; cannot infer qudit dims.")
    base, exponent = decomposition
    return ((base,) * exponent, (base,) * exponent)


def get_custom_gates_from_program(program: Program) -> CustomGateMap:
    """Extract custom gate definitions from a Quil program.

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
                parameter_map: dict[Parameter | MemoryReference, float] = {
                    Parameter(p.name): arg for p, arg in zip(defgate.parameters, args, strict=False)
                }
                matrix = jnp.asarray(
                    [[substitute(element, parameter_map) for element in row] for row in defgate.matrix],
                    dtype=complex,
                )
                return qx.Unitary.from_matrix(matrix, _qudit_dims(matrix.shape[0]))

            custom_gates[defgate.name] = parametric_gate
        else:
            matrix = jnp.asarray(defgate.matrix, dtype=complex)
            custom_gates[defgate.name] = qx.Unitary.from_matrix(matrix, _qudit_dims(matrix.shape[0]))
    return custom_gates


def get_instruction_unitary(
    inst: Gate,
    custom_gates: CustomGateMap | None = None,
) -> qx.Unitary:
    """Get the unitary matrix associated with a gate instruction.

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
        result = qx.Unitary.from_matrix(result.matrix, result.dims)
    return result


@runtime_checkable
class ChannelProtocol(Protocol):
    """Shared behavior for gate noise channels backed by a superoperator ``process``.

    A gate channel attaches a CPTP ``process`` (a ``qx.SuperOp`` that *includes* the ideal
    gate) to a :class:`~pyquil.quilbase.Gate`, with fidelity metrics measured against the
    ideal ``unitary``. Concrete subclasses supply the three attributes below; the
    ``process`` may be a stored field (:class:`SuperopChannel`) or derived from a generator
    (:class:`Channel`). Every method here depends only on those three attributes,
    so it works identically regardless of how ``process`` is produced.

    Operations that leave the superoperator/Lindbladian structure — composition (``@``),
    :meth:`pauli_twirl`, :meth:`to_coherent_channel`, :meth:`to_stochastic_channel` — return
    a plain :class:`SuperopChannel`, since their result is a generic superoperator and not
    necessarily a Lindbladian generator.
    """

    # Provided by concrete subclasses (as dataclass fields or cached properties).
    if TYPE_CHECKING:
        inst: Gate
        process: qx.SuperOp
        unitary: qx.Unitary

    @cached_property
    def qubits(self) -> list[int]:
        """The qubits which the channel applies to."""
        return self.inst.get_qubit_indices()

    @cached_property
    def num_qubits(self) -> int:
        """The number of qubits the channel acts on."""
        return len(self.qubits)

    # ──────────────────────────────────────────────
    # Cached representation conversions
    # ──────────────────────────────────────────────

    def as_post_gate_noise(self) -> qx.SuperOp:
        r"""The noise as a superoperator applied *after* the ideal gate.

        A noisy gate channel can be viewed either as noise applied after the ideal gate
        (*post-gate*) or before it (*pre-gate*):

        - **post-gate**: :math:`\mathcal{E} = \Lambda_{\text{post}} \circ \mathcal{U}`, so
          :math:`\Lambda_{\text{post}} = \mathcal{E} \circ \mathcal{U}^\dagger` — this method.
        - **pre-gate**: :math:`\mathcal{E} = \mathcal{U} \circ \Lambda_{\text{pre}}`, so
          :math:`\Lambda_{\text{pre}} = \mathcal{U}^\dagger \circ \mathcal{E}`.

        The two coincide only when the noise commutes with the gate; in general they are
        related by conjugation, :math:`\Lambda_{\text{post}} = \mathcal{U} \circ
        \Lambda_{\text{pre}} \circ \mathcal{U}^\dagger`, and share the same fidelity metrics.
        This channel adopts the post-gate convention throughout.
        """
        return qx.to_superop(self.process @ self.unitary.h)

    # ──────────────────────────────────────────────
    # Fidelity properties
    # ──────────────────────────────────────────────

    @cached_property
    def fidelity(self) -> float:
        r"""Average gate fidelity :math:`F_{\mathrm{avg}}` of the channel relative to the ideal gate."""
        return float(qx.process_fidelity_to_average_fidelity(self.pauli_fidelity, dims=self.unitary.dims[0]))

    @cached_property
    def infidelity(self) -> float:
        r"""Average gate infidelity :math:`1 - F_{\mathrm{avg}}`."""
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
        return float(qx.stochastic_infidelity(self.as_post_gate_noise()))

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
        return float(qx.unitarity(self.as_post_gate_noise()))

    # ──────────────────────────────────────────────
    # SuperopChannel analysis methods
    # ──────────────────────────────────────────────

    def _as_channel(self, process: qx.SuperOp) -> SuperopChannel:
        """Wrap a derived superoperator as a plain :class:`SuperopChannel` for this gate."""
        return SuperopChannel(inst=self.inst, process=process, unitary=self.unitary)

    def pauli_twirl(self) -> SuperopChannel:
        """Return a Pauli-twirled version of this channel.

        Pauli twirling projects the channel onto the Pauli diagonal, eliminating
        off-diagonal coherences in the Pauli-Liouville representation. The
        resulting channel is a stochastic Pauli channel with the same diagonal
        error rates.
        """
        ptm = qx.to_pauli_liouville(self.process)
        # Keep only the diagonal of the PTM
        twirled_ptm_matrix = jnp.diag(jnp.diag(ptm.matrix))
        twirled_superop = qx.to_superop(qx.PauliLiouville.from_matrix(twirled_ptm_matrix, self.process.dims))
        return self._as_channel(twirled_superop)

    @cached_property
    def _unitary_error_component(self) -> Array:
        """Extract the dominant unitary from the noise-only channel.

        Uses eigendecomposition + SVD polar decomposition to find the closest
        unitary to the noise channel.
        """
        choi_matrix = qx.to_choi(self.as_post_gate_noise()).matrix
        d = 2**self.num_qubits

        # Dominant eigenvector of the Choi matrix
        eigenvalues, eigenvectors = jnp.linalg.eigh(choi_matrix)
        dominant_eigenvector = eigenvectors[:, jnp.argmax(jnp.abs(eigenvalues))]

        # SVD polar decomposition to extract the closest unitary
        u, _, vh = jnp.linalg.svd(dominant_eigenvector.reshape(d, d).T)
        return u @ vh

    def to_coherent_channel(self) -> SuperopChannel:
        """Isolate the coherent (unitary) component of the noise.

        Extracts the dominant unitary from the noise Choi matrix via polar
        decomposition and returns a channel consisting of that unitary error
        composed with the ideal gate.
        """
        u_error = self._unitary_error_component
        u_error_qx = qx.Unitary.from_matrix(u_error, self.process.dims)
        coherent_superop = qx.to_superop(u_error_qx @ self.unitary)
        return self._as_channel(coherent_superop)

    def to_stochastic_channel(self) -> SuperopChannel:
        r"""Isolate the stochastic (incoherent) component of the noise.

        The full channel decomposes as
        :math:`\mathcal{E} = \mathcal{S} \circ \mathcal{U}_{\mathrm{err}} \circ \mathcal{U}_{\mathrm{gate}}`.
        This method factors out the coherent unitary error and returns
        :math:`\mathcal{S} \circ \mathcal{U}_{\mathrm{gate}}`.
        """
        u_error = self._unitary_error_component
        # Get the noise-only superoperator and compose with U_err†
        noise_superop = self.as_post_gate_noise().matrix
        u_err_inv_superop = jnp.kron(u_error.conj(), u_error.conj().T)
        stochastic_noise_superop = noise_superop @ u_err_inv_superop
        # Recompose with the ideal gate
        ideal_superop = jnp.kron(self.unitary.matrix, self.unitary.matrix.conj())
        stochastic_superop = stochastic_noise_superop @ ideal_superop
        return self._as_channel(qx.SuperOp.from_matrix(stochastic_superop, self.process.dims))

    def is_pauli(self) -> bool:
        """Check if the noise channel is a Pauli (stochastic Pauli) channel.

        A Pauli channel has a diagonal Pauli transfer matrix (noise-only part).
        """
        ptm = qx.to_pauli_liouville(self.as_post_gate_noise()).matrix
        mask = ~jnp.eye(ptm.shape[0], dtype=bool)
        return bool(jnp.allclose(ptm[mask], 0))

    def to_pauli_vector(self) -> Array:
        """Convert the noise channel to a Pauli error probability vector.

        Returns the vector of probabilities for each Pauli error in lexicographic
        order (II, IX, IY, IZ, XI, XX, ...). The vector sums to 1.0.
        """
        noise_superop = self.as_post_gate_noise().matrix
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

    # ──────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────

    def plot(self, noise_only: bool = True, show_identity: bool = False) -> Figure:
        """Plot the Pauli transfer matrix of the channel.

        :param noise_only: If True (default), plot the noise-only channel (the post-gate
            noise, with the ideal gate unitary factored out; see :meth:`as_post_gate_noise`).
            If False, plot the full channel including the gate unitary.
        :param show_identity: If True, include the identity component in the noise-only plot.
            If False (default), visualize the generator of the noise channel via the matrix
            logarithm of the PTM.  For near-identity noise this approximates PTM - I, but
            correctly captures the Lie-algebraic structure of the channel.
            Only applies when ``noise_only=True``.
        :return: A Plotly Figure.
        """
        if noise_only:
            channel = self.as_post_gate_noise()
            if not show_identity:
                ptm = qx.to_pauli_liouville(channel)
                log_ptm = scipy_logm(np.asarray(ptm.matrix))
                channel = qx.PauliLiouville.from_matrix(jnp.array(log_ptm), channel.dims)
            title_prefix = "Noise SuperopChannel"
        else:
            channel = self.process
            title_prefix = "Full SuperopChannel"

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
    # Dunder methods
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        """Return a simplified string representation showing the gate and process fidelity."""
        return f"<{self.inst.out()} ~ ({100 * self.pauli_fidelity:.2f}%)>"

    def __eq__(self, other: object) -> bool:
        """Check equality by concrete type, instruction, and exact process/ideal-gate matrices.

        Equality is exact (no fidelity tolerance): two channels are equal only if they are the
        same concrete class, share the same instruction, and have bit-for-bit identical process
        and target-unitary matrices. Making tolerance decisions on the user's behalf is
        deliberately avoided.
        """
        if type(self) is not type(other):
            return False
        if self.inst != other.inst:  # type: ignore[attr-defined]
            return False
        return bool(
            jnp.array_equal(self.process.matrix, other.process.matrix)  # type: ignore[attr-defined]
            and jnp.array_equal(self.unitary.matrix, other.unitary.matrix)  # type: ignore[attr-defined]
        )

    __hash__ = None  # type: ignore[assignment]

    def __matmul__(self, other: ChannelProtocol) -> SuperopChannel:
        r"""Compose two channels: ``channel_B @ channel_A``.

        Both channels share the same gate instruction. The composition factors
        out one copy of the gate unitary so the result represents the sequential
        application of the two noisy processes:

        :math:`\mathcal{E}_B \circ \mathcal{U}^\dagger \circ \mathcal{E}_A`

        This is the natural composition: if ``channel_A`` already includes the
        gate, applying ``channel_B`` after it should not double-count the gate.
        The result is a plain :class:`SuperopChannel` (a superoperator composition).
        """
        if not isinstance(other, ChannelProtocol):
            return NotImplemented
        if self.inst != other.inst:
            raise ValueError(f"Cannot compose channels for different gates: {self.inst.out()} vs {other.inst.out()}")
        # E_B @ U† @ E_A  (factor out one gate unitary between the two channels)
        u_dag_superop = qx.to_superop(self.unitary.h)
        composed_superop = qx.to_superop(self.process @ u_dag_superop @ other.process)
        return self._as_channel(composed_superop)

    def __or__(self, other: ChannelProtocol | MeasurementChannel) -> CycleChannel:
        """Tensor product of two channels on disjoint qubits, producing a CycleChannel.

        The result represents a cycle containing both operations acting in parallel
        on disjoint qubits. The DefCircuit encodes the parallel operations as
        formal instructions.

        :param other: Another gate channel or MeasurementChannel on disjoint qubits.
        :return: A CycleChannel representing the tensor product.
        """
        if not isinstance(other, (ChannelProtocol, MeasurementChannel)):
            return NotImplemented

        # Validate disjoint qubits
        self_qubits = set(self.qubits)
        other_qubits = set(other.qubits)
        if self_qubits & other_qubits:
            raise ValueError(f"Cannot tensor channels with overlapping qubits: {self_qubits & other_qubits}")

        return _build_cycle_channel([self, other])


@dataclass(frozen=True, eq=False)
class SuperopChannel(ChannelProtocol):
    """A noise channel that stores a superoperator directly, for a specific gate.

    This is the special case of :class:`ChannelProtocol` whose ``process`` is a stored
    ``qx.SuperOp`` (rather than derived from a Lindbladian generator, as :class:`Channel`). It is
    what the manifold-leaving operations (composition ``@``, :meth:`pauli_twirl`,
    :meth:`to_coherent_channel`, :meth:`to_stochastic_channel`) return, and is useful when only a
    raw superoperator is available. Prefer :class:`Channel` and its ``from_*`` constructors when a
    generator description is available.

    The superoperator *includes* the gate unitary, so the channel replaces the gate rather than
    being applied after it, and can be converted to alternative representations (Choi, Kraus,
    Pauli-Liouville) via ``quax``. Fidelity metrics are computed relative to ``unitary``.
    """

    inst: Gate
    """Quil gate to which the channel applies."""

    process: qx.SuperOp
    """The noisy process (superoperator) for the gate, including the gate unitary."""

    unitary: qx.Unitary
    """The noiseless unitary of the gate."""

    # ──────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────

    @classmethod
    def from_pauli_noise(
        cls: type[SuperopChannel],
        inst: Gate,
        pauli_noise: dict[str, float],
        custom_gates: CustomGateMap | None = None,
    ) -> SuperopChannel:
        r"""Create a traditional post-gate stochastic Pauli error channel from error probabilities.

        This is the one-shot mixed-Pauli model applied *after* the ideal gate:
        :math:`\mathcal{E}(\rho) = \sum_i p_i\, P_i U \rho U^\dagger P_i^\dagger`, where the
        probabilities ``p_i`` (plus the implicit identity term) sum to 1. Unlike
        :meth:`Channel.from_pauli_generators`, the resulting channel reproduces the given error
        probabilities exactly (no exponentiation).

        :param inst: The gate to which the channel applies.
        :param pauli_noise: Pauli error probabilities, e.g. ``{"IX": 0.01, "ZZ": 0.02}``. Must sum
            to at most 1.0; the remainder is assigned to the identity (no-error) term.
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A SuperopChannel instance.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        num_qubits = len(unitary.dims[0])

        total_error_rate = 0.0
        for pauli, error_rate in pauli_noise.items():
            if error_rate < 0.0:
                raise ValueError(f"Pauli term '{pauli}' has negative error rate {error_rate}.")
            if len(pauli) != num_qubits:
                raise ValueError(f"Pauli term '{pauli}' has length {len(pauli)}, expected {num_qubits}.")
            total_error_rate += error_rate
        if total_error_rate > 1.0:
            raise ValueError(f"Pauli error rates must sum to at most 1.0, got {total_error_rate}.")

        all_pauli_terms = tuple("".join(term) for term in product("IXYZ", repeat=num_qubits))
        pauli_error_rates: list[float] = []
        for term in reversed(all_pauli_terms):
            if term in pauli_noise:
                error_rate = pauli_noise[term]
            elif all(p == "I" for p in term):
                error_rate = 1 - sum(pauli_error_rates)
            else:
                error_rate = 0
            pauli_error_rates.append(error_rate)
        pauli_error_rates = list(reversed(pauli_error_rates))

        # Kraus operators are the Pauli tensor products scaled by sqrt(probability), in
        # lexicographic (I, X, Y, Z) order matching pauli_error_rates, applied after the gate.
        single_pauli_matrices = qx.ensembles.PAULIS.matrix  # (4, 2, 2): I, X, Y, Z
        pauli_op_matrices = jnp.stack(
            [reduce(jnp.kron, paulis) for paulis in product(single_pauli_matrices, repeat=num_qubits)]
        )
        coeffs = jnp.sqrt(jnp.array(pauli_error_rates, dtype=float))
        kraus_matrices = coeffs[:, None, None] * pauli_op_matrices
        kraus_map = qx.KrausMap.from_matrix(kraus_matrices, unitary.dims)

        process_superop = qx.to_superop(kraus_map @ unitary)
        return cls(inst=inst, process=process_superop, unitary=unitary)

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize SuperopChannel to a JSON string.

        :return: JSON string representation.
        """
        data = {
            "inst": self.inst.out(),
            "superop": _pack_operator(self.process),
            "unitary": _pack_operator(self.unitary),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls: type[SuperopChannel], json_str: str) -> SuperopChannel:
        """Deserialize a SuperopChannel from a JSON string.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: SuperopChannel instance.
        """
        data = json.loads(json_str)
        inst = _parse_quil_instruction(data["inst"])
        if not isinstance(inst, Gate):
            raise TypeError(f"SuperopChannel JSON must contain a gate instruction, got {type(inst).__name__}.")

        superop_data = data["superop"]
        superop = qx.SuperOp.from_matrix(_unpack_complex_array(superop_data), _unpack_dims(superop_data["dims"]))

        u_data = data["unitary"]
        unitary = qx.Unitary.from_matrix(_unpack_complex_array(u_data), _unpack_dims(u_data["dims"]))

        return cls(inst=inst, process=superop, unitary=unitary)


@runtime_checkable
class _LindbladianBacked(Protocol):
    """Mixin for channels whose ``process`` is generated by a ``qx.Lindbladian``.

    Provides the ``process`` superoperator (``evolve(lindbladian, gate_time)``) and a helper for
    the CPTP-safe power operation. Concrete subclasses supply ``lindbladian`` and ``gate_time``.
    """

    if TYPE_CHECKING:
        lindbladian: qx.Lindbladian
        gate_time: float

    @cached_property
    def process(self) -> qx.SuperOp:
        """The channel superoperator, obtained by evolving the generator for ``gate_time``."""
        return qx.evolve(self.lindbladian, self.gate_time)

    def _scaled_noise_generator(self, power: float, *, gate_hamiltonian: qx.Observable | None) -> qx.Lindbladian:
        r"""Scale the dissipative + coherent-noise part of the generator by ``power``, keeping the gate.

        Physically powers the noise: each jump rate :math:`\gamma_k \to \text{power} \cdot \gamma_k`
        (so :math:`L_k \to \sqrt{\text{power}}\,L_k`) and the coherent-noise Hamiltonian
        :math:`H_{\text{noise}} \to \text{power} \cdot H_{\text{noise}}`, while the coherent gate
        generator ``gate_hamiltonian`` (``None`` for a purely dissipative reset) is preserved. The
        result is CPTP for ``power >= 0``.
        """
        if power < 0:
            raise ValueError(f"Lindbladian channel power must be non-negative, got {power}.")
        hamiltonian = self.lindbladian.hamiltonian
        if gate_hamiltonian is not None and hamiltonian is not None:
            noise_hamiltonian: qx.Observable | None = hamiltonian - gate_hamiltonian
        else:
            noise_hamiltonian = hamiltonian
        scaled_hamiltonian = noise_hamiltonian * float(power) if noise_hamiltonian is not None else None
        if gate_hamiltonian is not None:
            scaled_hamiltonian = (
                gate_hamiltonian if scaled_hamiltonian is None else scaled_hamiltonian + gate_hamiltonian
            )
        scaled_jumps = self.lindbladian.jump_operators * float(np.sqrt(power))
        return qx.Lindbladian(hamiltonian=scaled_hamiltonian, jump_operators=scaled_jumps)


@dataclass(frozen=True, eq=False)
class Channel(_LindbladianBacked, ChannelProtocol):
    r"""A noisy quantum gate: the ideal gate together with the noise that accompanies it.

    ``Channel`` is the primary way to describe gate noise. Rather than a raw error matrix, it
    holds a physical *generator* — a GKSL (Lindblad) master-equation model that combines the
    ideal gate's coherent evolution with dissipation (relaxation, dephasing, leakage) and
    coherent errors (over-rotations, unwanted couplings). Evolving that generator for
    ``gate_time`` yields the noisy ``process`` (a CPTP superoperator) that *replaces* the ideal
    gate in a circuit.

    Build one from whatever noise data is available via the ``from_*`` constructors — an average
    or process fidelity, a depolarizing constant, T1/T2 coherence times, Pauli generator rates, a
    mixture of unitary errors, a random coherent error, or a Lindbladian directly. Once built, a
    channel reports a full suite of fidelity and error metrics (:attr:`fidelity`,
    :attr:`pauli_fidelity`, :attr:`coherent_infidelity`, :attr:`stochastic_infidelity`,
    :attr:`unitarity`, ...), can be visualized with :meth:`plot`, decomposed into coherent and
    stochastic parts, Pauli-twirled, composed, and serialized.

    Because the noise is stored as a generator, two operations are natural and always CPTP-safe:

    - :meth:`__pow__` scales the *noise* (jump rates and coherent-noise Hamiltonian) while keeping
      the ideal gate, sweeping noise strength in a physically meaningful way.
    - :meth:`__add__` combines the *noise* of two channels on the same gate, keeping the gate.

    ``gate_time`` defaults to ``1.0`` (dimensionless). It may instead be a physical duration (e.g.
    ``~40e-9`` s), in which case the Hamiltonian and jump operators are in physical units; the gate
    Hamiltonian is scaled so that evolving for ``gate_time`` reproduces the ideal ``unitary``.
    """

    inst: Gate
    """Quil gate to which the channel applies."""

    lindbladian: qx.Lindbladian
    """The GKSL generator for the gate, including the (scaled) gate Hamiltonian."""

    unitary: qx.Unitary
    """The noiseless unitary of the gate."""

    gate_time: float = 1.0
    """Evolution time for ``evolve(lindbladian, gate_time)``. Default 1.0 (dimensionless)."""

    @cached_property
    def gate_hamiltonian(self) -> qx.Observable:
        """Coherent generator whose evolution over ``gate_time`` yields ``unitary``."""
        return qx.unitary_to_hamiltonian(self.unitary) * (1.0 / self.gate_time)

    @cached_property
    def _noise_lindbladian(self) -> qx.Lindbladian:
        """The generator with the coherent gate Hamiltonian factored out (dissipation + coherent noise)."""
        hamiltonian = self.lindbladian.hamiltonian
        noise_hamiltonian = hamiltonian - self.gate_hamiltonian if hamiltonian is not None else None
        return qx.Lindbladian(hamiltonian=noise_hamiltonian, jump_operators=self.lindbladian.jump_operators)

    # ──────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────

    @classmethod
    def from_lindbladian(
        cls: type[Channel],
        inst: Gate,
        noise_lindbladian: qx.Lindbladian,
        gate_time: float = 1.0,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        """Create a channel from a noise-only Lindbladian, folding in the gate.

        :param inst: The gate to which the channel applies.
        :param noise_lindbladian: The noise generator (e.g. from ``qx.lindbladians``), *without*
            the gate Hamiltonian. Its rates are interpreted per unit time and evolved for ``gate_time``.
        :param gate_time: Evolution time (default 1.0).
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        gate_hamiltonian = qx.unitary_to_hamiltonian(unitary) * (1.0 / gate_time)
        noise_hamiltonian = noise_lindbladian.hamiltonian
        total_hamiltonian = gate_hamiltonian if noise_hamiltonian is None else noise_hamiltonian + gate_hamiltonian
        lindbladian = qx.Lindbladian(hamiltonian=total_hamiltonian, jump_operators=noise_lindbladian.jump_operators)
        return cls(inst=inst, lindbladian=lindbladian, unitary=unitary, gate_time=gate_time)

    @classmethod
    def from_gate_fidelity(
        cls: type[Channel],
        inst: Gate,
        fidelity: float,
        gate_time: float = 1.0,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        """Create a depolarizing Lindbladian channel from an average gate fidelity."""
        unitary = get_instruction_unitary(inst, custom_gates)
        p = qx.average_fidelity_to_depolarizing_constant(fidelity, unitary.dims[0])
        return cls.from_depolarizing_constant(inst, p, gate_time, custom_gates)

    @classmethod
    def from_pauli_fidelity(
        cls: type[Channel],
        inst: Gate,
        pauli_fidelity: float,
        gate_time: float = 1.0,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        """Create a depolarizing Lindbladian channel from a process (Pauli) fidelity."""
        unitary = get_instruction_unitary(inst, custom_gates)
        p = qx.process_fidelity_to_depolarizing_constant(pauli_fidelity, unitary.dims[0])
        return cls.from_depolarizing_constant(inst, p, gate_time, custom_gates)

    @classmethod
    def from_depolarizing_constant(
        cls: type[Channel],
        inst: Gate,
        depolarizing_constant: float,
        gate_time: float = 1.0,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        r"""Create a depolarizing Lindbladian channel from a depolarization constant.

        Matches :meth:`SuperopChannel.from_depolarizing_constant`: the depolarizing constant :math:`p`
        parameterizes :math:`\mathcal{D}_p(\rho) = p\,\rho + (1-p)\,I/d`. The rate is chosen so that
        evolving for ``gate_time`` shrinks every traceless operator by exactly ``p``.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        dims = unitary.dims[0]
        d = int(np.prod(dims))
        shrink = float(np.clip(depolarizing_constant, np.finfo(float).tiny, 1.0))
        gamma_unit_time = -np.log(shrink) * (d**2 - 1) / d**2
        noise = qx.lindbladians.depolarizing(gamma_unit_time / gate_time, dims)
        return cls.from_lindbladian(inst, noise, gate_time, custom_gates)

    @classmethod
    def from_coherence_times(
        cls: type[Channel],
        inst: Gate,
        gate_duration: float,
        t1s: list[float],
        t2s: list[float] | None = None,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        """Create a decoherence channel from T1/T2 coherence times, evolved over ``gate_duration``.

        The gate and thermal relaxation happen together over ``gate_duration`` (``gate_time`` is set
        to ``gate_duration``). Physical rates come from ``qx.lindbladians.thermal_relaxation``.

        :param inst: The target instruction.
        :param gate_duration: The duration of the gate (used as ``gate_time``).
        :param t1s: The T1 time(s) of the qudits.
        :param t2s: The T2 time(s) of the qudits. Default to ``2 * t1``.
        """
        qubits = inst.get_qubit_indices()
        num_sys = len(qubits)
        if num_sys != len(t1s):
            raise ValueError(f"Expected {num_sys} T1 values for {inst.out()}, got {len(t1s)}.")
        if t2s is None:
            t2s = [2 * t1 for t1 in t1s]
        elif num_sys != len(t2s):
            raise ValueError(f"Expected {num_sys} T2 values for {inst.out()}, got {len(t2s)}.")

        t1_array = jnp.asarray(t1s)
        tphi_array = 1 / (1 / jnp.asarray(t2s) - 1 / t1_array)
        per_qubit = [
            qx.lindbladians.thermal_relaxation(t1, tphi) for t1, tphi in zip(t1_array, tphi_array, strict=True)
        ]
        noise = reduce(lambda a, b: a | b, per_qubit)
        return cls.from_lindbladian(inst, noise, gate_duration, custom_gates)

    @classmethod
    def from_pauli_generators(
        cls: type[Channel],
        inst: Gate,
        pauli_generators: dict[str, float],
        gate_time: float = 1.0,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        """Create a Pauli-dissipation channel from Pauli generator rates.

        Each Pauli term ``P`` with rate ``r`` contributes a jump operator ``sqrt(r) * P`` to the
        Lindbladian generator, so the rates are per-unit-time *generator* rates (not one-shot
        probabilities); over ``gate_time`` the channel is the corresponding continuously-generated
        Pauli channel. For a traditional post-gate stochastic Pauli error model with exact
        probabilities, use :meth:`SuperopChannel.from_pauli_noise`.

        :param inst: The gate to which the channel applies.
        :param pauli_generators: Pauli generator rates, e.g. ``{"IX": 0.01, "ZZ": 0.02}``.
        :param gate_time: Evolution time (default 1.0).
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        num_qubits = len(unitary.dims[0])
        single_pauli_matrices = qx.ensembles.PAULIS.matrix  # (4, 2, 2): I, X, Y, Z
        pauli_index = {"I": 0, "X": 1, "Y": 2, "Z": 3}

        jump_matrices = []
        for pauli, rate in pauli_generators.items():
            if rate < 0.0:
                raise ValueError(f"Pauli term '{pauli}' has negative rate {rate}.")
            if len(pauli) != num_qubits:
                raise ValueError(f"Pauli term '{pauli}' has length {len(pauli)}, expected {num_qubits}.")
            op = reduce(jnp.kron, [single_pauli_matrices[pauli_index[p]] for p in pauli])
            jump_matrices.append(jnp.sqrt(rate) * op)

        d = 2**num_qubits
        stacked = jnp.stack(jump_matrices) if jump_matrices else jnp.zeros((1, d, d), dtype=complex)
        jump_operators = qx.Operator.from_matrix(stacked, unitary.dims)
        noise = qx.Lindbladian(hamiltonian=None, jump_operators=jump_operators)
        return cls.from_lindbladian(inst, noise, gate_time, custom_gates)

    @classmethod
    def from_mixture(
        cls: type[Channel],
        inst: Gate,
        constituents: list[qx.Unitary],
        probabilities: list[float],
        gate_time: float = 1.0,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        r"""Create a mixture channel from a set of unitary errors with given probabilities.

        Each error unitary ``V_i`` with probability ``p_i`` contributes a jump operator
        ``sqrt(p_i) * V_i`` to the generator; over ``gate_time`` this is the exponentiated
        mixture channel composed with the ideal gate.

        :param inst: The gate to which the channel applies.
        :param constituents: Unitary error operators to mix.
        :param probabilities: Probability of each unitary error.
        :param gate_time: Evolution time (default 1.0).
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        ideal = get_instruction_unitary(inst, custom_gates)
        if len(constituents) != len(probabilities):
            raise ValueError("The number of constituents and probabilities must match.")
        if any(p < 0.0 for p in probabilities):
            raise ValueError("Mixture probabilities must be non-negative.")

        d = int(np.prod(ideal.dims[0]))
        jump_matrices = [jnp.sqrt(p) * v.matrix for p, v in zip(probabilities, constituents, strict=True)]
        stacked = jnp.stack(jump_matrices) if jump_matrices else jnp.zeros((1, d, d), dtype=complex)
        jump_operators = qx.Operator.from_matrix(stacked, ideal.dims)
        noise = qx.Lindbladian(hamiltonian=None, jump_operators=jump_operators)
        return cls.from_lindbladian(inst, noise, gate_time, custom_gates)

    @classmethod
    def from_random_coherent_error(
        cls: type[Channel],
        inst: Gate,
        process_fidelity: float,
        rng: np.random.Generator | None = None,
        gate_time: float = 1.0,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        r"""Create a channel with a random coherent (unitary) error at the specified process fidelity.

        A random unitary close to identity is generated with the given process fidelity and composed
        with the ideal gate; the result is a purely coherent :class:`Channel` (Hamiltonian only,
        no jump operators).

        :param inst: The gate to which the channel applies.
        :param process_fidelity: The process fidelity of the coherent error, :math:`F_e \in [0, 1]`.
        :param rng: NumPy random number generator for reproducibility.
        :param gate_time: Evolution time (default 1.0).
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        if rng is None:
            rng = np.random.default_rng()

        ideal = get_instruction_unitary(inst, custom_gates)
        num_qubits = len(ideal.dims[0])
        d = 2**num_qubits

        angle = jnp.arccos(2 * process_fidelity - 1) / (2 * jnp.pi)
        id_coeff = 1 - float(angle)
        coeffs = rng.random(4**num_qubits - 1)
        coeffs = (1 - id_coeff) / np.sqrt(np.sum(np.square(coeffs))) * coeffs

        pauli_matrices = qx.ensembles.PAULIS.matrix  # shape (4, 2, 2)
        pauli_sum = jnp.eye(d, dtype=complex) * id_coeff
        pauli_products = list(itertools.product(pauli_matrices, repeat=num_qubits))[1:]
        for paulis, coefficient in zip(pauli_products, coeffs, strict=False):
            pauli_sum = pauli_sum + reduce(jnp.kron, paulis) * coefficient

        error_unitary = jax_expm(-1j * jnp.pi * pauli_sum)
        phase = jnp.exp(-1j * jnp.angle(error_unitary[0, 0]))
        error_unitary = error_unitary * phase

        noisy_unitary = qx.Unitary.from_matrix(error_unitary @ ideal.matrix, ideal.dims)
        hamiltonian = qx.unitary_to_hamiltonian(noisy_unitary) * (1.0 / gate_time)
        zero_jumps = qx.Operator.from_matrix(jnp.zeros((1, d, d), dtype=complex), ideal.dims)
        lindbladian = qx.Lindbladian(hamiltonian=hamiltonian, jump_operators=zero_jumps)
        return cls(inst=inst, lindbladian=lindbladian, unitary=ideal, gate_time=gate_time)

    # ──────────────────────────────────────────────
    # Lindbladian-native operations
    # ──────────────────────────────────────────────

    def __pow__(self, power: float) -> Channel:
        """Scale the noise to a (non-negative) ``power`` while preserving the gate.

        ``power = 0`` yields the ideal gate, ``1`` leaves the channel unchanged, and ``> 1``
        strengthens the noise. Unlike a fractional matrix power of a superoperator, this is always
        CPTP because it acts on the generator's jump operators and coherent-noise Hamiltonian.
        """
        if not isinstance(power, (int, float)):
            return NotImplemented
        scaled = self._scaled_noise_generator(power, gate_hamiltonian=self.gate_hamiltonian)
        return replace(self, lindbladian=scaled)

    def __add__(self, other: Channel) -> Channel:
        """Combine the *noise* of two channels on the same gate, keeping the gate.

        The gate Hamiltonian is factored out of each operand, the noise generators are summed
        (jump operators concatenated, coherent-noise Hamiltonians added), and the gate is folded
        back in. Adding two ``RX(pi/2)`` channels therefore yields an ``RX(pi/2)`` channel whose
        noise is the union of the two.
        """
        if not isinstance(other, Channel):
            return NotImplemented
        if self.inst != other.inst:
            raise ValueError(f"Cannot add channels for different gates: {self.inst.out()} vs {other.inst.out()}")
        combined_noise = self._noise_lindbladian + other._noise_lindbladian
        gate_hamiltonian = self.gate_hamiltonian
        combined_hamiltonian = (
            gate_hamiltonian if combined_noise.hamiltonian is None else combined_noise.hamiltonian + gate_hamiltonian
        )
        combined = qx.Lindbladian(hamiltonian=combined_hamiltonian, jump_operators=combined_noise.jump_operators)
        return replace(self, lindbladian=combined)

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize Channel to a JSON string."""
        hamiltonian = self.lindbladian.hamiltonian
        data = {
            "inst": self.inst.out(),
            "gate_time": self.gate_time,
            "unitary": _pack_operator(self.unitary),
            "hamiltonian": None if hamiltonian is None else _pack_operator(hamiltonian),
            "jump_operators": _pack_operator(self.lindbladian.jump_operators),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls: type[Channel], json_str: str) -> Channel:
        """Deserialize a Channel from a JSON string."""
        data = json.loads(json_str)
        inst = _parse_quil_instruction(data["inst"])
        if not isinstance(inst, Gate):
            raise TypeError(f"Channel JSON must contain a gate instruction, got {type(inst).__name__}.")

        u_data = data["unitary"]
        unitary = qx.Unitary.from_matrix(_unpack_complex_array(u_data), _unpack_dims(u_data["dims"]))

        ham_data = data["hamiltonian"]
        hamiltonian = (
            None
            if ham_data is None
            else qx.Observable.from_matrix(_unpack_complex_array(ham_data), _unpack_dims(ham_data["dims"]))
        )
        jump_data = data["jump_operators"]
        jump_operators = qx.Operator.from_matrix(_unpack_complex_array(jump_data), _unpack_dims(jump_data["dims"]))
        lindbladian = qx.Lindbladian(hamiltonian=hamiltonian, jump_operators=jump_operators)
        return cls(inst=inst, lindbladian=lindbladian, unitary=unitary, gate_time=data["gate_time"])


@dataclass(frozen=True)
class MeasurementChannel:
    """A measurement noise channel attaches a quantum instrument to a specific measurement operation.

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
        return [qubit.index if hasattr(qubit, "index") else int(qubit)]

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
    ) -> MeasurementChannel:
        """Create a readout quantum instrument with optional asymmetry.

        Produces a perfectly QND measurement with the given classification fidelity.
        Error is distributed only between adjacent levels: P(j+1|j) and P(j|j+1).
        Non-adjacent confusion is zero.

        :param inst: The measurement instruction.
        :param fidelity: The average readout fidelity.
        :param asymmetry: Value between -1 and +1. Zero is symmetric.
            Positive biases toward upward confusion P(j+1|j), negative toward downward P(j|j+1).
        :param dim: The dimension of the measured system (2 for qubits, 3 for qutrits, etc.).
        :return: A MeasurementChannel instance.
        :raises ValueError: If ``dim < 2``.
        """
        if dim < 2:
            raise ValueError(f"Measured system dimension must be at least 2, got dim={dim}.")

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
    ) -> MeasurementChannel:
        """Create a MeasurementChannel from a confusion matrix and a transition matrix.

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
    ) -> MeasurementChannel:
        """Create a MeasurementChannel from a Bloch sphere measurement axis.

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
    ) -> MeasurementChannel:
        """Create a MeasurementChannel for a binary discriminator.

        Models a measurement that reports a single classical *bit* for a
        ``dim``-level system by thresholding: levels ``[0, threshold)`` yield
        outcome ``0`` and levels ``[threshold, dim)`` yield outcome ``1``.  The
        resulting instrument therefore always has exactly two outcomes, so leaked
        levels are lumped in with whichever side of the threshold they fall on
        (the usual case being a "dark" ground state vs. everything "bright").

        For example, ``threshold=1, dim=2`` is an ordinary qubit readout
        (``{0}`` -> 0, ``{1}`` -> 1); ``threshold=1, dim=3`` discriminates
        ``{0}`` vs ``{1, 2}`` (ground vs. excited-or-leaked); ``threshold=2,
        dim=3`` discriminates ``{0, 1}`` vs ``{2}`` (i.e. flags leakage only).

        An optional ``fidelity`` parameter degrades the ideal discriminator with
        uniform classification noise.

        :param inst: The measurement instruction.
        :param dim: The dimension of the measured system.
        :param threshold: The split point: levels below it report 0, levels at or
            above it report 1.  Must satisfy ``1 <= threshold < dim``.
        :param fidelity: Additional classification fidelity applied on top of the
            discrimination (1.0 = perfect discriminator).
        :return: A MeasurementChannel instance.
        """
        if not (1 <= threshold < dim):
            raise ValueError(f"threshold must satisfy 1 <= threshold < dim, got threshold={threshold}, dim={dim}")

        # Ideal two-outcome confusion matrix of shape (num_outcomes=2, dim):
        # column j (prepared level j) puts all its weight on outcome 0 if
        # j < threshold, else on outcome 1.  Two rows so the instrument has
        # exactly two outcomes (never a phantom, zero-probability outcome).
        confusion = jnp.zeros((2, dim))
        for j in range(dim):
            confusion = confusion.at[int(j >= threshold), j].set(1.0)

        # Optionally degrade with uniform noise across the two outcomes.
        if fidelity < 1.0:
            confusion = fidelity * confusion + (1 - fidelity) * jnp.ones((2, dim)) / 2

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
        return self.process.confusion_matrix  # type: ignore[no-any-return]

    @cached_property
    def transition_matrix(self) -> Array:
        """The post-measurement transition matrix.

        Shape ``(d, d)``. Entry ``[k, j]`` is P(ending in k | input j),
        marginalized over all measurement outcomes.
        """
        return self.process.transition_matrix  # type: ignore[no-any-return]

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
        """Plot the quantum instrument using the quax visualization.

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
        """Serialize MeasurementChannel to a JSON string.

        :return: JSON string representation.
        """
        # Store per-outcome superoperator matrices.
        instrument_data = []
        for i in range(self.process.num_outcomes):
            superop_i, _ = self.process.outcome_superop(i)
            instrument_data.append(_pack_operator(superop_i))

        data = {
            "inst": self.inst.out(),
            "instruments": instrument_data,
            "measured_qudits": list(self.process.measured_qudits),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls: type[MeasurementChannel], json_str: str) -> MeasurementChannel:
        """Deserialize a MeasurementChannel from a JSON string.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: MeasurementChannel instance.
        """
        data = json.loads(json_str)
        inst = _parse_quil_instruction(data["inst"])
        if not isinstance(inst, Measurement):
            raise TypeError(
                f"MeasurementChannel JSON must contain a measurement instruction, got {type(inst).__name__}."
            )
        measured_qudits = tuple(data["measured_qudits"])

        superop_matrices = []
        instrument_dims = None
        for inst_data in data["instruments"]:
            arr = _unpack_complex_array(inst_data)
            op_dims = _unpack_dims(inst_data["dims"])
            if instrument_dims is None:
                instrument_dims = op_dims
            elif instrument_dims != op_dims:
                raise ValueError("All serialized measurement outcomes must have the same dims.")
            superop_matrices.append(arr)

        if instrument_dims is None:
            raise ValueError("MeasurementChannel JSON must contain at least one outcome superoperator.")

        instrument = qx.QuantumInstrument.from_matrix(jnp.stack(superop_matrices), instrument_dims, measured_qudits)
        return cls(inst=inst, process=instrument)

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        """Return a simplified string representation."""
        return f"<MEASURE({self.classification_fidelity:.2f}) {self.qubits[0]} ~ QND({100 * self.non_demolition_fidelity:.2f}%)>"

    def __eq__(self, other: object) -> bool:
        """Check equality by instruction and exact instrument matrix (no tolerance)."""
        if not isinstance(other, MeasurementChannel):
            return False
        if self.inst != other.inst:
            return False
        return bool(jnp.array_equal(self.process.matrix, other.process.matrix))

    __hash__ = None  # type: ignore[assignment]

    def __matmul__(self, other: MeasurementChannel) -> MeasurementChannel:
        """Compose two measurement channels on the same qubit.

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

    def __or__(self, other: SuperopChannel | MeasurementChannel) -> CycleChannel:
        """Tensor product of two channels on disjoint qubits, producing a CycleChannel.

        :param other: Another SuperopChannel or MeasurementChannel on disjoint qubits.
        :return: A CycleChannel representing the tensor product.
        """
        if not isinstance(other, (SuperopChannel, MeasurementChannel)):
            return NotImplemented

        self_qubits = set(self.qubits)
        other_qubits = set(other.qubits)
        if self_qubits & other_qubits:
            raise ValueError(f"Cannot tensor channels with overlapping qubits: {self_qubits & other_qubits}")

        return _build_cycle_channel([self, other])


@runtime_checkable
class _ResetChannelBase(Protocol):
    """Shared behavior for reset noise channels backed by a superoperator ``process``.

    A reset channel replaces a targeted reset with a CPTP ``process`` (a ``qx.SuperOp`` that
    *includes* the ideal reset). Unlike gate channels there is no unitary; fidelity is
    measured against the ideal reset ``qx.gates.RESET``. Concrete subclasses supply ``inst`` and
    ``process`` — a stored field for :class:`SuperopResetChannel`, or derived from a generator for
    :class:`ResetChannel`.
    """

    if TYPE_CHECKING:
        inst: ResetQubit
        process: qx.SuperOp

    def __post_init__(self) -> None:
        """Validate that the channel is attached to a targeted reset."""
        if not isinstance(self.inst, ResetQubit):
            raise TypeError(f"{type(self).__name__} only supports targeted ResetQubit instructions.")

    @cached_property
    def qubits(self) -> list[int]:
        """The qubit(s) that the reset applies to."""
        qubit = self.inst.qubit
        if qubit is None:
            return []
        return [qubit.index if hasattr(qubit, "index") else int(qubit)]

    @cached_property
    def _ideal_reset(self) -> qx.SuperOp:
        """The ideal reset superoperator matching this channel's dimension."""
        return qx.gates.RESET(dim=self.process.dims[0][0])

    @cached_property
    def fidelity(self) -> float:
        r"""Process fidelity of the reset channel relative to the ideal reset :math:`F_e \in [0, 1]`."""
        return float(qx.process_fidelity(self.process, self._ideal_reset))

    def as_post_gate_noise(self) -> qx.SuperOp:
        r"""The noise as a superoperator applied *after* the ideal reset.

        The full channel factors as :math:`\mathcal{E} = \mathcal{N} \circ \mathcal{R}`, where
        :math:`\mathcal{R}` is the ideal reset and :math:`\mathcal{N}` the post-reset noise.
        The ideal reset is not invertible, so :math:`\mathcal{N}` is recovered with its
        Moore-Penrose pseudo-inverse, :math:`\mathcal{N} = \mathcal{E} \circ \mathcal{R}^{+}`.
        This is exact for channels built as noise-after-reset (e.g.
        :meth:`SuperopResetChannel.from_reset_fidelity`); for relaxation-type resets it is the
        representative that agrees with the channel on the reset's output subspace
        (:math:`\mathcal{N} \circ \mathcal{R} = \mathcal{E}` always holds).
        """
        ideal = self._ideal_reset
        noise_matrix = self.process.matrix @ jnp.linalg.pinv(ideal.matrix)
        return qx.SuperOp.from_matrix(noise_matrix, self.process.dims)

    def plot(self, noise_only: bool = False) -> Figure:
        """Plot the Pauli transfer matrix of the reset channel.

        :param noise_only: If True, plot the post-reset noise (see :meth:`as_post_gate_noise`)
            instead of the full process. Defaults to False (the full channel).
        :return: A Plotly Figure.
        """
        channel = self.as_post_gate_noise() if noise_only else self.process
        title_prefix = "Reset noise" if noise_only else "Reset channel"
        fig = qx.plot(channel)
        qubit_str = str(self.qubits[0]) if self.qubits else "?"
        fig.update_layout(title=(f"{title_prefix} RESET {qubit_str}<br><sub>F_χ={self.fidelity * 100:.2f}%</sub>"))
        return fig

    def __str__(self) -> str:
        """Return a simplified string representation."""
        qubit_str = str(self.qubits[0]) if self.qubits else "?"
        return f"<RESET({self.fidelity:.2f}) {qubit_str}>"

    def __eq__(self, other: object) -> bool:
        """Check equality by concrete type, instruction, and exact process matrix (no tolerance)."""
        if type(self) is not type(other):
            return False
        if self.inst != other.inst:  # type: ignore[attr-defined]
            return False
        return bool(jnp.array_equal(self.process.matrix, other.process.matrix))  # type: ignore[attr-defined]

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, eq=False)
class SuperopResetChannel(_ResetChannelBase):
    """A reset noise channel attaches a superoperator to a specific reset operation.

    The ``process`` field is a ``qx.SuperOp`` which *includes* the ideal reset, so the channel
    replaces the reset instruction rather than being applied after it.
    """

    inst: ResetQubit
    """The reset operation to which the channel applies."""

    process: qx.SuperOp
    """A superoperator representation of the noisy reset (including ideal reset)."""

    # ──────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────

    @classmethod
    def from_reset_fidelity(
        cls: type[SuperopResetChannel],
        inst: ResetQubit,
        fidelity: float,
        dim: int = 2,
    ) -> SuperopResetChannel:
        r"""Create a SuperopResetChannel with depolarizing noise scaled to the given process fidelity.

        The ideal reset maps every state to :math:`|0\rangle\langle 0|`; noise is a depolarizing
        channel applied *after* it, so the process is ``depolarizing @ RESET`` — every state is
        reset and then shrunk toward the maximally-mixed state by a factor ``fidelity``.

        :param inst: The reset instruction.
        :param fidelity: Process fidelity of the reset channel, :math:`F \in [0, 1]`.
            1.0 yields an ideal reset; values below 1 introduce depolarizing noise.
        :param dim: Hilbert-space dimension (2 for qubits).
        :return: A SuperopResetChannel instance.
        """
        if not isinstance(inst, ResetQubit):
            raise TypeError("SuperopResetChannel only supports targeted ResetQubit instructions.")

        # Depolarizing rate whose evolution shrinks every traceless operator by exactly ``fidelity``.
        d2 = dim * dim
        shrink = float(np.clip(fidelity, np.finfo(float).tiny, 1.0))
        gamma = -np.log(shrink) * (d2 - 1) / d2
        depol = qx.channels.depolarizing(gamma, (dim,))
        noisy_superop = qx.to_superop(depol @ qx.gates.RESET(dim=dim))
        return cls(inst=inst, process=noisy_superop)

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize SuperopResetChannel to a JSON string.

        :return: JSON string representation.
        """
        data = {
            "inst": self.inst.out(),
            "superop": _pack_operator(self.process),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls: type[SuperopResetChannel], json_str: str) -> SuperopResetChannel:
        """Deserialize a SuperopResetChannel from a JSON string.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: SuperopResetChannel instance.
        """
        data = json.loads(json_str)
        inst = _parse_quil_instruction(data["inst"])
        if not isinstance(inst, ResetQubit):
            raise TypeError(
                f"SuperopResetChannel JSON must contain a targeted reset instruction, got {type(inst).__name__}."
            )
        superop_data = data["superop"]
        process = qx.SuperOp.from_matrix(_unpack_complex_array(superop_data), _unpack_dims(superop_data["dims"]))
        return cls(inst=inst, process=process)


@dataclass(frozen=True, eq=False)
class ResetChannel(_LindbladianBacked, _ResetChannelBase):
    """A reset channel modeled as finite-time relaxation toward the ground state.

    The ``process`` is ``qx.evolve(lindbladian, gate_time)`` for a purely dissipative relaxation
    generator (e.g. amplitude damping / thermal relaxation). The ideal reset is the strong-damping
    (``gate_time -> infinity``) limit, so finite times give a physically-grounded *noisy* reset.
    Like :class:`Channel`, :meth:`__pow__` scales the relaxation strength (there is no
    gate Hamiltonian to preserve).
    """

    inst: ResetQubit
    """The reset operation to which the channel applies."""

    lindbladian: qx.Lindbladian
    """The dissipative generator whose evolution over ``gate_time`` relaxes toward the ground state."""

    gate_time: float = 1.0
    """Evolution time for ``evolve(lindbladian, gate_time)``. Default 1.0 (dimensionless)."""

    @classmethod
    def from_lindbladian(
        cls: type[ResetChannel],
        inst: ResetQubit,
        lindbladian: qx.Lindbladian,
        gate_time: float = 1.0,
    ) -> ResetChannel:
        """Create a reset channel directly from a dissipative relaxation generator."""
        return cls(inst=inst, lindbladian=lindbladian, gate_time=gate_time)

    @classmethod
    def from_amplitude_damping(
        cls: type[ResetChannel],
        inst: ResetQubit,
        gamma: float,
        gate_time: float = 1.0,
        dim: int = 2,
    ) -> ResetChannel:
        """Create a reset channel from an amplitude-damping (decay-to-ground) generator.

        :param gamma: Amplitude-damping rate. Larger ``gamma * gate_time`` gives a more complete reset.
        :param dim: Hilbert-space dimension (2 for qubits).
        """
        return cls.from_lindbladian(inst, qx.lindbladians.amplitude_damping(gamma, (dim,)), gate_time)

    @classmethod
    def from_coherence_times(
        cls: type[ResetChannel],
        inst: ResetQubit,
        duration: float,
        t1: float,
        t2: float | None = None,
    ) -> ResetChannel:
        """Create a reset channel from T1/T2 relaxation over ``duration`` (used as ``gate_time``)."""
        t2_value = 2 * t1 if t2 is None else t2
        tphi = 1 / (1 / t2_value - 1 / t1)
        return cls.from_lindbladian(inst, qx.lindbladians.thermal_relaxation(t1, tphi), duration)

    def __pow__(self, power: float) -> ResetChannel:
        """Scale the relaxation to a (non-negative) ``power``; there is no gate to preserve."""
        if not isinstance(power, (int, float)):
            return NotImplemented
        scaled = self._scaled_noise_generator(power, gate_hamiltonian=None)
        return replace(self, lindbladian=scaled)

    def to_json(self) -> str:
        """Serialize ResetChannel to a JSON string."""
        hamiltonian = self.lindbladian.hamiltonian
        data = {
            "inst": self.inst.out(),
            "gate_time": self.gate_time,
            "hamiltonian": None if hamiltonian is None else _pack_operator(hamiltonian),
            "jump_operators": _pack_operator(self.lindbladian.jump_operators),
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls: type[ResetChannel], json_str: str) -> ResetChannel:
        """Deserialize a ResetChannel from a JSON string."""
        data = json.loads(json_str)
        inst = _parse_quil_instruction(data["inst"])
        if not isinstance(inst, ResetQubit):
            raise TypeError(f"ResetChannel JSON must contain a targeted reset instruction, got {type(inst).__name__}.")
        ham_data = data["hamiltonian"]
        hamiltonian = (
            None
            if ham_data is None
            else qx.Observable.from_matrix(_unpack_complex_array(ham_data), _unpack_dims(ham_data["dims"]))
        )
        jump_data = data["jump_operators"]
        jump_operators = qx.Operator.from_matrix(_unpack_complex_array(jump_data), _unpack_dims(jump_data["dims"]))
        lindbladian = qx.Lindbladian(hamiltonian=hamiltonian, jump_operators=jump_operators)
        return cls(inst=inst, lindbladian=lindbladian, gate_time=data["gate_time"])


@dataclass(frozen=True)
class CycleChannel:
    """A cycle noise channel attaches superoperators to a specific cycle.

    Cycles can include gates and measurements. The constituent channels are stored
    directly, allowing fidelity metrics and serialization to be derived from them.
    """

    inst: Gate
    """The cycle to which the channel applies."""

    defcircuit: DefCircuit
    """The DefCircuit representing the logical cycle to which instruction represents."""

    channels: tuple[ChannelProtocol | MeasurementChannel, ...]
    """Constituent channels (one per operation in the cycle) on disjoint qubits."""

    def __post_init__(self) -> None:
        """Validate that every instruction in the cycle body has a corresponding channel.

        Downstream consumers (the resolver, the stim converter) use only ``channels`` and
        ignore ``defcircuit``; a missing channel would silently drop that operation's noise.
        Operations are matched by identity (name, params, concrete qubits), independent of
        the DefCircuit's formal-argument naming.
        """
        if len(self.expanded_instructions) != len(self.channels):
            raise ValueError(
                "CycleChannel is incomplete: every instruction in the cycle's DefCircuit "
                "body must have a corresponding channel. "
                f"\nDefCircuit body: {self.expanded_instructions}"
                f"\nChannels:        {self.channels}"
            )
        for instruction, channel in zip(self.expanded_instructions, self.channels, strict=True):
            if str(instruction) != str(channel.inst):
                raise ValueError(
                    "CycleChannel is incomplete: every instruction in the cycle's DefCircuit "
                    "body must have a corresponding channel. "
                    f"\nDefCircuit body: {instruction}"
                    f"\nChannels:        {channel.inst}"
                )

    # ──────────────────────────────────────────────
    # Derived properties
    # ──────────────────────────────────────────────

    @cached_property
    def expanded_instructions(self) -> list[Gate | Measurement | ResetQubit]:
        """Return the expanded instructions of the defcircuit."""
        qarg_to_qubit = dict(zip(self.defcircuit.qubit_variables, self.inst.get_qubit_indices(), strict=False))
        instructions: list[Gate | Measurement | ResetQubit] = []
        for inst in self.defcircuit.instructions:
            match inst:
                case Measurement():
                    instructions.append(Measurement(qubit=qarg_to_qubit[inst.qubit], classical_reg=inst.classical_reg))  # type: ignore[index]
                case ResetQubit():
                    instructions.append(ResetQubit(qarg_to_qubit[inst.qubit]))  # type: ignore[index]
                case Gate():
                    instructions.append(Gate(inst.name, inst.params, [qarg_to_qubit[q] for q in inst.qubits]))  # type: ignore[index]
                case _:
                    raise TypeError(f"Unsupported instruction type in defcircuit: {type(inst).__name__}")
        return instructions

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
            if isinstance(ch, ChannelProtocol):
                f *= ch.pauli_fidelity
        return f

    @cached_property
    def fidelity(self) -> float:
        """Product of average gate fidelities over all gate channels in the cycle.

        Measurement channels do not contribute a gate fidelity and are skipped.
        """
        f = 1.0
        for ch in self.channels:
            if isinstance(ch, ChannelProtocol):
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
        """Serialize CycleChannel to a JSON string.

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
    def from_json(cls: type[CycleChannel], json_str: str) -> CycleChannel:
        """Deserialize a CycleChannel from a JSON string.

        The ``inst`` and ``defcircuit`` fields are reconstructed from the constituent
        channels, consistent with how :func:`_build_cycle_channel` builds them.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: CycleChannel instance.
        """
        data = json.loads(json_str)
        _type_map: dict[str, type[ChannelProtocol | MeasurementChannel]] = {
            "SuperopChannel": SuperopChannel,
            "Channel": Channel,
            "MeasurementChannel": MeasurementChannel,
        }
        constituent_channels: list[ChannelProtocol | MeasurementChannel] = [
            _type_map[ch_data["type"]].from_json(ch_data["data"]) for ch_data in data["channels"]
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

    __hash__ = None  # type: ignore[assignment]


def _channel_to_formal_inst(channel: ChannelProtocol | MeasurementChannel) -> Gate | Measurement:
    """Convert a channel's instruction to use formal arguments for DefCircuit."""
    if isinstance(channel, ChannelProtocol):
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
    channels: list[ChannelProtocol | MeasurementChannel],
) -> CycleChannel:
    """Build a CycleChannel from a list of SuperopChannel/MeasurementChannel on disjoint qubits."""
    all_qubits = sorted(q for ch in channels for q in ch.qubits)
    cycle_name = "CYCLE"
    formal_insts = [_channel_to_formal_inst(ch) for ch in channels]

    defcircuit = DefCircuit(
        name=cycle_name,
        parameters=[],
        qubits=[FormalArgument(f"q{q}") for q in all_qubits],
        instructions=list(formal_insts),
    )
    inst = Gate(name=cycle_name, params=[], qubits=all_qubits)
    return CycleChannel(inst=inst, defcircuit=defcircuit, channels=tuple(channels))

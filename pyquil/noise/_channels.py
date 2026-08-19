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

This module defines the noise channel dataclasses used to describe noise in quantum circuits:

- :class:`Channel` and :class:`SuperopChannel` for gates (Lindbladian-generated and raw
  superoperator, respectively),
- :class:`ResetChannel` and :class:`SuperopResetChannel` for targeted resets,
- :class:`MeasurementChannel` for measurements, and
- :class:`CycleChannel` for a set of operations acting in parallel on disjoint qubits.

It also provides helper functions for resolving gate unitaries and extracting custom gate
definitions from Quil programs.
"""

from __future__ import annotations

import itertools
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cached_property, reduce
from itertools import product
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict, cast, final

import jax.numpy as jnp
import numpy as np
import quax as qx
from jax import Array
from jax.scipy.linalg import expm as jax_expm
from quil.expression import Expression as QuilExpression
from quil.instructions import Instruction as RSInstruction
from scipy.linalg import logm as scipy_logm
from scipy.optimize import brentq

from pyquil.quilatom import Expression, FormalArgument, MemoryReference, Parameter, ParameterDesignator, substitute
from pyquil.quilbase import DefCircuit, DefGate, Gate, Measurement, Reset, ResetQubit, _integer_base_and_exponent

if TYPE_CHECKING:
    from plotly.graph_objs import Figure

    from pyquil import Program

logger = logging.getLogger(__name__)

_DEFAULT_GATE_TIME = 1.0
"""Default evolution time for Lindbladian-backed channels.

``1.0`` is *dimensionless*: the generator's rates are interpreted per unit gate, so a
depolarizing rate of ``0.01`` means "1% per gate". Pass a physical duration instead (e.g.
``40e-9`` seconds) when the Hamiltonian and jump operators carry physical units.
"""

_IMAGINARY_PARAM_TOLERANCE = 1e-12
"""Largest imaginary part tolerated in a gate parameter before it is rejected."""

CustomGateMap: TypeAlias = dict[str, qx.Unitary | Callable[..., qx.Unitary]]
"""Lookup map from gate name to a unitary (fixed gates) or a factory (parametric gates).

The callable form is deliberately left as ``Callable[..., qx.Unitary]``: quax gate factories
are not uniformly positional-only, so a stricter signature would not hold for every entry.
"""


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


class PackedArray(TypedDict):
    """JSON payload for a complex array: flattened real/imaginary pairs plus its shape."""

    _complex_array: list[list[float]]
    shape: list[int]


class PackedOperator(PackedArray):
    """JSON payload for a quax operator: a :class:`PackedArray` plus its per-subsystem dims."""

    dims: list[list[int]]


def _pack_complex_array(array: Array | np.ndarray) -> PackedArray:
    """Pack a complex array into JSON-compatible real/imaginary pairs."""
    np_array = np.asarray(array)
    return {
        "_complex_array": [[float(value.real), float(value.imag)] for value in np_array.flat],
        "shape": list(np_array.shape),
    }


def _unpack_complex_array(data: PackedArray) -> Array:
    """Unpack a complex array from :func:`_pack_complex_array` data."""
    shape = tuple(data["shape"])
    return jnp.array([complex(*pair) for pair in data["_complex_array"]], dtype=complex).reshape(shape)


def _unpack_dims(data: list[list[int]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Unpack (and validate) quax operator dims from their JSON representation."""
    if len(data) != 2:
        raise ValueError(f"Serialized operator dims must contain output and input dims, got {data}.")
    return (tuple(int(dim) for dim in data[0]), tuple(int(dim) for dim in data[1]))


def _pack_operator(operator: qx.SuperOp | qx.Unitary | qx.Choi | qx.Observable | qx.Operator) -> PackedOperator:
    """Pack a quax operator matrix with explicit dimension metadata.

    ``dims`` is a plain tuple of tuples, which :func:`json.dumps` already encodes as nested
    lists, so no separate packing step is needed.
    """
    data = _pack_complex_array(operator.matrix)
    return {**data, "dims": [list(operator.dims[0]), list(operator.dims[1])]}


def _evaluate_parameter_designators(params: Sequence[ParameterDesignator]) -> list[float]:
    """Evaluate gate parameters to concrete float values.

    This performs *arithmetic* evaluation only — it collapses expressions such as ``pi / 2`` to a
    number. It does not substitute free parameters: a gate still carrying an unbound
    ``%theta`` cannot be evaluated and is rejected. Substitute parameters first (e.g. with
    :func:`pyquil.quilatom.substitute`) if the gate is parametric.

    Gate parameters are required to be real, matching the standard gate set and the wider
    literature. A parameter with an imaginary part beyond :data:`_IMAGINARY_PARAM_TOLERANCE` is
    a caller error and is rejected rather than silently truncated.

    :param params: The gate parameters (numbers, or expressions over them).
    :return: A list of concrete float values.
    :raises ValueError: If a parameter is unbound, or has a non-negligible imaginary part.
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

        imaginary_part = float(np.imag(value))
        if abs(imaginary_part) > _IMAGINARY_PARAM_TOLERANCE:
            raise ValueError(
                f"Gate parameter {p!r} has a non-negligible imaginary part ({imaginary_part!r}). "
                "Gate parameters must be real."
            )
        fixed_params.append(float(np.real(value)))
    return fixed_params


def _operator_dims_from_dimension(dimension: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Infer square operator dims ``((d,)*k, (d,)*k)`` from a matrix dimension ``d**k``.

    Supports qudits: the base ``d`` is the qudit dimension (2 for qubits, 3 for qutrits, ...)
    and ``k`` the number of qudits. Raises if ``dimension`` is not a prime power.
    """
    decomposition = _integer_base_and_exponent(dimension)
    if decomposition is None:
        raise ValueError(f"Matrix dimension {dimension} is not a prime power; cannot infer qudit dims.")
    base, exponent = decomposition
    return ((base,) * exponent, (base,) * exponent)


def _require_qubit_dims(dims: tuple[int, ...], context: str) -> int:
    """Assert that every subsystem is a qubit, returning the number of qubits.

    The Pauli basis used by the Pauli-flavored constructors and analyses is defined for
    two-level systems only. Rather than let a qudit fall through to an opaque array-shape
    error, reject it here with a message that names the offending operation.
    """
    if any(dim != 2 for dim in dims):
        raise ValueError(
            f"{context} is defined for qubits only, but the instruction acts on subsystems with "
            f"dimensions {dims}. Use a Lindbladian constructor such as "
            "Channel.from_depolarizing_constant or Channel.from_lindbladian for qudits."
        )
    return len(dims)


def _pure_dephasing_times(t1s: Sequence[float], t2s: Sequence[float]) -> list[float]:
    r"""Convert coherence times :math:`(T_1, T_2)` to pure-dephasing times :math:`T_\varphi`.

    ``quax.lindbladians.thermal_relaxation`` takes :math:`T_\varphi`, not the :math:`T_2` that
    hardware reports. The two are related by

    .. math:: \frac{1}{T_2} = \frac{1}{2 T_1} + \frac{1}{T_\varphi}
        \quad\Longrightarrow\quad T_\varphi = \frac{1}{1/T_2 - 1/(2 T_1)}.

    Note the factor of two on :math:`T_1`: :math:`T_2 = 2 T_1` means *no* pure dephasing
    (:math:`T_\varphi = \infty`), not infinite dephasing.

    :raises ValueError: If any :math:`T_1` or :math:`T_2` is non-positive, or any
        :math:`T_2 > 2 T_1`, for which the pure-dephasing rate would be negative and unphysical.
    """
    tphis: list[float] = []
    for t1, t2 in zip(t1s, t2s, strict=True):
        if t1 <= 0.0 or t2 <= 0.0:
            raise ValueError(f"Coherence times must be positive, got T1={t1}, T2={t2}.")
        dephasing_rate = 1.0 / t2 - 1.0 / (2.0 * t1)
        if dephasing_rate < 0.0:
            raise ValueError(
                f"T2 must not exceed 2*T1 (got T1={t1}, T2={t2}, so 2*T1={2 * t1}); the implied "
                "pure-dephasing rate would be negative and the channel non-physical."
            )
        # T2 == 2*T1 exactly means no pure dephasing at all: an infinite Tphi, whose jump
        # operator sqrt(1 / (2 Tphi)) Z vanishes.
        tphis.append(float("inf") if dephasing_rate == 0.0 else 1.0 / dephasing_rate)
    return tphis


# Grid resolution used to bracket the rotation angle in :func:`_random_coherent_error_unitary`.
# The angle is searched over [0, 2*pi] because a spectral-norm-1 generator has eigenvalues in
# [-1, 1], so |Tr(exp(-i*theta*H))/d| completes at least one full oscillation over that range.
_COHERENT_ERROR_ANGLE_GRID = 512
_COHERENT_ERROR_MAX_ANGLE = 2 * np.pi


def _random_coherent_error_unitary(
    num_qubits: int, dimension: int, process_fidelity: float, rng: np.random.Generator
) -> Array:
    r"""Draw a random unitary error with exactly the requested process fidelity to the identity.

    Builds a random traceless Hermitian generator :math:`H` from the non-identity Paulis,
    normalizes it to unit spectral norm, and solves :math:`|\mathrm{Tr}(e^{-i\theta H})/d|^2 = F`
    for the smallest positive :math:`\theta` by bracketing on a grid and refining with Brent's
    method. Solving numerically rather than in closed form makes the achieved fidelity exact for
    any number of qubits.
    """
    if process_fidelity >= 1.0:
        return jnp.eye(dimension, dtype=complex)

    pauli_matrices = qx.ensembles.PAULIS.matrix  # shape (4, 2, 2): I, X, Y, Z
    # Skip the all-identity product: it only contributes a global phase.
    pauli_products = list(itertools.product(pauli_matrices, repeat=num_qubits))[1:]
    coefficients = rng.random(len(pauli_products))
    coefficients = coefficients / np.sqrt(np.sum(np.square(coefficients)))

    generator = jnp.zeros((dimension, dimension), dtype=complex)
    for paulis, coefficient in zip(pauli_products, coefficients, strict=True):
        generator = generator + reduce(jnp.kron, paulis) * coefficient
    # Unit spectral norm puts the eigenvalues in [-1, 1], making the angle grid below meaningful
    # regardless of how many qubits the gate acts on.
    generator = generator / jnp.linalg.norm(generator, 2)

    def achieved_fidelity(theta: float) -> float:
        error_unitary = jax_expm(-1j * theta * generator)
        return float(jnp.abs(jnp.trace(error_unitary) / dimension) ** 2)

    # Walk out along the angle until the achieved fidelity crosses the target, then refine. The
    # first crossing gives the error closest to the identity.
    angles = np.linspace(0.0, _COHERENT_ERROR_MAX_ANGLE, _COHERENT_ERROR_ANGLE_GRID)
    previous = 1.0
    for low, high in zip(angles[:-1], angles[1:], strict=True):
        current = achieved_fidelity(float(high))
        if current <= process_fidelity <= previous:
            # brentq returns a bare root unless full_output=True, which we do not request.
            root = cast(float, brentq(lambda t: achieved_fidelity(t) - process_fidelity, low, high, xtol=1e-14))
            return jnp.asarray(jax_expm(-1j * float(root) * generator))
        previous = current

    raise ValueError(
        f"Could not reach a process fidelity of {process_fidelity} with the drawn error direction; "
        "the requested fidelity may be unreachably low for this number of qubits."
    )


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
                return qx.Unitary.from_matrix(matrix, _operator_dims_from_dimension(matrix.shape[0]))

            custom_gates[defgate.name] = parametric_gate
        else:
            matrix = jnp.asarray(defgate.matrix, dtype=complex)
            custom_gates[defgate.name] = qx.Unitary.from_matrix(matrix, _operator_dims_from_dimension(matrix.shape[0]))
    return custom_gates


def get_instruction_unitary(
    inst: Gate,
    custom_gates: CustomGateMap | None = None,
) -> qx.Unitary:
    """Get the unitary matrix associated with a gate instruction.

    Looks up the gate by name — first in ``custom_gates`` (if provided), then in the
    standard quax gate table ``qx.gates.QUANTUM_GATES``. Parametric gates are supported
    provided all parameters are concrete numeric values.

    ``DAGGER`` modifiers are applied (an odd number of them conjugate-transposes the
    result). ``CONTROLLED`` and ``FORKED`` are rejected: each adds a qudit to the
    instruction, so honouring them means building a larger operator than the named gate,
    which is not yet implemented.

    :param inst: The gate instruction.
    :param custom_gates: Optional dictionary of additional gate definitions (e.g. from
        :func:`get_custom_gates_from_program`). Takes precedence over the standard gate set.
    :return: The unitary matrix, including the effect of any ``DAGGER`` modifiers.
    :raises ValueError: If any gate parameter is symbolic, or the gate carries a
        ``CONTROLLED`` or ``FORKED`` modifier.
    :raises KeyError: If the gate name is not found in either the custom or standard gate set.
    """
    name = inst.name
    dagger_count = _validate_and_count_dagger_modifiers(inst)

    # Look up gate definition: custom gates take precedence
    if custom_gates is not None and name in custom_gates:
        gate_def = custom_gates[name]
    elif name in qx.gates.QUANTUM_GATES:
        gate_def = qx.gates.QUANTUM_GATES[name]
    else:
        raise KeyError(f"Unknown gate '{name}'. Provide it via custom_gates (e.g. custom_gates={{'{name}': matrix}}).")

    if inst.params:
        fixed_params = _evaluate_parameter_designators(inst.params)
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

    # DAGGER is an involution, so only the parity matters.
    if dagger_count % 2 == 1:
        result = qx.Unitary.from_matrix(result.h.matrix, result.dims)
    return result


def _validate_and_count_dagger_modifiers(inst: Gate) -> int:
    """Return the number of ``DAGGER`` modifiers on *inst*, rejecting the others.

    ``CONTROLLED`` and ``FORKED`` each consume an extra qudit from the front of the
    instruction's qubit list, so the operator they denote is larger than the named gate's
    and cannot be produced by a table lookup alone. Rejecting them explicitly keeps an
    unsupported modifier from being silently dropped, which would return the *unmodified*
    gate and quietly simulate the wrong circuit.

    :raises ValueError: If ``inst`` carries a ``CONTROLLED`` or ``FORKED`` modifier.
    """
    dagger_count = 0
    for modifier in inst.modifiers:
        name = str(modifier).upper()
        if name == "DAGGER":
            dagger_count += 1
        else:
            raise ValueError(
                f"Gate modifier {name} is not supported (in {inst.out()!r}). Only DAGGER is "
                "implemented; CONTROLLED and FORKED add a qudit to the instruction and need a "
                "larger operator than the named gate provides. Expand the modifier into an "
                "explicit gate definition instead."
            )
    return dagger_count


class ChannelBase(ABC):
    """Shared behavior for noise channels backed by a superoperator ``process``.

    This is the base of every *gate* channel — :class:`Channel` and
    :class:`SuperopChannel` — and is the type to branch on when handling a gate channel
    generically::

        if isinstance(channel, ChannelBase):
            apply(channel.process)

    Testing against a concrete class instead silently misses the other one: most channel
    operations (``@``, :meth:`pauli_twirl`, :meth:`to_coherent_channel`,
    :meth:`to_stochastic_channel`) return a :class:`SuperopChannel`, so a consumer that only
    checks for :class:`Channel` would treat their output as noiseless.

    Most noisy operations in quantum programs can be represented as superoperators,
    including all Gates and Resets.

    A gate channel attaches a CPTP ``process`` (a ``qx.SuperOp`` that *includes* the ideal
    gate) to a :class:`~pyquil.quilbase.Gate`, with fidelity metrics measured against the
    ``ideal_unitary``. Concrete subclasses supply the three attributes below; the
    ``process`` may be a stored field (:class:`SuperopChannel`) or derived from a generator
    (:class:`Channel`). Every method here depends only on those three attributes,
    so it works identically regardless of how ``process`` is produced.

    Operations that leave the superoperator/Lindbladian structure — composition (``@``),
    :meth:`pauli_twirl`, :meth:`to_coherent_channel`, :meth:`to_stochastic_channel` — return
    a plain :class:`SuperopChannel`, since their result is a generic superoperator and not
    necessarily a Lindbladian generator.
    """

    # Provided by concrete subclasses (as dataclass fields or cached properties).
    #
    # These are declared under ``TYPE_CHECKING`` rather than as abstract properties on purpose.
    # An abstract property here would collide with the dataclass fields on SuperopChannel: the
    # inherited ``property`` object becomes the field's default value, and instantiation then
    # fails with "Can't instantiate abstract class ... without an implementation for abstract
    # method 'process'". Abstract properties would work for Channel (where ``process`` is a
    # cached_property) but not for SuperopChannel, so the base cannot require them.
    if TYPE_CHECKING:
        inst: Gate
        process: qx.SuperOp
        ideal_unitary: qx.Unitary

    # ──────────────────────────────────────────────
    # Serialization contract (enforced on every concrete channel)
    # ──────────────────────────────────────────────

    @abstractmethod
    def to_json(self) -> str:
        """Serialize the channel to a JSON string."""
        ...

    @classmethod
    @abstractmethod
    def from_json(cls, json_str: str) -> ChannelBase:
        """Deserialize a channel from a JSON string produced by :meth:`to_json`."""
        ...

    @property
    def qubits(self) -> list[int]:
        """The qubits which the channel applies to."""
        return self.inst.get_qubit_indices()

    @property
    def num_qubits(self) -> int:
        """The number of qubits the channel acts on."""
        return len(self.qubits)

    @property
    def dims(self) -> tuple[int, ...]:
        """Per-subsystem Hilbert-space dimensions, e.g. ``(2, 2)`` for a two-qubit gate."""
        return tuple(self.ideal_unitary.dims[0])

    # ──────────────────────────────────────────────
    # Cached representation conversions
    # ──────────────────────────────────────────────

    @cached_property
    def error_process(self) -> qx.SuperOp:
        r"""The error, as a superoperator applied *after* the ideal gate.

        Where :attr:`process` is the *full* noisy operation (it includes the ideal gate and
        replaces it in a circuit), this is the error alone.

        A noisy gate channel can be viewed either as noise applied after the ideal gate
        (*post-gate*) or before it (*pre-gate*):

        - **post-gate**: :math:`\mathcal{E} = \Lambda_{\text{post}} \circ \mathcal{U}`, so
          :math:`\Lambda_{\text{post}} = \mathcal{E} \circ \mathcal{U}^\dagger` — this property.
        - **pre-gate**: :math:`\mathcal{E} = \mathcal{U} \circ \Lambda_{\text{pre}}`, so
          :math:`\Lambda_{\text{pre}} = \mathcal{U}^\dagger \circ \mathcal{E}`.

        The two coincide only when the error commutes with the gate; in general they are
        related by conjugation, :math:`\Lambda_{\text{post}} = \mathcal{U} \circ
        \Lambda_{\text{pre}} \circ \mathcal{U}^\dagger`, and share the same fidelity metrics.
        This channel adopts the post-gate convention throughout.
        """
        return qx.to_superop(self.process @ self.ideal_unitary.h)

    # ──────────────────────────────────────────────
    # Fidelity properties
    # ──────────────────────────────────────────────

    @cached_property
    def average_gate_fidelity(self) -> float:
        r"""Average gate fidelity :math:`F_{\mathrm{avg}}` of the channel relative to the ideal gate."""
        return float(qx.process_fidelity_to_average_fidelity(self.process_fidelity, dims=self.dims))

    @cached_property
    def average_gate_infidelity(self) -> float:
        r"""Average gate infidelity :math:`1 - F_{\mathrm{avg}}`."""
        return 1.0 - self.average_gate_fidelity

    @cached_property
    def process_fidelity(self) -> float:
        """Process fidelity (entanglement fidelity) :math:`F_e` relative to the ideal gate.

        Also called the Pauli fidelity. Named to match ``quax.process_fidelity``.
        """
        process, unitary = qx.promote_hilbert_space(self.process, qx.to_superop(self.ideal_unitary))
        return float(qx.process_fidelity(process, unitary))

    @cached_property
    def process_infidelity(self) -> float:
        """Process infidelity :math:`1 - F_e`."""
        return 1.0 - self.process_fidelity

    @cached_property
    def stochastic_infidelity(self) -> float:
        """Stochastic (incoherent) component of the process infidelity."""
        return float(qx.stochastic_infidelity(self.error_process))

    @cached_property
    def stochastic_fidelity(self) -> float:
        """Stochastic fidelity :math:`1 - e_S`."""
        return 1.0 - self.stochastic_infidelity

    @cached_property
    def coherent_infidelity(self) -> float:
        """Coherent component of the process infidelity: :math:`e_C = e - e_S`."""
        return self.process_infidelity - self.stochastic_infidelity

    @cached_property
    def coherent_fidelity(self) -> float:
        """Coherent fidelity :math:`1 - e_C`."""
        return 1.0 - self.coherent_infidelity

    @cached_property
    def unitarity(self) -> float:
        """Unitarity of the channel."""
        return float(qx.unitarity(self.error_process))

    # ──────────────────────────────────────────────
    # SuperopChannel analysis methods
    # ──────────────────────────────────────────────

    def _to_superop_channel(self, process: qx.SuperOp) -> SuperopChannel:
        """Wrap a derived superoperator as a plain :class:`SuperopChannel` for this gate."""
        return SuperopChannel(inst=self.inst, process=process, ideal_unitary=self.ideal_unitary)

    def _unitary_error(self) -> qx.Unitary:
        """Return the dominant coherent error as a quax ``Unitary`` on this channel's subsystems."""
        return qx.Unitary.from_matrix(self._unitary_error_component, self.process.dims)

    def pauli_twirl(self) -> SuperopChannel:
        r"""Return a Pauli-twirled version of this channel.

        Twirling is applied to the *error* (:attr:`error_process`), whose Pauli-Liouville
        representation is projected onto its diagonal, and the ideal gate is then recomposed.
        The result is the ideal gate followed by a stochastic Pauli channel carrying the
        original diagonal error rates. Process fidelity is preserved exactly.

        For a Clifford gate this is precisely the standard Pauli twirl of the noisy gate,

        .. math:: \tilde{\mathcal{E}} = \frac{1}{4^n} \sum_k \mathcal{P}'^\dagger_k \circ
            \mathcal{E} \circ \mathcal{P}_k, \qquad P'_k = U P_k U^\dagger

        because :math:`\mathcal{U} \circ \mathcal{P}_k = \mathcal{P}'_k \circ \mathcal{U}` lets the
        gate be pulled out of the average, and for Clifford :math:`U` the conjugated set
        :math:`\{P'_k\}` is again the Pauli group. (Verified numerically to machine precision for
        ``X``, ``H``, ``RX(pi/2)``, ``CZ`` and ``CNOT``.)

        For a **non-Clifford** gate the two differ, since :math:`\{U P_k U^\dagger\}` is no longer
        the Pauli group; this method is then defined as the Pauli twirl of the error, which is the
        quantity randomized compiling actually implements.

        .. note::
            Twirling the full :attr:`process` instead would destroy the gate: zeroing the
            off-diagonal entries of a gate's own transfer matrix does not leave the gate behind.
        """
        ptm = qx.to_pauli_liouville(self.error_process)
        # Keep only the diagonal of the error PTM, then put the ideal gate back.
        twirled_ptm_matrix = jnp.diag(jnp.diag(ptm.matrix))
        twirled_error = qx.PauliLiouville.from_matrix(twirled_ptm_matrix, self.error_process.dims)
        twirled_superop = qx.to_superop(qx.to_superop(twirled_error) @ qx.to_superop(self.ideal_unitary))
        return self._to_superop_channel(twirled_superop)

    @cached_property
    def _unitary_error_component(self) -> Array:
        """Extract the dominant unitary from the error-only channel.

        Uses eigendecomposition + SVD polar decomposition to find the closest
        unitary to the error channel.
        """
        choi_matrix = qx.to_choi(self.error_process).matrix
        d = int(np.prod(self.dims))

        # Dominant eigenvector of the Choi matrix
        eigenvalues, eigenvectors = jnp.linalg.eigh(choi_matrix)
        dominant_eigenvector = eigenvectors[:, jnp.argmax(jnp.abs(eigenvalues))]

        # SVD polar decomposition to extract the closest unitary
        u, _, vh = jnp.linalg.svd(dominant_eigenvector.reshape(d, d).T)
        return u @ vh

    def to_coherent_channel(self) -> SuperopChannel:
        r"""Isolate the coherent (unitary) component of the error.

        Extracts the dominant unitary :math:`U_{\mathrm{err}}` from the error Choi matrix via
        polar decomposition and returns :math:`\mathcal{U}_{\mathrm{err}} \circ
        \mathcal{U}_{\mathrm{gate}}`.

        Together with :meth:`to_stochastic_channel` this splits the channel into its coherent
        and stochastic parts; composing the two errors reproduces the full error.
        """
        coherent_superop = qx.to_superop(self._unitary_error() @ self.ideal_unitary)
        return self._to_superop_channel(coherent_superop)

    def to_stochastic_channel(self) -> SuperopChannel:
        r"""Isolate the stochastic (incoherent) component of the error.

        The full channel decomposes as
        :math:`\mathcal{E} = \mathcal{S} \circ \mathcal{U}_{\mathrm{err}} \circ \mathcal{U}_{\mathrm{gate}}`.
        This method factors out the coherent unitary error and returns
        :math:`\mathcal{S} \circ \mathcal{U}_{\mathrm{gate}}`.
        """
        # S = Lambda_post ∘ U_err†, then recompose with the ideal gate. Composing through quax
        # keeps the superoperator convention consistent; building the Kronecker products by hand
        # here silently used the opposite convention and produced a non-CPTP result.
        stochastic_superop = qx.to_superop(self.error_process @ self._unitary_error().h @ self.ideal_unitary)
        return self._to_superop_channel(stochastic_superop)

    def is_pauli(self) -> bool:
        """Check if the error channel is a Pauli (stochastic Pauli) channel.

        A Pauli channel has a diagonal Pauli transfer matrix (error-only part).
        """
        _require_qubit_dims(self.dims, "is_pauli")
        ptm = qx.to_pauli_liouville(self.error_process).matrix
        mask = ~jnp.eye(ptm.shape[0], dtype=bool)
        return bool(jnp.allclose(ptm[mask], 0))

    def to_pauli_vector(self) -> Array:
        """Project the error channel onto the Pauli basis.

        Returns the coefficient of each Pauli error in lexicographic order
        (II, IX, IY, IZ, XI, XX, ...) — equivalently, the error rates of the
        :meth:`pauli_twirl` of this channel.

        For a Pauli channel (:meth:`is_pauli`) these are genuine probabilities summing to 1.
        For a channel with coherent error they are the magnitudes of quasi-probabilities, so
        treat them as a twirled approximation rather than an exact error budget.
        """
        num_qubits = _require_qubit_dims(self.dims, "to_pauli_vector")
        noise_superop = self.error_process.matrix
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

    def plot(self, only_noise: bool = True, show_identity: bool = False) -> Figure:
        """Plot the Pauli transfer matrix of the channel.

        :param only_noise: If True (default), plot the error-only channel (the post-gate
            error, with the ideal gate unitary factored out; see :attr:`error_process`).
            If False, plot the full channel including the gate unitary.
        :param show_identity: If True, include the identity component in the noise-only plot.
            If False (default), visualize the generator of the noise channel via the matrix
            logarithm of the PTM.  For near-identity noise this approximates PTM - I, but
            correctly captures the Lie-algebraic structure of the channel.
            Only applies when ``only_noise=True``.
        :return: A Plotly Figure.
        """
        if only_noise:
            channel = self.error_process
            if not show_identity:
                ptm = qx.to_pauli_liouville(channel)
                log_ptm = scipy_logm(np.asarray(ptm.matrix))
                channel = qx.PauliLiouville.from_matrix(jnp.array(log_ptm), channel.dims)
            title_prefix = "Error channel"
        else:
            channel = self.process
            title_prefix = "Full channel"

        fig = qx.plot(channel)
        fig.update_layout(
            title=(
                f"{title_prefix} for {self.inst.out()}<br>"
                f"𝜀={self.process_infidelity * 100:.2f}%, "
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
        return f"<{self.inst.out()} ~ ({100 * self.process_fidelity:.2f}%)>"

    def __eq__(self, other: object) -> bool:
        """Check equality by concrete type, instruction, and (approximate) process/unitary matrices.

        Two channels are equal iff they are the same concrete class, share the same instruction,
        and have element-wise-close ``process`` and ``ideal_unitary`` matrices (via
        :func:`jax.numpy.allclose`). The comparison is approximate rather than bit-exact so it
        tolerates the small floating-point differences between otherwise-equivalent constructions.
        """
        if type(self) is not type(other):
            return False
        if self.inst != other.inst:
            return False
        return bool(
            jnp.allclose(self.process.matrix, other.process.matrix)
            and jnp.allclose(self.ideal_unitary.matrix, other.ideal_unitary.matrix)
        )

    __hash__ = None  # type: ignore[assignment]

    def __matmul__(self, other: ChannelBase) -> SuperopChannel:
        r"""Compose two channels: ``channel_B @ channel_A``.

        Both channels share the same gate instruction. The composition factors
        out one copy of the gate unitary so the result represents the sequential
        application of the two noisy processes:

        :math:`\mathcal{E}_B \circ \mathcal{U}^\dagger \circ \mathcal{E}_A`

        This is the natural composition: if ``channel_A`` already includes the
        gate, applying ``channel_B`` after it should not double-count the gate.

        The composition is *exact* — it is carried out on superoperators — and therefore
        returns a plain :class:`SuperopChannel` even when both operands are Lindbladian-backed:
        the composition of two Lindbladian-generated channels is not itself the evolution of a
        Lindbladian generator unless the two generators commute. To combine noise at the
        generator level (which is CPTP-safe and stays a :class:`Channel`), add the channels with
        :meth:`Channel.__add__` instead.
        """
        if not isinstance(other, ChannelBase):
            return NotImplemented
        if self.inst != other.inst:
            raise ValueError(f"Cannot compose channels for different gates: {self.inst.out()} vs {other.inst.out()}")
        # E_B @ U† @ E_A  (factor out one gate unitary between the two channels)
        u_dag_superop = qx.to_superop(self.ideal_unitary.h)
        composed_superop = qx.to_superop(self.process @ u_dag_superop @ other.process)
        return self._to_superop_channel(composed_superop)

    def __or__(self, other: CycleConstituent | CycleChannel) -> CycleChannel:
        """Tensor product with another channel on disjoint qubits, producing a CycleChannel.

        The result represents a cycle containing both operations acting in parallel
        on disjoint qubits. The DefCircuit encodes the parallel operations as
        formal instructions.

        :param other: Another gate, reset, or measurement channel — or a
            :class:`CycleChannel`, whose constituents are flattened in — on disjoint qubits.
        :return: A CycleChannel representing the tensor product.
        """
        return _tensor_into_cycle(self, other)


@final
@dataclass(frozen=True, eq=False)
class SuperopChannel(ChannelBase):
    """A noise channel that stores a superoperator directly, for a specific gate.

    This is the special case of :class:`ChannelBase` whose ``process`` is a stored
    ``qx.SuperOp`` (rather than derived from a Lindbladian generator, as :class:`Channel`). It is
    what the manifold-leaving operations (composition ``@``, :meth:`pauli_twirl`,
    :meth:`to_coherent_channel`, :meth:`to_stochastic_channel`) return, and is useful when only a
    raw superoperator is available. Prefer :class:`Channel` and its ``from_*`` constructors when a
    generator description is available.

    The superoperator *includes* the gate unitary, so the channel replaces the gate rather than
    being applied after it, and can be converted to alternative representations (Choi, Kraus,
    Pauli-Liouville) via ``quax``. Fidelity metrics are computed relative to ``ideal_unitary``,
    and the error alone is available as :attr:`~ChannelBase.error_process`.
    """

    inst: Gate
    """Quil gate to which the channel applies."""

    process: qx.SuperOp
    """The noisy process (superoperator) for the gate, including the gate unitary."""

    ideal_unitary: qx.Unitary
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
            The identity term is implicit: it takes the remaining probability, so passing an
            all-identity key (``"I"``, ``"II"``, ...) is rejected.
        :param pauli_noise: Pauli error probabilities, e.g. ``{"IX": 0.01, "ZZ": 0.02}``. Must sum
            to at most 1.0; the remainder is assigned to the identity (no-error) term.
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A SuperopChannel instance.
        :raises ValueError: If a rate is negative, the rates sum above 1.0, a term has the wrong
            length, or an all-identity term is supplied.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        num_qubits = _require_qubit_dims(unitary.dims[0], "SuperopChannel.from_pauli_noise")

        identity_term = "I" * num_qubits
        total_error_rate = 0.0
        for pauli, error_rate in pauli_noise.items():
            if error_rate < 0.0:
                raise ValueError(f"Pauli term '{pauli}' has negative error rate {error_rate}.")
            if len(pauli) != num_qubits:
                raise ValueError(f"Pauli term '{pauli}' has length {len(pauli)}, expected {num_qubits}.")
            if pauli == identity_term:
                raise ValueError(
                    f"Pauli term '{pauli}' is the identity, whose probability is implicit: it receives "
                    "whatever the given error rates leave over. Pass error terms only."
                )
            total_error_rate += error_rate
        if total_error_rate > 1.0:
            raise ValueError(f"Pauli error rates must sum to at most 1.0, got {total_error_rate}.")

        # Rates in lexicographic (I, X, Y, Z) order, with the identity taking the remainder so the
        # resulting Kraus map is trace-preserving.
        all_pauli_terms = tuple("".join(term) for term in product("IXYZ", repeat=num_qubits))
        pauli_error_rates = [pauli_noise.get(term, 0.0) for term in all_pauli_terms]
        pauli_error_rates[all_pauli_terms.index(identity_term)] = 1.0 - total_error_rate

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
        return cls(inst=inst, process=process_superop, ideal_unitary=unitary)

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
            "unitary": _pack_operator(self.ideal_unitary),
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

        return cls(inst=inst, process=superop, ideal_unitary=unitary)


class _LindbladianBacked(ABC):  # noqa: B024  (abstract mixin; its contract, `lindbladian`/`gate_time`, is supplied as dataclass fields, not @abstractmethods)
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


@final
@dataclass(frozen=True, eq=False)
class Channel(_LindbladianBacked, ChannelBase):
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
    channel reports a full suite of fidelity and error metrics (:attr:`average_gate_fidelity`,
    :attr:`process_fidelity`, :attr:`coherent_infidelity`, :attr:`stochastic_infidelity`,
    :attr:`unitarity`, ...), can be visualized with :meth:`plot`, decomposed into coherent and
    stochastic parts, Pauli-twirled, composed, and serialized.

    Because the noise is stored as a generator, two operations are natural and always CPTP-safe:

    - :meth:`__pow__` scales the *noise* (jump rates and coherent-noise Hamiltonian) while keeping
      the ideal gate, sweeping noise strength in a physically meaningful way.
    - :meth:`__add__` combines the *noise* of two channels on the same gate, keeping the gate.

    Note that ``@`` (:meth:`~ChannelBase.__matmul__`) is different from ``+``: it is the exact
    superoperator composition of the two noisy processes and returns a :class:`SuperopChannel`,
    because composing two Lindbladian evolutions is not itself a Lindbladian evolution unless the
    generators commute.

    ``gate_time`` defaults to :data:`_DEFAULT_GATE_TIME` (dimensionless). It may instead be a
    physical duration (e.g. ``~40e-9`` s), in which case the Hamiltonian and jump operators are in
    physical units; the gate Hamiltonian is scaled so that evolving for ``gate_time`` reproduces
    the ideal ``ideal_unitary``.
    """

    inst: Gate
    """Quil gate to which the channel applies."""

    lindbladian: qx.Lindbladian
    """The GKSL generator for the gate, including the (scaled) gate Hamiltonian."""

    ideal_unitary: qx.Unitary
    """The noiseless unitary of the gate."""

    gate_time: float = _DEFAULT_GATE_TIME
    """Evolution time for ``evolve(lindbladian, gate_time)``. See :data:`_DEFAULT_GATE_TIME`."""

    @cached_property
    def gate_hamiltonian(self) -> qx.Observable:
        """Coherent generator whose evolution over ``gate_time`` yields ``ideal_unitary``."""
        return qx.unitary_to_hamiltonian(self.ideal_unitary) * (1.0 / self.gate_time)

    @cached_property
    def target_lindbladian(self) -> qx.Lindbladian:
        """The ideal gate as a purely-coherent generator (``gate_hamiltonian`` with zero dissipation).

        Together with :attr:`noise_lindbladian` this decomposes the full ``lindbladian`` into its
        target (gate) and noise parts: ``lindbladian == noise_lindbladian + target_lindbladian``.
        """
        d = int(np.prod(self.dims))
        zero_jumps = qx.Operator.from_matrix(jnp.zeros((1, d, d), dtype=complex), self.ideal_unitary.dims)
        return qx.Lindbladian(hamiltonian=self.gate_hamiltonian, jump_operators=zero_jumps)

    @cached_property
    def noise_lindbladian(self) -> qx.Lindbladian:
        """The generator with the coherent gate Hamiltonian factored out (dissipation + coherent noise).

        Together with :attr:`target_lindbladian` this decomposes the full ``lindbladian`` into its
        target (gate) and noise parts: ``lindbladian == noise_lindbladian + target_lindbladian``.
        """
        hamiltonian = self.lindbladian.hamiltonian
        noise_hamiltonian = hamiltonian - self.gate_hamiltonian if hamiltonian is not None else None
        return qx.Lindbladian(hamiltonian=noise_hamiltonian, jump_operators=self.lindbladian.jump_operators)

    # ──────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────

    @classmethod
    def _from_noise_lindbladian(
        cls: type[Channel],
        inst: Gate,
        unitary: qx.Unitary,
        noise_lindbladian: qx.Lindbladian,
        gate_time: float,
    ) -> Channel:
        """Fold an already-resolved gate unitary and a noise generator into a channel.

        The shared tail of every ``from_*`` constructor. Taking the resolved ``unitary`` means the
        gate is looked up exactly once per construction, rather than once per delegation hop.
        """
        gate_hamiltonian = qx.unitary_to_hamiltonian(unitary) * (1.0 / gate_time)
        noise_hamiltonian = noise_lindbladian.hamiltonian
        total_hamiltonian = gate_hamiltonian if noise_hamiltonian is None else noise_hamiltonian + gate_hamiltonian
        lindbladian = qx.Lindbladian(hamiltonian=total_hamiltonian, jump_operators=noise_lindbladian.jump_operators)
        return cls(inst=inst, lindbladian=lindbladian, ideal_unitary=unitary, gate_time=gate_time)

    @classmethod
    def from_lindbladian(
        cls: type[Channel],
        inst: Gate,
        noise_lindbladian: qx.Lindbladian,
        gate_time: float = _DEFAULT_GATE_TIME,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        """Create a channel from a noise-only Lindbladian, folding in the gate.

        :param inst: The gate to which the channel applies.
        :param noise_lindbladian: The noise generator (e.g. from ``qx.lindbladians``), *without*
            the gate Hamiltonian. Its rates are interpreted per unit time and evolved for ``gate_time``.
        :param gate_time: Evolution time (see :data:`_DEFAULT_GATE_TIME`).
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        return cls._from_noise_lindbladian(inst, unitary, noise_lindbladian, gate_time)

    @classmethod
    def from_gate_fidelity(
        cls: type[Channel],
        inst: Gate,
        fidelity: float,
        gate_time: float = _DEFAULT_GATE_TIME,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        """Create a depolarizing Lindbladian channel from an average gate fidelity."""
        unitary = get_instruction_unitary(inst, custom_gates)
        p = float(qx.average_fidelity_to_depolarizing_constant(fidelity, unitary.dims[0]))
        return cls._from_depolarizing_constant(inst, unitary, p, gate_time)

    @classmethod
    def from_pauli_fidelity(
        cls: type[Channel],
        inst: Gate,
        pauli_fidelity: float,
        gate_time: float = _DEFAULT_GATE_TIME,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        """Create a depolarizing Lindbladian channel from a process (Pauli) fidelity."""
        unitary = get_instruction_unitary(inst, custom_gates)
        p = float(qx.process_fidelity_to_depolarizing_constant(pauli_fidelity, unitary.dims[0]))
        return cls._from_depolarizing_constant(inst, unitary, p, gate_time)

    @classmethod
    def _from_depolarizing_constant(
        cls: type[Channel],
        inst: Gate,
        unitary: qx.Unitary,
        depolarizing_constant: float,
        gate_time: float,
    ) -> Channel:
        """Build a depolarizing channel from an already-resolved unitary."""
        dims = unitary.dims[0]
        d = int(np.prod(dims))
        if not 0.0 <= depolarizing_constant <= 1.0:
            raise ValueError(
                f"Depolarizing constant must lie in [0, 1], got {depolarizing_constant}. "
                "It is the factor by which every traceless operator shrinks: 1.0 is noiseless, "
                "0.0 is complete depolarization."
            )
        # Floor at the smallest positive float so p == 0 (complete depolarization) yields a large
        # finite rate rather than log(0).
        shrink = max(depolarizing_constant, np.finfo(float).tiny)
        gamma_unit_time = -np.log(shrink) * (d**2 - 1) / d**2
        noise = qx.lindbladians.depolarizing(gamma_unit_time / gate_time, dims)
        return cls._from_noise_lindbladian(inst, unitary, noise, gate_time)

    @classmethod
    def from_depolarizing_constant(
        cls: type[Channel],
        inst: Gate,
        depolarizing_constant: float,
        gate_time: float = _DEFAULT_GATE_TIME,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        r"""Create a depolarizing Lindbladian channel from a depolarization constant.

        The depolarizing constant :math:`p` parameterizes
        :math:`\mathcal{D}_p(\rho) = p\,\rho + (1-p)\,I/d`. The rate is chosen so that
        evolving for ``gate_time`` shrinks every traceless operator by exactly ``p``.

        :param inst: The gate to which the channel applies.
        :param depolarizing_constant: The shrink factor :math:`p \in [0, 1]`; 1.0 is noiseless.
        :param gate_time: Evolution time (see :data:`_DEFAULT_GATE_TIME`).
        :param custom_gates: Optional dictionary of custom gate definitions.
        :raises ValueError: If ``depolarizing_constant`` lies outside :math:`[0, 1]`.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        return cls._from_depolarizing_constant(inst, unitary, depolarizing_constant, gate_time)

    @classmethod
    def from_coherence_times(
        cls: type[Channel],
        inst: Gate,
        gate_duration: float,
        t1s: list[float],
        t2s: list[float] | None = None,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        r"""Create a decoherence channel from T1/T2 coherence times, evolved over ``gate_duration``.

        The gate and thermal relaxation happen together over ``gate_duration`` (``gate_time`` is set
        to ``gate_duration``). Physical rates come from ``qx.lindbladians.thermal_relaxation``,
        which takes the *pure-dephasing* time :math:`T_\varphi` rather than :math:`T_2`; the
        conversion is :math:`T_\varphi = 1/(1/T_2 - 1/(2 T_1))` (see :func:`_pure_dephasing_times`).

        :param inst: The target instruction.
        :param gate_duration: The duration of the gate (used as ``gate_time``).
        :param t1s: The T1 time(s) of the qudits.
        :param t2s: The T2 time(s) of the qudits. Defaults to ``2 * t1`` (no pure dephasing).
        :raises ValueError: If the number of T1/T2 values does not match the instruction's qudits,
            or if any :math:`T_2 > 2 T_1`.
        """
        qubits = inst.get_qubit_indices()
        num_sys = len(qubits)
        if num_sys != len(t1s):
            raise ValueError(f"Expected {num_sys} T1 values for {inst.out()}, got {len(t1s)}.")
        if t2s is None:
            t2s = [2 * t1 for t1 in t1s]
        elif num_sys != len(t2s):
            raise ValueError(f"Expected {num_sys} T2 values for {inst.out()}, got {len(t2s)}.")

        per_qubit = [
            qx.lindbladians.thermal_relaxation(t1, tphi)
            for t1, tphi in zip(t1s, _pure_dephasing_times(t1s, t2s), strict=True)
        ]
        noise = reduce(lambda a, b: a | b, per_qubit)
        unitary = get_instruction_unitary(inst, custom_gates)
        return cls._from_noise_lindbladian(inst, unitary, noise, gate_duration)

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
        :param gate_time: Evolution time (see :data:`_DEFAULT_GATE_TIME`).
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        :raises ValueError: If the gate acts on non-qubit subsystems, or a rate is negative.
        """
        unitary = get_instruction_unitary(inst, custom_gates)
        num_qubits = _require_qubit_dims(unitary.dims[0], "Channel.from_pauli_generators")
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

        d = int(np.prod(unitary.dims[0]))
        stacked = jnp.stack(jump_matrices) if jump_matrices else jnp.zeros((1, d, d), dtype=complex)
        jump_operators = qx.Operator.from_matrix(stacked, unitary.dims)
        noise = qx.Lindbladian(hamiltonian=None, jump_operators=jump_operators)
        return cls._from_noise_lindbladian(inst, unitary, noise, gate_time)

    @classmethod
    def from_mixture(
        cls: type[Channel],
        inst: Gate,
        constituents: list[qx.Unitary],
        rates: list[float],
        gate_time: float = _DEFAULT_GATE_TIME,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        r"""Create a channel whose error is generated by a set of unitary jump operators.

        Each error unitary :math:`V_i` with rate :math:`r_i` contributes a jump operator
        :math:`\sqrt{r_i}\,V_i` to the generator, so this is the continuously-generated
        counterpart of a mixed-unitary channel, in the same sense that
        :meth:`from_pauli_generators` is the counterpart of
        :meth:`SuperopChannel.from_pauli_noise`.

        .. note::
            This is *not* the one-shot mixture
            :math:`\sum_i r_i V_i \rho V_i^\dagger + (1 - \sum_i r_i)\rho`. Exponentiating the
            generator produces a Poissonian sum over *products* of the :math:`V_i`, so the two
            agree only to first order in the rates. For an exact mixed-Pauli channel use
            :meth:`SuperopChannel.from_pauli_noise`.

        :param inst: The gate to which the channel applies.
        :param constituents: Unitary error operators.
        :param rates: Generator rate of each unitary error, per unit time.
        :param gate_time: Evolution time (see :data:`_DEFAULT_GATE_TIME`).
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        :raises ValueError: If the lengths disagree, a rate is negative, or a constituent's dims
            do not match the gate's.
        """
        ideal = get_instruction_unitary(inst, custom_gates)
        if len(constituents) != len(rates):
            raise ValueError(f"Got {len(constituents)} constituents but {len(rates)} rates; they must match.")
        if any(r < 0.0 for r in rates):
            raise ValueError("Mixture rates must be non-negative.")
        for constituent in constituents:
            if constituent.dims != ideal.dims:
                raise ValueError(
                    f"Constituent unitary has dims {constituent.dims}, expected {ideal.dims} " f"to match {inst.out()}."
                )

        d = int(np.prod(ideal.dims[0]))
        jump_matrices = [jnp.sqrt(r) * v.matrix for r, v in zip(rates, constituents, strict=True)]
        stacked = jnp.stack(jump_matrices) if jump_matrices else jnp.zeros((1, d, d), dtype=complex)
        jump_operators = qx.Operator.from_matrix(stacked, ideal.dims)
        noise = qx.Lindbladian(hamiltonian=None, jump_operators=jump_operators)
        return cls._from_noise_lindbladian(inst, ideal, noise, gate_time)

    @classmethod
    def from_random_coherent_error(
        cls: type[Channel],
        inst: Gate,
        process_fidelity: float,
        rng: np.random.Generator | None = None,
        gate_time: float = _DEFAULT_GATE_TIME,
        custom_gates: CustomGateMap | None = None,
    ) -> Channel:
        r"""Create a channel with a random coherent (unitary) error at the specified process fidelity.

        A random traceless Hermitian direction :math:`H` is drawn over the non-identity Paulis and
        normalized; the error is :math:`V(\theta) = \exp(-i\theta H)` with :math:`\theta` chosen so
        that :math:`|\mathrm{Tr}(V)/d|^2` equals ``process_fidelity`` exactly. The smallest such
        :math:`\theta` is taken, keeping the error as close to the identity as the target allows.
        The result is a purely coherent :class:`Channel` (Hamiltonian only, no jump operators).

        :math:`\theta` is found numerically. A closed form exists only for a single qudit, where
        the three non-identity Paulis anticommute so that :math:`H^2 \propto I`; for multi-qubit
        gates the eigenvalues of :math:`H` depend on the drawn direction, and the closed form
        misses the target by roughly 1% of the infidelity.

        :param inst: The gate to which the channel applies.
        :param process_fidelity: The process fidelity of the coherent error, :math:`F_e \in (0, 1]`.
        :param rng: NumPy random number generator for reproducibility.
        :param gate_time: Evolution time (see :data:`_DEFAULT_GATE_TIME`).
        :param custom_gates: Optional dictionary of custom gate definitions.
        :return: A Channel instance.
        :raises ValueError: If the gate acts on non-qubit subsystems, if ``process_fidelity`` lies
            outside :math:`(0, 1]`, or if no rotation angle reaches the requested fidelity.
        """
        if rng is None:
            rng = np.random.default_rng()

        ideal = get_instruction_unitary(inst, custom_gates)
        num_qubits = _require_qubit_dims(ideal.dims[0], "Channel.from_random_coherent_error")
        d = int(np.prod(ideal.dims[0]))

        if not 0.0 < process_fidelity <= 1.0:
            raise ValueError(f"process_fidelity must lie in (0, 1], got {process_fidelity}.")

        error_unitary = _random_coherent_error_unitary(num_qubits, d, process_fidelity, rng)

        noisy_unitary = qx.Unitary.from_matrix(error_unitary @ ideal.matrix, ideal.dims)
        hamiltonian = qx.unitary_to_hamiltonian(noisy_unitary) * (1.0 / gate_time)
        zero_jumps = qx.Operator.from_matrix(jnp.zeros((1, d, d), dtype=complex), ideal.dims)
        lindbladian = qx.Lindbladian(hamiltonian=hamiltonian, jump_operators=zero_jumps)
        return cls(inst=inst, lindbladian=lindbladian, ideal_unitary=ideal, gate_time=gate_time)

    # ──────────────────────────────────────────────
    # Lindbladian-native operations
    # ──────────────────────────────────────────────

    def __pow__(self, power: float) -> Channel:
        """Scale the noise to a (non-negative) ``power`` while preserving the gate.

        ``power = 0`` yields the ideal gate, ``1`` leaves the channel unchanged, and ``> 1``
        strengthens the noise. Unlike a fractional matrix power of a superoperator, this is always
        CPTP because it acts on the generator's jump operators and coherent-noise Hamiltonian.

        .. note::
            This scales the noise; it is not the matrix power of the channel's superoperator
            (which is not CPTP in general, and not even well defined for a channel that is not
            infinitely divisible). Consistently with :meth:`__add__`, ``channel ** 2`` is the
            channel with twice the noise, i.e. ``channel + channel``.
        """
        if not isinstance(power, (int, float)):
            return NotImplemented
        scaled = self._scaled_noise_generator(power, gate_hamiltonian=self.gate_hamiltonian)
        return replace(self, lindbladian=scaled)

    def __add__(self, other: Channel) -> Channel:
        r"""Combine the *noise* of two channels on the same gate, keeping the gate.

        The gate Hamiltonian is factored out of each operand (via :attr:`noise_lindbladian`), the
        noise generators are summed (jump operators concatenated, coherent-noise Hamiltonians
        added), and the gate is folded back in. Adding two ``RX(pi/2)`` channels therefore yields an
        ``RX(pi/2)`` channel whose noise is the union of the two. The result is always CPTP and
        stays a :class:`Channel`.

        This is a generator-level operation and is *not* the same as composing the two channels
        with ``@``. Composition applies one process after the other,
        :math:`\mathcal{E}_B \circ \mathcal{U}^\dagger \circ \mathcal{E}_A`, which equals
        :math:`\mathrm{evolve}(L_A + L_B - H_{\mathrm{gate}}, t)` only when the two generators
        commute. Use ``+`` to describe a gate subject to several noise mechanisms at once, and
        ``@`` to describe one noisy operation followed by another.

        Both operands must share the same ``gate_time``: jump operators carry rates *per unit
        time*, so adding generators calibrated against different durations would silently rescale
        one operand's noise.

        :raises ValueError: If the gates or the ``gate_time`` values differ.
        """
        if not isinstance(other, Channel):
            return NotImplemented
        if self.inst != other.inst:
            raise ValueError(f"Cannot combine channels for different gates: {self.inst.out()} vs {other.inst.out()}")
        if self.gate_time != other.gate_time:
            raise ValueError(
                f"Cannot combine channels with different gate times ({self.gate_time} vs {other.gate_time}): "
                "jump operators are per-unit-time rates, so the sum would reinterpret one operand's "
                "noise in the other's time base. Rebuild one channel at the other's gate_time first."
            )
        combined_noise = self.noise_lindbladian + other.noise_lindbladian
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
            "unitary": _pack_operator(self.ideal_unitary),
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
        return cls(inst=inst, lindbladian=lindbladian, ideal_unitary=unitary, gate_time=data["gate_time"])


@final
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

    @property
    def qubits(self) -> list[int]:
        """The qubits which the measurement applies to."""
        return list(self.inst.get_qubit_indices())

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

        ``fidelity`` is the *average* over levels: for ``dim > 2`` the interior levels carry error
        in both directions, so their individual accuracies are lower than the edge levels'.

        :param inst: The measurement instruction.
        :param fidelity: The average readout fidelity, in :math:`[0, 1]`.
        :param asymmetry: Value between -1 and +1. Zero is symmetric.
            Positive biases toward upward confusion P(j+1|j), negative toward downward P(j|j+1).
        :param dim: The dimension of the measured system (2 for qubits, 3 for qutrits, etc.).
        :return: A MeasurementChannel instance.
        :raises ValueError: If ``dim < 2``, if ``fidelity`` or ``asymmetry`` is out of range, or if
            the two combine to a confusion matrix with a negative diagonal — which requires
            ``dim * (1 - fidelity) * (1 + abs(asymmetry)) / (2 * (dim - 1)) <= 1``.
        """
        if dim < 2:
            raise ValueError(f"Measured system dimension must be at least 2, got dim={dim}.")
        if not 0.0 <= fidelity <= 1.0:
            raise ValueError(f"Readout fidelity must lie in [0, 1], got {fidelity}.")
        if not -1.0 <= asymmetry <= 1.0:
            raise ValueError(f"Readout asymmetry must lie in [-1, 1], got {asymmetry}.")

        # The largest single off-diagonal entry is error_factor * (1 + |asymmetry|); if that
        # exceeds 1 the corresponding diagonal entry would go negative.
        worst_case_error = dim * (1 - fidelity) * (1 + abs(asymmetry)) / (2 * (dim - 1))
        if worst_case_error > 1.0:
            raise ValueError(
                f"fidelity={fidelity} and asymmetry={asymmetry} cannot be realized for dim={dim}: "
                f"they imply a confusion probability of {worst_case_error:.4f} between adjacent "
                "levels. Reduce the asymmetry or raise the fidelity."
            )

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
        """Check equality by instruction and (approximate) instrument matrix (via :func:`jax.numpy.allclose`)."""
        if not isinstance(other, MeasurementChannel):
            return False
        if self.inst != other.inst:
            return False
        return bool(jnp.allclose(self.process.matrix, other.process.matrix))

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

    def __or__(self, other: CycleConstituent | CycleChannel) -> CycleChannel:
        """Tensor product with another channel on disjoint qubits, producing a CycleChannel.

        :param other: Another gate, reset, or measurement channel — or a
            :class:`CycleChannel`, whose constituents are flattened in — on disjoint qubits.
        :return: A CycleChannel representing the tensor product.
        """
        return _tensor_into_cycle(self, other)


class ResetChannelBase(ABC):
    """Shared behavior for reset noise channels backed by a superoperator ``process``.

    This is the base of every *reset* channel — :class:`ResetChannel` and
    :class:`SuperopResetChannel` — and, as with :class:`ChannelBase`, is the type to branch on
    rather than either concrete class.

    Like the rest of this module it is not yet re-exported from ``pyquil.noise``; import it
    from ``pyquil.noise._channels`` until the quax noise API becomes public in pyQuil v5.

    A reset channel replaces a targeted reset with a CPTP ``process`` (a ``qx.SuperOp`` that
    *includes* the ideal reset). Unlike gate channels there is no unitary; fidelity is
    measured against the ideal reset ``qx.gates.RESET``. Concrete subclasses supply ``inst`` and
    ``process`` — a stored field for :class:`SuperopResetChannel`, or derived from a generator for
    :class:`ResetChannel`.
    """

    # Declared under TYPE_CHECKING rather than as abstract properties for the same reason as in
    # :class:`ChannelBase` — an abstract property collides with the subclass dataclass field.
    if TYPE_CHECKING:
        inst: ResetQubit
        process: qx.SuperOp

    @abstractmethod
    def to_json(self) -> str:
        """Serialize the reset channel to a JSON string."""
        ...

    @classmethod
    @abstractmethod
    def from_json(cls, json_str: str) -> ResetChannelBase:
        """Deserialize a reset channel from a JSON string produced by :meth:`to_json`."""
        ...

    def __post_init__(self) -> None:
        """Validate that the channel is attached to a targeted reset."""
        if not isinstance(self.inst, ResetQubit):
            raise TypeError(f"{type(self).__name__} only supports targeted ResetQubit instructions.")

    @property
    def qubits(self) -> list[int]:
        """The qubit(s) that the reset applies to."""
        indices = self.inst.get_qubit_indices()
        return [] if indices is None else list(indices)

    @cached_property
    def _ideal_reset(self) -> qx.SuperOp:
        """The ideal reset superoperator matching this channel's dimension."""
        return qx.gates.RESET(dim=self.process.dims[0][0])

    @cached_property
    def process_fidelity(self) -> float:
        r"""Process fidelity of the reset channel relative to the ideal reset :math:`F_e \in [0, 1]`.

        Named to match ``quax.process_fidelity`` — and, deliberately, *not* ``fidelity``: on gate
        channels that name means the average gate fidelity, which is a different quantity.
        """
        return float(qx.process_fidelity(self.process, self._ideal_reset))

    @cached_property
    def error_process(self) -> qx.SuperOp:
        r"""The error, as a superoperator applied *after* the ideal reset.

        The full channel factors as :math:`\mathcal{E} = \mathcal{N} \circ \mathcal{R}`, where
        :math:`\mathcal{R}` is the ideal reset and :math:`\mathcal{N}` the post-reset error.
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

    def plot(self, only_noise: bool = False) -> Figure:
        """Plot the Pauli transfer matrix of the reset channel.

        :param only_noise: If True, plot the post-reset error (see :attr:`error_process`)
            instead of the full process. Defaults to False (the full channel).
        :return: A Plotly Figure.
        """
        channel = self.error_process if only_noise else self.process
        title_prefix = "Reset error" if only_noise else "Reset channel"
        fig = qx.plot(channel)
        qubit_str = str(self.qubits[0]) if self.qubits else "?"
        fig.update_layout(
            title=(f"{title_prefix} RESET {qubit_str}<br><sub>F_χ={self.process_fidelity * 100:.2f}%</sub>")
        )
        return fig

    def __str__(self) -> str:
        """Return a simplified string representation."""
        qubit_str = str(self.qubits[0]) if self.qubits else "?"
        return f"<RESET({self.process_fidelity:.2f}) {qubit_str}>"

    def __eq__(self, other: object) -> bool:
        """Check equality by concrete type, instruction, and (approximate) process matrix.

        Two reset channels are equal iff they are the same concrete class, share the same
        instruction, and have element-wise-close ``process`` matrices (via :func:`jax.numpy.allclose`).
        The comparison is approximate rather than bit-exact so it tolerates the small floating-point
        differences between otherwise-equivalent constructions.
        """
        if type(self) is not type(other):
            return False
        if self.inst != other.inst:
            return False
        return bool(jnp.allclose(self.process.matrix, other.process.matrix))

    __hash__ = None  # type: ignore[assignment]

    def __or__(self, other: CycleConstituent | CycleChannel) -> CycleChannel:
        """Tensor product with another channel on disjoint qubits, producing a CycleChannel.

        :param other: Another gate, reset, or measurement channel — or a
            :class:`CycleChannel`, whose constituents are flattened in — on disjoint qubits.
        :return: A CycleChannel representing the tensor product.
        """
        return _tensor_into_cycle(self, other)


@final
@dataclass(frozen=True, eq=False)
class SuperopResetChannel(ResetChannelBase):
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
        r"""Create a SuperopResetChannel whose :attr:`~ResetChannelBase.process_fidelity` is ``fidelity``.

        The ideal reset maps every state to :math:`|0\rangle\langle 0|`; noise is a depolarizing
        channel applied *after* it, so the process is ``depolarizing @ RESET`` — every state is
        reset and then shrunk toward the maximally-mixed state.

        The depolarizing shrink factor is *not* the resulting fidelity. Because the ideal reset is
        rank one, the process fidelity of ``depolarizing_p @ RESET`` against ``RESET`` is
        :math:`F = (1 + (d - 1) p) / d`, so the shrink factor is obtained by inverting that:
        :math:`p = (d F - 1) / (d - 1)`. The channel therefore reports back exactly the
        ``fidelity`` requested here.

        :param inst: The reset instruction.
        :param fidelity: Process fidelity of the reset channel, :math:`F \in [1/d, 1]`.
            1.0 yields an ideal reset; values below 1 introduce depolarizing noise. The floor is
            :math:`1/d` because a fully depolarizing reset still has that much overlap with the
            ideal one.
        :param dim: Hilbert-space dimension (2 for qubits).
        :return: A SuperopResetChannel instance.
        :raises ValueError: If ``dim < 2`` or ``fidelity`` lies outside :math:`[1/d, 1]`.
        """
        if not isinstance(inst, ResetQubit):
            raise TypeError("SuperopResetChannel only supports targeted ResetQubit instructions.")
        if dim < 2:
            raise ValueError(f"Reset dimension must be at least 2, got dim={dim}.")
        minimum_fidelity = 1.0 / dim
        if not minimum_fidelity <= fidelity <= 1.0:
            raise ValueError(
                f"Reset process fidelity must lie in [{minimum_fidelity}, 1] for dim={dim}, got {fidelity}. "
                f"A fully depolarized reset already has process fidelity {minimum_fidelity}."
            )

        # Invert F = (1 + (d - 1) p) / d to get the depolarizing shrink factor, then pick the rate
        # whose evolution shrinks every traceless operator by exactly that factor.
        depolarizing_constant = (dim * fidelity - 1.0) / (dim - 1.0)
        d2 = dim * dim
        shrink = max(depolarizing_constant, np.finfo(float).tiny)
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


@final
@dataclass(frozen=True, eq=False)
class ResetChannel(_LindbladianBacked, ResetChannelBase):
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

    gate_time: float = _DEFAULT_GATE_TIME
    """Evolution time for ``evolve(lindbladian, gate_time)``. See :data:`_DEFAULT_GATE_TIME`."""

    @classmethod
    def from_lindbladian(
        cls: type[ResetChannel],
        inst: ResetQubit,
        lindbladian: qx.Lindbladian,
        gate_time: float = _DEFAULT_GATE_TIME,
    ) -> ResetChannel:
        """Create a reset channel directly from a dissipative relaxation generator."""
        return cls(inst=inst, lindbladian=lindbladian, gate_time=gate_time)

    @classmethod
    def from_amplitude_damping(
        cls: type[ResetChannel],
        inst: ResetQubit,
        gamma: float,
        gate_time: float = _DEFAULT_GATE_TIME,
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
        """Create a reset channel from T1/T2 relaxation over ``duration`` (used as ``gate_time``).

        :param inst: The targeted reset instruction.
        :param duration: How long the reset is allowed to relax (used as ``gate_time``).
        :param t1: The T1 relaxation time.
        :param t2: The T2 coherence time. Defaults to ``2 * t1`` (no pure dephasing). Must not
            exceed ``2 * t1``; see :func:`_pure_dephasing_times`.
        """
        t2_value = 2 * t1 if t2 is None else t2
        (tphi,) = _pure_dephasing_times([t1], [t2_value])
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


CycleConstituent: TypeAlias = ChannelBase | ResetChannelBase | MeasurementChannel
"""A single operation within a cycle: a gate channel, a reset channel, or a measurement channel."""


@final
@dataclass(frozen=True)
class CycleChannel:
    """A cycle noise channel attaches superoperators to a specific cycle.

    Cycles can include gates, resets, and measurements. The constituent channels are stored
    directly, allowing fidelity metrics and serialization to be derived from them.
    """

    inst: Gate
    """The cycle to which the channel applies."""

    defcircuit: DefCircuit
    """The DefCircuit representing the logical cycle to which instruction represents."""

    channels: tuple[CycleConstituent, ...]
    """Constituent channels (one per operation in the cycle) on disjoint qubits."""

    def __post_init__(self) -> None:
        """Validate that every instruction in the cycle body has a corresponding channel.

        Downstream consumers (the resolver, the stim converter) use only ``channels`` and
        ignore ``defcircuit``; a missing channel would silently drop that operation's noise.
        Operations are matched by instruction equality (name, params, concrete qubits),
        independent of the DefCircuit's formal-argument naming.
        """
        if len(self.expanded_instructions) != len(self.channels):
            raise ValueError(
                "CycleChannel is incomplete: every instruction in the cycle's DefCircuit "
                "body must have a corresponding channel. "
                f"\nDefCircuit body: {[str(i) for i in self.expanded_instructions]}"
                f"\nChannels:        {[str(c.inst) for c in self.channels]}"
            )
        for instruction, channel in zip(self.expanded_instructions, self.channels, strict=True):
            if instruction != channel.inst:
                raise ValueError(
                    "CycleChannel is inconsistent: each instruction in the cycle's DefCircuit "
                    "body must match its corresponding channel's instruction. "
                    f"\nDefCircuit body: {instruction}"
                    f"\nChannel:         {channel.inst}"
                )

    # ──────────────────────────────────────────────
    # Derived properties
    # ──────────────────────────────────────────────

    @cached_property
    def expanded_instructions(self) -> tuple[Gate | Measurement | ResetQubit, ...]:
        """The defcircuit's body with its formal arguments substituted for concrete qubits.

        Returns a tuple rather than a list: this is a cached property on a frozen dataclass, so a
        mutable result would let a caller corrupt the cache.
        """
        formal_arguments = self.defcircuit.qubit_variables
        cycle_qubits = self.inst.get_qubit_indices()
        if len(formal_arguments) != len(cycle_qubits):
            raise ValueError(
                f"Cycle instruction {self.inst.out()} supplies {len(cycle_qubits)} qubit(s) but "
                f"DEFCIRCUIT {self.defcircuit.name} declares {len(formal_arguments)} formal "
                f"argument(s) ({[str(a) for a in formal_arguments]})."
            )
        qarg_to_qubit = dict(zip(formal_arguments, cycle_qubits, strict=True))

        def resolve(qubit: Any) -> int:
            if qubit not in qarg_to_qubit:
                raise ValueError(
                    f"DEFCIRCUIT {self.defcircuit.name} body references {qubit}, which is not one "
                    f"of its formal arguments ({[str(a) for a in formal_arguments]})."
                )
            return qarg_to_qubit[qubit]

        instructions: list[Gate | Measurement | ResetQubit] = []
        for inst in self.defcircuit.instructions:
            match inst:
                case Measurement():
                    instructions.append(Measurement(qubit=resolve(inst.qubit), classical_reg=inst.classical_reg))
                case ResetQubit():
                    instructions.append(ResetQubit(resolve(inst.qubit)))
                case Gate():
                    instructions.append(Gate(inst.name, inst.params, [resolve(q) for q in inst.qubits]))
                case _:
                    raise TypeError(f"Unsupported instruction type in defcircuit: {type(inst).__name__}")
        return tuple(instructions)

    @cached_property
    def operator(self) -> tuple[qx.SuperOp | qx.QuantumInstrument, ...]:
        """Tuple of process superoperators, one per constituent channel."""
        return tuple(ch.process for ch in self.channels)

    @property
    def qubits(self) -> list[int]:
        """All qubits in the cycle, derived from the instruction."""
        return self.inst.get_qubit_indices()

    @property
    def _gate_channels(self) -> tuple[ChannelBase, ...]:
        """The gate channels in the cycle; the only constituents carrying a gate fidelity."""
        return tuple(ch for ch in self.channels if isinstance(ch, ChannelBase))

    @cached_property
    def process_fidelity(self) -> float:
        """Product of process (Pauli) fidelities over all gate channels in the cycle.

        The product is *exact*: process fidelity is multiplicative over a tensor product, and the
        constituent channels act on disjoint subsystems by construction.

        .. warning::
            Reset and measurement channels carry no gate fidelity and are skipped entirely, so a
            cycle made only of measurements reports 1.0 here however noisy its readout is. Read
            readout quality off the constituent :class:`MeasurementChannel` objects
            (:attr:`~MeasurementChannel.classification_fidelity` and friends) instead.
        """
        f = 1.0
        for ch in self._gate_channels:
            f *= ch.process_fidelity
        return f

    @cached_property
    def average_gate_fidelity(self) -> float:
        """Average gate fidelity of the cycle's gate channels, taken as a whole.

        Unlike the process fidelity, average gate fidelity is *not* multiplicative over a tensor
        product: the process-to-average conversion depends on the total dimension. So this converts
        once, from the exact :attr:`process_fidelity` product, using the combined dimensions of
        every gate channel in the cycle — rather than multiplying the constituents' average
        fidelities, which would overstate the result.

        The same caveat as :attr:`process_fidelity` applies: reset and measurement channels are
        skipped.
        """
        gate_channels = self._gate_channels
        if not gate_channels:
            return 1.0
        dims = tuple(dim for ch in gate_channels for dim in ch.dims)
        return float(qx.process_fidelity_to_average_fidelity(self.process_fidelity, dims=dims))

    @cached_property
    def average_gate_infidelity(self) -> float:
        """``1 - average_gate_fidelity``."""
        return 1.0 - self.average_gate_fidelity

    @cached_property
    def process_infidelity(self) -> float:
        """``1 - process_fidelity``."""
        return 1.0 - self.process_fidelity

    # ──────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize CycleChannel to a JSON string.

        The cycle instruction and its ``DEFCIRCUIT`` are written out alongside the constituent
        channels, so a cycle built elsewhere (with its own name, formal arguments, or body
        ordering) survives a round trip rather than being rebuilt as a generic ``CYCLE``.

        :return: JSON string representation.
        """
        data = {
            "inst": self.inst.out(),
            "defcircuit": self.defcircuit.out(),
            "channels": [{"type": type(ch).__name__, "data": ch.to_json()} for ch in self.channels],
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls: type[CycleChannel], json_str: str) -> CycleChannel:
        """Deserialize a CycleChannel from a JSON string.

        :param json_str: JSON string as produced by :meth:`to_json`.
        :return: CycleChannel instance.
        """
        data = json.loads(json_str)
        _type_map: dict[str, type[CycleConstituent]] = {
            "SuperopChannel": SuperopChannel,
            "Channel": Channel,
            "MeasurementChannel": MeasurementChannel,
            "SuperopResetChannel": SuperopResetChannel,
            "ResetChannel": ResetChannel,
        }
        constituent_channels: list[CycleConstituent] = [
            _type_map[ch_data["type"]].from_json(ch_data["data"]) for ch_data in data["channels"]
        ]

        # Older payloads carried only the constituents; rebuild the generic cycle for those.
        if "defcircuit" not in data or "inst" not in data:
            return _build_cycle_channel(constituent_channels)

        inst = _parse_quil_instruction(data["inst"])
        if not isinstance(inst, Gate):
            raise TypeError(f"CycleChannel JSON must contain a gate instruction, got {type(inst).__name__}.")
        defcircuit = _parse_defcircuit(data["defcircuit"])
        return cls(inst=inst, defcircuit=defcircuit, channels=tuple(constituent_channels))

    # ──────────────────────────────────────────────
    # Dunder methods
    # ──────────────────────────────────────────────

    def __str__(self) -> str:
        """Return a simplified string representation showing the gate and process fidelity."""
        return f"<{self.inst.out()} ~ ({100 * self.process_fidelity:.2f}%)>"

    def __eq__(self, other: object) -> bool:
        """Check equality by cycle instruction, DefCircuit, and constituent channels."""
        if not isinstance(other, CycleChannel):
            return False
        return self.inst == other.inst and self.defcircuit == other.defcircuit and self.channels == other.channels

    __hash__ = None  # type: ignore[assignment]

    def __or__(self, other: CycleConstituent | CycleChannel) -> CycleChannel:
        """Tensor another channel into this cycle, producing a wider CycleChannel.

        Lets cycles be built up left to right (``a | b | c``). A :class:`CycleChannel` operand has
        its constituents flattened in, so the result is always a flat cycle.

        :param other: Another gate, reset, or measurement channel, or another CycleChannel, on
            qubits disjoint from this cycle's.
        :return: A CycleChannel containing every constituent operation.
        """
        return _tensor_into_cycle(self, other)


def _channel_to_formal_inst(channel: CycleConstituent) -> Gate | Measurement | ResetQubit:
    """Convert a channel's instruction to use formal arguments for DefCircuit."""
    if isinstance(channel, ChannelBase):
        inst = channel.inst
        return Gate(
            name=inst.name,
            params=inst.params,
            qubits=[FormalArgument(f"q{q}") for q in inst.get_qubit_indices()],
            modifiers=inst.modifiers,  # type: ignore[arg-type]
        )
    elif isinstance(channel, ResetChannelBase):
        qubit_idx = channel.qubits[0]
        return ResetQubit(qubit=FormalArgument(f"q{qubit_idx}"))
    elif isinstance(channel, MeasurementChannel):
        qubit_idx = channel.qubits[0]
        return Measurement(
            qubit=FormalArgument(f"q{qubit_idx}"),
            classical_reg=None,
        )
    raise TypeError(f"Unsupported channel type: {type(channel)}")


def _build_cycle_channel(
    channels: list[CycleConstituent],
) -> CycleChannel:
    """Build a CycleChannel from gate/reset/measurement channels on disjoint qubits."""
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


def _tensor_into_cycle(
    left: CycleConstituent | CycleChannel,
    right: CycleConstituent | CycleChannel,
) -> CycleChannel:
    """Tensor two channels on disjoint qubits into a single flat :class:`CycleChannel`.

    Shared by every ``__or__`` implementation so that ``|`` is closed over cycles: either operand
    may itself be a :class:`CycleChannel`, in which case its constituents are flattened in and
    ``a | b | c`` builds one three-operation cycle rather than failing on the second ``|``.
    """
    if not isinstance(left, (ChannelBase, ResetChannelBase, MeasurementChannel, CycleChannel)):
        return NotImplemented
    if not isinstance(right, (ChannelBase, ResetChannelBase, MeasurementChannel, CycleChannel)):
        return NotImplemented

    overlap = set(left.qubits) & set(right.qubits)
    if overlap:
        raise ValueError(f"Cannot tensor channels with overlapping qubits: {overlap}")

    def constituents(channel: CycleConstituent | CycleChannel) -> list[CycleConstituent]:
        return list(channel.channels) if isinstance(channel, CycleChannel) else [channel]

    return _build_cycle_channel(constituents(left) + constituents(right))


def _parse_defcircuit(quil_str: str) -> DefCircuit:
    """Parse a ``DEFCIRCUIT`` block back into a pyquil :class:`~pyquil.quilbase.DefCircuit`."""
    rs_inst = RSInstruction.parse(quil_str)
    if not rs_inst.is_circuit_definition():
        raise ValueError(f"Expected a DEFCIRCUIT definition, got: {quil_str}")
    return DefCircuit._from_rs_circuit_definition(rs_inst.to_circuit_definition())

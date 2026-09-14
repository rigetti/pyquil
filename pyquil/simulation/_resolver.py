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
r"""Compilation from Quil to a :class:`~pyquil.simulation._circuit.Circuit`.

Turning a program into something a simulator can evolve happens in two steps:

1. **Expansion** (:func:`expand_program`) turns a program into a flat sequence of operators
   and the physical qubits each acts on.  Noise channels replace the instructions they are
   attached to, ``DEFGATE`` and ``DEFCIRCUIT`` definitions are expanded, and a gate whose
   angle refers to declared memory becomes a :class:`ParametricGate` that is
   evaluated once the parameter values are known.
2. **Resolution** (:meth:`Resolution.resolve`) binds a parameter vector and returns the
   :class:`~pyquil.simulation._circuit.Circuit` of concrete operators placed on a register.

Everything after that — merging operators, converting representations, evolving a state — works
on the circuit and knows nothing about Quil.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias

import jax.numpy as jnp
import numpy as np
import quax as qx
from jax import Array

from pyquil.noise._channels import (
    ChannelBase,
    CycleChannel,
    MeasurementChannel,
    ResetChannelBase,
    _reject_gate_modifiers,
    get_custom_gates_from_program,
    get_instruction_unitary,
)
from pyquil.noise._noise_model import (
    NoiseModelLike,
)
from pyquil.quil import Program
from pyquil.quilatom import (
    Add,
    BinaryExp,
    Div,
    Function,
    MemoryReference,
    Mul,
    Parameter,
    Pow,
    Qubit,
    Sub,
    _contained_mrefs,
    substitute,
)
from pyquil.quilbase import (
    AbstractInstruction,
    ArithmeticBinaryOp,
    ClassicalComparison,
    ClassicalConvert,
    ClassicalExchange,
    ClassicalLoad,
    ClassicalMove,
    ClassicalStore,
    DefCircuit,
    Gate,
    Jump,
    JumpUnless,
    JumpWhen,
    LogicalBinaryOp,
    Measurement,
    Reset,
    ResetQubit,
    UnaryClassicalInstruction,
)
from pyquil.simulation._circuit import Circuit, CircuitOp, Placement

# ──────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────

#: A ``(register_name, offset)`` pair naming one scalar of classical memory, e.g. ``("theta", 0)``.
ParameterRef: TypeAlias = tuple[str, int]


#: The Quil arithmetic functions, as JAX functions.  ``CIS(x)`` is ``exp(i x)``.
_FUNCTIONS: dict[str, Callable[[Array], Array]] = {
    "SIN": jnp.sin,
    "COS": jnp.cos,
    "SQRT": jnp.sqrt,
    "EXP": jnp.exp,
    "CIS": lambda x: jnp.exp(1j * x),
}
_BINARY_OPERATORS: dict[type[BinaryExp], Callable[[Array, Array], Array]] = {
    Add: jnp.add,
    Sub: jnp.subtract,
    Mul: jnp.multiply,
    Div: jnp.divide,
    Pow: jnp.power,
}


@dataclass(frozen=True, slots=True)
class ParameterExpression:
    """A gate argument given as a Quil arithmetic expression over memory references.

    Any expression Quil allows is supported: ``+ - * / ^``, the functions ``SIN``, ``COS``,
    ``SQRT``, ``EXP`` and ``CIS``, and real or complex literals -- for example
    ``RX(theta[0]/2 + pi)``, ``RZ(2*SIN(phi[1]))``, or ``CPH(CIS(theta[0])) 0`` for a
    ``DEFGATE CPH(%z)`` taking a complex parameter.  Evaluation goes through JAX, so an
    expression can be jitted and differentiated through.

    There are two ways in.  :attr:`evaluate` takes the *narrowed* vector of only the values this
    expression reads, which is what the simulator gathers once per vectorised batch; calling the
    instance takes the whole circuit-wide parameter vector and gathers from it first.

    :param slot_indices: Slots of the parameter vector the expression reads, in order of first
        appearance.
    :param key: The expression with its references replaced by ``%0``, ``%1``, ... in that
        order.  Two expressions with the same key have the same shape and constants and differ
        only in which slots they read, which is what lets the simulator evaluate them together
        in one vectorised operation.
    :param is_complex: Whether the value may be complex (the expression contains ``CIS`` or a
        complex literal).  Everything else is evaluated in real arithmetic; note that Quil
        would evaluate ``SQRT`` of a negative number or a fractional power of one as complex,
        which real arithmetic reports as ``nan``.
    :param evaluate: Evaluates the expression from the values of :attr:`slot_indices`, in that
        order.
    """

    slot_indices: tuple[int, ...]
    key: str
    is_complex: bool
    evaluate: Callable[[Array], Array]

    def __call__(self, params: Array) -> Array:
        """Evaluate the expression from the whole circuit-wide parameter vector.

        Gathers :attr:`slot_indices` out of ``params`` and hands the narrowed vector to
        :attr:`evaluate`.
        """
        return self.evaluate(params[jnp.asarray(self.slot_indices)])


def _literal_value(expression: Any) -> float | complex:
    """Evaluate a parameter-free expression (a number, ``pi/2``, ``SIN(pi/4)``, ``1i``) to a scalar."""
    value = complex(np.asarray(expression, dtype=complex).item()) if not isinstance(expression, complex) else expression
    return value.real if value.imag == 0 else value


def _build_expression(node: Any, references: list[MemoryReference]) -> tuple[Callable[[Array], Array], str, bool]:
    """Compile one node of a Quil expression tree.

    :param node: The node to compile; recursion handles its operands.
    :param references: The memory references seen so far, in first-appearance order.  New ones are
        appended, and a reference's position here is the index into the narrowed value vector that
        the compiled closure reads.
    :returns: The closure, the shape key, and whether the value may be complex.
    :raises ValueError: If the node is an unbound ``DEFGATE`` parameter or an unknown function.
        The message names only the offending node; :func:`expand_program` adds the instruction.
    """
    if isinstance(node, MemoryReference):
        if node not in references:
            references.append(node)
        index = references.index(node)
        return (lambda slot_values, index=index: slot_values[index]), f"%{index}", False
    if isinstance(node, Parameter):
        raise ValueError(f"Unbound DEFGATE parameter {node}.")
    if isinstance(node, BinaryExp):
        left, left_key, left_complex = _build_expression(node.op1, references)
        right, right_key, right_complex = _build_expression(node.op2, references)
        operator = _BINARY_OPERATORS[type(node)]
        return (
            (lambda slot_values: operator(left(slot_values), right(slot_values))),
            f"({left_key}{node.operator.strip()}{right_key})",
            left_complex or right_complex,
        )
    if isinstance(node, Function):
        if node.name not in _FUNCTIONS:
            raise ValueError(f"Unknown Quil function {node.name!r}.")
        function = _FUNCTIONS[node.name]
        inner, inner_key, inner_complex = _build_expression(node.expression, references)
        return (
            (lambda slot_values: function(inner(slot_values))),
            f"{node.name}({inner_key})",
            inner_complex or node.name == "CIS",
        )
    literal = _literal_value(node)
    return (lambda slot_values, literal=literal: jnp.asarray(literal)), repr(literal), isinstance(literal, complex)


def _compile_expression(expression: Any, slot_of: Callable[[MemoryReference], int]) -> ParameterExpression:
    """Compile a Quil expression over memory references into a :class:`ParameterExpression`.

    :param expression: The gate parameter; must contain at least one memory reference.
    :param slot_of: Maps a memory reference to its slot in the parameter vector.
    :raises ValueError: If the expression contains an unbound ``DEFGATE`` parameter or an unknown
        function.
    """
    references: list[MemoryReference] = []
    evaluate, key, is_complex = _build_expression(expression, references)
    return ParameterExpression(tuple(slot_of(ref) for ref in references), key, is_complex, evaluate)


@dataclass(frozen=True, slots=True)
class ParametricGate:
    """A parametric gate whose matrix depends on runtime parameters.

    Calling an instance with the flat parameter vector returns the gate's ``qx.Unitary``.  The
    constructor and argument layout are exposed so that gates of the same kind can be built
    together in one vectorised operation.

    :param gate_fn: The quax gate constructor (e.g. ``qx.gates.RX``), or a parametric
        ``DEFGATE`` callable.
    :param arguments: One entry per gate argument: a literal number, or a
        :class:`ParameterExpression` reading the parameter vector.  Gates that read the same
        memory reference share a slot; see :func:`expand_program`.
    """

    gate_fn: Callable[..., qx.Operator]
    arguments: tuple[float | complex | ParameterExpression, ...]

    def __call__(self, params: Array) -> qx.Unitary:
        """Build the gate for one parameter vector."""
        resolved: list[Any] = [arg(params) if isinstance(arg, ParameterExpression) else arg for arg in self.arguments]
        result = self.gate_fn(*resolved)
        if not isinstance(result, qx.Unitary):
            result = qx.Unitary.from_matrix(result.matrix, result.dims)
        return result


#: An expanded item is either a concrete operator or a ParametricGate that resolves parameters
#: into a Unitary.
ExpandedOp: TypeAlias = CircuitOp | ParametricGate


# ──────────────────────────────────────────────────────────
# DEFCIRCUIT expansion
# ──────────────────────────────────────────────────────────


def expand_defcircuit_body(
    inst: Gate,
    defcircuit: DefCircuit,
    circuit_definitions: dict[str, DefCircuit],
) -> Iterator[Gate | Measurement | ResetQubit | Reset]:
    """Yield concrete instructions from a DEFCIRCUIT invocation.

    Substitutes formal qubit/parameter arguments with the concrete values
    from ``inst``.  Handles nested DEFCIRCUITs via recursion.

    :param inst: The Gate that invokes the DEFCIRCUIT.
    :param defcircuit: The DefCircuit definition to expand.
    :param circuit_definitions: All known DEFCIRCUIT definitions (for nested expansion).
    :yields: Concrete instructions with physical qubits and resolved parameters.
    """
    if len(inst.qubits) != len(defcircuit.qubit_variables):
        raise ValueError(
            f"{inst.out()!r} passes {len(inst.qubits)} qubit(s) but DEFCIRCUIT "
            f"{defcircuit.name} declares {len(defcircuit.qubit_variables)} "
            f"({[str(a) for a in defcircuit.qubit_variables]})."
        )
    if len(inst.params) != len(defcircuit.parameters):
        raise ValueError(
            f"{inst.out()!r} passes {len(inst.params)} parameter(s) but DEFCIRCUIT "
            f"{defcircuit.name} declares {len(defcircuit.parameters)} "
            f"({[str(a) for a in defcircuit.parameters]})."
        )
    qarg_to_arg_map = {qarg: q for q, qarg in zip(inst.qubits, defcircuit.qubit_variables, strict=True)}
    parg_to_arg_map = {parg: param for param, parg in zip(inst.params, defcircuit.parameters, strict=True)}

    def resolve_qubit(qarg: Any) -> Any:
        """Map a formal argument to its concrete qubit.

        A DEFCIRCUIT body must reference only the circuit's own formal arguments. Quil permits
        a literal qubit in a body (``DEFCIRCUIT C q: X q; X 3``), but simulating one is a trap:
        the qubit is invisible to ``Program.get_qubit_indices`` (pyQuil issue #1868), so it
        silently escapes the register the simulator sizes itself for. Rejecting it here is
        clearer than the downstream failure; the check can be dropped once #1868 is fixed.
        """
        if qarg not in qarg_to_arg_map:
            raise ValueError(
                f"DEFCIRCUIT {defcircuit.name} body references {qarg}, which is not one of its "
                f"formal arguments ({[str(a) for a in defcircuit.qubit_variables]}). Literal "
                "qubits in a DEFCIRCUIT body are not supported; parameterize the circuit over "
                "all the qubits it touches."
            )
        return qarg_to_arg_map[qarg]

    for circuit_inst in defcircuit.instructions:
        if isinstance(circuit_inst, Gate):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubits = [resolve_qubit(qarg) for qarg in circuit_inst.qubits]
            if hasattr(circuit_inst, "params"):
                circuit_inst.params = [substitute(param, parg_to_arg_map) for param in circuit_inst.params]  # type: ignore[arg-type]
            if circuit_inst.name in circuit_definitions:
                yield from expand_defcircuit_body(
                    circuit_inst, circuit_definitions[circuit_inst.name], circuit_definitions
                )
            else:
                yield circuit_inst
        elif isinstance(circuit_inst, Measurement):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubit = resolve_qubit(circuit_inst.qubit)
            yield circuit_inst
        elif isinstance(circuit_inst, ResetQubit):
            circuit_inst = deepcopy(circuit_inst)
            circuit_inst.qubit = resolve_qubit(circuit_inst.qubit)
            yield circuit_inst
        else:
            yield deepcopy(circuit_inst)  # type: ignore[misc]


# ══════════════════════════════════════════════════════════
# Expander
# ══════════════════════════════════════════════════════════

#: Instructions the simulator refuses.  Each of these changes *which* quantum operations run
#: (control flow) or what classical memory holds (and hence, via ``MEASURE`` targets and gate
#: parameters, what the program means), and none can be honoured by a straight-line simulation.
#: Everything else that is not a gate, measurement or reset is ignored: declarations and
#: definitions, pragmas, labels, ``HALT``/``NOP``/``WAIT``, and all Quil-T pulse-level
#: instructions, which are taken to be the physical realisation of the logical program rather
#: than a change to it.
UNSUPPORTED_INSTRUCTIONS: tuple[type[AbstractInstruction], ...] = (
    Jump,
    JumpWhen,
    JumpUnless,
    UnaryClassicalInstruction,
    LogicalBinaryOp,
    ArithmeticBinaryOp,
    ClassicalMove,
    ClassicalExchange,
    ClassicalConvert,
    ClassicalLoad,
    ClassicalStore,
    ClassicalComparison,
)


def _measure_registers(program: Program) -> set[str]:
    """Return the set of register names that are targets of MEASURE instructions."""
    regs: set[str] = set()
    for inst in program.instructions:
        if isinstance(inst, Measurement):
            cr = inst.classical_reg
            if cr is not None:
                regs.add(cr.name)
    return regs


def expand_program(
    program: Program,
    noise_model: NoiseModelLike | None = None,
    qubit_dimensions: Mapping[int, int] | None = None,
) -> tuple[tuple[ExpandedOp, ...], tuple[tuple[int, ...], ...], tuple[ParameterRef, ...]]:
    """Expand a program into operators and physical qubit tuples.

    Fixed (non-parameterized) operations are returned as concrete quax types
    (``Unitary``, ``SuperOp``, ``QuantumInstrument``).  Only parameterized
    gates are returned as :class:`ParametricGate` callables.

    DEFCIRCUIT invocations are expanded:

    * If a cycle invocation matches a :class:`CycleChannel` in the noise
      model, the cycle is expanded using the channel's constituent operators.
    * Otherwise the DEFCIRCUIT body is expanded via qubit/param substitution
      and each resulting instruction is resolved individually.

    The noise model is fully resolved during expansion: noisy gates become
    ``SuperOp``, noisy measurements become ``QuantumInstrument``, and noisy
    resets become ``SuperOp``.  A channel is looked up by instruction equality
    (name, parameters, qubits and modifiers), so ``RX(pi/2) 0`` and ``RX(pi/2) 1``
    are distinct keys.

    A ``MEASURE`` becomes a :class:`quax.QuantumInstrument`, whose outcome-labelled operators
    let a trajectory simulator sample the result.  A simulator that does not record outcomes
    replaces it by its total channel, the dephasing map obtained by summing over outcomes
    (:meth:`~pyquil.simulation._circuit.Circuit.to_superops`).

    **Parameter layout.**  Each *distinct* memory reference appearing as a gate argument
    (``theta[0]``, say) is assigned one slot of the flat parameter vector, in order of first
    use.  A reference used by several gates therefore maps to a single slot, so a gradient with
    respect to that slot is already the total derivative — nothing has to be summed by hand.

    :param program: Quil program (may contain DEFCIRCUITs).
    :param noise_model: Optional noise model.
    :param qubit_dimensions: Optional mapping from physical qubit id to its
        Hilbert-space dimension. Used for ideal measurement and reset operators,
        whose quax constructors otherwise default to qubit dimension.
    :return: Tuple of ``(ops, qubit_tuples, parameters)`` where each op is either a concrete
        quax operator or a :class:`ParametricGate`, each qubit tuple contains physical qubit
        IDs, and ``parameters`` lists the distinct ``(register_name, offset)`` references in
        slot order.
    :raises ValueError: If the program contains control flow or classical memory
        instructions (see :data:`UNSUPPORTED_INSTRUCTIONS`), a gate modifier, an
        expression-valued or feed-forward gate parameter, or a DEFCIRCUIT body that names a
        literal qubit.
    """
    # Program-level derivations. These are independent of qubit dimensions, so
    # ``resolve_program``'s two passes each recompute them; measured at 1-4 ms (under 6% of a
    # resolve), which is not worth caching across the passes.
    circuit_definitions: dict[str, DefCircuit] = {
        inst.name: inst for inst in program.instructions if isinstance(inst, DefCircuit)
    }
    custom_gates = get_custom_gates_from_program(program) or None
    measure_regs = _measure_registers(program)
    all_qubits = sorted(program.get_qubit_indices())

    ops: list[ExpandedOp] = []
    qubit_tuples: list[tuple[int, ...]] = []
    # Distinct memory references in first-use order; the value is the slot in the parameter
    # vector.  ``dict`` preserves insertion order, which is what makes the slots stable.
    slots: dict[ParameterRef, int] = {}

    def _emit_op(op: ExpandedOp, qubits: tuple[int, ...]) -> None:
        ops.append(op)
        qubit_tuples.append(qubits)

    def _resolve_gate(inst: Gate) -> tuple[ExpandedOp, tuple[int, ...]]:
        """Resolve a single gate instruction to an operator or callable."""
        # Modifiers are rejected on every path, parametric included: dropping ``DAGGER`` from
        # ``DAGGER RX(theta[0]) 0`` would simulate ``RX(+theta)`` and return a plausible wrong
        # state.  ``get_instruction_unitary`` performs the same check for the fixed path.
        _reject_gate_modifiers(inst)
        qubits = tuple(inst.get_qubit_indices())

        # Check noise model first. Match on ChannelBase, not a concrete class: gate channels
        # come in both Lindbladian-backed (Channel) and raw-superoperator (SuperopChannel)
        # flavors, and every derived operation (composition, twirling, coherent/stochastic
        # splitting) returns the latter. Narrowing to one of them silently drops the other's
        # noise and simulates the ideal gate instead.
        channel = noise_model.get_channel(inst) if noise_model is not None else None
        if isinstance(channel, ChannelBase):
            return channel.process, qubits
        if isinstance(channel, CycleChannel):
            raise ValueError(f"CycleChannel for {inst.name} was not expanded before gate resolution.")

        # Parameterized gate → callable that resolves params at call time.
        if any(_contained_mrefs(p) for p in inst.params):  # type: ignore[arg-type]
            gate_name = inst.name
            if custom_gates is not None and gate_name in custom_gates:
                gate_def, is_builtin = custom_gates[gate_name], False
            elif gate_name in qx.gates.QUANTUM_GATES:
                gate_def, is_builtin = qx.gates.QUANTUM_GATES[gate_name], True
            else:
                raise KeyError(f"Unknown gate '{gate_name}'.")
            if isinstance(gate_def, qx.Unitary):
                raise ValueError(f"Gate '{gate_name}' is not parametric but {inst.out()!r} passes parameters.")

            arguments: list[float | complex | ParameterExpression] = []
            for p in inst.params:
                argument: float | complex | ParameterExpression
                if not _contained_mrefs(p):  # type: ignore[arg-type]
                    # A literal: a compile-time constant for this gate.
                    argument = _literal_value(p)
                    is_complex = isinstance(argument, complex)
                else:
                    feed_forward = [m for m in _contained_mrefs(p) if m.name in measure_regs]  # type: ignore[arg-type]
                    if feed_forward:
                        # Classically-conditioned angle: the value is only known mid-circuit.
                        raise ValueError(
                            f"Gate parameter {p} in {inst.out()!r} reads memory region "
                            f"'{feed_forward[0].name}', which is written by a MEASURE in this program. "
                            "Feed-forward (classically-conditioned) parameters are not supported."
                        )
                    try:
                        argument = _compile_expression(
                            p, lambda ref: slots.setdefault((ref.name, ref.offset), len(slots))
                        )
                    except ValueError as error:
                        # The compiler sees one expression; name the instruction it came from.
                        raise ValueError(f"Gate parameter {p} in {inst.out()!r}: {error}") from error
                    is_complex = argument.is_complex
                # Checked for literals too: quax's built-in constructors take real angles, and a
                # complex one otherwise surfaces much later as an opaque error from deep in quax.
                if is_complex and is_builtin:
                    raise ValueError(
                        f"Gate parameter {p} in {inst.out()!r} is complex-valued (it contains CIS or a complex "
                        f"literal), but the built-in gate {gate_name} takes real angles. Complex arguments are "
                        "only supported for DEFGATE gates."
                    )
                arguments.append(argument)

            return ParametricGate(gate_def, tuple(arguments)), qubits

        # Fixed gate → resolve to Unitary now.
        unitary = get_instruction_unitary(inst, custom_gates=custom_gates)
        return unitary, qubits

    def _dimension_for(qubit: int) -> int:
        return qubit_dimensions.get(qubit, 2) if qubit_dimensions is not None else 2

    def _resolve_measurement(inst: Measurement) -> tuple[CircuitOp, tuple[int, ...]]:
        """Resolve a measurement instruction to a ``QuantumInstrument``."""
        qubits = tuple(inst.get_qubit_indices())
        channel = None
        if noise_model is not None:
            # Readout channels are keyed by the measured qubit, not by where the bit is stored:
            # ``MEASURE 0 ro[3]`` and ``MEASURE 0`` read the same qubit with the same error.
            channel = noise_model.get_channel(inst)
            if channel is None and inst.classical_reg is not None:
                channel = noise_model.get_channel(Measurement(inst.qubit, None))
        instrument = (
            channel.process
            if isinstance(channel, MeasurementChannel)
            else qx.gates.MEASURE(dim=_dimension_for(qubits[0]))
        )
        return instrument, qubits

    def _resolve_reset_qubit(inst: ResetQubit) -> tuple[CircuitOp, tuple[int, ...]]:
        """Resolve a targeted reset instruction."""
        qubits = tuple(inst.get_qubit_indices())  # type: ignore[arg-type]
        channel = noise_model.get_channel(inst) if noise_model is not None else None
        # ResetChannelBase, not ResetChannel: see the note in _resolve_gate. SuperopResetChannel
        # is the other reset flavor and must not fall through to the ideal reset.
        if isinstance(channel, ResetChannelBase):
            return channel.process, qubits
        return qx.gates.RESET(dim=_dimension_for(qubits[0])), qubits

    def _emit_instruction(inst: Gate | Measurement | ResetQubit | Reset) -> None:
        """Resolve and emit a single instruction."""
        match inst:
            case Gate():
                op, qubits = _resolve_gate(inst)
                _emit_op(op, qubits)
            case Measurement():
                op, qubits = _resolve_measurement(inst)
                _emit_op(op, qubits)
            case ResetQubit():
                op, qubits = _resolve_reset_qubit(inst)
                _emit_op(op, qubits)
            case Reset():
                # A global RESET is the same physical operation as a targeted one on every
                # qubit, so it must consult the noise model per qubit rather than always
                # emitting the ideal reset -- otherwise a model carrying reset channels is
                # silently ignored for `RESET` while being honoured for `RESET <q>`.
                for q in all_qubits:
                    op, qubits = _resolve_reset_qubit(ResetQubit(Qubit(q)))
                    _emit_op(op, qubits)

    for inst in program.instructions:
        if isinstance(inst, UNSUPPORTED_INSTRUCTIONS):
            raise ValueError(
                f"{str(inst)!r} cannot be simulated: control flow and classical memory "
                "instructions change which quantum operations run, and this simulator evolves a "
                "straight-line program. Resolve the instruction's effect before simulating."
            )
        if isinstance(inst, DefCircuit):
            continue

        if isinstance(inst, Gate) and inst.name in circuit_definitions:
            channel = noise_model.get_channel(inst) if noise_model is not None else None

            if isinstance(channel, CycleChannel):
                # Expand using the channel's constituent operators.  A MeasurementChannel
                # constituent carries a QuantumInstrument and is emitted as one, exactly as a
                # standalone MEASURE is.
                for sub_ch in channel.channels:
                    # Use the channel's own `qubits` property rather than reaching through to
                    # `inst.get_qubit_indices()`, whose return type varies across the constituent
                    # families (a list for gates, a set-or-None for resets).
                    _emit_op(sub_ch.process, tuple(sub_ch.qubits))
            else:
                # Expand DEFCIRCUIT body and resolve each instruction.
                for expanded_inst in expand_defcircuit_body(inst, circuit_definitions[inst.name], circuit_definitions):
                    _emit_instruction(expanded_inst)
        elif isinstance(inst, (Gate, Measurement, ResetQubit, Reset)):
            _emit_instruction(inst)

    return tuple(ops), tuple(qubit_tuples), tuple(slots)


# ══════════════════════════════════════════════════════════
# Qubit remapping
# ══════════════════════════════════════════════════════════


def remap_qubits(
    qubit_tuples: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    qubit_indices: Mapping[int, int],
) -> tuple[tuple[int, ...], ...]:
    """Remap physical qubit IDs to 0-based indices.

    :param qubit_tuples: Physical qubit tuples from :func:`expand_program`.
    :param qubit_indices: Mapping from physical qubit id → 0-based index.
    :return: Remapped qubit tuples.
    :raises ValueError: If an operation touches a qubit outside ``qubit_indices``.
    """
    remapped: list[tuple[int, ...]] = []
    for qubits in qubit_tuples:
        missing = [q for q in qubits if q not in qubit_indices]
        if missing:
            raise ValueError(
                f"Operation on qubit(s) {missing} but the simulated register is "
                f"{sorted(qubit_indices)}. Pass qubits=[...] listing every qubit the program "
                "touches."
            )
        remapped.append(tuple(qubit_indices[q] for q in qubits))
    return tuple(remapped)


# ══════════════════════════════════════════════════════════
# Resolver
# ══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Resolution:
    """An expanded program, ready to be resolved for any parameter values.

    Holds the expanded operators, where each acts, the register dimensions and the parameter
    layout.  :meth:`resolve` binds a parameter vector and returns the corresponding
    :class:`~pyquil.simulation._circuit.Circuit`.

    :param dims: Inferred per-qudit dimensions (e.g. ``(2, 2, 3)``).
    :param ops: Expanded operators, one per operation, in program order.
    :param subsystems: 0-based qudit tuple each operator acts on, in operand order.
    :param parameters: The distinct ``(register_name, offset)`` references, one per slot of the
        parameter vector, in slot order.
    """

    dims: tuple[int, ...]
    ops: tuple[ExpandedOp, ...]
    subsystems: tuple[tuple[int, ...], ...]
    parameters: tuple[ParameterRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(int(d) for d in self.dims))
        object.__setattr__(self, "ops", tuple(self.ops))
        object.__setattr__(self, "subsystems", tuple(tuple(int(q) for q in sub) for sub in self.subsystems))
        object.__setattr__(self, "parameters", tuple((str(name), int(offset)) for name, offset in self.parameters))
        if len(self.ops) != len(self.subsystems):
            raise ValueError(f"{len(self.ops)} operator(s) but {len(self.subsystems)} subsystem(s).")

    @property
    def num_parameters(self) -> int:
        """The length of the parameter vector :meth:`resolve` expects."""
        return len(self.parameters)

    def placements(self, params: Array | None = None) -> tuple[Placement, ...]:
        """Bind *params* to produce one concrete operator per expanded operation.

        Fixed operators pass straight through; :class:`ParametricGate` entries are called with
        the parameter vector to build their ``Unitary``.

        :param params: Flat parameter vector, laid out as :attr:`parameters`.  Omit for a
            parameter-free program.
        :return: ``(operator, subsystem)`` pairs in program order.
        """
        if params is None:
            params = jnp.zeros(self.num_parameters, dtype=float)
        return tuple(
            (op(params) if isinstance(op, ParametricGate) else op, subsystem)
            for op, subsystem in zip(self.ops, self.subsystems, strict=True)
        )

    def resolve(self, params: Array | None = None) -> Circuit:
        """Bind *params* to produce the program's :class:`~pyquil.simulation._circuit.Circuit`.

        This is the hand-off out of Quil: past this point there are no gate names, no memory
        references and no instructions, only operators placed on a register.

        :param params: Flat parameter vector, laid out as :attr:`parameters`.  Omit for a
            parameter-free program.
        :return: The circuit, on a register of ``dims``.
        """
        return Circuit(dims=self.dims, ops=self.placements(params))


def resolve_program(
    program: Program,
    noise_model: NoiseModelLike | None = None,
    qubits: list[int] | None = None,
    dims: tuple[int, ...] | None = None,
) -> Resolution:
    """Expand a program and build its parameter-resolving template.

    Operators are returned in their most specific native type:

    * Ideal gates → ``qx.Unitary`` (parametric gates as a :class:`ParametricGate` callable)
    * Noisy gates (``Channel``) → ``qx.SuperOp``
    * Expanded cycle gates with ``CycleChannel`` noise → constituent ``qx.SuperOp``
    * Measurements → ``qx.QuantumInstrument``
    * Noisy/ideal resets → ``qx.SuperOp``

    Register dimensions are inferred from the gates and channels acting on each qudit (a
    qutrit gate or channel makes its qudit three-dimensional), so that ideal measurement and
    reset operators are built at the right size.  Pass *dims* to fix them instead.

    :param program: Quil program (may contain DEFCIRCUITs and DEFGATEs).
    :param noise_model: Optional noise model.
    :param qubits: Optional explicit qubit list, defining the register order. If ``None``,
        the program's qubits in ascending order.
    :param dims: Optional pre-determined per-qudit dimensions, in ``qubits`` order.
    :return: A :class:`Resolution`.
    """
    if qubits is None:
        qubits = sorted(program.get_qubit_indices())
    qubit_indices = {q: i for i, q in enumerate(qubits)}

    def expand(
        qubit_dimensions: Mapping[int, int] | None,
    ) -> tuple[tuple[ExpandedOp, ...], tuple[tuple[int, ...], ...], tuple[ParameterRef, ...]]:
        ops, phys_qubits, parameters = expand_program(program, noise_model, qubit_dimensions=qubit_dimensions)
        return ops, remap_qubits(phys_qubits, qubit_indices), parameters

    if dims is None:
        ops, subsystems, parameters = expand(None)
        probe = Resolution(dims=(), ops=ops, subsystems=subsystems, parameters=parameters)
        dims = Circuit.infer_dims(probe.placements(jnp.zeros(len(parameters))), len(qubits))

    qubit_dimensions = {q: dims[i] for q, i in qubit_indices.items()}
    ops, subsystems, parameters = expand(qubit_dimensions)
    return Resolution(dims, ops, subsystems, parameters)

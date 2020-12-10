from numbers import Real
import numpy as np

from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, Expression, Sub, Div
from pyquil.quilbase import (
    Declare,
    Gate,
    SetFrequency,
    ShiftFrequency,
    SetScale,
    SetPhase,
    ShiftPhase,
)
from rpcq.messages import ParameterSpec, ParameterAref, RewriteArithmeticResponse

from typing import Dict, Union, List, no_type_check


# TODO
# The various reassignments to the variable expr make it difficult
# to tie the typing down.
@no_type_check
def rewrite_arithmetic(prog: Program) -> RewriteArithmeticResponse:
    """Rewrite compound arithmetic expressions.

    The basic motivation is that a parametric program may have gates with
    compound arguments which cannot be evaluated natively on the underlying
    control hardware. The solution provided here is to translate a program like

      DECLARE theta REAL
      DECLARE beta REAL
      RZ(3 * theta) 0
      RZ(beta+theta) 0

    into something like

      DECLARE theta REAL
      DECLARE beta REAL
      DECLARE __P REAL[2]
      RZ(__P[0]) 0
      RZ(__P[1]) 0

    along with a "recalculation table" mapping new memory references to their
    corresponding arithmetic expressions,

      {
        ParameterAref('__P', 0): "((3.0)*theta[0])",
        ParameterAref('__P', 1): "(beta[0]+theta[0])"
      }

    When executing the parametric program with specific values for `theta` and
    `beta`, the PyQuil client will patch in values for `__P` by evaluating the
    expressions in the recalculation table.

    :param prog: A program.
    :returns: A RewriteArithmeticResponse, containing the updated program along
      with its memory descriptors and a recalculation table.

    """

    def spec(inst: Declare) -> ParameterSpec:
        return ParameterSpec(type=inst.memory_type, length=inst.memory_size)

    def aref(ref: MemoryReference) -> ParameterAref:
        return ParameterAref(name=ref.name, index=ref.offset)

    updated = prog.copy_everything_except_instructions()
    old_descriptors = {inst.name: spec(inst) for inst in prog if isinstance(inst, Declare)}
    recalculation_table: Dict[ParameterAref, str] = {}
    seen_exprs: Dict[str, MemoryReference] = {}

    # generate a unique name. it's nice to do this in a deterministic fashion
    # rather than globbing in a UUID
    suffix = len(old_descriptors)
    while f"__P{suffix}" in old_descriptors:
        suffix += 1
    mref_name = f"__P{suffix}"
    mref_idx = 0

    def expr_mref(expr: object) -> MemoryReference:
        """ Get a suitable MemoryReference for a given expression. """
        nonlocal mref_idx
        expr = str(expr)
        if expr in seen_exprs:
            return seen_exprs[expr]
        new_mref = MemoryReference(mref_name, mref_idx)
        seen_exprs[expr] = new_mref
        mref_idx += 1
        recalculation_table[aref(new_mref)] = expr
        return new_mref

    for inst in prog:
        if isinstance(inst, Gate):
            new_params: List[Union[Real, MemoryReference]] = []
            for param in inst.params:
                if isinstance(param, Real):
                    new_params.append(param)
                elif isinstance(param, Expression):
                    # Quil gate angles are in radians,
                    # but downstream processing expects revolutions
                    expr = str(Div(param, 2 * np.pi))
                    new_params.append(expr_mref(expr))
                else:
                    raise ValueError(f"Unknown parameter type {type(param)} in {inst}.")
            updated.inst(Gate(inst.name, new_params, inst.qubits))
        elif isinstance(inst, (SetFrequency, ShiftFrequency)):
            if isinstance(inst.freq, Real):
                updated.inst(inst)
                continue
            try:
                fdefn = prog.frames[inst.frame]
            except KeyError:
                raise ValueError(f"Unable to rewrite {inst} without DEFFRAME {inst.frame}.")
            if fdefn.sample_rate is None:
                raise ValueError(f"Unable to rewrite {inst} on frame with undefined SAMPLE-RATE.")
            if fdefn.center_frequency:
                expr = Sub(inst.freq, fdefn.center_frequency)
            else:
                expr = inst.freq
            expr = Div(expr, fdefn.sample_rate)
            expr = str(expr)
            updated.inst(inst.__class__(inst.frame, expr_mref(expr)))
        elif isinstance(inst, (SetPhase, ShiftPhase)):
            if isinstance(inst.phase, Real):
                updated.inst(inst)
            else:
                # Quil phases are in radians
                # but downstream processing expects revolutions
                expr = str(Div(inst.phase, 2 * np.pi))
                updated.inst(inst.__class__(inst.frame, expr_mref(expr)))
        elif isinstance(inst, SetScale):
            if isinstance(inst.scale, Real):
                updated.inst(inst)
            else:
                # scale is in [-4,4)
                # binary patching assumes periodic with period 1
                # so we divide by 8...
                expr = str(Div(inst.scale, 8))
                updated.inst(SetScale(inst.frame, expr_mref(expr)))
        else:
            updated.inst(inst)

    if mref_idx > 0:
        updated._instructions.insert(0, Declare(mref_name, "REAL", mref_idx))

    return RewriteArithmeticResponse(
        quil=updated.out(),
        original_memory_descriptors=old_descriptors,
        recalculation_table=recalculation_table,
    )

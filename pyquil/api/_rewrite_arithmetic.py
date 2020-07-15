from numbers import Real

from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, Expression
from pyquil.quilbase import (
    Declare,
    Gate,
)
from rpcq.messages import ParameterSpec, ParameterAref, RewriteArithmeticResponse

from typing import Dict


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
    recalculation_table = {}
    seen_exprs: Dict[str, MemoryReference] = {}

    # generate a unique name. it's nice to do this in a deterministic fashion
    # rather than globbing in a UUID
    suffix = len(old_descriptors)
    while f"__P{suffix}" in old_descriptors:
        suffix += 1
    mref_name = f"__P{suffix}"
    mref_idx = 0

    for inst in prog:
        if isinstance(inst, Gate):
            new_params = []
            for param in inst.params:
                if isinstance(param, (Real, MemoryReference)):
                    new_params.append(param)
                elif isinstance(param, Expression):
                    expr = str(param)
                    if expr in seen_exprs:
                        new_params.append(seen_exprs[expr])
                    else:
                        new_mref = MemoryReference(mref_name, mref_idx)
                        seen_exprs[expr] = new_mref
                        mref_idx += 1
                        recalculation_table[aref(new_mref)] = expr
                        new_params.append(new_mref)
                else:
                    raise ValueError(f"Unknown parameter type {type(param)} in {inst}.")
            updated.inst(Gate(inst.name, new_params, inst.qubits))
        else:
            updated.inst(inst)

    if mref_idx > 0:
        updated._instructions.insert(0, Declare(mref_name, "REAL", mref_idx))

    return RewriteArithmeticResponse(
        quil=updated.out(),
        original_memory_descriptors=old_descriptors,
        recalculation_table=recalculation_table,
    )

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
"""Quil arithmetic expressions as gate arguments in the quax-based simulators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyquil.gates import FSIM, PHASE, RX, RY, RZ, H
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, Parameter, quil_cis, quil_cos, quil_exp, quil_sin, quil_sqrt
from pyquil.quilbase import Declare, DefGate, Gate
from pyquil.simulation._resolver import ParameterExpression, _compile_expression, expand_program
from pyquil.simulation._simulator import DensityMatrixSimulator, PureStateVectorSimulator

THETA = MemoryReference("theta", 0)
PHI = MemoryReference("theta", 1)


def _program(*gates):
    program = Program()
    program += Declare("theta", "REAL", 2)
    for gate in gates:
        program += gate
    return program


def _state(program, **memory):
    sim = PureStateVectorSimulator(program)
    return np.asarray(sim.compute(sim.linearize(memory)).matrix).reshape(-1)


def _compile(expression):
    return _compile_expression(expression, lambda ref: ref.offset)


class TestCompileExpression:
    @pytest.mark.parametrize(
        ("expression", "key", "slots"),
        [
            (THETA, "%0", (0,)),
            (THETA / 2 + np.pi, "((%0/2.0)+3.141592653589793)", (0,)),
            (2 * PHI, "(2.0*%0)", (1,)),
            (-THETA, "(-1.0*%0)", (0,)),
            (THETA + THETA, "(%0+%0)", (0,)),
            (THETA * PHI, "(%0*%1)", (0, 1)),
            (PHI - THETA, "(%0-%1)", (1, 0)),
            (quil_sin(THETA), "SIN(%0)", (0,)),
            (quil_cos(2 * THETA), "COS((2.0*%0))", (0,)),
            (quil_sqrt(THETA), "SQRT(%0)", (0,)),
            (quil_exp(-THETA), "EXP((-1.0*%0))", (0,)),
            (quil_cis(THETA), "CIS(%0)", (0,)),
            (THETA**2, "(%0^2.0)", (0,)),
        ],
    )
    def test_key_and_slots(self, expression, key, slots):
        compiled = _compile(expression)
        assert compiled.key == key
        assert compiled.slot_indices == slots

    def test_same_shape_different_slots_share_a_key(self):
        assert _compile(quil_sin(THETA)).key == _compile(quil_sin(PHI)).key
        assert _compile(THETA / 2).key == _compile(PHI / 2).key
        assert _compile(THETA / 2).key != _compile(THETA / 3).key

    @pytest.mark.parametrize(
        ("expression", "is_complex"),
        [(THETA, False), (quil_sin(THETA), False), (THETA**2, False), (quil_cis(THETA), True), (1j * THETA, True)],
    )
    def test_complex_flag(self, expression, is_complex):
        assert _compile(expression).is_complex is is_complex

    @pytest.mark.parametrize(
        ("expression", "theta", "expected"),
        [
            (THETA / 2 + np.pi, 0.4, 0.2 + np.pi),
            (quil_sin(THETA) * quil_cos(THETA), 0.7, np.sin(0.7) * np.cos(0.7)),
            (quil_sqrt(THETA), 0.81, 0.9),
            (quil_exp(-THETA), 1.5, np.exp(-1.5)),
            (THETA**2 - 2**THETA, 1.5, 1.5**2 - 2**1.5),
            (quil_cis(THETA), 0.3, np.exp(0.3j)),
            ((1 + 2j) * THETA, 0.5, 0.5 + 1j),
            (quil_sqrt(2) * THETA, 1.0, np.sqrt(2)),
        ],
    )
    def test_evaluates_like_quil(self, expression, theta, expected):
        compiled = _compile(expression)
        np.testing.assert_allclose(compiled(jnp.array([theta, 0.0])), expected, atol=1e-12)

    def test_unbound_defgate_parameter_reports_clearly(self):
        with pytest.raises(ValueError, match="Unbound DEFGATE parameter"):
            _compile(THETA + Parameter("p"))


class TestSimulation:
    @pytest.mark.parametrize(
        ("gate", "literal"),
        [
            (RX(THETA / 2 + np.pi, 0), lambda t, p: RX(t / 2 + np.pi, 0)),
            (RZ(2 * PHI, 0), lambda t, p: RZ(2 * p, 0)),
            (RX(-THETA, 0), lambda t, p: RX(-t, 0)),
            (RY(np.pi - PHI, 0), lambda t, p: RY(np.pi - p, 0)),
            (RX(THETA + THETA, 0), lambda t, p: RX(2 * t, 0)),
            (RX(THETA * PHI, 0), lambda t, p: RX(t * p, 0)),
            (RX(quil_sin(THETA), 0), lambda t, p: RX(np.sin(t), 0)),
            (RY(np.pi * quil_cos(PHI), 0), lambda t, p: RY(np.pi * np.cos(p), 0)),
            (RZ(quil_sqrt(THETA), 0), lambda t, p: RZ(np.sqrt(t), 0)),
            (RX(quil_exp(-THETA), 0), lambda t, p: RX(np.exp(-t), 0)),
            (RX(THETA**2, 0), lambda t, p: RX(t**2, 0)),
            (RX(quil_sqrt(2) * THETA, 0), lambda t, p: RX(np.sqrt(2) * t, 0)),
        ],
        ids=[
            "half_plus_pi",
            "double",
            "negated",
            "pi_minus",
            "same_reference_twice",
            "product_of_references",
            "sin",
            "pi_cos",
            "sqrt",
            "exp",
            "power",
            "sqrt2_constant",
        ],
    )
    def test_matches_the_literal_gate(self, gate, literal):
        theta, phi = 0.37, 1.21
        got = _state(_program(H(0), gate), theta=[theta, phi])
        expected = _state(Program(H(0), literal(theta, phi)))
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_parsed_quilc_style_program(self):
        """The arithmetic quilc emits for a compiled parametric program simulates directly."""
        program = Program(
            "DECLARE theta REAL[1]\nRX((theta[0]/2)+pi) 0\nRZ(-2*theta[0]) 0\nRY(SIN(theta[0])*COS(pi/3)) 0"
        )
        theta = 0.6
        got = _state(program, theta=[theta])
        expected = _state(
            Program(RX(theta / 2 + np.pi, 0), RZ(-2 * theta, 0), RY(np.sin(theta) * np.cos(np.pi / 3), 0))
        )
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_cis_and_complex_literals_in_a_defgate_argument(self):
        """A DEFGATE with a complex parameter accepts CIS and complex arithmetic."""
        z = Parameter("z")
        defgate = DefGate("CPH", [[1, 0], [0, z]], [z])
        theta = 0.8

        def run(argument):
            program = _program()
            program += defgate
            program += H(0)
            program += Gate("CPH", [argument], [0])
            return _state(program, theta=[theta, 0.0])

        expected = _state(Program(H(0), PHASE(theta, 0)))
        np.testing.assert_allclose(run(quil_cis(THETA)), expected, atol=1e-12)
        # 1i * CIS(theta) == CIS(theta + pi/2)
        np.testing.assert_allclose(
            run(1j * quil_cis(THETA)), _state(Program(H(0), PHASE(theta + np.pi / 2, 0))), atol=1e-12
        )
        # A complex literal times a real reference, folded through a real result: (1i)*(-1i) == 1.
        np.testing.assert_allclose(run(quil_cis(1j * (-1j) * THETA)), expected, atol=1e-12)

    def test_complex_argument_to_a_builtin_gate_reports_clearly(self):
        with pytest.raises(ValueError, match="complex-valued"):
            PureStateVectorSimulator(_program(RX(quil_cis(THETA), 0)))
        with pytest.raises(ValueError, match="complex-valued"):
            PureStateVectorSimulator(_program(RX(1j * THETA, 0)))

    def test_complex_literal_beside_an_expression_argument_reports_clearly(self):
        """A complex *literal* is rejected too, not only a complex expression."""
        with pytest.raises(ValueError, match="complex-valued"):
            PureStateVectorSimulator(_program(FSIM(THETA, 1j, 0, 1)))

    def test_unbound_defgate_parameter_names_the_instruction(self):
        with pytest.raises(ValueError, match=r"RX\(.*\) 0'?: Unbound DEFGATE parameter"):
            PureStateVectorSimulator(_program(RX(THETA + Parameter("p"), 0)))

    def test_feed_forward_parameter_still_rejected(self):
        from pyquil.gates import MEASURE

        program = Program()
        program += Declare("ro", "BIT", 1)
        program += MEASURE(0, ("ro", 0))
        program += RX(2 * MemoryReference("ro", 0), 0)
        with pytest.raises(ValueError, match="written by a MEASURE"):
            DensityMatrixSimulator(program)

    def test_same_shape_expressions_share_one_slot_each_and_batch(self):
        program = _program(RX(quil_sin(THETA), 0), RX(quil_sin(PHI), 1), RX(THETA, 2))
        sim = PureStateVectorSimulator(program)
        assert sim.parameters == (("theta", 0), ("theta", 1))
        theta, phi = 0.5, 0.9
        got = _state(program, theta=[theta, phi])
        expected = _state(Program(RX(np.sin(theta), 0), RX(np.sin(phi), 1), RX(theta, 2)))
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_grad_and_jit(self):
        """d/dtheta P(|1>) for RX(SIN(theta)) is sin(sin theta) cos(theta) / 2."""
        sim = DensityMatrixSimulator(_program(RX(quil_sin(THETA), 0)))

        def excited(params):
            return jnp.real(sim.compute(params).matrix[1, 1])

        theta = 0.8
        params = jnp.array([theta])
        np.testing.assert_allclose(jax.jit(excited)(params), np.sin(np.sin(theta) / 2) ** 2, atol=1e-12)
        grad = jax.grad(excited)(params)
        np.testing.assert_allclose(grad, [np.sin(np.sin(theta)) * np.cos(theta) / 2], atol=1e-12)

    def test_expand_program_records_the_expression(self):
        ops, _, parameters = expand_program(_program(RX(3 * THETA - 1, 0)))
        (gate,) = ops
        assert parameters == (("theta", 0),)
        (argument,) = gate.arguments
        assert isinstance(argument, ParameterExpression)
        assert argument.slot_indices == (0,)
        assert argument.key == "((3.0*%0)-1.0)"

    def test_literal_expressions_are_folded(self):
        """A parameter-free expression such as SIN(pi/4) is a literal, not a parameter."""
        program = Program("RX(SIN(pi/4)) 0")
        sim = PureStateVectorSimulator(program)
        assert sim.parameters == ()
        np.testing.assert_allclose(_state(program), _state(Program(RX(np.sin(np.pi / 4), 0))), atol=1e-12)

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
"""Affine gate-parameter expressions (``a * theta + b``) in the quax-based simulators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyquil.gates import RX, RY, RZ, H
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, quil_sin
from pyquil.quilbase import Declare
from pyquil.simulation._resolver import _affine_form, expand_program
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


class TestAffineForm:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            (THETA, (THETA, 1.0, 0.0)),
            (THETA / 2 + np.pi, (THETA, 0.5, np.pi)),
            (2 * PHI, (PHI, 2.0, 0.0)),
            (-THETA, (THETA, -1.0, 0.0)),
            (np.pi - PHI, (PHI, -1.0, np.pi)),
            (THETA + THETA, (THETA, 2.0, 0.0)),
            ((3 * THETA - 1) / 4, (THETA, 0.75, -0.25)),
            (0.7, (None, 0.0, 0.7)),
        ],
    )
    def test_recognises_affine_expressions(self, expression, expected):
        ref, scale, offset = _affine_form(expression)
        assert ref == expected[0]
        assert scale == pytest.approx(expected[1])
        assert offset == pytest.approx(expected[2])

    @pytest.mark.parametrize(
        "expression",
        [THETA * PHI, THETA * THETA, quil_sin(THETA), THETA**2, 2 / THETA, THETA + PHI],
        ids=["product", "square", "sin", "power", "reciprocal", "two_references"],
    )
    def test_rejects_non_affine_expressions(self, expression):
        assert _affine_form(expression) is None


class TestSimulation:
    @pytest.mark.parametrize(
        ("gate", "literal"),
        [
            (RX(THETA / 2 + np.pi, 0), lambda t, p: RX(t / 2 + np.pi, 0)),
            (RZ(2 * PHI, 0), lambda t, p: RZ(2 * p, 0)),
            (RX(-THETA, 0), lambda t, p: RX(-t, 0)),
            (RY(np.pi - PHI, 0), lambda t, p: RY(np.pi - p, 0)),
            (RX(THETA + THETA, 0), lambda t, p: RX(2 * t, 0)),
        ],
        ids=["half_plus_pi", "double", "negated", "pi_minus", "same_reference_twice"],
    )
    def test_matches_the_literal_gate(self, gate, literal):
        theta, phi = 0.37, 1.21
        got = _state(_program(H(0), gate), theta=[theta, phi])
        expected = _state(Program(H(0), literal(theta, phi)))
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_parsed_quilc_style_program(self):
        """The arithmetic quilc emits for a compiled parametric program simulates directly."""
        program = Program("DECLARE theta REAL[1]\nRX((theta[0]/2)+pi) 0\nRZ(-2*theta[0]) 0")
        theta = 0.6
        got = _state(program, theta=[theta])
        expected = _state(Program(RX(theta / 2 + np.pi, 0), RZ(-2 * theta, 0)))
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_shares_a_slot_and_batches_with_a_plain_reference(self):
        """``RX(theta[0]/2)`` and ``RX(theta[1])`` use one slot each and one gate batch."""
        program = _program(RX(THETA / 2, 0), RX(PHI, 1), RX(THETA, 2))
        sim = PureStateVectorSimulator(program)
        assert sim.parameters == (("theta", 0), ("theta", 1))
        theta, phi = 0.5, 0.9
        got = _state(program, theta=[theta, phi])
        expected = _state(Program(RX(theta / 2, 0), RX(phi, 1), RX(theta, 2)))
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_grad_and_jit(self):
        """d/dtheta P(|1>) for RX(theta/2 + pi) is -sin(theta/2)/4 (the pi flips the population)."""
        sim = DensityMatrixSimulator(_program(RX(THETA / 2 + np.pi, 0)))

        def excited(params):
            return jnp.real(sim.compute(params).matrix[1, 1])

        theta = 0.8
        params = jnp.array([theta])
        np.testing.assert_allclose(jax.jit(excited)(params), np.cos(theta / 4) ** 2, atol=1e-12)
        grad = jax.grad(excited)(params)
        np.testing.assert_allclose(grad, [-np.sin(theta / 2) / 4], atol=1e-12)

    def test_expand_program_records_scale_and_offset(self):
        ops, _, parameters = expand_program(_program(RX(3 * THETA - 1, 0)))
        (gate,) = ops
        assert parameters == (("theta", 0),)
        assert gate.param_indices == (0,)
        assert gate.scales == (3.0,)
        assert gate.offsets == (-1.0,)

    @pytest.mark.parametrize("expression", [THETA * PHI, quil_sin(THETA), THETA**2], ids=["product", "sin", "power"])
    def test_non_affine_parameter_reports_clearly(self, expression):
        with pytest.raises(ValueError, match="not an affine expression"):
            PureStateVectorSimulator(_program(RX(expression, 0)))

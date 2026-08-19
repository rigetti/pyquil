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

"""jit/grad contract for the grad-able simulators.

Both :class:`PureStateVectorSimulator` and :class:`DensityMatrixSimulator` advertise
``compute`` as directly usable under ``jax.jit`` and ``jax.grad``. These tests pin that
claim: gradients are checked against central finite differences, and jitted results against
their eager counterparts. Without them the scan/switch machinery could silently lose
differentiability (e.g. by introducing a Python-level branch on a traced value) while every
value-based test kept passing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyquil.gates import CNOT, RX, RY, RZ, H, X
from pyquil.noise._channels import Channel
from pyquil.noise._noise_model import NoiseModel
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare
from pyquil.simulation._simulator import DensityMatrixSimulator, PureStateVectorSimulator

_Z = np.diag([1.0, -1.0]).astype(complex)
_FD_STEP = 1e-6
_FD_TOL = 1e-6


def _parametric_program(n_params, n_qubits=1):
    """A program with ``n_params`` distinct runtime angles, entangling when multi-qubit."""
    program = Program(Declare("theta", "REAL", n_params))
    for i in range(n_params):
        gate = (RX, RY, RZ)[i % 3]
        program += gate(MemoryReference("theta", i), i % n_qubits)
        if n_qubits > 1 and i % n_qubits == n_qubits - 1:
            program += CNOT(0, n_qubits - 1)
    return program


def _central_difference(fn, params, index, step=_FD_STEP):
    plus = params.at[index].add(step)
    minus = params.at[index].add(-step)
    return (fn(plus) - fn(minus)) / (2 * step)


def _z_on_first_qubit(dims):
    """Observable Z on the most significant subsystem, identity elsewhere (big-endian)."""
    observable = _Z
    for d in dims[1:]:
        observable = np.kron(observable, np.eye(d))
    return jnp.asarray(observable)


class TestStateVectorJitAndGrad:
    @pytest.fixture
    def sim(self):
        return PureStateVectorSimulator(_parametric_program(3), qubits=[0])

    @staticmethod
    def _expectation(sim, params):
        psi = sim.compute(params).matrix.reshape(-1)
        observable = _z_on_first_qubit(sim.dims)
        return jnp.real(jnp.vdot(psi, observable @ psi))

    def test_jit_matches_eager(self, sim):
        params = jnp.array([0.3, 1.1, 0.7])
        eager = np.asarray(sim.compute(params).matrix)
        jitted = np.asarray(jax.jit(sim.compute)(params).matrix)
        np.testing.assert_allclose(jitted, eager, atol=1e-12)

    def test_jit_recompiles_correctly_for_new_values(self, sim):
        """A jitted callable must track parameter values, not bake the first ones in."""
        jitted = jax.jit(sim.compute)
        for values in ([0.3, 1.1, 0.7], [0.9, 0.2, 1.4], [0.0, 0.0, 0.0]):
            params = jnp.array(values)
            np.testing.assert_allclose(
                np.asarray(jitted(params).matrix), np.asarray(sim.compute(params).matrix), atol=1e-12
            )

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_grad_matches_finite_difference(self, sim, index):
        params = jnp.array([0.3, 1.1, 0.7])

        def fn(p):
            return self._expectation(sim, p)

        analytic = np.asarray(jax.grad(fn)(params))[index]
        numeric = float(_central_difference(fn, params, index))
        assert analytic == pytest.approx(numeric, abs=_FD_TOL)

    def test_grad_under_jit_matches_grad(self, sim):
        params = jnp.array([0.3, 1.1, 0.7])

        def fn(p):
            return self._expectation(sim, p)

        np.testing.assert_allclose(
            np.asarray(jax.jit(jax.grad(fn))(params)), np.asarray(jax.grad(fn)(params)), atol=1e-10
        )

    def test_grad_of_analytically_known_expectation(self):
        """For ``RY(theta)`` on |0>, <Z> = cos(theta) so the derivative is -sin(theta)."""
        sim = PureStateVectorSimulator(_parametric_program(1), qubits=[0])

        def fn(p):
            return self._expectation(sim, p)

        for theta in (0.0, 0.4, 1.3, np.pi / 2):
            params = jnp.array([theta])
            assert float(fn(params)) == pytest.approx(np.cos(theta), abs=1e-9)
            assert float(jax.grad(fn)(params)[0]) == pytest.approx(-np.sin(theta), abs=1e-9)

    def test_second_derivative(self):
        """``RY``'s <Z> is cos(theta), so the second derivative is -cos(theta)."""
        sim = PureStateVectorSimulator(_parametric_program(1), qubits=[0])

        def fn(theta):
            return self._expectation(sim, jnp.array([theta]))

        for theta in (0.4, 1.3):
            assert float(jax.grad(jax.grad(fn))(theta)) == pytest.approx(-np.cos(theta), abs=1e-7)

    def test_vmap_over_parameter_batch(self):
        sim = PureStateVectorSimulator(_parametric_program(1), qubits=[0])
        batch = jnp.array([[0.1], [0.5], [1.2]])
        batched = jax.vmap(lambda p: sim.compute(p).matrix)(batch)
        for i, theta in enumerate([0.1, 0.5, 1.2]):
            np.testing.assert_allclose(
                np.asarray(batched[i]).reshape(-1),
                np.asarray(sim.compute(jnp.array([theta])).matrix).reshape(-1),
                atol=1e-12,
            )

    def test_grad_through_multi_qubit_entangling_program(self):
        sim = PureStateVectorSimulator(_parametric_program(4, n_qubits=2), qubits=[0, 1])
        params = jnp.array([0.3, 1.1, 0.7, 0.2])

        def fn(p):
            return self._expectation(sim, p)

        analytic = np.asarray(jax.grad(fn)(params))
        for index in range(params.size):
            assert analytic[index] == pytest.approx(float(_central_difference(fn, params, index)), abs=_FD_TOL)

    def test_unitary_is_jittable_and_differentiable(self):
        sim = PureStateVectorSimulator(_parametric_program(2), qubits=[0])
        params = jnp.array([0.3, 1.1])
        np.testing.assert_allclose(
            np.asarray(jax.jit(sim.unitary)(params).matrix), np.asarray(sim.unitary(params).matrix), atol=1e-12
        )

        def trace_real(p):
            return jnp.real(jnp.trace(sim.unitary(p).matrix))

        analytic = np.asarray(jax.grad(trace_real)(params))
        for index in range(params.size):
            assert analytic[index] == pytest.approx(float(_central_difference(trace_real, params, index)), abs=_FD_TOL)


class TestDensityMatrixJitAndGrad:
    @pytest.fixture
    def sim(self):
        return DensityMatrixSimulator(_parametric_program(3), qubits=[0])

    @staticmethod
    def _expectation(sim, params):
        rho = sim.compute(params).matrix
        return jnp.real(jnp.trace(rho @ _z_on_first_qubit(sim.dims)))

    def test_jit_matches_eager(self, sim):
        params = jnp.array([0.3, 1.1, 0.7])
        np.testing.assert_allclose(
            np.asarray(jax.jit(sim.compute)(params).matrix), np.asarray(sim.compute(params).matrix), atol=1e-12
        )

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_grad_matches_finite_difference(self, sim, index):
        params = jnp.array([0.3, 1.1, 0.7])

        def fn(p):
            return self._expectation(sim, p)

        analytic = np.asarray(jax.grad(fn)(params))[index]
        assert analytic == pytest.approx(float(_central_difference(fn, params, index)), abs=_FD_TOL)

    def test_grad_of_analytically_known_expectation(self):
        sim = DensityMatrixSimulator(_parametric_program(1), qubits=[0])

        def fn(p):
            return self._expectation(sim, p)

        for theta in (0.0, 0.4, 1.3):
            assert float(fn(jnp.array([theta]))) == pytest.approx(np.cos(theta), abs=1e-9)
            assert float(jax.grad(fn)(jnp.array([theta]))[0]) == pytest.approx(-np.sin(theta), abs=1e-9)

    def test_grad_survives_a_noise_model(self):
        """A gradient must flow through a noisy (non-unitary) operation.

        The noise sits on a *literal-angle* gate downstream of the parametric one, because a
        channel cannot currently be attached to a runtime-parametric gate at all -- see
        :meth:`test_noise_model_does_not_apply_to_parametric_gates`.

        Depolarizing with constant p scales the Bloch vector by p, so <Z> and hence its
        derivative are both scaled by exactly p.
        """
        shrink = 0.8
        program = Program(Declare("theta", "REAL", 1), RY(MemoryReference("theta", 0), 0), X(0))
        noiseless = DensityMatrixSimulator(program, qubits=[0])
        noisy = DensityMatrixSimulator(
            program,
            qubits=[0],
            noise_model=NoiseModel.from_channels([Channel.from_depolarizing_constant(X(0), shrink)]),
        )

        def fn(sim):
            return lambda p: self._expectation(sim, p)

        params = jnp.array([0.7])
        clean_grad = float(jax.grad(fn(noiseless))(params)[0])
        noisy_grad = float(jax.grad(fn(noisy))(params)[0])
        assert noisy_grad == pytest.approx(float(_central_difference(fn(noisy), params, 0)), abs=_FD_TOL)
        assert noisy_grad == pytest.approx(shrink * clean_grad, abs=1e-6)
        assert abs(noisy_grad) < abs(clean_grad), "noise must damp the gradient"

    def test_noise_model_does_not_apply_to_parametric_gates(self):
        """Noise on a runtime-parametric gate is intentionally unsupported.

        Channels are keyed by instruction, and a program's parametric instruction is
        ``RX(theta[0]) 0`` -- not equal to any concrete ``RX(angle) 0``. A channel also cannot be
        *built* from the symbolic gate, since resolving its ideal unitary needs a number. A noise
        model passed alongside a parametric program is therefore inert for that gate.

        This is a deliberate design decision, not a gap to be closed. On real hardware the only
        genuinely continuous gate is ``RZ``, which is virtual -- implemented as a frame change
        with no pulse and hence no duration and no noise. Every other nominally parametric gate
        (``RX``, ``RY``, two-qubit rotations) is calibrated at a fixed set of angles in practice,
        so its noise is naturally described by a channel keyed on that concrete instruction.

        The test pins the behaviour so a future change to it is a considered one.
        """
        program = Program(Declare("theta", "REAL", 1), RX(MemoryReference("theta", 0), 0))
        symbolic = program.instructions[-1]
        noise_model = NoiseModel.from_channels([Channel.from_depolarizing_constant(RX(0.6, 0), 0.8)])

        # The concrete channel does not match the symbolic instruction...
        assert noise_model.get_channel(symbolic) is None
        # ...so the noisy and noiseless simulations coincide.
        params = jnp.array([0.6])
        noisy = np.asarray(DensityMatrixSimulator(program, qubits=[0], noise_model=noise_model).compute(params).matrix)
        clean = np.asarray(DensityMatrixSimulator(program, qubits=[0]).compute(params).matrix)
        np.testing.assert_allclose(noisy, clean, atol=1e-12)
        # ...and building a channel from the symbolic gate is not possible either.
        with pytest.raises(ValueError, match="Cannot evaluate expression"):
            Channel.from_depolarizing_constant(symbolic, 0.8)

    def test_grad_through_measurement_dephasing(self):
        """MEASURE is a dephasing superop, so it must stay differentiable."""
        from pyquil.gates import MEASURE

        program = Program(Declare("theta", "REAL", 1), RY(MemoryReference("theta", 0), 0), MEASURE(0, None))
        sim = DensityMatrixSimulator(program, qubits=[0])

        def fn(p):
            return self._expectation(sim, p)

        params = jnp.array([0.7])
        # Populations survive dephasing, so <Z> is still cos(theta).
        assert float(fn(params)) == pytest.approx(np.cos(0.7), abs=1e-9)
        assert float(jax.grad(fn)(params)[0]) == pytest.approx(-np.sin(0.7), abs=1e-7)

    def test_grad_through_multi_qubit_noisy_program(self):
        program = _parametric_program(4, n_qubits=2)
        sim = DensityMatrixSimulator(
            program,
            qubits=[0, 1],
            noise_model=NoiseModel.from_channels([Channel.from_depolarizing_constant(CNOT(0, 1), 0.9)]),
        )
        params = jnp.array([0.3, 1.1, 0.7, 0.2])

        def fn(p):
            return self._expectation(sim, p)

        analytic = np.asarray(jax.grad(fn)(params))
        for index in range(params.size):
            assert analytic[index] == pytest.approx(float(_central_difference(fn, params, index)), abs=_FD_TOL)

    def test_vmap_over_parameter_batch(self):
        sim = DensityMatrixSimulator(_parametric_program(1), qubits=[0])
        batch = jnp.array([[0.1], [0.5], [1.2]])
        batched = jax.vmap(lambda p: sim.compute(p).matrix)(batch)
        for i, theta in enumerate([0.1, 0.5, 1.2]):
            np.testing.assert_allclose(
                np.asarray(batched[i]), np.asarray(sim.compute(jnp.array([theta])).matrix), atol=1e-12
            )

    def test_parameter_free_program_is_jittable(self):
        """The constant-stack fast path must still work under jit."""
        sim = DensityMatrixSimulator(Program(H(0), CNOT(0, 1)), qubits=[0, 1])
        eager = np.asarray(sim.compute().matrix)
        np.testing.assert_allclose(np.asarray(jax.jit(sim.compute)().matrix), eager, atol=1e-12)
        # And repeated calls (which reuse the cached stack) stay correct.
        np.testing.assert_allclose(np.asarray(sim.compute().matrix), eager, atol=1e-12)

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
"""Scaling guards for the quax-backed simulators.

Two things these protect that no correctness test can:

* **Compile time must not scale with circuit depth.** Same-shaped gates are built under one
  ``jax.vmap`` and merged operations dispatch through one ``jax.lax.switch``, so the traced
  graph grows with the number of distinct gate *kinds* and *subsystems*, not with the number
  of gates.  If that batching breaks, every numerical test still passes — a slow graph
  computes the right answer — and only the equation count moves.  ``test_graph_size_is_flat_*``
  asserts it directly, and is the cheapest early warning available.
* **Construction and first evaluation are both on the critical path.** Preparation is eager,
  so work moved out of the first ``compute`` and into the constructor. Benchmarking either in
  isolation is misleading; ``test_first_shot_*`` measures the sum, which is what a caller who
  simulates a program once actually pays.
"""

import jax
import pytest

from pyquil import Program
from pyquil.gates import CZ, RX
from pyquil.quil import Declare
from pyquil.quilatom import MemoryReference
from pyquil.simulation._simulator import DensityMatrixSimulator, PureStateVectorSimulator


def brickwork(n_qubits: int, n_layers: int, *, parametric: bool) -> Program:
    """A layered circuit with alternating-offset entanglers, so merges are non-trivial."""
    program = Program()
    if parametric:
        program += Declare("theta", "REAL", n_qubits * n_layers)
    slot = 0
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            angle = MemoryReference("theta", slot) if parametric else 0.1 + 0.01 * slot
            program += RX(angle, qubit)
            slot += 1
        for qubit in range(layer % 2, n_qubits - 1, 2):
            program += CZ(qubit, qubit + 1)
    return program


def _params(simulator, n_qubits, n_layers):
    return simulator.linearize({"theta": [0.1 + 0.001 * i for i in range(n_qubits * n_layers)]})


def _graph_size(simulator, params):
    return len(jax.make_jaxpr(simulator.compute)(params).jaxpr.eqns)


@pytest.mark.parametrize(
    ("simulator_cls", "n_qubits"),
    [(PureStateVectorSimulator, 5), (DensityMatrixSimulator, 3)],
    ids=["state_vector", "density_matrix"],
)
def test_graph_size_is_flat_in_depth(simulator_cls, n_qubits):
    """Ten times the gates must not mean a bigger traced graph."""
    sizes = []
    for n_layers in (2, 20):
        program = brickwork(n_qubits, n_layers, parametric=True)
        simulator = simulator_cls(program)
        sizes.append(_graph_size(simulator, _params(simulator, n_qubits, n_layers)))
    assert sizes[0] == sizes[1], (
        f"traced graph grew with depth ({sizes[0]} -> {sizes[1]} equations): gate batching has "
        "regressed, even though every numerical test will still pass"
    )


def test_graph_size_is_flat_in_gate_count_for_constants():
    """A parameter-free program's stack is a compile-time constant, so its graph is tiny."""
    sizes = []
    for n_layers in (2, 20):
        simulator = PureStateVectorSimulator(brickwork(5, n_layers, parametric=False))
        sizes.append(_graph_size(simulator, None))
    assert sizes[0] == sizes[1]
    assert sizes[0] < 10, f"expected a near-empty graph for a constant program, got {sizes[0]}"


@pytest.mark.parametrize(
    ("simulator_cls", "n_qubits", "n_layers"),
    [(PureStateVectorSimulator, 8, 10), (DensityMatrixSimulator, 4, 8)],
    ids=["state_vector", "density_matrix"],
)
def test_first_shot_constant_program(benchmark, simulator_cls, n_qubits, n_layers):
    """Construct plus first evaluation, for a program simulated once."""
    program = brickwork(n_qubits, n_layers, parametric=False)
    simulator_cls(program)  # warm JAX dispatch so the measurement is steady-state

    def first_shot():
        simulator = simulator_cls(program)
        return jax.block_until_ready(simulator.compute())

    benchmark(first_shot)


@pytest.mark.parametrize(
    ("simulator_cls", "n_qubits", "n_layers"),
    [(PureStateVectorSimulator, 8, 10), (DensityMatrixSimulator, 4, 8)],
    ids=["state_vector", "density_matrix"],
)
def test_compile_parametric_program(benchmark, simulator_cls, n_qubits, n_layers):
    """Lowering and compiling a parametric program, the dominant cost for a fresh circuit."""
    program = brickwork(n_qubits, n_layers, parametric=True)
    simulator = simulator_cls(program)
    params = _params(simulator, n_qubits, n_layers)
    jax.jit(simulator.compute).lower(params).compile()  # warm XLA

    def compile_once():
        return jax.jit(simulator.compute).lower(params).compile()

    benchmark.pedantic(compile_once, iterations=1, rounds=3)


@pytest.mark.parametrize(
    ("simulator_cls", "n_qubits", "n_layers"),
    [(PureStateVectorSimulator, 8, 10), (DensityMatrixSimulator, 4, 8)],
    ids=["state_vector", "density_matrix"],
)
def test_steady_state_run(benchmark, simulator_cls, n_qubits, n_layers):
    """Evaluating an already-compiled parametric program."""
    program = brickwork(n_qubits, n_layers, parametric=True)
    simulator = simulator_cls(program)
    params = _params(simulator, n_qubits, n_layers)
    compiled = jax.jit(simulator.compute).lower(params).compile()
    jax.block_until_ready(compiled(params))

    benchmark(lambda: jax.block_until_ready(compiled(params)))

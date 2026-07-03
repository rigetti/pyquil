.. _simulation_architecture:

=========================================
Noisy simulation architecture
=========================================

.. note::

   The simulators described here live in the experimental, private modules
   ``pyquil.simulation._simulator`` and ``pyquil.simulation._resolver`` (and the
   noise model in ``pyquil.noise._noise_model`` / ``pyquil.noise._channels``).
   The API is not yet stable and the import paths are private. It is documented
   here because the design is intended to become the default simulation backend
   in a future major release, replacing the NumPy reference simulators.

   These modules depend on `JAX <https://jax.readthedocs.io>`_ (via the
   ``rigetti-quax`` package), which provides the operator algebra and the
   ``jit``/``grad``/``vmap`` machinery the simulators are built on.


Goal of the module
==================

The module simulates the action of a (possibly noisy) Quil program on a quantum
register and returns the resulting quantum state or measurement statistics. It
is designed to solve two problems with the existing simulators simultaneously:

* **Expressiveness.** Device-realistic noise is not limited to a fixed menu of
  Kraus channels. The module represents noise as arbitrary completely-positive,
  trace-preserving (CPTP) maps attached to individual instructions — coherent
  errors, stochastic Pauli channels, thermal relaxation, leakage to higher
  levels (qutrits and beyond), readout confusion, and reset infidelity — and
  composes them exactly.

* **Performance and differentiability.** Every stage is expressed in JAX so that
  the entire forward simulation is a single traceable function. It can be
  ``jax.jit``-compiled (amortizing compilation across a parameter sweep) and
  ``jax.grad``-differentiated (exact gradients of an output observable with
  respect to gate parameters), and trajectories can be vectorized with
  ``jax.vmap`` and sharded across devices.

The unit of noise is a **channel** keyed to a program instruction. A
:class:`~pyquil.noise._noise_model.NoiseModel` is, conceptually, a partial map
from instructions to channels,

.. math::

   \mathcal{N} : \text{instruction} \longmapsto \mathcal{E},

queried during simulation via ``NoiseModel.get_channel(inst)``. A channel's
``process`` is a superoperator that *includes* the ideal gate, so the channel
**replaces** the instruction rather than being appended after it:

.. math::

   \mathcal{E} \;=\; \Lambda \circ \mathcal{U},

where :math:`\mathcal{U}(\rho) = U \rho U^\dagger` is the ideal gate and
:math:`\Lambda` is the noise. An instruction with no channel is simulated
ideally.


Operator vocabulary
===================

The pipeline manipulates a small set of ``quax`` operator types. Each carries
explicit per-qudit dimensions (e.g. ``((2, 2), (2, 2))`` for a two-qubit
operator, ``((3,), (3,))`` for a qutrit), so qubit and qudit systems are treated
uniformly.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Type
     - Meaning
   * - ``Unitary``
     - An ideal gate :math:`U`, acting as :math:`\rho \mapsto U \rho U^\dagger`.
   * - ``SuperOp``
     - A general linear map on density operators in column-stacking
       (Liouville) form, the canonical representation of a noisy ``Channel``.
   * - ``KrausMap``
     - A channel as a set of Kraus operators :math:`\{K_i\}` with
       :math:`\sum_i K_i^\dagger K_i = I`; the form a state-vector trajectory
       samples from.
   * - ``QuantumInstrument``
     - A measurement: a collection of outcome-labelled CP maps whose sum is
       trace-preserving. Models classification (confusion) error and
       post-measurement back-action.

Conversions (``to_superop``, ``to_kraus``, ``to_choi``, ``to_pauli_liouville``)
are provided by ``quax`` and are used by the adapters described below.


The simulation pipeline
=======================

A simulator is an **object constructed from a program**, not a function called
on one. The reason is that efficient simulation requires several closures whose
structure is fixed by the program (and noise model) but whose inputs are the
runtime parameters. Building them once, at construction time, lets the
expensive analysis (circuit expansion, dependency analysis, operator merging,
trace/compile) be shared across every subsequent evaluation — a parameter sweep,
a gradient computation, or a batch of Monte-Carlo trajectories.

Construction runs four conceptual stages, each materialized as a closure:

.. code-block:: text

   Program (+ NoiseModel)
        │
        ▼
   ┌──────────────┐   MemoryMap ──► flat parameter vector θ
   │  Linearizer  │
        │
        ▼
   ┌──────────────┐   θ ──► [(operator, subsystem), ...]
   │   Resolver   │   (consults the noise model; ideal gates stay parametric)
        │
        ▼
   ┌──────────────┐   merge adjacent operators on the program DAG
   │  Compressor  │   up to `max_subsystem_size` qubits
        │
        ▼
   ┌──────────────┐   apply the operator stack to the initial state
   │  Calculator  │   (jit/grad/vmap-friendly)
        │
        ▼
   StateVector / DensityMatrix / (StateVector, outcomes)

The first three stages are shared infrastructure in ``_resolver.py``; the last
is specialized per simulator in ``_simulator.py``.

Linearizer
----------

A Quil program references classical memory by name and offset (e.g.
``theta[0]``). The linearizer flattens a :class:`~pyquil.api.MemoryMap` into the
dense parameter vector :math:`\theta \in \mathbb{R}^{n}` that the rest of the
pipeline (and ``jax.grad``) operates on. The layout — which ``(register,
offset)`` pair occupies each slot — is discovered during expansion and fixed for
the life of the object, so ``linearize`` is a cheap gather.

Resolver
--------

The resolver turns :math:`\theta` into an ordered list of
``(operator, subsystem)`` pairs, one per operation, where ``subsystem`` is the
tuple of (zero-based) qudit indices the operator acts on. It is produced by
:func:`~pyquil.simulation._resolver.resolve_program`, which returns a
``Resolution`` bundling the inferred dimensions, the expanded operators, their
subsystems, the parameter layout, and the resolve closure.

Expansion does several things at once:

* **Noise resolution.** Each instruction is looked up in the noise model. A
  noisy gate becomes its ``SuperOp``; a noisy measurement becomes a
  ``QuantumInstrument``; a noisy reset becomes a ``SuperOp``. Instructions with
  no channel resolve to their ideal operator.

* **Most-specific typing.** Operators are kept in their tightest native type —
  ideal gates as ``Unitary``, channels as ``SuperOp``, measurements as
  ``QuantumInstrument``. This lets the cheapest backend (pure state vector)
  avoid density-matrix arithmetic whenever a program happens to be noiseless,
  and lets each backend choose how to adapt the rest (see *Adapters*).

* **Parametric closures.** A gate whose angle is a runtime memory reference is
  *not* resolved to a number. It is wrapped in a ``ParametricGate`` that, given
  :math:`\theta`, constructs the gate matrix. This keeps gate construction
  inside the traced/differentiated graph, which is what makes ``jax.grad`` with
  respect to gate angles work.

* **DEFCIRCUIT and cycle expansion.** ``DEFCIRCUIT`` bodies are expanded with
  formal-argument substitution. When a circuit invocation matches a
  :class:`~pyquil.noise._channels.CycleChannel` in the noise model — a single
  channel describing the joint noise of a whole parallel cycle — the cycle is
  replaced by the channel's constituent operators directly.

* **Dimension inference.** The register dimension of each qudit is inferred from
  the operators that act on it (e.g. a ``TX`` gate or a qutrit channel promotes a
  line to dimension 3). The program is expanded twice: once with default qubit
  dimensions to infer the true dimensions, then again with those dimensions so
  that *ideal* measurement and reset operators are built at the correct size.
  Passing ``dims`` explicitly skips the first pass.

Dependency DAG
--------------

The subsystem list induces a dependency DAG
(:func:`~pyquil.simulation._resolver.build_dag`): one node per operation, with an
edge :math:`u \to v` whenever :math:`u` and :math:`v` share a qubit and :math:`u`
precedes :math:`v` in program order. The DAG encodes exactly the orderings that
must be preserved; everything else is free to be reordered or merged.

Compressor
----------

Applying operators one at a time is wasteful: a depth-:math:`D`,
:math:`N`-qubit program issues many small one- and two-qubit operators, and
under ``jit`` each distinct operator shape becomes a distinct branch in the
compiled graph. The compressor
(:func:`~pyquil.simulation._resolver.compressor_from_dag`) performs **greedy edge
contraction** on the DAG, fusing adjacent operators into a single operator on the
union of their qubits, up to a cap of ``max_subsystem_size`` qubits.

Key properties:

* **Small-first priority.** Candidate merges are taken from a priority queue
  ordered by the size of the resulting subsystem, so one-qubit gates are
  absorbed into neighbouring multi-qubit groups first. This reduces the number
  of *distinct* subsystem shapes, which is what governs compile time.

* **Convexity / barrier safety.** A merge is rejected if it would create a cycle
  in the contracted (quotient) graph — i.e. if some operation lies on a
  dependency path *between* the two groups. This is what prevents two gates that
  straddle a mid-circuit measurement from being fused, which would silently
  reorder the measurement. Measurements (and any explicit barrier) are marked as
  non-mergeable.

* **Order-preserving emission.** The merged groups are emitted in a
  lexicographic topological order keyed by program index, guaranteeing that
  measurement operators appear in the compressed list in the same order as the
  ``MEASURE`` instructions in the program, regardless of how gates were fused.

Setting ``max_subsystem_size=0`` disables merging entirely (useful for
debugging or for exact per-instruction inspection).

Adapters
--------

The resolver is backend-agnostic: it yields each operator in its most specific
type. Each simulator then adapts the compressed list to the representation it
evolves:

* **Density matrix** (:func:`~pyquil.simulation._resolver.adapt_for_density_matrix`):
  everything becomes a ``SuperOp`` (a ``QuantumInstrument`` is collapsed to its
  total channel, since the density-matrix backend does not branch on outcomes).

* **Trajectory** (:func:`~pyquil.simulation._resolver.adapt_for_trajectory`): a
  ``SuperOp`` is converted to a (truncated) ``KrausMap``; ``Unitary``,
  ``KrausMap``, and ``QuantumInstrument`` pass through unchanged.

Calculator
----------

The calculator applies the operator stack to the initial state. Two strategies
appear, both designed so that the compiled graph scales with the number of
*distinct subsystem shapes* rather than the number of operations:

* **Scan + switch.** Operators are stacked into one array and applied with a
  :func:`jax.lax.scan`; the loop body dispatches each operator to a
  :func:`jax.lax.switch` branch selected by its base subsystem. Only one branch
  per distinct subsystem is traced.

* **Vectorized construction.** For state-vector evolution, gate matrices of the
  same *kind* (same constructor, constant arguments, and embedding) are built in
  a single ``jax.vmap`` and then folded within each merge group by a segmented
  matrix-product scan. The traced graph is then proportional to the number of
  gate kinds, not the number of gates. For parameter-free programs the operator
  stack is a compile-time constant and is materialized once and reused.

Because the whole calculator is a pure JAX function of :math:`\theta` (and, for
trajectories, a PRNG key), ``jax.jit`` and ``jax.grad`` compose with it directly.


The three simulators
====================

All three share the pipeline above and differ only in the state they evolve and
the operations they admit.

.. list-table::
   :header-rows: 1
   :widths: 26 30 14 14 16

   * - Simulator
     - Use case
     - Noise
     - Measurements / resets
     - Differentiable
   * - ``PureStateVectorSimulator``
     - Gate-only programs
     - No
     - No
     - ``jit`` + ``grad``
   * - ``DensityMatrixSimulator``
     - Any program, optional noise
     - Yes
     - Resets (measurements as total channel)
     - ``jit`` + ``grad``
   * - ``TrajectorySimulator``
     - Monte-Carlo sampling
     - Yes
     - Yes
     - ``jit`` (per batch)

The qubit ceilings are set by memory: a state vector holds :math:`2^{N}`
amplitudes, so pure-state and trajectory simulation are practical to roughly
:math:`N \lesssim 26`; a density matrix holds :math:`4^{N}` entries, limiting the
density-matrix backend to roughly :math:`N \lesssim 13`.

All simulators take the program (and, where relevant, a ``noise_model`` and
``max_subsystem_size``) at construction, and expose ``linearize``, ``resolve``,
``compress``, and ``compute``. ``compute`` is the entry point and takes the flat
parameter vector from ``linearize``.

Pure state vector
-----------------

For unitary, noiseless, measurement-free programs, evolve a pure state
:math:`|\psi\rangle = U_D \cdots U_1 |0\rangle`. This is the cheapest backend and
the natural target for gradient-based circuit optimization, and it can return the
full program unitary in addition to the state.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from pyquil import Program
   from pyquil.gates import H, CNOT, RX
   from pyquil.simulation._simulator import PureStateVectorSimulator

   # A Bell state (no runtime parameters).
   sim = PureStateVectorSimulator(Program(H(0), CNOT(0, 1)))
   psi = sim.compute(jnp.array([]))          # final state vector

   # The full 4x4 program unitary.
   U = sim.unitary(jnp.array([]))

   # A parametric program, jit-compiled and differentiated.
   from pyquil.quilatom import MemoryReference
   from pyquil.quilbase import Declare

   p = Program(Declare("theta", "REAL", 1), RX(MemoryReference("theta", 0), 0))
   sim = PureStateVectorSimulator(p)

   def excited_pop(theta):
       psi = sim.compute(jnp.array([theta]))
       amps = psi.matrix.reshape(-1)
       return jnp.abs(amps[1]) ** 2          # P(|1>)

   grad_pop = jax.jit(jax.grad(excited_pop))
   print(grad_pop(0.3))                       # exact d P(|1>) / d theta

Density matrix
--------------

For noisy, deterministic evolution, propagate the density matrix
:math:`\rho \mapsto \mathcal{E}_D \circ \cdots \circ \mathcal{E}_1 (\rho)` exactly.
This is the backend to use for expectation values and process metrics under
noise, since it tracks the full mixed state without sampling. Measurements are
applied as their total (outcome-averaged) channel.

.. code-block:: python

   import jax.numpy as jnp
   from pyquil import Program
   from pyquil.gates import RX
   from pyquil.noise._channels import Channel
   from pyquil.noise._noise_model import NoiseModel
   from pyquil.simulation._simulator import DensityMatrixSimulator

   gate = RX(jnp.pi, 0)
   noise = NoiseModel.from_channels([
       Channel.from_gate_fidelity(inst=gate, fidelity=0.99),
   ])

   sim = DensityMatrixSimulator(Program(gate), noise_model=noise)
   rho = sim.compute(jnp.array([]))           # final density matrix (a quax DensityMatrix)

A device-realistic model can be built directly from an instruction set
architecture with :meth:`NoiseModel.from_isa <pyquil.noise._noise_model.NoiseModel.from_isa>`,
which converts per-gate fidelities to depolarizing channels and per-qubit
readout fidelities to symmetric confusion.

Trajectory
----------

For programs with mid-circuit measurements, resets, and feed-forward-style
sampling, unravel the dynamics into pure-state **quantum trajectories**: each
trajectory samples a Kraus operator (or measurement outcome) at every noisy step
and evolves a single state vector, so the cost is that of a state vector rather
than a density matrix. Averaging over trajectories recovers the density-matrix
result; the individual trajectories *are* the sampled measurement records.

The number of trajectories is set by the shape of the PRNG key: a scalar key
runs one trajectory, while a batch of keys (from ``jax.random.split``) runs that
many in parallel via ``vmap``. A measurement is handled by flattening its
``QuantumInstrument`` into a single Kraus axis, so sampling a Kraus index also
selects the outcome.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from pyquil import Program
   from pyquil.gates import H, MEASURE
   from pyquil.quilatom import MemoryReference
   from pyquil.quilbase import Declare
   from pyquil.simulation._simulator import TrajectorySimulator

   p = Program(Declare("ro", "BIT", 1), H(0), MEASURE(0, MemoryReference("ro", 0)))
   sim = TrajectorySimulator(p)
   params = jnp.array([])

   # A batch of 1000 trajectories in parallel.
   keys = jax.random.split(jax.random.key(0), 1000)
   psi_batch, outcomes = sim.compute(params, keys)
   # outcomes has shape (1000, n_measurements); ~50/50 for an H gate.

   # Or, scalable sampling that streams batches and keeps only the outcomes:
   shots = sim.sample(params, num_trajectories=100_000, batch_size=2_000)

``sample`` runs trajectories in fixed-size batches, discarding state vectors
between batches so the total number of shots is unbounded by memory. When
multiple JAX devices are available, each batch is run data-parallel via
:func:`jax.pmap` — one independent kernel replica per device, with no
cross-device communication. In that case ``batch_size`` is interpreted **per
device**, so ``n`` devices run ``n * batch_size`` trajectories per batch and
each device's memory footprint matches a single-device run.


Choosing a simulator
====================

* Use **``PureStateVectorSimulator``** for ideal, measurement-free circuits —
  variational ansätze, unitary verification, gradient-based optimization. It is
  the fastest and supports ``jax.grad`` and the full-unitary readout.

* Use **``DensityMatrixSimulator``** when you need the *exact* noisy state or a
  noise-averaged expectation value at modest qubit count (:math:`\lesssim 13`),
  with no sampling noise. It is also differentiable.

* Use **``TrajectorySimulator``** when the program contains mid-circuit
  measurements or resets, when you want sampled bitstrings rather than a state,
  or when the qubit count is too large for a density matrix but a state vector
  still fits. Increase the trajectory count to reduce sampling error.

In all cases, ``max_subsystem_size`` trades compile time against runtime: larger
groups mean fewer, denser operator applications (faster steady-state runtime) at
the cost of larger merged matrices and longer compilation. The default (2) is a
reasonable balance for circuits dominated by one- and two-qubit gates.

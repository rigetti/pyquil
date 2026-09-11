.. _simulation_architecture:

=========================================
Noisy simulation architecture
=========================================

.. note::

   **Experimental.** The simulators described here live in the private modules
   ``pyquil.simulation._simulator``, ``pyquil.simulation._resolver`` and
   ``pyquil.simulation._circuit`` (and the noise model in ``pyquil.noise``).
   The API is not stable: names, signatures and return types may change in any
   release before pyQuil 5, and the import paths are private on purpose. It is
   documented here because the design is intended to become the default
   simulation backend in a future major release, replacing the NumPy reference
   simulators, and because using it in real work is how the API will be settled.

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

* **Qudits.** The register is not assumed to be a collection of qubits. Every stage carries
  explicit per-qudit dimensions, so a program may act on qutrits, on a mixed register (say
  ``(3, 3, 2)``), or on any other combination, and an operator may move population *out* of
  the computational subspace. This is the primary motivation for the module: leakage is a
  significant error mechanism on real hardware, and it cannot be represented at all in a
  simulator whose state is a tensor of two-level systems. Dimensions are inferred from the
  program (see `Resolver`_), so qubit-only programs need no extra ceremony.

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
   ┌──────────────┐   θ ──► Circuit
   │   Resolver   │   (consults the noise model; ideal gates stay parametric)
        │
        ▼   ─── the Quil/quantum-information boundary ───
        │
   ┌──────────────┐   merge adjacent operators, up to `max_subsystem_size`
   │  Compressor  │   qudits — MergePlan
        │
        ▼
   ┌──────────────┐   apply the operator stack to the initial state
   │  Calculator  │   (jit/grad/vmap-friendly)
        │
        ▼
   StateVector / DensityMatrix / (StateVector, outcomes)

The linearizer and resolver are Quil-specific and live in ``_resolver.py``. The
compressor is not: merge planning and operator fusion are quantum information,
not language, so they live in quax (:class:`~pyquil.simulation._circuit.Circuit`,
:class:`~pyquil.simulation._circuit.MergePlan`) and are shared with any other caller. The calculator is
specialized per simulator in ``_simulator.py``.

The dividing line is the :class:`~pyquil.simulation._circuit.Circuit` the resolver produces. A circuit
carries concrete operators and their placements and nothing else — no gate
names, no memory references, no control flow — so quax never needs a notion of
a program, and pyquil keeps sole ownership of what a program means.

Linearizer
----------

A Quil program references classical memory by name and offset (e.g.
``theta[0]``). The linearizer flattens a :class:`~pyquil.api.MemoryMap` into the
dense parameter vector :math:`\theta \in \mathbb{R}^{n}` that the rest of the
pipeline (and ``jax.grad``) operates on. Here :math:`n` is the number of
**distinct** memory references the program's gates read, in order of first use:
a reference used by several gates occupies a single slot, so the program

.. code-block::

   DECLARE theta REAL[2]
   DECLARE phi REAL[1]
   RX(theta[0]) 0
   RX(theta[0]) 1
   RZ(phi[0]) 0

has :math:`n = 2` with slots ``theta[0], phi[0]`` — not three — and the gradient
with respect to slot 0 is already the total derivative through both ``RX`` gates.
The layout is discovered during expansion and fixed for the life of the object;
``sim.parameters`` lists it and ``sim.parameter_index("theta", 0)`` looks a slot
up, which is how the component of a gradient belonging to one ``DECLARE`` entry is
found. ``linearize`` itself is a pure gather over the region arrays, so it can be
jitted and differentiated through: ``jax.grad(lambda t: loss(sim.compute(
sim.linearize({"theta": t}))))`` differentiates with respect to the declared
memory directly.

Resolver
--------

The resolver turns :math:`\theta` into a :class:`~pyquil.simulation._circuit.Circuit`: an ordered
sequence of concrete operators, each placed on a tuple of (zero-based) qudit
indices, over a register of inferred dimensions. It is produced by
:func:`~pyquil.simulation._resolver.resolve_program`, which returns a
``Resolution`` bundling the inferred dimensions, the expanded operators, their
subsystems, the parameter layout, and the resolve closure.

This is the last stage that knows anything about Quil. ``Resolution`` is
therefore the program *template*: it owns the parameter layout, because a slot
in :math:`\theta` is meaningful only relative to a ``DECLARE``, and
:meth:`~pyquil.simulation._resolver.Resolution.resolve` binds a parameter vector
to hand back a circuit that no longer refers to any of it.

Expansion does several things at once:

* **Noise resolution.** Each instruction is looked up in the noise model. A
  noisy gate becomes its ``SuperOp``; a noisy measurement becomes a
  ``QuantumInstrument``; a noisy reset becomes a ``SuperOp``. Instructions with
  no channel resolve to their ideal operator. The lookup is by instruction
  *equality* — gate name, parameters, qubits and modifiers together — so
  ``RX(pi/2) 0`` and ``RX(pi/2) 1`` are distinct keys, and a channel attached to
  ``RX(pi/2) 0`` does not apply to ``RX(pi/4) 0``.

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

  A body must reference **only its own formal arguments**. Quil allows a literal qubit in a
  body (``DEFCIRCUIT C q: X q; X 3``), but such a qubit is invisible to
  ``Program.get_qubit_indices`` (`pyQuil issue #1868
  <https://github.com/rigetti/pyquil/issues/1868>`_), so it escapes the register the simulator
  sizes itself for. It is rejected rather than silently mis-simulated; the check is a stopgap
  until that issue is fixed.

* **Unsupported instructions raise; non-quantum ones are ignored.** Control flow (``JUMP``,
  ``JUMP-WHEN``, ``JUMP-UNLESS``) and classical memory instructions (``MOVE``, ``ADD``, ``EQ``,
  ``LOAD``, ...) change which quantum operations run or what a measured register means, and a
  straight-line simulation cannot honour them, so they are rejected with a clear error.
  Declarations and definitions, ``PRAGMA``, labels, ``HALT``/``NOP``/``WAIT`` and every Quil-T
  pulse-level instruction (``PULSE``, ``DELAY``, ``FENCE``, ``SHIFT-PHASE``, ...) are ignored:
  they are the physical realisation of the logical program, not a change to it, and rejecting
  them would make compiled programs unsimulatable.

* **Gate modifiers are not supported.** ``DAGGER``, ``CONTROLLED`` and ``FORKED`` are all
  rejected. ``CONTROLLED`` and ``FORKED`` each add a qudit to the instruction, so the operator
  they denote is larger than the named gate and cannot come from a table lookup; ``DAGGER``
  could be honoured by conjugate-transposing, but is rejected too so that the supported gate
  set is one rule rather than a per-modifier exception list. Expand a modifier into an explicit
  gate — or, for ``DAGGER``, the inverse rotation angle — before simulating. Rejecting is the
  important part: silently dropping a modifier would simulate ``RX(+theta)`` for
  ``DAGGER RX(theta)`` and return a plausible wrong answer.

* **Dimension inference.** The register dimension of each qudit is inferred from
  the operators that act on it (e.g. a ``TX`` gate or a qutrit channel promotes a
  line to dimension 3). The program is expanded twice: once with default qubit
  dimensions to infer the true dimensions, then again with those dimensions so
  that *ideal* measurement and reset operators are built at the correct size.
  (:func:`~pyquil.simulation._resolver.resolve_program` accepts ``dims`` to skip
  the first pass; the simulators always infer.)

  .. note::
     This double expansion is the least elegant part of the pipeline, and it is worth being
     explicit about why it is here. The dimensions are a property of the *resolved operators*,
     but resolving an ideal ``MEASURE`` or ``RESET`` requires already knowing the dimension of
     the line it acts on — there is nothing in ``MEASURE 0`` itself that says whether qubit 0 is
     a qubit or a qutrit. That circularity has to be broken somewhere.

     Two alternatives were considered and rejected. Requiring callers to declare register
     dimensions up front removes the second pass but makes every qutrit program carry
     bookkeeping the module could have derived — which is also why the simulators take no
     ``dims`` argument: an all-qutrit register holding only qubit gates never populates the
     extra level, so nothing observable depends on declaring it. Defaulting silently to dimension 2 and promoting later is worse still:
     an ideal reset built at the wrong size fails deep inside a tensor contraction, far from the
     instruction that caused it. Expanding twice keeps the ergonomics and localizes the cost,
     which is small — the second pass reuses nothing but is cheap relative to compilation.

Dependency DAG
--------------

The subsystem list induces a dependency DAG (:func:`~pyquil.simulation._circuit.dependency_edges`): one
node per operation, with an edge :math:`u \to v` whenever :math:`u` and :math:`v`
share a qubit and :math:`u` precedes :math:`v` in program order. The DAG encodes
exactly the orderings that must be preserved; everything else is free to be
reordered or merged.

Because every edge runs from a lower operation index to a higher one, program
order is always a valid topological order, which is why the planner can compose
a merged group's members by ascending index and get program order for free.

Compressor
----------

Applying operators one at a time is wasteful: a depth-:math:`D`,
:math:`N`-qubit program issues many small one- and two-qubit operators, and
under ``jit`` each distinct operator shape becomes a distinct branch in the
compiled graph. The compressor (:meth:`~pyquil.simulation._circuit.MergePlan.greedy`) performs **greedy
edge contraction** on the DAG, fusing adjacent operators into a single operator
on the union of their qubits, up to a cap of ``max_subsystem_size`` qubits.

Planning is separated from merging, and this matters more than it looks.
:meth:`~pyquil.simulation._circuit.MergePlan.greedy` consumes only the subsystem list and the size cap
— no operators, no dimensions, no parameters — and returns a plan as *data*:
which operations fuse into which group, the distinct base subsystems, and each
group's base index. The vectorized construction described under `Calculator`_
reads that plan and builds the fused operator stack under ``jax.vmap`` **without
ever materialising the individual operators**, which is what makes its compile
time proportional to the number of gate kinds rather than the number of gates.
Had the plan been hidden inside an opaque ``optimize()`` call, that path would
not be expressible from outside quax. :meth:`~pyquil.simulation._circuit.MergePlan.apply` performs the
eager merge for callers that want the operators themselves.

Greedy contraction is one strategy, not the definition of a plan.
:meth:`~pyquil.simulation._circuit.MergePlan.from_partition` accepts *any* partition of the
operations into groups, checks that every group is convex (the quotient DAG of the partition
must be acyclic) and within budget, and emits the groups in topological order. ``greedy``
computes its partition and hands it over, and a future strategy — layer-wise fusion,
light-cone pruning, independent-register splitting — is a few lines of networkx on the
dependency DAG that produce a partition.

Key properties:

* **Small-first priority.** Candidate merges are taken from a priority queue
  ordered by the size of the resulting subsystem, so one-qubit gates are
  absorbed into neighbouring multi-qubit groups first. This reduces the number
  of *distinct* subsystem shapes, which is what governs compile time.

* **Convexity.** A merge is rejected if it would create a cycle in the contracted (quotient)
  graph — i.e. if some operation lies on a dependency path *between* the two groups. This is
  what prevents two gates that straddle a mid-circuit measurement from being fused, which would
  silently reorder the measurement.

* **Atomicity.** Convexity protects *order*, but it is not sufficient on its own, because it
  reasons about the circuit's overall channel. Fusing a sampled ``QuantumInstrument`` into a
  neighbour preserves that channel perfectly and is therefore permitted by the convexity check
  — yet the outcome is no longer separately observable, which for a trajectory simulator is
  exactly the thing being computed. Such operations are passed to the planner as ``atomic``
  and are never merged.

  Which operations those are is not something expansion decides. A ``MEASURE`` is always
  expanded to a ``QuantumInstrument``, the representation that keeps the most information, and
  each backend then adapts it: the differentiable simulators collapse every instrument to its
  total channel in ``_prepare_ops``, after which a measurement merges with its neighbours like
  any other superoperator and nothing is atomic; a trajectory backend keeps its instruments, and
  they are pinned automatically because they are still there.

  Collapsing an instrument to its total channel is exactly
  :meth:`~pyquil.simulation._circuit.Circuit.to_superops`, which is why expansion needs no mode parameter: resolving a
  measurement as a dephasing superoperator up front and collapsing an instrument afterwards give
  bit-identical results, so the choice belongs to whoever evolves the circuit.

* **Emission order is not application order.** Groups are emitted in a topological order of the
  quotient graph, keyed by each group's earliest member. That respects every dependency, but it
  does *not* guarantee that two independent operations keep their relative order: a group
  blocked behind a wide merge can be emitted after an operation that came later in the program.
  Since operations sharing no qubit commute, this is physically harmless — but it means a
  caller must **not** use a group's position as an operation's identity. Measurement outcome
  columns are labelled from the operation indices carried in
  :attr:`~pyquil.simulation._circuit.MergePlan.groups`, which are unaffected by fusion.

Setting ``max_subsystem_size=0`` disables merging entirely (useful for
debugging or for exact per-instruction inspection).

Adapters
--------

The resolver is backend-agnostic: it yields each operator in its most specific
type. Each simulator then converts the merged circuit to the representation it
evolves, using the methods on :class:`~pyquil.simulation._circuit.Circuit`:

* **Density matrix** (:meth:`~pyquil.simulation._circuit.Circuit.to_superops`): everything becomes a
  ``SuperOp`` (a ``QuantumInstrument`` is collapsed to its total channel, since
  the density-matrix backend does not branch on outcomes).

* **Trajectory** (:meth:`~pyquil.simulation._circuit.Circuit.to_kraus_maps`): a ``SuperOp`` is
  converted to a (truncated) ``KrausMap``; ``Unitary``, ``KrausMap``, and
  ``QuantumInstrument`` pass through unchanged, each already being applicable to
  a state vector either deterministically or by sampling.

Calculator
----------

The calculator applies the operator stack to the initial state. Two strategies
appear, both designed so that the compiled graph scales with the number of
*distinct subsystem shapes* rather than the number of operations:

* **Scan + switch.** Operators are stacked into one array and applied with a
  :func:`jax.lax.scan`; the loop body dispatches each operator to a
  :func:`jax.lax.switch` branch selected by its base subsystem. Only one branch
  per distinct subsystem is traced.

* **Vectorized construction.** Operator matrices of the same *kind* (same constructor,
  constant arguments, and embedding) are built in a single ``jax.vmap`` and then folded within
  each merge group by a segmented matrix-product scan. The traced graph is then proportional to
  the number of gate kinds, not the number of gates. Both the state-vector and density-matrix
  simulators use this path — the latter lifts each operator to a superoperator after embedding.

  This is the most intricate code in the module, and it earns its keep. Building the stack the
  obvious way, with a Python comprehension over the compressed operators, puts one traced
  operation per gate into the graph, and XLA compile time grows superlinearly: at 20 qubits and
  20 layers that is **180 s against 0.35 s**, a 518x difference.

Because the whole calculator is a pure JAX function of :math:`\theta` (and, for
trajectories, a PRNG key), ``jax.jit`` and ``jax.grad`` compose with it directly.

The parameter vector is **optional**. ``compute()`` may be called with no argument for a
program that declares no runtime parameters; for a parametric program the vector is required,
must have one entry per parametric gate *occurrence*, and is most easily built with
``sim.linearize(memory_map)``. A wrong length or a missing vector is reported as such rather
than surfacing as an indexing error from JAX.


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
   * - ``TrajectorySimulator`` [#planned]_
     - Monte-Carlo sampling
     - Yes
     - Yes
     - ``jit`` (per batch)

API shape
---------

All simulators share one shape, chosen to stay close to the simulator APIs of
Cirq, Qiskit and PennyLane where the design allows it:

* **Construction** takes the program (and, where relevant, a ``noise_model`` and
  ``max_subsystem_size``). Cirq and Qiskit construct a simulator without a
  program and pass it at run time; here the program is bound at construction
  because building the fused operator stack and its ``jit`` closures *is* the
  expensive step, and it is reused across every evaluation. ``qubits`` fixes the
  register order (big-endian) and must be exactly the qubits the program acts on.
* **Parameters**: ``parameters``, ``parameter_index`` and ``linearize`` describe
  and build the flat parameter vector (see `Linearizer`_).
* **Introspection**: ``resolve(params)`` returns the program's
  :class:`~pyquil.simulation._circuit.Circuit`, ``compress(circuit)`` the merged one,
  and ``plan`` the :class:`~pyquil.simulation._circuit.MergePlan` behind it.
* **Evaluation**: ``compute(params)`` is the entry point and returns a quax state
  (``StateVector`` or ``DensityMatrix``); ``params`` may be omitted for a program
  with no runtime parameters. The trajectory simulator adds a PRNG key and returns
  the sampled outcomes alongside the state, and offers ``sample`` for outcomes
  alone.

Samples are not the only thing worth computing. Because ``compute`` returns a
quax state and is itself a pure JAX function, any quax metric composes with it
under ``jit`` and ``grad`` — fidelity to a target, purity, entanglement measures,
Pauli expectation values, or a composite loss built from several — without a
PennyLane-style "measurement process" layer in between:

.. code-block:: python

   import jax
   import quax as qx

   sim = DensityMatrixSimulator(program, noise_model=noise_model)
   target = qx.zero_state_matrix(dims=sim.dims)

   def infidelity(params):
       return 1.0 - qx.fidelity(sim.compute(params), target)

   value, grad = jax.value_and_grad(infidelity)(sim.linearize(memory_map))
   d_theta0 = grad[sim.parameter_index("theta", 0)]

Memory
------

What limits register size is the size of the state itself. For a register of
:math:`N` qudits of dimension :math:`d`, at ``complex128`` (16 bytes per entry):

.. math::

   \text{state vector} = 16\, d^{N}\ \text{bytes}
   \qquad
   \text{density matrix} = 16\, d^{2N}\ \text{bytes}

A trajectory simulation holds one state vector per trajectory in the batch, so
its cost is the state-vector figure times the per-device batch size — which is
why batch size, not register size, is the knob for fitting it into memory.

.. list-table:: State size for qubits (:math:`d = 2`)
   :header-rows: 1
   :widths: 20 40 40

   * - :math:`N`
     - State vector
     - Density matrix
   * - 10
     - 16 KiB
     - 16 MiB
   * - 13
     - 128 KiB
     - 1 GiB
   * - 16
     - 1 MiB
     - 64 GiB
   * - 20
     - 16 MiB
     - 16 TiB
   * - 26
     - 1 GiB
     - —
   * - 30
     - 16 GiB
     - —

Read the ceiling off whatever budget you actually have rather than from a fixed
number of qubits: 1 GiB of state is reached at :math:`N = 26` for a state vector
and at :math:`N = 13` for a density matrix, and each further qubit costs
:math:`2\times` and :math:`4\times` respectively. Qutrits move the base rather
than the shape — :math:`3^{N}` and :math:`9^{N}` — so a 13-qutrit state vector
(24 MiB) sits between a 20- and a 21-qubit one.

These are *dense* representations, so the cost depends only on the register, not
on how entangled it is: a product state occupies exactly as much memory as a
maximally entangled one, and none of the three backends differ in this respect.
Splitting a program into independent subsystems and simulating each in its own
smaller register is a real and large saving, but it is a decision above this
layer — the simulators here evolve whatever register they are given.

Beyond the state, the compressed operator stack holds
:math:`n_\text{groups} \times W^{2}` entries, with :math:`W = d_\text{max}` for
unitary evolution and :math:`W = d_\text{max}^{2}` for superoperator evolution,
where :math:`d_\text{max}` is the largest merge group's Hilbert-space dimension.
At the default ``max_subsystem_size=2`` this is negligible against the state;
raising it grows the stack as :math:`d^{4k}` for a :math:`k`-qudit budget in the
density-matrix case, which is the practical reason not to set it high.

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
   psi = sim.compute()                        # final state vector

   # The full 4x4 program unitary.
   U = sim.unitary()

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
   rho = sim.compute()                        # final density matrix (a quax DensityMatrix)

A device-realistic model can be built directly from an instruction set
architecture with :meth:`NoiseModel.from_isa <pyquil.noise._noise_model.NoiseModel.from_isa>`,
which takes a QCS ``InstructionSetArchitecture`` and converts per-gate fidelities to
depolarizing channels and per-qubit readout fidelities to symmetric confusion.  (For the
legacy rpcq-derived ``CompilerISA``, use
:meth:`NoiseModel.from_compiler_isa <pyquil.noise._noise_model.NoiseModel.from_compiler_isa>`,
which is deprecated for removal in pyQuil v5.)

Trajectory
----------

.. note::
   ``TrajectorySimulator`` is **not yet available**.  This section describes the design that
   the instrument representation and the merge plan's ``atomic`` set exist to support; the
   simulator itself lands in a follow-up change.  The code block below will not run against
   this release.

.. [#planned] Planned; see the note under `Trajectory`_.

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

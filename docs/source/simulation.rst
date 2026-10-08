.. _simulation:

=====================
Simulating programs
=====================

pyQuil simulates Quil programs locally, in-process, with three simulators built on
`JAX <https://jax.readthedocs.io>`_ and `quax <https://rigetti.gitlab.io/application_benchmarking/quax/>`_, Rigetti's
JAX-based quantum information library. No simulator server needs to be running. Because every simulator is a pure JAX
function of the program's parameters, you can ``jax.jit``-compile a parameter sweep, take exact gradients with
``jax.grad``, and batch runs with ``jax.vmap``.

This page shows how to use the simulators. :ref:`simulation_architecture` explains how they work: the pipeline from
program to operators, operator merging, memory requirements, and numerical precision. :ref:`noise` covers how to build
the noise models the simulators accept.

.. note::

   Some quax conversions need JAX's 64-bit mode, so turn it on before you build a simulator or a noise model:

   .. code-block:: python

      import jax

      jax.config.update("jax_enable_x64", True)

   Alternatively, set the environment variable ``JAX_ENABLE_X64=1``. See :ref:`simulation_architecture` for the
   reasons.

Choosing a simulator
====================

.. list-table::
   :header-rows: 1
   :widths: 26 44 30

   * - Simulator
     - Use it for
     - Returns
   * - :class:`~pyquil.simulation.PureStateVectorSimulator`
     - Ideal programs made only of gates, with no noise, measurements or resets. This is the fastest simulator and
       it is differentiable.
     - A quax ``StateVector``, or the program's full ``Unitary``
   * - :class:`~pyquil.simulation.DensityMatrixSimulator`
     - Exact noisy states, exact readout statistics and noisy expectation values, at modest qubit counts (about 13
       qubits or fewer). Differentiable.
     - A quax ``DensityMatrix``, or the outcome distribution of the program's final measurements
   * - :class:`~pyquil.simulation.TrajectorySimulator`
     - Sampled bitstrings, mid-circuit measurements and resets, and noisy registers too large for a density matrix.
       Can be ``jax.jit``-compiled but is not differentiable.
     - Sampled measurement outcomes, plus the state of each trajectory

All three simulators share one pattern:

#. **Construct the simulator from a program.** Analysis and compilation happen here, once.
#. **Evaluate it** with ``compute()`` or one of the more specific methods below, as many times as you need.

Pass a noise model to :class:`~pyquil.simulation.DensityMatrixSimulator` or :class:`~pyquil.simulation.TrajectorySimulator`
with the ``noise_model`` argument.

Qubit ordering
==============

The simulators order the register **big-endian**: by default the first (most significant) qubit in a state is the
lowest-numbered qubit in the program. The ``qubits`` argument chooses a different order. It must list exactly the
qubits the program acts on.

.. testcode:: ordering

   import numpy as np
   from pyquil import Program
   from pyquil.gates import I, X
   from pyquil.simulation import PureStateVectorSimulator

   program = Program()
   program += X(0)
   program += I(1)

   # Basis order |q0 q1>: amplitude 1 on |10>.
   print(np.round(PureStateVectorSimulator(program).compute().matrix.reshape(-1).real, 3))

   # Basis order |q1 q0>: amplitude 1 on |01>.
   print(np.round(PureStateVectorSimulator(program, qubits=[1, 0]).compute().matrix.reshape(-1).real, 3))

.. testoutput:: ordering

   [0. 0. 1. 0.]
   [0. 1. 0. 0.]

See :ref:`basis_ordering` for how this convention applies to gate matrices.

Inspecting the state
====================

:class:`~pyquil.simulation.PureStateVectorSimulator` returns the exact state, with no sampling. The program may
contain gates only.

.. testcode:: statevector

   import numpy as np
   from pyquil import Program
   from pyquil.gates import CNOT, H
   from pyquil.simulation import PureStateVectorSimulator

   program = Program()
   program += H(0)
   program += CNOT(0, 1)

   sim = PureStateVectorSimulator(program)
   psi = sim.compute()                       # a quax StateVector
   print(sim.qubits, sim.dims)
   print(np.round(psi.matrix.reshape(-1).real, 3))

   U = sim.unitary()                         # the full 4x4 program unitary
   print(U.matrix.shape)

.. testoutput:: statevector

   (0, 1) (2, 2)
   [0.707 0.    0.    0.707]
   (4, 4)

``compute`` returns a quax quantum object, not a bare array. ``psi.matrix`` holds its amplitudes, and any quax metric
(fidelity, purity, expectation values) accepts it directly.

``unitary()`` builds the full operator on the whole register. Its cost grows exponentially with the number of
qubits, so use it to check small programs, not to simulate large ones.

Sampling bitstrings
===================

:class:`~pyquil.simulation.TrajectorySimulator` samples measurement records the way a QPU produces them.

``sample`` returns an integer array of shape ``(num_trajectories, n_measurements)``. Column ``j`` holds the outcome of
the ``j``-th ``MEASURE`` in program order.

.. testcode:: sampling

   import jax
   import numpy as np
   from pyquil import Program
   from pyquil.gates import CNOT, H, MEASURE
   from pyquil.simulation import TrajectorySimulator

   program = Program()
   ro = program.declare("ro", "BIT", 2)
   program += H(0)
   program += CNOT(0, 1)
   program += MEASURE(0, ro[0])
   program += MEASURE(1, ro[1])

   sim = TrajectorySimulator(program)
   shots = sim.sample(num_trajectories=1000, key=jax.random.key(0))
   print(shots.shape)
   print(np.unique(np.asarray(shots), axis=0))   # only 00 and 11 occur

.. testoutput:: sampling

   (1000, 2)
   [[0 0]
    [1 1]]

Pass a ``key`` to make a run reproducible. If you omit it, every call draws fresh randomness, so repeated calls
accumulate independent samples.

``sample`` runs trajectories in batches and spreads each batch across the available JAX devices, so the number of
shots you can request is not limited by device memory.

Mid-circuit measurements and resets
-----------------------------------

A trajectory follows a single measurement record, so ``MEASURE`` and ``RESET`` can appear anywhere in a program:

.. testcode:: midcircuit

   import jax
   from pyquil import Program
   from pyquil.gates import H, MEASURE, RESET, X
   from pyquil.simulation import TrajectorySimulator

   program = Program()
   ro = program.declare("ro", "BIT", 2)
   program += H(0)
   program += MEASURE(0, ro[0])   # random
   program += RESET(0)
   program += X(0)
   program += MEASURE(0, ro[1])   # always 1

   shots = TrajectorySimulator(program).sample(num_trajectories=100, key=jax.random.key(1))
   print(bool((shots[:, 1] == 1).all()))

.. testoutput:: midcircuit

   True

``compute`` gives lower-level access. It takes a PRNG key and returns the final state of each trajectory together with
its outcomes. A scalar key runs one trajectory, and a batch of keys from ``jax.random.split`` runs one trajectory per
key in parallel:

.. code-block:: python

   keys = jax.random.split(jax.random.key(0), 100)
   states, outcomes = sim.compute(key=keys)    # outcomes has shape (100, n_measurements)

Exact outcome probabilities
===========================

:class:`~pyquil.simulation.DensityMatrixSimulator` evolves the full mixed state, so it computes noisy results exactly,
without sampling error.

``outcome_probabilities`` returns the joint distribution of the program's *terminal* measurements: those that no
later instruction acts on. Readout error from the noise model is included.

.. testcode:: densitymatrix

   import numpy as np
   from pyquil import Program
   from pyquil.gates import CNOT, H, MEASURE
   from pyquil.noise import Channel, MeasurementChannel, NoiseModel
   from pyquil.simulation import DensityMatrixSimulator

   program = Program()
   ro = program.declare("ro", "BIT", 2)
   program += H(0)
   program += CNOT(0, 1)
   program += MEASURE(0, ro[0])
   program += MEASURE(1, ro[1])

   noise_model = NoiseModel.from_channels([
       Channel.from_gate_fidelity(CNOT(0, 1), fidelity=0.98),
       MeasurementChannel.from_readout_fidelity(MEASURE(0, None), fidelity=0.95),
       MeasurementChannel.from_readout_fidelity(MEASURE(1, None), fidelity=0.95),
   ])

   sim = DensityMatrixSimulator(program, noise_model=noise_model)
   probs = sim.outcome_probabilities()
   print(sim.measured_qubits)
   print(np.round(probs, 4))       # probs[a, b] = P(qubit 0 reads a, qubit 1 reads b)

.. testoutput:: densitymatrix

   (0, 1)
   [[0.4471 0.0529]
    [0.0529 0.4471]]

``compute`` returns the final ``DensityMatrix``. In that state each measurement is applied as its outcome-averaged
channel, and no classical record is kept.

A measurement that is followed by another operation on the same qubit is a *mid-circuit* measurement. It enters only
as that averaged channel and does not appear in ``outcome_probabilities``. To get mid-circuit outcome statistics, use
:class:`~pyquil.simulation.TrajectorySimulator`.

Parametric programs
===================

A program that reads ``DECLARE``\ d memory is simulated as a function of a flat parameter vector.

The simulator lays this vector out when you construct it: one slot for each *distinct* memory reference the program's
gates read, in the order each is first used. The following tools describe it:

* ``sim.parameters`` lists the layout.
* ``sim.parameter_index(name, offset)`` gives the slot of one memory reference.
* ``sim.linearize(memory_map)`` builds the vector from the same kind of memory map you would send to a QPU.

.. testcode:: parametric

   from pyquil import Program
   from pyquil.gates import RX, RZ
   from pyquil.simulation import PureStateVectorSimulator

   program = Program()
   theta = program.declare("theta", "REAL", 2)
   program += RX(theta[0], 0)
   program += RX(theta[0], 1)     # the same reference: one slot, not two
   program += RZ(theta[1], 0)

   sim = PureStateVectorSimulator(program)
   print(sim.parameters)
   print(sim.parameter_index("theta", 1))
   psi = sim.compute(sim.linearize({"theta": [0.3, 0.1]}))

.. testoutput:: parametric

   (('theta', 0), ('theta', 1))
   1

``compute`` and ``outcome_probabilities`` are pure JAX functions of that vector, so they can be compiled,
differentiated and vectorized:

.. testcode:: gradients

   import jax
   import jax.numpy as jnp
   from pyquil import Program
   from pyquil.gates import MEASURE, RX
   from pyquil.simulation import DensityMatrixSimulator

   program = Program()
   theta = program.declare("theta", "REAL", 1)
   ro = program.declare("ro", "BIT", 1)
   program += RX(theta[0], 0)
   program += MEASURE(0, ro[0])

   sim = DensityMatrixSimulator(program)

   def p_one(t):
       return sim.outcome_probabilities(sim.linearize({"theta": t}))[1]

   grad = jax.jit(jax.grad(p_one))(jnp.array([0.3]))          # exact d P(1) / d theta
   print(jnp.allclose(grad, jnp.sin(0.3) / 2))

   sweep = jax.jit(jax.vmap(p_one))(jnp.linspace(0, jnp.pi, 5)[:, None])
   print(jnp.round(sweep, 4))

.. testoutput:: gradients

   True
   [0.     0.1464 0.5    0.8536 1.    ]

:class:`~pyquil.simulation.TrajectorySimulator` can be compiled and vectorized too. Sampling is a discrete choice,
though, so ``jax.grad`` raises an error instead of returning a gradient that is identically zero. Use
:class:`~pyquil.simulation.DensityMatrixSimulator` when you need gradients.

Noise models do not apply to gates whose angle is a memory reference. See :ref:`noise` for why.

Simulating compiled programs
============================

The simulators accept any straight-line Quil program, including the native Quil that ``quilc`` produces. Gate angles
may be arithmetic expressions over memory references, such as ``RZ(theta[0]/2 + pi)``, which is the form ``quilc``
emits when it compiles a parametric program. Pulse-level Quil-T instructions are ignored.

To see how a program behaves once it is compiled to a device's native gates and topology, compile it with a compiler
and simulate the result:

.. testcode:: compiled

   from pyquil import Program, get_qc
   from pyquil.gates import CNOT, H
   from pyquil.simulation import PureStateVectorSimulator

   program = Program()
   program += H(0)
   program += CNOT(0, 1)

   qc = get_qc("9q-square")        # compiles with quilc against a 3x3 lattice; cannot run programs
   native = qc.compiler.quil_to_native_quil(program)

   ideal = PureStateVectorSimulator(program).compute()
   compiled = PureStateVectorSimulator(native).compute()

``quilc`` can move the program onto other qubits, and it may leave only a global phase difference. For both reasons,
compare states with a phase-insensitive metric such as fidelity, not elementwise.

Supported instructions
======================

The simulators run *straight-line* programs. Here is how each kind of instruction is handled:

* **Simulated:** gates, including ``DEFGATE`` gates and expanded ``DEFCIRCUIT`` bodies. ``MEASURE`` and ``RESET``
  are accepted by :class:`~pyquil.simulation.DensityMatrixSimulator` and
  :class:`~pyquil.simulation.TrajectorySimulator`.
* **Ignored:** declarations, ``PRAGMA``, labels, ``HALT``, ``NOP``, ``WAIT``, and pulse-level Quil-T instructions
  (``PULSE``, ``DELAY``, ``FENCE``, ``SHIFT-PHASE``, and so on). These describe *how* a program is realised on hardware,
  not *what* it does, so compiled programs still simulate.
* **Rejected with an error:**

  * Control flow (``JUMP``, ``JUMP-WHEN``, ``JUMP-UNLESS``) and classical memory instructions (``MOVE``, ``ADD``,
    ``EQ``, ``LOAD``, and so on).
  * Gate modifiers (``DAGGER``, ``CONTROLLED``, ``FORKED``). Expand them into explicit gates before simulating.

Qudits
======

The register does not have to be made of qubits. If a program uses a gate or a noise channel that acts on more than
two levels, such as a qutrit leakage channel, the simulators widen that qudit's dimension to match. ``sim.dims``
reports the dimension they inferred. See :ref:`noise` for leakage models, and :ref:`simulation_architecture` for how
dimensions are inferred.

Splitting a program
===================

``compute``, ``outcome_probabilities`` and ``TrajectorySimulator.compute`` all accept an ``initial_state``. This lets
many runs that share an expensive prefix evolve that prefix once. A typical case is a randomized-measurement
experiment, where one prepared state is measured in many random bases. See :ref:`simulation_architecture`.

Further reading
===============

* :doc:`simulation-demo`: a worked tutorial covering a Bell state, a variational algorithm optimized with
  ``jax.grad``, and the same algorithm under noise.
* :ref:`noise`: building channels and noise models, and how they are applied to a program's instructions.
* :ref:`simulation_architecture`: the simulation pipeline, memory requirements, numerical precision, and how to choose
  ``max_subsystem_size``.

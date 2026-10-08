.. _noise:

============
Noise models
============

Real quantum processors are noisy: gates are imperfect, qubits relax and dephase while they wait, and
readout misclassifies. pyQuil describes that noise with a **noise model** that attaches a quantum
channel to individual Quil instructions. The simulators (see :ref:`simulation`) consult it
while they run a program, so the same program can be simulated ideally or under any noise you can
write down.

This page is about the Quil side of noise: how to construct channels, how a noise model keys them
to instructions, and how the simulators apply them. The underlying quantum-information objects --
superoperators, Lindbladians and quantum instruments -- come from `quax
<https://rigetti.gitlab.io/application_benchmarking/quax/>`_, and their background is covered in
its documentation:

* `Quantum objects <https://rigetti.gitlab.io/application_benchmarking/quax/quantum-objects.html>`_
  -- the operator and superoperator types, and composing them with ``@`` and ``|``.
* `Superoperator representations
  <https://rigetti.gitlab.io/application_benchmarking/quax/examples/superoperator-representations.html>`_
  -- Liouville, Kraus, Choi and Pauli-Liouville forms of a channel.
* `Lindbladians <https://rigetti.gitlab.io/application_benchmarking/quax/lindbladians.html>`_ --
  generators of continuous-time noise and the common noise processes.
* `Quantum instruments <https://rigetti.gitlab.io/application_benchmarking/quax/quantum-instruments.html>`_
  -- measurements with classification error and back-action.
* `Promotion <https://rigetti.gitlab.io/application_benchmarking/quax/promotion.html>`_ -- embedding
  qubit operators in larger (qutrit) spaces, which is how leakage is modelled.


Overview
========

A :class:`~pyquil.noise.NoiseModel` is a partial map from instructions to channels,

.. math::

   \mathcal{N} : \text{instruction} \longmapsto \mathcal{E},

and every channel carries the instruction it belongs to in its ``inst`` field. There are three kinds,
one per kind of quantum instruction:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Channel
     - Instruction
     - ``process``
   * - :class:`~pyquil.noise.Channel`, :class:`~pyquil.noise.SuperopChannel`
     - a ``Gate``
     - a ``SuperOp`` that *includes* the ideal gate
   * - :class:`~pyquil.noise.MeasurementChannel`
     - ``MEASURE q``
     - a ``QuantumInstrument``: one CP map per outcome
   * - :class:`~pyquil.noise.ResetChannel`, :class:`~pyquil.noise.SuperopResetChannel`
     - ``RESET q``
     - a ``SuperOp`` that includes the ideal reset

A fourth, :class:`~pyquil.noise.CycleChannel`, describes the joint noise of a whole layer of
operations (see `Cycle noise`_).

Because a channel's process includes the ideal operation, it **replaces** the instruction rather than
being appended after it: for a gate :math:`U` with noise :math:`\Lambda`,

.. math::

   \mathcal{E} \;=\; \Lambda \circ \mathcal{U}, \qquad \mathcal{U}(\rho) = U \rho U^\dagger.

An instruction with no channel is simulated ideally, so a noise model only needs to mention the
instructions you want to be noisy.


Quickstart
==========

Build a channel for each noisy instruction, collect them in a noise model, and pass it to a
simulator:

.. testcode:: quickstart

   import numpy as np
   from pyquil import Program
   from pyquil.gates import CNOT, H, MEASURE
   from pyquil.noise import Channel, MeasurementChannel, NoiseModel
   from pyquil.simulation import DensityMatrixSimulator

   program = Program()  # a Bell state, measured
   program += H(0)
   program += CNOT(0, 1)
   program += MEASURE(0, None)
   program += MEASURE(1, None)

   noise_model = NoiseModel.from_channels(
       [
           Channel.from_gate_fidelity(H(0), fidelity=0.999),
           Channel.from_gate_fidelity(CNOT(0, 1), fidelity=0.99),
           MeasurementChannel.from_readout_fidelity(MEASURE(0, None), fidelity=0.97),
           MeasurementChannel.from_readout_fidelity(MEASURE(1, None), fidelity=0.97),
       ]
   )

   sim = DensityMatrixSimulator(program, noise_model=noise_model)
   probabilities = sim.outcome_probabilities()  # joint P(q0, q1), readout error included
   print(np.round(probabilities, 3))

.. testoutput:: quickstart

   [[0.468 0.032]
    [0.032 0.468]]

The same noise model works with every simulator. :class:`~pyquil.simulation.TrajectorySimulator`
samples bitstrings instead:

.. testcode:: quickstart

   import jax
   from pyquil.simulation import TrajectorySimulator

   shots = TrajectorySimulator(program, noise_model=noise_model).sample(
       num_trajectories=1000, key=jax.random.key(0)
   )
   print(shots.shape)  # one row per shot, one column per MEASURE

.. testoutput:: quickstart

   (1000, 2)


Gate channels
=============

:class:`~pyquil.noise.Channel` is the general-purpose gate channel. It stores the gate's noise as a
**Lindbladian** -- a generator of continuous-time evolution -- and its process is the gate
Hamiltonian and the noise generator evolved together for ``gate_time``:

.. math::

   \mathcal{E} = \exp\!\big((\mathcal{L}_\text{gate} + \mathcal{L}_\text{noise})\, t_\text{gate}\big).

Storing the generator rather than the finished process is what makes channels easy to combine and
scale (see `Combining channels`_). ``gate_time`` defaults to ``1.0``, so rates are *per gate*;
set it to the physical duration when rates come from physical quantities such as :math:`T_1`.

Every constructor takes the instruction first. The parameters of the gate must be concrete numbers --
see `Parametric gates are noiseless`_.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Constructor
     - Noise
   * - ``Channel.from_gate_fidelity(inst, fidelity)``
     - Depolarizing, from an average gate fidelity.
   * - ``Channel.from_pauli_fidelity(inst, pauli_fidelity)``
     - Depolarizing, from a process (Pauli, entanglement) fidelity.
   * - ``Channel.from_depolarizing_constant(inst, depolarizing_constant)``
     - Depolarizing, :math:`\rho \mapsto p\rho + (1-p) I/d`; ``1.0`` is noiseless.
   * - ``Channel.from_coherence_times(inst, gate_duration, t1s, t2s=None)``
     - Thermal relaxation and dephasing of each qudit during the gate.
   * - ``Channel.from_pauli_generators(inst, pauli_generators)``
     - Pauli dissipation with generator *rates*, e.g. ``{"XI": 0.01, "ZZ": 0.002}``.
   * - ``Channel.from_mixture(inst, constituents, rates)``
     - Dissipation generated by arbitrary unitary jump operators.
   * - ``Channel.from_random_coherent_error(inst, process_fidelity, rng=None)``
     - A random unitary over-rotation with an exact process fidelity.
   * - ``Channel.from_lindbladian(inst, noise_lindbladian)``
     - Any noise generator built with quax.

.. testcode:: gates

   import numpy as np
   import quax as qx
   from pyquil.gates import CZ, RX
   from pyquil.noise import Channel

   x90 = RX(np.pi / 2, 0)

   depolarizing = Channel.from_gate_fidelity(x90, fidelity=0.999)

   # 40 ns gate on a qubit with T1 = 30 us, T2 = 20 us; times in seconds.
   decoherence = Channel.from_coherence_times(x90, gate_duration=40e-9, t1s=[30e-6], t2s=[20e-6])

   # Correlated ZZ dephasing on a CZ, as a rate per gate.
   zz = Channel.from_pauli_generators(CZ(0, 1), {"ZZ": 0.002})

   # Any quax Lindbladian (here: amplitude damping at a rate of 0.001 per gate).
   damping = Channel.from_lindbladian(x90, qx.lindbladians.amplitude_damping(0.001, (2,)))

   print(f"{depolarizing.average_gate_fidelity:.4f}")

.. testoutput:: gates

   0.9990

.. note::

   ``from_pauli_generators`` and ``from_mixture`` take *generator rates*, not one-shot error
   probabilities. Exponentiating a generator produces products of errors as well as single errors, so
   the result agrees with a mixture of Pauli errors only to first order in the rates. For an exact
   post-gate Pauli channel with given probabilities, use ``SuperopChannel.from_pauli_noise`` below.

Channels from a superoperator
-----------------------------

:class:`~pyquil.noise.SuperopChannel` holds a process directly, for noise that is easiest to write as
a finished channel -- a measured process matrix, a channel from a pulse-level simulation, or a
hand-built Kraus map. Its fields are the instruction, the noisy ``process`` (including the gate), and
the ``ideal_unitary``:

.. testcode:: gates

   from pyquil.noise import SuperopChannel

   # An exact stochastic Pauli channel after the gate: X with probability 1%, Z with 0.5%.
   pauli = SuperopChannel.from_pauli_noise(x90, {"X": 0.01, "Z": 0.005})

   # Any CPTP map can be wrapped: here, amplitude damping as a Kraus map, applied after the gate.
   gamma = 0.01
   kraus = qx.KrausMap.from_matrix(
       np.array([[[1, 0], [0, np.sqrt(1 - gamma)]], [[0, np.sqrt(gamma)], [0, 0]]]), dims=((2,), (2,))
   )
   ideal = qx.gates.RX(np.pi / 2)
   custom = SuperopChannel(
       inst=x90,
       process=qx.to_superop(kraus) @ qx.to_superop(ideal),
       ideal_unitary=ideal,
   )

Gates defined with ``DEFGATE``
------------------------------

Constructors find a gate's ideal unitary in quax's gate table. For a gate the program defines itself,
pass its definitions with ``custom_gates``; :func:`~pyquil.noise.get_custom_gates_from_program`
collects them from a program:

.. testcode:: gates

   from pyquil import Program
   from pyquil.quilbase import DefGate
   from pyquil.noise import get_custom_gates_from_program

   sqrt_x = DefGate("SQRT-X", np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]))
   program = Program(sqrt_x, sqrt_x.get_constructor()(0))

   channel = Channel.from_gate_fidelity(
       sqrt_x.get_constructor()(0),
       fidelity=0.999,
       custom_gates=get_custom_gates_from_program(program),
   )


Combining channels
==================

Channels for the same instruction combine through their generators, and channels for different
instructions combine into a cycle:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Operation
     - Meaning
   * - ``a + b``
     - Both noise processes act simultaneously during the gate (generators add). Same ``inst`` and
       ``gate_time``; the result is a ``Channel``.
   * - ``a ** k``
     - The noise scaled by ``k``, with the gate unchanged: ``a ** 2 == a + a``, and ``a ** 0`` is
       the ideal gate. Useful for noise-strength sweeps and zero-noise extrapolation.
   * - ``a @ b``
     - Sequential composition: the result applies ``a``'s noise, then ``b``'s, around one gate.
       Same ``inst``; the result is a ``SuperopChannel``.
   * - ``a | b``
     - Channels on disjoint qubits, run in parallel as one cycle (a ``CycleChannel``).

.. testcode:: gates

   coherent = Channel.from_random_coherent_error(x90, process_fidelity=0.999, rng=np.random.default_rng(1))
   combined = coherent + depolarizing ** 0.5  # half the depolarizing noise, plus a coherent error

   print(f"{depolarizing.process_fidelity:.5f} {(depolarizing ** 2).process_fidelity:.5f}")

.. testoutput:: gates

   0.99850 0.99700

Analysing a channel
-------------------

Every gate channel reports standard metrics of its *error* -- the noise with the ideal gate divided
out, also available as ``error_process``:

* ``average_gate_fidelity``, ``process_fidelity`` and their infidelities;
* ``unitarity``, and the split of the infidelity into ``coherent_infidelity`` and
  ``stochastic_infidelity``;
* ``to_coherent_channel()`` and ``to_stochastic_channel()``, the two parts as channels in their own
  right, and ``pauli_twirl()``, the Pauli-twirled channel with the same process fidelity;
* ``is_pauli()`` and ``to_pauli_vector()`` for Pauli channels on qubits;
* ``plot()``, a Plotly heat map of the error's Pauli transfer matrix.

.. testcode:: gates

   print(f"{combined.process_infidelity:.2e}")
   print(combined.pauli_twirl().is_pauli())

.. testoutput:: gates

   1.75e-03
   True

These metrics are defined in the quax documentation; pyQuil computes them on the channel's error
process.


Measurement channels
====================

A :class:`~pyquil.noise.MeasurementChannel` attaches a `quantum instrument
<https://rigetti.gitlab.io/application_benchmarking/quax/quantum-instruments.html>`_ to a
measurement. An instrument models both **classification error** -- reporting the wrong outcome -- and
**back-action** -- what the measurement does to the state it leaves behind. Two column-stochastic
matrices summarise it:

* ``confusion_matrix[i, j]``: the probability of reporting outcome ``i`` for a qudit in level ``j``;
* ``transition_matrix[k, j]``: the probability of leaving the qudit in level ``k`` from level ``j``
  (the identity for a quantum non-demolition measurement).

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Constructor
     - Measurement
   * - ``from_readout_fidelity(inst, fidelity, asymmetry=0.0, dim=2)``
     - Non-demolition, with confusion between adjacent levels; ``asymmetry`` biases errors upward
       (positive) or downward (negative).
   * - ``from_confusion_and_transition(inst, confusion_matrix, transition_matrix)``
     - Independent control of classification and back-action.
   * - ``from_axis(inst, theta=0.0, phi=0.0, sharpness=1.0)``
     - Along a Bloch-sphere axis; ``sharpness < 1`` is a weak measurement.
   * - ``from_binary_discriminator(inst, dim, threshold, fidelity=1.0)``
     - A one-bit readout of a ``dim``-level system: levels below ``threshold`` read ``0``.

.. testcode:: measurement

   import numpy as np
   from pyquil.gates import MEASURE
   from pyquil.noise import MeasurementChannel

   # |1> is misread as 0 more often than |0> as 1 (T1 decay during readout).
   readout = MeasurementChannel.from_readout_fidelity(MEASURE(0, None), fidelity=0.95, asymmetry=-0.4)
   print(np.round(readout.confusion_matrix, 3))

   # Classification is perfect, but 2% of |1> decays to |0> during the measurement.
   demolition = MeasurementChannel.from_confusion_and_transition(
       MEASURE(1, None),
       confusion_matrix=np.eye(2),
       transition_matrix=np.array([[1.0, 0.02], [0.0, 0.98]]),
   )
   print(f"{demolition.non_demolition_fidelity:.2f}")

.. testoutput:: measurement

   [[0.97 0.07]
    [0.03 0.93]]
   0.99

Measurement channels are keyed by the **qubit**, not by the classical address: a channel keyed by
``MEASURE 0`` (``MEASURE(0, None)``) also applies to ``MEASURE 0 ro[0]`` and ``MEASURE 0 ro[3]``,
because readout error is a property of the qubit and not of the bit it is stored in. An exact match,
such as a channel keyed by ``MEASURE 0 ro[1]`` itself, takes precedence.


Reset channels
==============

A reset channel replaces an active ``RESET q`` with a noisy one. The ideal reset sends every state to
:math:`|0\rangle`; a noisy reset leaves some residual population behind.

* :meth:`SuperopResetChannel.from_reset_fidelity <pyquil.noise.SuperopResetChannel.from_reset_fidelity>`
  -- an ideal reset followed by depolarization, with the requested process fidelity.
* :meth:`ResetChannel.from_amplitude_damping <pyquil.noise.ResetChannel.from_amplitude_damping>` and
  :meth:`ResetChannel.from_coherence_times <pyquil.noise.ResetChannel.from_coherence_times>` -- reset
  as finite-time relaxation toward the ground state; the longer the reset relaxes, the better it is.

.. testcode:: reset

   from pyquil.noise import ResetChannel, SuperopResetChannel
   from pyquil.quilbase import ResetQubit

   reset_0 = SuperopResetChannel.from_reset_fidelity(ResetQubit(0), fidelity=0.99)
   reset_1 = ResetChannel.from_coherence_times(ResetQubit(1), duration=100e-6, t1=30e-6)
   print(f"{reset_0.process_fidelity:.2f}")

.. testoutput:: reset

   0.99

Reset channels are keyed by the targeted ``RESET q``. A program-wide ``RESET`` (no qubit) is applied
as a reset of every qubit in the program, each looked up as ``RESET q``.


How channels are applied
========================

The simulators look up every instruction of the program in the noise model as they expand it (see
the resolver in :ref:`simulation_architecture`). The rules are:

* **Lookup is by instruction equality**: name, parameters, qubits and modifiers. ``RX(pi/2) 0`` and
  ``RX(pi/2) 1`` are different keys, a channel for ``RX(pi/2) 0`` does not apply to ``RX(pi/4) 0``,
  and a channel for ``CZ 0 1`` does not apply to ``CZ 1 0``, even though the gate is symmetric. A
  program compiled by ``quilc`` uses whatever operand order the compiler chose, so key two-qubit
  channels in both orders if you are unsure.
* **Measurements fall back to the qubit**: ``MEASURE q ro[k]`` uses the channel keyed by
  ``MEASURE q ro[k]`` if there is one, and otherwise the channel keyed by ``MEASURE q``.
* **Instructions without a channel are ideal.**
* **Gate modifiers are rejected.** ``DAGGER``, ``CONTROLLED`` and ``FORKED`` are not supported by the
  simulators, noisy or not.
* A ``DEFCIRCUIT`` call is either replaced by a matching ``CycleChannel`` or expanded, after which
  each instruction in its body is looked up on its own.

What a simulator does with a channel depends on the state it evolves:

* :class:`~pyquil.simulation.DensityMatrixSimulator` applies every channel exactly. A measurement
  acts on the state as its total, outcome-averaged channel, while the readout error of the *terminal*
  measurements is included in ``outcome_probabilities``.
* :class:`~pyquil.simulation.TrajectorySimulator` converts each channel to Kraus operators and samples
  one per trajectory; measurement instruments are sampled outcome by outcome, so the returned
  bitstrings include classification error and the state after a mid-circuit measurement includes
  back-action.
* :class:`~pyquil.simulation.PureStateVectorSimulator` does not accept a noise model.

Parametric gates are noiseless
------------------------------

A gate whose parameter reads classical memory, such as ``RX(theta[0]) 0``, is **never noisy**: the
noise model is not consulted for it, and a channel cannot be constructed for it. This is deliberate.
On Rigetti hardware the only continuously-parametrised gate is ``RZ``, which is implemented virtually
as a frame change and has no pulse, no duration and no error; every other rotation is calibrated at a
fixed set of angles, so its noise is described by a channel keyed on the concrete instruction, such as
``RX(pi/2) 0``. Bind parameters before simulating if you need noise on them.

.. testcode:: keying

   import numpy as np
   from pyquil import Program
   from pyquil.gates import MEASURE, RX
   from pyquil.noise import Channel, NoiseModel

   noise_model = NoiseModel.from_channels([Channel.from_gate_fidelity(RX(np.pi, 0), fidelity=0.99)])

   print(noise_model.get_channel(RX(np.pi, 0)) is not None)
   print(noise_model.get_channel(RX(np.pi, 1)))  # different qubit: ideal
   print(noise_model.get_channel(RX(np.pi / 2, 0)))  # different angle: ideal

.. testoutput:: keying

   True
   None
   None


Cycle noise
===========

Some noise is not a property of one instruction: crosstalk, correlated errors, or idling qubits that
decohere while their neighbours are driven. A :class:`~pyquil.noise.CycleChannel` describes the noise
of a whole parallel **cycle** of operations, keyed to a call of a ``DEFCIRCUIT`` that contains it.
When the simulator meets that call, it applies the cycle's channels in place of the circuit's body.

A cycle holds one channel per instruction in the circuit body, in order, on disjoint qubits. The
simplest way to build one is ``|``, which also generates a ``DEFCIRCUIT`` named ``CYCLE`` whose body
is the channels' instructions; add that definition and a call to it to the program:

.. testcode:: cycle

   import numpy as np
   from pyquil import Program
   from pyquil.gates import I, MEASURE, RX
   from pyquil.noise import Channel, MeasurementChannel, NoiseModel
   from pyquil.simulation import DensityMatrixSimulator

   cycle = (
       Channel.from_gate_fidelity(RX(np.pi, 0), fidelity=0.99)
       | Channel.from_coherence_times(I(1), gate_duration=40e-9, t1s=[20e-6])  # an idling neighbour
       | MeasurementChannel.from_readout_fidelity(MEASURE(2, None), fidelity=0.95)
   )
   print(cycle.inst)

   program = Program(cycle.defcircuit, cycle.inst)
   rho = DensityMatrixSimulator(program, noise_model=NoiseModel.from_channels([cycle])).compute()

.. testoutput:: cycle

   CYCLE 0 1 2

To key noise to circuits you already have -- one ``DEFCIRCUIT`` per layer of an error-correction round,
say -- construct the cycle directly from the definition, the call, and one channel per body
instruction (with formal arguments replaced by the call's qubits):

.. testcode:: cycle

   from pyquil.noise import CycleChannel
   from pyquil.quilatom import FormalArgument, Qubit
   from pyquil.quilbase import DefCircuit, Gate

   a, b = FormalArgument("a"), FormalArgument("b")
   layer = DefCircuit("LAYER", [], [a, b], [RX(np.pi, a), RX(np.pi, b)])
   call = Gate("LAYER", [], [Qubit(4), Qubit(5)])

   layer_noise = CycleChannel(
       inst=call,
       defcircuit=layer,
       channels=(
           Channel.from_gate_fidelity(RX(np.pi, 4), fidelity=0.995),
           Channel.from_gate_fidelity(RX(np.pi, 5), fidelity=0.990),
       ),
   )
   print(f"{layer_noise.process_fidelity:.4f}")

.. testoutput:: cycle

   0.9776

A ``CycleChannel`` is checked against its circuit when it is built, so a body instruction without a
channel -- which would silently drop that operation's noise -- raises an error. A circuit body must
reference only its formal arguments.


Qudits and leakage
==================

Noise models are not limited to qubits. A transmon has more than two levels, and population that
leaks out of the computational subspace is a significant error on real hardware. Channels carry
explicit per-qudit dimensions, and the simulators infer the dimension of each line of the register
from the operators that act on it, so a single qutrit channel is enough to promote a qubit to a
qutrit throughout the simulation.

* Constructors built on Lindbladians -- ``from_gate_fidelity``, ``from_depolarizing_constant``,
  ``from_coherence_times``, ``from_lindbladian`` -- work for any dimension, given a gate whose unitary
  is a qudit operator (quax's qutrit gates, or a ``DEFGATE``).
* ``from_lindbladian`` with a qutrit generator models leakage during a gate; quax's
  `promotion <https://rigetti.gitlab.io/application_benchmarking/quax/promotion.html>`_ embeds qubit
  gates and noise in a qutrit space.
* Measurement and reset constructors take ``dim``. ``from_binary_discriminator`` models a readout
  that reports one bit for a qutrit -- with ``threshold=2``, it flags leakage only.
* Pauli-based constructors and analyses (``from_pauli_generators``, ``from_random_coherent_error``,
  ``SuperopChannel.from_pauli_noise``, ``is_pauli``, ``to_pauli_vector``) are defined for qubits only
  and say so if given a qudit.

.. testcode:: qudits

   import jax
   import numpy as np
   from pyquil import Program
   from pyquil.gates import MEASURE
   from pyquil.noise import Channel, MeasurementChannel, NoiseModel
   from pyquil.quilbase import Gate
   from pyquil.simulation import TrajectorySimulator

   tx = Gate("TX", [], [0])  # quax's qutrit X: |0> -> |2>
   noise_model = NoiseModel.from_channels(
       [
           Channel.from_gate_fidelity(tx, fidelity=0.98),
           MeasurementChannel.from_readout_fidelity(MEASURE(0, None), fidelity=0.95, dim=3),
       ]
   )

   program = Program(tx, MEASURE(0, None))
   shots = TrajectorySimulator(program, noise_model=noise_model).sample(num_trajectories=1000, key=jax.random.key(0))
   print(sorted(set(np.asarray(shots).ravel().tolist())))  # outcomes run over the three levels

.. testoutput:: qudits

   [0, 1, 2]


Noise models
============

:class:`~pyquil.noise.NoiseModel` is an immutable mapping from instructions to channels. Build one
from channels, and extend or merge models without mutating them:

.. testcode:: models

   import numpy as np
   from pyquil.gates import CZ, MEASURE, RX
   from pyquil.noise import Channel, MeasurementChannel, NoiseModel

   gates = NoiseModel.from_channels(
       [Channel.from_gate_fidelity(RX(np.pi / 2, q), fidelity=0.999) for q in range(2)]
   )
   readout = NoiseModel.from_channels(
       [MeasurementChannel.from_readout_fidelity(MEASURE(q, None), fidelity=0.97) for q in range(2)]
   )

   noise_model = (gates + readout).with_channels([Channel.from_gate_fidelity(CZ(0, 1), fidelity=0.99)])
   print(len(noise_model.channels))

.. testoutput:: models

   5

``from_channels``, ``+`` and ``with_channels`` all raise if two channels are keyed to the same
instruction, rather than silently keeping one of them.

From a device
-------------

:meth:`NoiseModel.from_isa <pyquil.noise.NoiseModel.from_isa>` builds a model from a QCS
instruction set architecture: every native gate with a reported fidelity becomes a depolarizing
``Channel`` with that average gate fidelity, and every qubit's readout fidelity becomes a symmetric
``MeasurementChannel``. Gates are keyed by the operand order the ISA reports.

.. code-block:: python

   from qcs_sdk.qpu.isa import get_instruction_set_architecture
   from pyquil.noise import NoiseModel

   noise_model = NoiseModel.from_isa(get_instruction_set_architecture("Ankaa-3"))

A device model is a starting point: it treats every gate error as depolarizing and says nothing
about coherent errors, crosstalk or leakage, which you can add with ``with_channels`` or ``+``.

Uniform noise and custom models
-------------------------------

The simulators accept anything with a ``get_channel(inst)`` method -- the
:class:`~pyquil.noise.NoiseModelLike` protocol -- so a noise model can also compute its channels on
demand. :class:`~pyquil.noise.DepolarizingNoiseModel` is one: it returns the same depolarizing
channel for every gate, whatever its qubits or parameters, and leaves measurements and resets ideal.

.. testcode:: models

   from pyquil import Program
   from pyquil.noise import DepolarizingNoiseModel
   from pyquil.simulation import DensityMatrixSimulator

   program = Program(RX(np.pi / 2, 0), CZ(0, 1))
   rho = DensityMatrixSimulator(program, noise_model=DepolarizingNoiseModel(0.99)).compute()

A custom model returns a channel for each instruction it considers noisy and ``None`` otherwise. Cache
the channels it builds: the simulator asks once per instruction in the program.

Saving and loading
------------------

Noise models and channels serialize to JSON, so a model can be built once -- from a characterization
experiment, say -- and stored alongside the results:

.. testcode:: models

   restored = NoiseModel.from_json(noise_model.to_json())
   print(restored == noise_model)

.. testoutput:: models

   True


Estimating program fidelity
===========================

Before simulating, a quick estimate of how much a noise model will hurt a program is often useful.
:func:`~pyquil.noise.estimate_program_fidelity` multiplies the process fidelities of the program's
noisy gates; :func:`~pyquil.noise.estimate_program_observable_fidelity` first restricts the program to
the backward light cone of an observable, so gates that cannot affect it do not count. Both ignore
readout and reset error, and both are estimates: they assume the errors are uncorrelated and do not
interfere.

.. testcode:: models

   from pyquil.noise import estimate_program_fidelity, estimate_program_observable_fidelity
   from pyquil.paulis import sZ

   program = Program(RX(np.pi / 2, 0), RX(np.pi / 2, 1), CZ(0, 1), RX(np.pi / 2, 1))
   print(f"{estimate_program_fidelity(program, noise_model):.4f}")
   print(f"{estimate_program_observable_fidelity(program, noise_model, sZ(0)):.4f}")

.. testoutput:: models

   0.9831
   0.9845


Migrating from the Kraus-map noise model
========================================

pyQuil v4's noise model was a set of Kraus maps attached to a program as QVM pragmas, together with
helpers for readout correction. It was removed in pyQuil v5 along with the QVM. The table maps its
functions to the current API.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - pyQuil v4
     - pyQuil v5
   * - ``Program.define_noisy_gate(name, qubits, kraus_ops)``,
       ``append_kraus_to_gate``
     - ``SuperopChannel(inst=..., process=..., ideal_unitary=...)``, with the process built from a
       ``qx.KrausMap`` (see `Channels from a superoperator`_), in a ``NoiseModel``.
   * - ``damping_kraus_map``, ``dephasing_kraus_map``, ``damping_after_dephasing``
     - ``Channel.from_coherence_times``, or ``Channel.from_lindbladian`` with
       ``qx.lindbladians.amplitude_damping`` / ``thermal_relaxation``.
   * - ``pauli_kraus_map``, ``merge_with_pauli_noise``
     - ``SuperopChannel.from_pauli_noise``.
   * - ``tensor_kraus_maps``, ``combine_kraus_maps``
     - ``|`` (parallel) and ``@`` (sequential) on channels, or ``|`` and ``@`` on quax operators.
   * - ``add_decoherence_noise``, ``decoherence_noise_with_asymmetric_ro``
     - ``Channel.from_coherence_times`` per gate, ``MeasurementChannel.from_readout_fidelity`` with
       ``asymmetry``, and ``ResetChannel`` for resets; or ``NoiseModel.from_isa`` for a device.
   * - ``Program.define_noisy_readout(qubit, p00, p11)``
     - ``MeasurementChannel.from_confusion_and_transition`` with
       ``confusion_matrix=[[p00, 1 - p11], [1 - p00, p11]]`` and an identity transition matrix.
   * - ``apply_noise_model(program, noise_model)``
     - Pass the model to the simulator: ``DensityMatrixSimulator(program, noise_model=...)``.
   * - ``get_qc(..., noisy=True)``, ``qc.qam.gate_noise``, ``qc.qam.measurement_noise``
     - ``DepolarizingNoiseModel`` or a ``NoiseModel`` of ``SuperopChannel.from_pauli_noise`` and
       ``MeasurementChannel`` entries.
   * - ``estimate_bitstring_probs``, ``correct_bitstring_probs``, ``corrupt_bitstring_probs``,
       ``estimate_assignment_probs``, ``bitstring_probs_to_z_moments``
     - Removed. ``DensityMatrixSimulator.outcome_probabilities`` gives exact noisy distributions;
       readout mitigation is provided by ``rigetti-qpu-hybrid-benchmark``.

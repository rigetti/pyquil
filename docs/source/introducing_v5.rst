.. _introducing_v5:

=====================
Introducing pyQuil v5
=====================

pyQuil v5 replaces the Quantum Virtual Machine (QVM) with simulators that run in-process, built on
`JAX <https://docs.jax.dev>`_ and `quax <https://rigetti.gitlab.io/application_benchmarking/quax/>`_. They need no
server, model noise as arbitrary channels attached to individual instructions, simulate qudits and leakage, and can be
``jax.jit``-compiled and differentiated with ``jax.grad``. Alongside the QVM, v5 removes the APIs that were deprecated in
pyQuil v4, and it orders multi-qubit states big-endian.

Read through the :doc:`changes` for the complete list of changes. This page describes how to move code written for
pyQuil v4 onto v5.

*********************
What has been removed
*********************

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Removed in v5
     - Replacement
   * - ``pyquil.api.QVM``, ``pyquil.pyqvm.PyQVM``, and QVM-backed quantum computers from ``get_qc`` (a ``-qvm`` or
       ``-pyqvm`` name, or ``as_qvm=True``)
     - :py:class:`~pyquil.simulation.TrajectorySimulator` to sample measurement outcomes, or
       :py:class:`~pyquil.simulation.DensityMatrixSimulator` for exact outcome probabilities. See :ref:`simulation`.
   * - ``pyquil.api.WavefunctionSimulator``, ``pyquil.wavefunction``, and the ``NumpyWavefunctionSimulator``,
       ``ReferenceWavefunctionSimulator`` and ``ReferenceDensitySimulator`` simulators and their helper functions in
       ``pyquil.simulation``
     - :py:class:`~pyquil.simulation.PureStateVectorSimulator` for the state of a noiseless program, or
       :py:class:`~pyquil.simulation.DensityMatrixSimulator` for the state of a noisy one.
   * - ``pyquil.api.QVMCompiler``
     - ``get_qc`` with a generic lattice name such as ``"9q-square"`` or ``"8q"``, which compiles with ``quilc`` but
       can't run programs; or :py:class:`~pyquil.api.QuilcCompiler` for a topology of your own.
   * - The Kraus-map noise model in ``pyquil.noise`` (``NoiseModel``, ``KrausModel``, ``add_decoherence_noise``,
       ``append_kraus_to_gate``, ``apply_noise_model``, ``decoherence_noise_with_asymmetric_ro``, and the Kraus-map
       helpers)
     - The channel-based noise model in :py:mod:`pyquil.noise`: :py:class:`~pyquil.noise.Channel`,
       :py:class:`~pyquil.noise.MeasurementChannel`, :py:class:`~pyquil.noise.NoiseModel` and friends. See
       :ref:`noise`.
   * - ``Program.define_noisy_gate``, ``Program.define_noisy_readout`` and ``pyquil.quil.merge_with_pauli_noise``
     - Noise lives in a noise model passed to a simulator, not in the program: a
       :py:class:`~pyquil.noise.Channel` for a gate, a :py:class:`~pyquil.noise.MeasurementChannel` for readout.
   * - The readout helpers ``estimate_bitstring_probs``, ``correct_bitstring_probs``, ``corrupt_bitstring_probs``,
       ``bitstring_probs_to_z_moments`` and ``estimate_assignment_probs``
     - None in pyQuil. :py:meth:`DensityMatrixSimulator.outcome_probabilities
       <pyquil.simulation.DensityMatrixSimulator.outcome_probabilities>` gives exact outcome probabilities with readout
       error included.
   * - ``pyquil.experiment``, ``QuantumComputer.run_experiment`` and ``QuantumComputer.calibrate``
     - `rigetti-qpu-hybrid-benchmark <https://pypi.org/project/rigetti-qpu-hybrid-benchmark/>`_.
   * - ``pyquil.operator_estimation``
     - The Estimator interface of rigetti-qpu-hybrid-benchmark.
   * - ``pyquil.latex``
     - None.

:py:func:`~pyquil.api.local_forest_runtime` now starts only ``quilc``, and the ``qvm_url`` setting and
``QCS_SETTINGS_APPLICATIONS_QVM_URL`` environment variable are no longer read.

If you still need the QVM itself, `qcs-sdk-python <https://github.com/rigetti/qcs-sdk-rust>`_ continues to support
executing Quil programs on a QVM server; see the documentation for
`qcs_sdk.qvm <https://rigetti.github.io/qcs-sdk-rust/qcs_sdk/qvm.html>`_.

*******************
Sampling a program
*******************

In pyQuil v4, a program was sampled by running it on a QVM-backed quantum computer:

.. code:: python

    # pyQuil v4
    from pyquil import Program, get_qc
    from pyquil.gates import CNOT, H, MEASURE

    p = Program()
    ro = p.declare("ro", "BIT", 2)
    p += H(0)
    p += CNOT(0, 1)
    p += MEASURE(0, ro[0])
    p += MEASURE(1, ro[1])
    p.wrap_in_numshots_loop(1000)

    qc = get_qc("2q-qvm")
    bitstrings = qc.run(qc.compile(p)).get_register_map()["ro"]

In pyQuil v5, a simulator is constructed from the program, and the number of shots is an argument to ``sample``:

.. testcode:: v5-sampling

    import jax

    from pyquil import Program
    from pyquil.gates import CNOT, H, MEASURE
    from pyquil.simulation import TrajectorySimulator

    p = Program()
    ro = p.declare("ro", "BIT", 2)
    p += H(0)
    p += CNOT(0, 1)
    p += MEASURE(0, ro[0])
    p += MEASURE(1, ro[1])

    bitstrings = TrajectorySimulator(p).sample(num_trajectories=1000, key=jax.random.key(0))
    print(bitstrings.shape)

.. testoutput:: v5-sampling

    (1000, 2)

Each row is one shot and each column one ``MEASURE``, in program order. A few differences are worth knowing:

* **No compilation is needed.** The simulators accept any gate pyQuil defines, so there is no need to compile a program
  before simulating it. To simulate what will actually run on a QPU, compile the program first and simulate the
  result; see :ref:`simulation`.
* **Only straight-line programs.** The simulators reject classical control flow (``JUMP``, ``JUMP-WHEN``,
  ``JUMP-UNLESS``) and classical arithmetic (``MOVE``, ``ADD``, ...), and gate modifiers (``DAGGER``,
  ``CONTROLLED``, ``FORKED``). They ignore Quil-T instructions such as ``DELAY`` and ``PULSE``.
* **Reproducibility is explicit.** Pass a JAX PRNG key to get the same samples every time; omit it for fresh ones.

*******************************
Inspecting a program's state
*******************************

In pyQuil v4, the ``WavefunctionSimulator`` returned the state of a program:

.. code:: python

    # pyQuil v4
    from pyquil.api import WavefunctionSimulator

    wavefunction = WavefunctionSimulator().wavefunction(Program(X(0), I(1)))
    amplitudes = wavefunction.amplitudes
    probabilities = wavefunction.probabilities()

In pyQuil v5, :py:class:`~pyquil.simulation.PureStateVectorSimulator` returns the state as a quax ``StateVector``:

.. testcode:: v5-state

    import numpy as np

    from pyquil import Program
    from pyquil.gates import I, X
    from pyquil.simulation import PureStateVectorSimulator

    state = PureStateVectorSimulator(Program(X(0), I(1))).compute()
    amplitudes = state.matrix.reshape(-1)
    probabilities = np.abs(amplitudes) ** 2
    print(probabilities)

.. testoutput:: v5-state

    [0. 0. 1. 0.]

Note where the excited qubit's amplitude lands: at index 2, :math:`\ket{10}`, not index 1. That is the change of
basis ordering described next.

.. _v5_endianness:

***************************
Big-endian basis ordering
***************************

pyQuil v4, the QVM and the ``WavefunctionSimulator`` ordered states little-endian: qubit 0 was the *least*
significant bit, so ``X 0`` on two qubits gave :math:`\ket{01}`, at index 1. pyQuil v5 orders them **big-endian**, as
is usual in the quantum computing literature: qubit 0 is the *most* significant bit, so the same program gives
:math:`\ket{10}`, at index 2. Gate matrices follow the same convention, so the full matrix of ``CNOT 0 1`` is the
textbook one. See :ref:`basis_ordering`.

Code that reads amplitudes or probabilities by index, or builds a bitstring from them, needs updating. Sampled
measurement outcomes are not affected: they are indexed by ``MEASURE`` instruction, not by basis state.

*********************
Compiling a program
*********************

``get_qc`` no longer returns a QVM. A generic lattice name -- the old ``-qvm`` names without the suffix, such as
``"9q-square"`` or ``"8q"`` -- returns a :py:class:`~pyquil.api.QuantumComputer` that compiles with ``quilc`` but
can't run programs. The name of a QPU returns one that runs programs on it, as before.

.. code:: python

    # pyQuil v4
    qc = get_qc("9q-square-qvm")
    executable = qc.compile(program)
    qc.run(executable)

    # pyQuil v5
    qc = get_qc("9q-square")
    native = qc.compiler.quil_to_native_quil(program)
    TrajectorySimulator(native).sample(num_trajectories=1000)

In pyQuil v4, ``get_qc(QPU_NAME, as_qvm=True)`` simulated a QPU's topology on a QVM. In v5, compile against the QPU
and simulate the compiled program; to include the QPU's noise, build a noise model from its instruction set
architecture with :py:meth:`NoiseModel.from_isa <pyquil.noise.NoiseModel.from_isa>` and pass it to the simulator.

.. code:: python

    from qcs_sdk.qpu.isa import get_instruction_set_architecture

    from pyquil import get_qc
    from pyquil.noise import NoiseModel
    from pyquil.simulation import DensityMatrixSimulator

    qc = get_qc("Ankaa-3")
    native = qc.compiler.quil_to_native_quil(program)
    noise_model = NoiseModel.from_isa(get_instruction_set_architecture(quantum_processor_id="Ankaa-3"))
    probabilities = DensityMatrixSimulator(native, noise_model=noise_model).outcome_probabilities()

******************
Modelling noise
******************

The Kraus-map noise model, which attached noise to a program through ``PRAGMA`` instructions only the QVM understood,
is replaced by a noise model that maps Quil instructions to quantum channels and is passed to a simulator alongside
the program:

.. code:: python

    # pyQuil v4
    from pyquil.noise import add_decoherence_noise

    noisy_program = add_decoherence_noise(program, T1=30e-6, T2=30e-6)
    qc.run(qc.compile(noisy_program))

    # pyQuil v5
    from pyquil.noise import Channel, NoiseModel
    from pyquil.simulation import DensityMatrixSimulator

    noise_model = NoiseModel.from_channels([
        Channel.from_coherence_times(gate, gate_duration=50e-9, t1s=[30e-6], t2s=[30e-6])
        for gate in single_qubit_gates
    ])
    DensityMatrixSimulator(program, noise_model=noise_model).outcome_probabilities()

:ref:`noise` describes how to build channels and noise models and how the simulators apply them, and ends with a guide
to translating each part of the Kraus-map noise model.

*************************
Simulation in more depth
*************************

The simulators are documented in :ref:`simulation`, worked through in the simulation tutorial, and their design --
how a program becomes a sequence of operators, and how that sequence is compiled and evaluated -- is described in
:ref:`simulation_architecture`.

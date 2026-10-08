.. _getting_started:

===============
Getting started
===============

.. _prerequisites:

**************
Pre-requisites
**************
To make full use of pyQuil, you'll want to have the Quil Compiler (``quilc``) installed. If you don't have it installed yet,
refer to `Rigetti's guide on installing the Quil SDK locally <https://docs.rigetti.com/qcs/getting-started/installing-locally>`_.
pyQuil simulates programs itself, in-process, with the JAX-based simulators described in :ref:`simulation`, so no separate
simulator needs to be installed or started.

.. note::

    If you're running from a Rigetti-provisioned JupyterLab IDE, the Quil SDK is already installed. Continue to
    :ref:`run_your_first_program`.

Upgrading or installing pyQuil
==============================
Before you install, it's recommended to activate a Python virtual environment. Then, install pyQuil using
`pip <https://pip.pypa.io/en/stable/quickstart/>`_:

::

    pip install pyquil

For those of you that already have pyQuil, you can upgrade with:

::

    pip install --upgrade pyquil

If you would like to stay up to date with the latest changes and bug fixes, you can also opt to install a pre-release version of pyQuil with:

::

    pip install --pre pyquil

.. note::

    pyQuil requires Python >=3.11, <3.13.

.. testcode:: verify-min-version
    :hide:

    # The above note and this test should be updated whenever
    # the minimum supported Python version is changed.

    import os
    import toml

    with open(f"{os.getcwd()}/../pyproject.toml", "r") as file:
        t = toml.load(file)
        print(t["tool"]["poetry"]["dependencies"]["python"])

.. testoutput:: verify-min-version
    :hide:

    >=3.11, <3.13

.. note::

   Some of pyQuil's core dependencies are powered by Rust. These dependencies have been pre-built for the most common platforms so that
   building from source isn't required. However, if you are on a less common platform, or choose to build pyQuil from source, you will need
   to `install Rust <https://www.rust-lang.org/tools/install>`_.

.. _server:

Setting up the compiler server
==============================

pyQuil compiles programs to native Quil by making requests to ``quilc`` running in server mode. If you have it installed
locally, launch it in its own terminal window:

.. code:: sh

   quilc -S

.. note::

    For more information about the compiler, refer to its manual page by using ``man quilc``.

That's it! You're all set up to use pyQuil locally. ``quilc`` is only needed to compile programs; building and simulating
programs works without it.

.. _run_your_first_program:

**********************
Run your first program
**********************
The program we will create prepares a fully entangled state between two qubits, called a `Bell State <https://www.wikiwand.com/en/Bell_state>`_.
This state is in an equal superposition between :math:`\ket{00}` and :math:`\ket{11}`, meaning that it's equally likely that a measurement will
result in measuring both qubits in the ground state or both qubits in the excited state.

First, import the essentials:

.. testcode:: first-program

    import jax
    import numpy as np

    from pyquil import Program
    from pyquil.gates import *
    from pyquil.simulation import PureStateVectorSimulator, TrajectorySimulator

The :py:class:`~pyquil.Program` class allows us to build a Quil program. We've also imported all (``*``) gates from the
``pyquil.gates`` module, which allows us to add operations to our program (:ref:`basics`), and two of pyQuil's simulators
(:ref:`simulation`).

Next, let's construct the Bell State program.

.. testcode:: first-program

    bell = Program(H(0), CNOT(0, 1))

We've accomplished this by driving qubit 0 into a superposition state (that's what the "H" gate does), and then creating
an entangled state between qubits 0 and 1 (that's what the "CNOT" gate does). A simulator can show us the resulting state
directly:

.. testcode:: first-program

    state = PureStateVectorSimulator(bell).compute()
    print(np.round(state.matrix.reshape(-1), 3))

.. testoutput:: first-program

    [0.707+0.j 0.   +0.j 0.   +0.j 0.707+0.j]

The four amplitudes belong to the basis states :math:`\ket{00}, \ket{01}, \ket{10}, \ket{11}`, with qubit 0 as the
leftmost (most significant) digit; see :ref:`basis_ordering`. On a real quantum computer we can't look at the state; we
measure it. Let's add measurements and sample the program ten times:

.. testcode:: first-program

    p = Program()
    ro = p.declare("ro", "BIT", 2)
    p += bell
    p += MEASURE(0, ro[0])
    p += MEASURE(1, ro[1])

    shots = TrajectorySimulator(p).sample(num_trajectories=10, key=jax.random.key(0))
    print(shots.shape)
    print(bool(np.all(shots[:, 0] == shots[:, 1])))

.. testoutput:: first-program

    (10, 2)
    True

Each row of ``shots`` is one run of the program, and each column one ``MEASURE``, in program order. The results are
random from shot to shot, but always agree between the two qubits.

Compiling for a quantum computer
================================

A quantum processor only implements a small set of *native* gates on the qubit pairs it physically connects. Before a
program runs on one, ``quilc`` rewrites it into that native gate set. :py:func:`~pyquil.get_qc` returns a
:py:class:`~pyquil.api.QuantumComputer`, and its compiler does the rewriting:

.. testcode:: first-program

    from pyquil import get_qc

    qc = get_qc("9q-square")
    native = qc.compiler.quil_to_native_quil(bell)

``"9q-square"`` names a generic nine-qubit lattice, which is useful for compiling and simulating locally; it can't run
programs. The name of a real quantum processor, such as ``"Ankaa-3"``, returns a :py:class:`~pyquil.api.QuantumComputer`
that runs programs on that QPU through `Rigetti's Quantum Cloud Services <https://docs.rigetti.com/qcs/>`_ (see
:ref:`the_quantum_computer`). The compiled program is still an ordinary :py:class:`~pyquil.Program`, so it can be
simulated as well, and it prepares the same state:

.. testcode:: first-program

    native_state = PureStateVectorSimulator(native).compute()

.. warning::

   If compiling hangs or fails, make sure the ``quilc`` server is running and reachable. First, review the
   `pre-requisites section <prerequisites>`_ and if that fails, see the `troubleshooting steps <timeouts>`_.

.. note::

    pyQuil also provides the :py:func:`~pyquil.api.local_forest_runtime()` context manager to ensure the ``quilc`` server
    is running by starting it as a subprocess if it isn't already.

    .. code:: python

        from pyquil import get_qc, Program
        from pyquil.gates import CNOT, Z
        from pyquil.api import local_forest_runtime

        prog = Program(Z(0), CNOT(0, 1))

        with local_forest_runtime():
            qc = get_qc("9q-square")
            native = qc.compiler.quil_to_native_quil(prog)

In the following sections, we'll cover gates, program construction & execution, and go into detail about simulation,
our QPUs, noise models and more. Let's start with the :ref:`basics`.

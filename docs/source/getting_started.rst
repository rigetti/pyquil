.. _getting_started:

===============
Getting started
===============

.. _prerequisites:

**************
Pre-requisites
**************
To make full use of pyQuil, you'll want to have both the Quantum Virtual Machine (QVM) and the Quil Compiler (quilc) installed. If you
don't have those installed yet, refer to `Rigetti's guide on installing the Quil SDK locally <https://docs.rigetti.com/qcs/getting-started/installing-locally>`_.

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

    pyQuil requires Python 3.8 or later.

.. note::

   Some of pyQuil's core dependencies are powered by Rust. These dependencies have been pre-built for the most common platforms so that
   building from source isn't required. However, if you are on a less common platform, or choose to build pyQuil from source, you will need
   to `install Rust <https://www.rust-lang.org/tools/install>`_.

.. testcode:: verify-min-version
    :hide:

    import os
    import toml

    with open(f"{os.getcwd()}/../pyproject.toml", "r") as file:
        t = toml.load(file)
        print(t["tool"]["poetry"]["dependencies"]["python"])

.. testoutput:: verify-min-version
    :hide:

    ^3.8...

.. _server:

Setting up server mode for pyQuil
=================================
To get started with pyQuil, ``quilc`` and ``qvm`` should both be running in server mode. If you have them installed locally
you can run them in their own terminal windows. First launch ``quilc``:

.. code:: sh

   quilc -S

Then, in another terminal window, launch the QVM:

.. code:: sh

   qvm -S

.. note::

    For more information about the QVM and the compiler, refer to their respective manual pages by using ``man quilc`` and ``man qvm``.

That's it! You're all set up to run pyQuil locally. Your programs will make requests to these server endpoints to compile your Quil
programs to native Quil, and to simulate those programs on the QVM.

.. _run_your_first_program:

**********************
Run your first program
**********************
Now that the QVM and the Quil compiler are running, you can start running pyQuil programs!

The program we will create prepares a fully entangled state between two qubits, called a `Bell State <https://www.wikiwand.com/en/Bell_state>`_.
This state is in an equal superposition between :math:`\ket{00}` and :math:`\ket{11}`, meaning that it's equally likely that a measurement will
result in measuring both qubits in the ground state or both qubits in the excited state.

First, import the essentials:

.. testcode:: first-program

    from pyquil import Program, get_qc
    from pyquil.gates import *
    from pyquil.quilbase import Declare

The :py:class:`~pyquil.Program` class allows us to build a Quil program. :py:func:`~pyquil.get_qc` connects us to a
:py:class:`~pyquil.api.QuantumComputer`, which specifies what our program should run on (see: :ref:`qvm`). We've also imported all (``*``)
gates from the ``pyquil.gates`` module, which allows us to add operations to our program (:ref:`basics`). :py:class:`~pyquil.quilbase.Declare`
allows us to declare classical memory regions so that we can receive data from the :py:class:`~pyquil.api.QuantumComputer`.

Next, let's construct the Bell State program.

.. testcode:: first-program

    p = Program(
        Declare("ro", "BIT", 2),
        H(0),
        CNOT(0, 1),
        MEASURE(0, ("ro", 0)),
        MEASURE(1, ("ro", 1)),
    ).wrap_in_numshots_loop(10)

We've accomplished this by driving qubit 0 into a superposition state (that's what the "H" gate does), and then creating
an entangled state between qubits 0 and 1 (that's what the "CNOT" gate does). Finally, we'll want to run our program:

.. testcode:: first-program

    # run the program on a QVM
    qc = get_qc('9q-square-qvm')
    result = qc.run(qc.compile(p)).readout_data.get("ro")
    print(result[0])
    print(result[1])

.. testoutput:: first-program
    :hide:

    [...]
    [...]

.. warning::

   If you run into an error running your program, or it hangs indefinitely when compiling, make sure that the ``quilc`` and ``QVM``
   servers are running and reachable. First, review the `pre-requisites section <prerequisites>`_ and if that fails, see the
   `troubleshooting steps <timeouts>`_.

Compare the two arrays of measurement results. The results will be consistent between the qubits and random from shot
to shot.

``qc`` is a simulated quantum computer. We've told our QVM to run the program specified above ten times and return
the results to us.

The calls to ``compile`` and ``run`` will make a request to the two servers we started up in the previous section:
first, to the ``quilc`` server instance to compile the Quil program into native Quil, and then to the ``qvm`` server
instance to simulate and return measurement results of the program 10 times. If you open up the terminal windows where
your servers are running, you should see output printed to the console regarding the requests you just made.

.. note::

    pyQuil also provides the :py:func:`~pyquil.api.local_forest_runtime()` context manager to ensure both ``quilc`` and ``qvm`` servers are running
    by starting them as subprocesses if they aren't already.

    .. testcode:: first-program

        from pyquil import get_qc, Program
        from pyquil.gates import CNOT, Z, MEASURE
        from pyquil.api import local_forest_runtime
        from pyquil.quilbase import Declare

        prog = Program(
            Declare("ro", "BIT", 2),
            Z(0),
            CNOT(0, 1),
            MEASURE(0, ("ro", 0)),
            MEASURE(1, ("ro", 1)),
        ).wrap_in_numshots_loop(10)

        with local_forest_runtime():
            qvm = get_qc('9q-square-qvm')
            bitstrings = qvm.run(qvm.compile(prog)).readout_data.get("ro")

In the following sections, we'll cover gates, program construction & execution, and go into detail about our Quantum
Virtual Machine, our QPUs, noise models and more. Let's start with the :ref:`basics`.

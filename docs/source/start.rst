.. _start:

Installation and Getting Started
================================
To make full use of the Rigetti Forest SDK, you will need pyQuil, the QVM, and the Quil Compiler. On this page, we will
take you through the process of installing all three of these. We also step you through
:ref:`running a basic pyQuil program <exampleprogram>`.

.. note::

    If you're running from a Rigetti-provisioned JupyterLab IDE, installation has been completed for you. Continue to
    :ref:`exampleprogram`.

Upgrading or Installing pyQuil
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pyQuil is our library for generating and executing Quil programs on the Rigetti Forest platform.

Before you install, we recommend that you activate a Python 3.7+ virtual environment. Then, install pyQuil using
`pip <https://pip.pypa.io/en/stable/quickstart/>`_:

::

    pip install pyquil


For those of you that already have pyQuil, you can upgrade with:

::

    pip install --upgrade pyquil

If you would like to stay up to date with the latest changes and bug fixes, you can also opt to install pyQuil from the
source `here <https://github.com/rigetti/pyquil>`__.

.. note::

    pyQuil requires Python 3.7 or later.

.. _sdkinstall:

Downloading the QVM and Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Forest 3.0 Downloadable SDK Preview currently contains:

-  The Quantum Virtual Machine (``qvm``) which allows high-performance simulation of Quil programs
-  The Quil Compiler (``quilc``) which allows compilation and optimization of Quil programs to native gate sets

The QVM and the compiler are packed as program binaries that are accessed through the command line. Both of them provide
support for direct command-line interaction, as well as a server mode. The :ref:`server mode <server>` is required for use with pyQuil.

`Download the Forest SDK here <https://qcs.rigetti.com/sdk-downloads>`__, where you can find links
for Windows, macOS, Linux (.deb), Linux (.rpm), and Linux (bare-bones).

All installation mechanisms, except the bare-bones package, require administrative privileges to install. To use the QVM
and Quil Compiler from the bare-bones package, you will have to install the prerequisite dependencies on your own.

.. note::

   You can also find the open source code for `quilc <http://github.com/quil-lang/quilc>`__ and `qvm <http://github.com/quil-lang/qvm>`__
   on GitHub, where you can find instructions for compiling, installing, and contributing to the compiler and QVM.

Installing the QVM and Compiler on Windows
------------------------------------------
Download the Windows distribution by clicking on the appropriate link on the `SDK download page <https://qcs.rigetti.com/sdk-downloads>`__.
Open the file ``forest-sdk.msi`` by double clicking on it in your Downloads folder, and follow the system prompts.

Upon successful installation, one should be able to open a new terminal window and run the following two commands:

::

    qvm --version
    quilc --version

To uninstall the Forest SDK, search for "Add or remove programs" in the Windows search bar. Click on "Add or remove programs" and, in the resulting window, search for "Forest SDK for Windows" in the list of applications and click on "Uninstall" to remove it.

Installing the QVM and Compiler on macOS
----------------------------------------
Download the macOS distribution by clicking on the appropriate link on the `SDK download page <https://qcs.rigetti.com/sdk-downloads>`__.
Mount the file ``forest-sdk.dmg`` by double clicking on it in your Downloads folder. From there, open ``forest-sdk.pkg`` by
double-clicking on it. Follow the installation instructions.

Upon successful installation, one should be able to open a new terminal window and run the following two commands:

::

    qvm --version
    quilc --version

To uninstall, delete the following files:

::

    /usr/local/bin/qvm
    /usr/local/bin/quilc
    /usr/local/share/man/man1/qvm.1
    /usr/local/share/man/man1/quilc.1


Installing the QVM and Compiler on Linux (deb)
----------------------------------------------

Download the Debian distribution by clicking on the appropriate link on the `SDK download page <https://qcs.rigetti.com/sdk-downloads>`__. Unpack the tarball and change to that directory
by doing (filling in ``<version>`` appropriately):

::

    tar -xf forest-sdk-linux-deb.tar.bz2
    cd forest-sdk-<version>-linux-deb

From here, run the following command:

::

    sudo ./forest-sdk-<version>-linux-deb.run

Upon successful installation, one should be able to run the following
two commands:

::

    qvm --version
    quilc --version

To uninstall, type:

::

    sudo apt remove forest-sdk

Installing the QVM and Compiler on Linux (rpm)
----------------------------------------------

Download the RPM-based distribution by clicking on the appropriate link on the `SDK download page <https://qcs.rigetti.com/sdk-downloads>`__. Unpack the tarball and change to that
directory by doing (filling in ``<version>`` appropriately):

::

    tar -xf forest-sdk-linux-rpm.tar.bz2
    cd forest-sdk-<version>-linux-rpm

From here, run the following command:

::

    sudo ./forest-sdk-<version>-linux-rpm.run

Upon successful installation, one should be able to run the following two commands:

::

    qvm --version
    quilc --version

To uninstall, type:

::

    sudo rpm -e forest-sdk
    # or
    sudo yum uninstall forest-sdk

Installing the QVM and Compiler on Linux (bare-bones)
-----------------------------------------------------

The bare-bones installation only contains the executable binaries and
manual pages, and doesn't contain any of the requisite dynamic
libraries. As such, installation doesn't require administrative or
``sudo`` privileges.

First, unpack the tarball and change to that directory by doing (filling in ``<version>`` appropriately):

::

    tar -xf forest-sdk-linux-barebones.tar.bz2
    cd forest-sdk-<version>-linux-barebones

From here, run the following command:

::

    ./forest-sdk-<version>-linux-barebones.run

Upon successful installation, this will have created a new directory ``rigetti`` in your home directory that contains all
of the binary and documentation artifacts.

This method of installation requires one, through whatever means, to install shared libraries for BLAS, LAPACK, and
libffi. On a Debian-derivative system, this could be accomplished with

::

   sudo apt-get install liblapack-dev libblas-dev libffi-dev libzmq3-dev

Or on any rhel-derivative systems (e.g. Amazon Linux) with

::

   sudo yum install -y lapack-devel blas-devel epel-release
   sudo yum install -y zeromq3-devel

To uninstall, remove the directory ``~/rigetti``.

.. _exampleprogram:

Getting Started
~~~~~~~~~~~~~~~
To get started using the SDK, you can either interact with the QVM and the compiler directly from the command line,
or you can run them in server mode and use them with pyQuil. In this section, we're going to explain how to do the latter.

For more information about directly interacting with the QVM and the compiler, refer to their respective manual pages.
After :ref:`installation <sdkinstall>`, you can read the manual pages by opening a new terminal window and typing ``man qvm`` (for the QVM)
or ``man quilc`` (for the compiler). Quit out of the manual page by typing ``q``.

.. _server:

Setting Up Server Mode for pyQuil
---------------------------------

.. note::
    This set up is only necessary to run pyQuil locally. If you're running from your JupyterLab notebook, this has
    already been done for you.

It's easy to start up local servers for the QVM and quilc on your laptop. You should have two terminal windows open
to run in the background. We recommend using a resource such as ``tmux`` for running and managing multiple programs in one
terminal.

::

    ### CONSOLE 1
    $ qvm -S

    Welcome to the Rigetti QVM
    (Configured with 10240 MiB of workspace and 8 workers.)
    [2018-09-20 15:39:50] Starting server on port 5000.


    ### CONSOLE 2
    $ quilc -S

    ... - Launching quilc.
    ... - Spawning server at (tcp://*:5555) .


That's it! You're all set up to run pyQuil locally. Your programs will make requests to these server endpoints to compile your Quil
programs to native Quil, and to simulate those programs on the QVM.

**NOTE**: Prior to quilc version 1.10 there existed two methods for communicating with the quilc server: over HTTP by creating the server with the ``-S`` flag, or over RPCQ by creating the server with the ``-R`` flag. The HTTP server mode was deprecated in early 2019, and removed in mid 2019. The ``-S`` and ``-R`` flags now both start the RPCQ server.


Run Your First Program
----------------------
Now that our local endpoints are up and running, we can start running pyQuil programs!
We will run a simple program on the Quantum Virtual Machine (QVM).

The program we will create prepares a fully entangled state between two qubits, called a Bell State. This state is in an equal
superposition between :math:`\ket{00}` and :math:`\ket{11}`, meaning that it is equally likely that a measurement will result in measuring
both qubits in the ground state or both qubits in the excited state.

To begin, start up python however you like. You can open a jupyter notebook (type ``jupyter notebook`` in your terminal),
open an interactive python notebook in your terminal (with ``ipython3``), or simply launch python in your terminal
(type ``python3``). Recall that you need Python 3.6+ to use pyQuil.

Import a few things from pyQuil:

.. code:: python

    from pyquil import Program, get_qc
    from pyquil.gates import *
    from pyquil.quilbase import Declare

The :py:class:`~pyquil.Program` object allows us to build up a Quil program. :py:func:`~pyquil.get_qc` connects us to a
:py:class:`~pyquil.api.QuantumComputer` object, which specifies what our program should run on (see: :ref:`qvm`). We've also imported all (``*``)
gates from the ``pyquil.gates`` module, which allows us to add operations to our program (:ref:`basics`). :py:class:`~pyquil.quilbase.Declare`
allows us to declare classical memory regions so that we can receive data from the :py:class:`~pyquil.api.QuantumComputer`.

.. note::

    pyQuil also provides a handy function for you to ensure that a local qvm and quilc are currently running in
    your environment. To make sure both are available you execute ``from pyquil.api import local_forest_runtime`` and then use
    :py:func:`~pyquil.api.local_forest_runtime()`. This will start qvm and quilc instances using subprocesses if they have not already been started.
    You can also use it as a context manager as in the following example:

    .. code:: python

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

Next, let's construct our Bell State.

.. code:: python

    # construct a Bell State program
    p = Program(
        Declare("ro", "BIT", 2),
        H(0),
        CNOT(0, 1),
        MEASURE(0, ("ro", 0)),
        MEASURE(1, ("ro", 1)),
    ).wrap_in_numshots_loop(10)

We've accomplished this by driving qubit 0 into a superposition state (that's what the "H" gate does), and then creating
an entangled state between qubits 0 and 1 (that's what the "CNOT" gate does). Finally, we'll want to run our program:

.. code:: python

    # run the program on a QVM
    qc = get_qc('9q-square-qvm')
    result = qc.run(qc.compile(p)).readout_data.get("ro")
    print(result[0])
    print(result[1])

Compare the two arrays of measurement results. The results will be correlated between the qubits and random from shot
to shot.

The ``qc`` is a simulated quantum computer. We've told our QVM to run the program specified above ten times and return
the results to us.

The calls to ``compile`` and ``run`` will make a request to the two servers we started up in the previous section:
first, to the ``quilc`` server instance to compile the Quil program into native Quil, and then to the ``qvm`` server
instance to simulate and return measurement results of the program 10 times. If you open up the terminal windows where
your servers are running, you should see output printed to the console regarding the requests you just made.


In the following sections, we'll cover gates, program construction & execution, and go into detail about our Quantum
Virtual Machine, our QPUs, noise models and more. Let's start with the :ref:`basics`.


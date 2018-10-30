.. _start:

Installation and Getting Started
================================
To make full use of the Rigetti Forest SDK, you will need pyQuil, the QVM, and the Quil Compiler. On this page, we will
take you through the process of installing all three of these.

Upgrading or Installing pyQuil
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PyQuil 2.0 is our library for generating and executing Quil programs on the Rigetti Forest platform.

Before you install, we recommend that you activate a Python 3.6+ virtual environment. Then, install pyQuil using
`pip <https://pip.pypa.io/en/stable/quickstart/>`_:

::

    pip install --pre pyquil


For those of you that already have pyQuil, you can upgrade by typing

::

    pip install --upgrade --pre pyquil

If you would like to stay up to date with the latest changes and bug fixes, you can also opt to install pyQuil from the
source `here <https://github.com/rigetticomputing/pyquil>`__.

.. note::

    PyQuil requires Python 3.6 or later.

Downloading the QVM and Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Forest 2.0 Downloadable SDK Preview currently contains:

-  The Rigetti Quantum Virtual Machine (``qvm``) which allows high-performance simulation of Quil programs
-  The Rigetti Quil Compiler (``quilc``) which allows compilation and optimization of Quil programs to native gate sets

The QVM and the compiler are packed as program binaries that are accessed through the command line. Both of them provide
support for direct command-line interaction, as well as a server mode. The server mode is required for use with pyQuil
(see section :ref:`server` for more information).

Request the Forest SDK `here <http://rigetti.com/forest>`__. You'll receive an email right away with the download links
for macOS, Linux (.deb), Linux (.rpm), and Linux (bare-bones).

All installation mechanisms, except the bare-bones package, require administrative privileges to install. To use the QVM
and Quil Compiler from the bare-bones package, you will have to install the prerequisite dependencies on your own.

Installing on macOS
-------------------
Mount the file ``forest-sdk.dmg`` by double clicking on it in your email. From there, open ``forest-sdk.pkg`` by
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

Download the Debian distribution by clicking on the link in your email. Unpack the tarball and change to that directory
by doing:

::

    tar -xf forest-sdk-linux-deb.tar.bz2
    cd forest-sdk-2.0rc2-linux-deb

From here, run the following command:

::

    sudo ./forest-sdk-2.0rc2-linux-deb.run

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

Download the RPM-based distribution by clicking on the link in your email. Unpack the tarball and change to that
directory by doing:

::

    tar -xf forest-sdk-linux-rpm.tar.bz2
    cd forest-sdk-2.0rc2-linux-rpm

From here, run the following command:

::

    sudo ./forest-sdk-2.0rc2-linux-rpm.run

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

First, unpack the tarball and change to that directory by doing:

::

    tar -xf forest-sdk-linux-barebones.tar.bz2
    cd forest-sdk-2.0rc2-linux-barebones

From here, run the following command:

::

    ./forest-sdk-2.0rc2-linux-barebones.run

Upon successful installation, this will have created a new directory ``rigetti`` in your home directory that contains all
of the binary and documentation artifacts.

This method of installation requires one, through whatever means, to install shared libraries for BLAS, LAPACK, and
libffi. On a Debian-derivative system, this could be accomplished with

::

   sudo apt-get install liblapack-dev libblas-dev libffi-dev

To uninstall, remove the directory ``~/rigetti``.

Getting Started
~~~~~~~~~~~~~~~
To get started using the SDK, you can either interact with the QVM and the compiler directly from the command line,
or you can run them in server mode and use them with pyQuil.

Using the QVM and Compiler
--------------------------
Refer to the manual pages for information on how to use the QVM and compiler directly. After installation, you can read
the manual pages by opening a new terminal window and typing ``man qvm`` (for the QVM) or ``man quilc`` (for the
compiler). Quit out of the manual page by typing ``q``.

.. _server:

Setting Up Server Mode for PyQuil
---------------------------------
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

    Welcome to the Rigetti Quil Compiler
    [2018-09-19 11:22:37] Starting server: 0.0.0.0 : 6000.


You're all set up to run pyQuil locally, which will make requests to these server endpoints to compile your Quil
programs to native Quil, and to simulate those programs on the QVM.

Example Program
---------------
Now that our local endpoints are up and running, we can start running pyQuil programs! Open a jupyter notebook (type
``jupyter notebook`` in your terminal), or launch python in your terminal (type ``python3``).

Now that you're in python, we can import a few things from pyQuil.

.. code:: python

    from pyquil import Program, get_qc
    from pyquil.gates import *

We've imported the ``Program`` object, which allows us to specify a Quil program. ``get-qc`` allows us to connect to a
``QuantumComputer`` object, which allows us to specify what our program should run on. We've also imported all (``*``)
gates from the ``pyquil.gates`` module, which allows us to add operations to our program.

Let's construct a basic program. A Bell State, for example, is a simple entangled state, where two qubits are entangled
in a superposition state. We will construct a Bell state such that the two qubits will be in identical states when
measured.

.. code:: python

    # construct a Bell State program
    p = Program(H(0), CNOT(0, 1))

We've accomplished this by driving qubit 0 into a superposition state (that's what the "H" gate does), and then created
an entangled state between qubits 0 and 1 (that's what the "CNOT" gate does). Next, we'll want to run our program:

.. code:: python

    # run the program on a QVM
    qvm = get_qc('9q-square-qvm')
    result = qvm.run_and_measure(p, trials=10)
    print(result)

The QVM object above is a simulated quantum computer. It's what you can connect to, using the downloadable Forest SDK.
By specifying we want to ``.run_and_measure``, we've told our QVM to run the program specified above, and to collapse the
state with a measurement. A measurement will give us the state of the qubits. "trials" refers to the number of times we
run the program- a Bell State will give you both (0,0) and (1, 1); see how many times you get each output.

Our Forest SDK comes with a few parts:

1. **Quil**: The Quantum Instruction Language standard. Instructions written in Quil can be executed on any
implementation of a quantum abstract machine, such as the quantum virtual machine (QVM), or on a real quantum processing
unit (QPU). More details regarding Quil can be found in the `whitepaper <https://arxiv.org/abs/1608.03355>`__.

2. **pyQuil**: A Python library to help write and run Quil code and quantum programs.

3. **QVM**: A `Quantum Virtual Machine <qvm.html>`_, which is an implementation of the quantum abstract machine on
classical hardware. The QVM lets you use a regular computer to simulate a small quantum computer.

4. **Quil Compiler**: In addition to running on the QVM or the QPU, users can directly use the Rigetti Quil
compiler, to investigate how arbitrary quantum programs can be compiled to target specific physical instruction set
architectures (ISAs).

5. **QPU**: pyQuil also includes some a special connection which lets you run experiments on Rigetti's prototype
superconducting quantum processors over the cloud.


In the following sections, we'll cover gates, program construction & execution, and go into detail about our Quantum
Virtual Machine, our QPUs, noise models and more. Jump to :ref:`basics` to continue.


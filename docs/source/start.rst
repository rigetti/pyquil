.. _start:

Installation and Getting Started
================================

Download the Forest SDK `here <http://rigetti.com/forest>`_. The SDK will pre-package pyQuil v2.0/Quil 1.1, a compatible
downloadable QVM, and Quil Compiler. You'll need to download these resources before constructing and executing pyQuil
programs.

We recommend installing pyQuil using package manager pip.


.. code::

    `pip install pyquil`

will install pyQuil you can install requirements directly by typing

.. code::

    `pip install -r requirements.txt`

in your pyquil folder.

For those of you that already have pyQuil, you can upgrade by typing

.. code::

    `pip install --upgrade pyquil`

in your pyquil folder.

.. note::

    PyQuil requires Python 3.

Connecting to Rigetti Forest
----------------------------

The expected locations of the QVM and Compiler endpoints are
configurable in pyQuil. When running on a QMI, these configuration
values are automatically managed so as to point to the relevant
Rigetti-internal endpoints. When running locally, these default to
values reasonable for a user running local instances of the Rigetti
toolchain on their laptop. Ideally, little-to-no work will be required
for setting up this configuration environment locally or remotely, or
for transferring functioning code from one configured environment to
another.

.. note::
    A config file is not necessary to run locally, though it may be useful in configuring your local set-up.

In general, these values are read out of a pair of configuration files
(stored at the location described by the environment variables
``FOREST_CONFIG`` and ``QCS_CONFIG``, or else at the respective default
locations ``~/.forest_config`` and ``~/.qcs_config``), which by default
have the following respective contents:

::

    [Rigetti Forest]
    url = https://api.rigetti.com/
    key = None
    user_id = None

    [QPU]
    exec_on_engage = :

::

    [Rigetti Forest]
    qpu_endpoint_address = None
    qvm_address = http://localhost:5000
    compiler_server_address = http://localhost:6000

These values control the following behaviors:

-  ``Rigetti Forest``: This section contains network endpoint
   information about the entire Rigetti Forest infrastructure, e.g.,
   where to find information about which QPU devices are available.
-  ``url``: This is the endpoint where pyQuil looks for device
   information and for the 2.0 endpoints.
-  ``key``: This stores the pre-2.0 API key.
-  ``user_id``: This stores a 2.0 user ID.
-  ``qpu_endpoint_address``: This is the endpoint where pyQuil will try to
   communicate with the QPU orchestrating service during QPU-engagement.
-  ``qvm_address``: This is the endpoint where pyQuil will try to
   communicate with the Rigetti Quantum Virtual Machine. On a QMI, this
   points to the provided QVM instance. On a local installation, this
   should be set to the server endpoint for a locally running QVM
   instance.
-  ``compiler_server_address``: This is the endpoint where pyQuil will
   try to communicate with the compiler server. On a QMI, this points to
   a provided compiler server instance. On a local installation, this
   should be set to the server endpoint for a locally running quilc
   instance.
-  ``QPU``: This section contains configuration information pertaining
   to QPU access.
-  ``exec_on_engage``: This is the shell command that the QMI will
   launch when the QMI becomes QPU-engaged.

    **NOTE:** PyQuil itself reads these values out using the helper
    class ``pyquil._config.PyquilConfig``. PyQuil users should not ever
    need to touch this class directly.


Getting Started
---------------

This toolkit provides some simple libraries for writing quantum programs. Before we learn about pyQuil, let's try to run
something on the simulator.

First, initialize a localQVM instance on your laptop. You should have two consoles open in your terminal to run in the
background.

.. code:: python

    ### CONSOLE 1
    $ quilc -S
    port triggered: 6000.
    [2018-09-19 11:22:37] Starting server: 0.0.0.0 : 6000.

    ### CONSOLE 2
    $ qvm -S
    Welcome to the Rigetti QVM
    (Configured with 2048 MiB of workspace and 8 workers.)
    [2018-09-20 15:39:50] Starting server on port 5000.


Now that our local endpoints are up and running, we can start running pyQuil programs! Open a jupyter notebook (type
..code::`jupyter notebook` in your terminal), or launch python in your terminal (type ..code::`python3`).

Now that you're in python, we can import a few things from pyquil.

.. code:: python

    from pyquil import Program, get_qc
    from pyquil.gates import *

We've imported the Program object, which allows us to specify a pyquil program. `get-qc` allows us to connect to a
"quantum computer" object, which allows us to specify something for our program to run on. We've also imported all (*)
gates from the pyquil.gates module, which allows us to construct a program.

Let's construct a basic program. A Bell State, for example, is a simple entangled state, where two qubits are entangled
in a superposition state, such that they will be in identical states when measured.

.. code:: python

    # construct a Bell State program
    p = Program(H(0), CNOT(0, 1))

We've accomplished this by driving qubit 0 into a superposition state (that's what the "H" gate does), and then created
an entangled state between qubits 0 and 1 (that's what the "CNOT" gate does). Next, we'll want to run our program:

.. code:: python

    # run the program on a QVM
    qvm = get_qc('9q-generic-qvm')
    result = qvm.run_and_measure(p, trials=10)
    print(result)

The QVM object above is a simulated quantum computer. It's what you can connect to, using the downloadable Forest SDK.
By specifying we want to `.run_and_measure`, we've told our QVM to run the program specified above, and to collapse the
state with a measurement. A measurement will give us the state of the qubits. "trials" refers to the number of times we
run the program- a Bell State will give you both (0,0) and (1, 1); see how many times you get each output.


Our Forest SDK comes with a few parts:

1. **Quil**: The Quantum Instruction Language standard. Instructions written in Quil can be executed on any
implementation of a quantum abstract machine, such as the quantum virtual machine (QVM), or on a real quantum processing
unit (QPU). More details regarding Quil can be found in the `whitepaper <https://arxiv.org/abs/1608.03355>`__.

2. **pyQuil**: A Python library to help write and run Quil code and quantum programs.

3. **QVM**: A `Quantum Virtual Machine <qvm.html>`_, which is an implementation of the quantum abstract machine on
classical hardware. The QVM lets you use a regular computer to simulate a small quantum computer.

4. **Quilc**: In addition to running on the QVM or the QPU, users can directly use the Quil compiler, to investigate how
arbitrary quantum programs can be compiled to target specific physical instruction set architectures (ISAs).

5. **QPU**: pyQuil also includes some a special connection which lets you run experiments on Rigetti's prototype
superconducting quantum processors over the cloud.


In the following sections, we'll cover gates, program construction & execution, and go into detail about our Quantum
Virtual Machine, our QPUs, noise models and more.
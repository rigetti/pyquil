.. _start:

Installation and Getting Started
================================

Downloads
---------

Install pyQuil using `pip <https://pip.pypa.io/en/stable/quickstart/>`_:


.. code::

    pip install --pre pyquil


For those of you that already have pyQuil, you can upgrade by typing

.. code::

    pip install --upgrade --pre pyquil

.. note::

    PyQuil requires Python 3.6 or later.


Installing the QVM and Compiler on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the Forest SDK `here <http://rigetti.com/forest>`_. The Forest Downloadable SDK 2.0 Preview currently contains:

-  The Rigetti Quantum Virtual Machine (``qvm``) which allows high-performance simulation of Quil programs
-  The Rigetti Quil Compiler (``quilc``) which allows compilation and optimization of Quil programs to native gate sets

The QVM and the compiler are packed as program binaries that are accessed through the command line. Both of them provide
support for direct command-line interaction, as well as a server mode. The server mode is required for use with pyQuil.

All installation mechanisms, except the bare-bones package, require administrative privileges to install. The bare-bones
package just contains binaries and documentation without any of the prerequisite dependencies.

Mount the file ``forest-sdk.dmg`` by double clicking on it. From there, open ``forest-sdk.pkg`` by double-clicking on it.
Follow the installation instructions.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, unpack the tarball and change to that directory by doing:

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, unpack the tarball and change to that directory by doing:

::

    tar -xf forest-sdk-linux-rpm.tar.bz2
    cd forest-sdk-2.0rc2-linux-rpm

From here, run the following command:

::

    sudo ./forest-sdk-2.0rc2-linux-rpm.run

Upon successful installation, one should be able to run the following
two commands:

::

    qvm --version
    quilc --version

To uninstall, type:

::

    sudo rpm -e forest-sdk
    # or
    sudo yum uninstall forest-sdk

Installing the QVM and Compiler on Linux (bare-bones)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Using the QVM and Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~
The QVM and the Quil compiler may either be used through pyQuil, or directly from the command line. Refer to the manual pages for
each of these for information on how to use them directly. After installation, one can read the manual pages by opening a new terminal window and
typing ``man qvm`` (for the QVM) or ``man quilc`` (for the compiler). One can quit out of the manual page by typing ``q``.

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
``jupyter notebook`` in your terminal), or launch python in your terminal (type ``python3``).

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
    qvm = get_qc('9q-square-qvm')
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

4. **Quil Compiler**: In addition to running on the QVM or the QPU, users can directly use the Rigetti Quil
compiler, to investigate how arbitrary quantum programs can be compiled to target specific physical instruction set
architectures (ISAs).

5. **QPU**: pyQuil also includes some a special connection which lets you run experiments on Rigetti's prototype
superconducting quantum processors over the cloud.


In the following sections, we'll cover gates, program construction & execution, and go into detail about our Quantum
Virtual Machine, our QPUs, noise models and more. Jump to :ref:`basics` to continue.


.. _quickstart:

Forest 2.0: Quick-Start & Migration Guide
-----------------------------------------

The goals of this guide are to cover changes to the Forest SDK (containing pyquil 2.0, new Quil, Quil Compiler, and QVM),
and to go through an example of migrating a VQE program from Forest 1.3 (pyQuil 1.9, Quil 1.0) to be able to run on the
new Forest SDK.


.. note::

    For installation & setup, follow the download instructions in the section :ref:`start` at the top of the page.

What's changed
--------------

With the new Forest SDK, users will be able to run pyQuil programs on a downloadable QVM and Quil Compiler!

In the following section, we'll cover the main changes to pyQuil, Quil, the Quil Compiler, and the QVM.

Overview of Updates to Quil and pyQuil
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary differences in the programming language Quil 1.0 (as appearing in pyQuil 1.3) and Quil 2 (as appearing in
2.0) amount to an enhanced memory model. Whereas the classical memory model in Quil 1.0 amounted to an flat bit array of
indefinite size, the memory model in  Quil 2 is segmented into typed, sized, named regions.

In terms of compatibility with Quil 1.0, this primarily changes how ``MEASURE`` instructions are formulated, since their
classical address targets must be modified to fit the new framework. In terms of new functionality, this allows angle
values to be read in from classical memory.

Quil 2 also introduces easier ways to manipulate gates by using gate modifiers. Two gate modifiers are supported currently,
`DAGGER` and `CONTROLLED`.

`DAGGER` can be written before a gate to refer to its inverse. For instance

.. code::

    DAGGER RX(pi/3) 0

would have the same effect as

.. code::

    RX(-pi/3) 0

`DAGGER` can be applied to any gate, but also circuits defined with `DEFCIRCUIT`. This allows for easy reversal of unitary circuits:

.. code::

    DEFCIRCUIT BELL:
        H 0
        CNOT 0 1

    # construct a Bell state
    BELL
    # disentangle, bringing us back to identity
    DAGGER BELL

Parametric programs
~~~~~~~~~~~~~~~~~~~

The main benefit for users of declared memory regions in Quil is that angle values for parametric gates can be loaded at
execution time on the QPU. Consider the following simple QAOA instance:

::

    DECLARE ro BIT[2]
    DECLARE beta REAL
    DECLARE gamma REAL

    H 0
    RZ(beta) 0
    H 0
    H 1
    RZ(beta) 1
    H 1

    CNOT 0 1
    RZ(gamma) 1
    CNOT 0 1

    MEASURE 0 ro[0]
    MEASURE 1 ro[1]

To generate a "landscape" plot as ``beta`` and ``gamma`` range, it was previously required to generate a different
program for each possible pair of values, substitute that pair in, send it to the compiler, and send the resulting
compiled program to the QPU for execution (and hence generate the expectation values). With Quil 2, this exact program
can be sent to the compiler, which returns a nativized Quil program that still has parametric gates with parameters
referencing the classical memory regions ``beta`` and ``gamma``. This program can then be loaded onto the QPU for
repeated execution with different values of ``beta`` and ``gamma``, without recompilation in between.

Details of updates to Quil
~~~~~~~~~~~~~~~~~~~~~~~~~~

Classical memory regions must be explicitly requested and named by a Quil program using ``DECLARE`` directive. A generic
``DECLARE`` directive has the following syntax:

``DECLARE region-name type([count])? (SHARING parent-region-name (OFFSET (offset-count offset-type)+))?``

The non-keyword items have the following allowable values:

-  ``region-name``: any non-keyword formal name.

-  ``type``: one of ``REAL``, ``BIT``, ``OCTET``, or ``INTEGER``

-  ``parent-region-name``: any non-keyword formal name previously used as ``region-name`` in a different ``DECLARE`` statement.

-  ``offset-count``: a nonnegative integer.

-  ``offset-type``: the same allowable values as ``type``.

Here are some examples:

::

    DECLARE beta REAL[32]
    DECLARE ro BIT[128]
    DECLARE beta-bits BIT[1436] SHARING beta
    DECLARE fourth-bit-in-beta1 BIT SHARING beta OFFSET 1 REAL 4 BIT

In order, the intention of these ``DECLARE`` statements is:

-  Allocate an array called ``beta`` of length 32, each entry of which is a ``REAL`` number.

-  Allocate an array called ``ro`` of length 128, each entry of which is a ``BIT``.

-  Name an array called ``beta-bits``, which is an overlay onto the existing array ``beta``, so that the bit representations of elements of ``beta`` can be directly examined and manipulated.

-  Name a single ``BIT`` called ``fourth-bit-in-beta1`` which overlays the fourth bit of the bit representation of the ``REAL`` value ``beta[1]``.


Backwards compatibility
~~~~~~~~~~~~~~~~~~~~~~~

Quil 1.0 is not compatible with Quil 2 in the following ways:

-  The unnamed memory references ``[n]`` and ``[n-m]`` have no direct equivalent in Quil 2 and must be replaced by named
    memory references. (This primarily affects ``MEASURE`` instructions.)

-  The classical memory manipulation instructions have been modified: the operands of ``AND`` have been reversed (so that
    in Quil 2, the left operand is the target address) and ``OR`` has been replaced by ``IOR`` and its operands reversed (so
    that, again, in Quil 2 the left operand is the target address).

In all other instances, Quil 1.0 will operate identically with Quil 2.

When confronted with program text conforming to Quil 1.0, pyQuil 2.0 will automatically rewrite ``MEASURE q [n]`` to
``MEASURE q ro[n]`` and insert a ``DECLARE`` statement which allocates a ``BIT``-array of the appropriate size named
``ro``.

Details of pyQuil and Forest updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Updates to Forest**

-  In Forest 1.3, job submission to the QPU was done from your workstation and the ability was gated by on user ID. In
    Forest 2.0, job submission to the QPU must be done from your remote virtual machine, called a QMI (*Quantum Machine Image*).

-  In Forest 1.3, user data persisted indefinitely in cloud storage and could be accessed using the assigned job ID. In
    Forest 2.0, user data is stored only transiently, and it is the user's responsibility to handle long-term data storage
    on their QMI.


**Updates to pyQuil**

-  In pyQuil 1.9, API calls were organized by endpoint (e.g., all simulation calls were passed to a ``QVMConnection``
    object). In pyQuil 2.0, API calls are organized by type (e.g., ``run`` calls are sent to a ``QuantumComputer`` but
    ``wavefunction`` calls are sent to a ``WavefunctionSimulator``).

-  In pyQuil 1.9, quantum program evaluation was asynchronous on the QPU and a mix of synchronuous or asynchronous on
    the QVM. In pyQuil 2.0, all quantum program evaluation is synchronous.

-  In pyQuil 1.9, each quantum program execution call started from scratch. In pyQuil 2.0, compiled program objects can be reused.

Backwards compatibility and migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyQuil 2.0 is not backwards compatible with pyQuil 1.9. However, the new API objects available in pyQuil 2.0 have
compatibility methods that make migration to pyQuil 2.0 easier.

.. note::
    Users writing new programs from scratch are encouraged to use the bare pyQuil 2.0 programming model over the
    compatibility methods. It is not possible to use the fanciest new features of Forest 2.0 (e.g., parametric execution
    of parametric programs) from within the compatibility model.

Whereas pyQuil 1.9 organized API calls around "connection objects" (viz., ``CompilerConnection``, ``QPUConnection``, and
``QVMConnection``), pyQuil 2.0 organizes API calls around function, so that QVM- and QPU-based objects can be more
easily swapped. These API objects fall into two groups:

-  ``QuantumComputer``: This wrapper object houses the typical ingredients for execution of a hybrid classical-quantum algorithm: an interface to a compiler, an interface to a quantum computational device, and some optional wrapper routines. ``QuantumComputer`` objects themselves can be manually initialized with these ingredients, or they can be requested by name from the Forest 2.0 service, which will populate these subfields with the appropriate objects for execution on a particular quantum device, real or simulated.

-  ``AbstractCompiler``: An interface to a compiler service. Compilers are responsible for two tasks: converting arbitrary Quil to "native" (or "device-specific") Quil, and converting native Quil to control system binaries.

-  ``QAM``: An interface to a quantum computational device. This can be populated by a connection to an actual QPU, or it can be populated by a connection to a QVM (**Quantum Virtual Machine**).

-  *Wrapper routines*: Execution of programs in pyQuil 1.9 was typically done with a single API call (e.g., ``.run()``). ``QuantumComputer`` exposes a near-identical interface for single runs of quantum programs, which wraps and hides the more low-level pyQuil 2.0 infrastructure.

-  ``WavefunctionSimulator``: This wrapper object houses the typical ingredients used for the debug process of wavefunction inspection. This is inherently **not** a procedure natively available on a quantum computational device, and so this wrapper either calls out to a QVM or functions as a repeated sampling wrapper from a physical quantum computational device.


Example: Computing the bond energy of molecular hydrogen, pyQuil 1.9 vs 2.0
---------------------------------------------------------------------------

By way of example, let's consider the following pyQuil 1.9 program,
which computes the natural bond distance in molecular hydrogen using a
VQE-type algorithm:

.. code:: python

    from pyquil.api import QVMConnection
    from pyquil.quil import Program


    def setup_forest_objects():
        qvm = QVMConnection()
        return qvm


    def build_wf_ansatz_prep(theta):
        program = Program(f"""
    # set up initial state
    X 0
    X 1

    # build the exponentiated operator
    RX(pi/2) 0
    H 1
    H 2
    H 3

    CNOT 0 1
    CNOT 1 2
    CNOT 2 3
    RZ({theta}) 3
    CNOT 2 3
    CNOT 1 2
    CNOT 0 1

    RX(-pi/2) 0
    H 1
    H 2
    H 3

    # measure out the results
    MEASURE 0 [0]
    MEASURE 1 [1]
    MEASURE 2 [2]
    MEASURE 3 [3]""")
        return program


    # some constants
    bond_step, bond_min, bond_max = 0.05, 0, 200
    angle_step, angle_min, angle_max = 0.1, 0, 63
    convolution_coefficients = [0.1698845197777728, 0.16988451977777283, -0.2188630663199042,
                                -0.2188630663199042]
    shots = 1000

    # set up the Forest object
    qvm = setup_forest_objects()

    # get all the unweighted expectations for all the sample wavefunctions
    occupations = list(range(angle_min, angle_max))
    indices = list(range(4))
    for offset in occupations:
        # set up the Program object, each time we have a new parameter
        program = build_wf_ansatz_prep(angle_min + offset * angle_step)
        bitstrings = qvm.run(program, indices, trials=shots)

        totals = [0, 0, 0, 0]
        for bitstring in bitstrings:
            for index in indices:
                totals[index] += bitstring[index]
        occupations[offset] = [t / shots for t in totals]

    # compute minimum energy as a function of bond length
    min_energies = list(range(bond_min, bond_max))
    for bond_length in min_energies:
        energies = []
        for offset in range(angle_min, angle_max):
            energy = 0
            for j in range(4):
                energy += occupations[offset][j] * convolution_coefficients[j]
            energies.append(energy)

        min_energies[bond_length] = min(energies)

    min_index = min_energies.index(min(min_energies))
    min_energy, relaxed_length = min_energies[min_index], min_index * bond_step

In order to port this code to pyQuil 2.0, we need change only one thing: the part referencing ``QVMConnection`` should be replaced by an equivalent part referencing a ``QuantumComputer`` connected to a ``QVM``. Specifically, the following
snippet

.. code:: python

    from pyquil.api import QVMConnection

    def setup_forest_objects():
        qvm = QVMConnection()
        return qvm

can be changed to

.. code:: python

    from pyquil.api import get_qc

    def setup_forest_objects():
        qc = get_qc("9q-square-qvm")
        return qc

and the references to ``qvm`` in the main body are changed to ``qc`` instead. Since the ``QuantumComputer`` object also
exposes a ``run`` routine and pyQuil itself automatically rewrites 1.9-style ``MEASURE`` instructions into 2.0-style
instructions, this is all we need to do.

If we are willing to be more intrusive, we can also take advantage of pyQuil 2.0's classical memory and parametric
programs. The first piece to change is the Quil program itself: we remove the argument ``theta`` from the Python
function ``build_wf_ansatz_prep``, with the intention of letting the QPU fill it in later. In turn, we modify the Quil
program itself to have a ``REAL`` memory parameter named ``theta``. We also declare a few ``BIT``\ s for our ``MEASURE``
instructions to target.

.. code:: python

    def build_wf_ansatz_prep():
        program = Program("""
    # set up memory
    DECLARE ro BIT[4]
    DECLARE theta REAL

    # set up initial state
    X 0
    X 1

    # build the exponentiated operator
    RX(pi/2) 0
    H 1
    H 2
    H 3

    CNOT 0 1
    CNOT 1 2
    CNOT 2 3
    RZ(theta) 3
    CNOT 2 3
    CNOT 1 2
    CNOT 0 1

    RX(-pi/2) 0
    H 1
    H 2
    H 3

    # measure out the results
    MEASURE 0 ro[0]
    MEASURE 1 ro[1]
    MEASURE 2 ro[2]
    MEASURE 3 ro[3]""")
        return program

Next, we modify the execution loop. Rather than reformulating the ``Program`` object each time, we build and compile it
once, then use the ``.load()`` method to transfer the parametric program to the (simulated) quantum device. We then set
only the angle value within the inner loop, and we change to using ``.run()`` and ``.wait()`` methods to manage control
between us and the quantum device.

More specifically, the old execution loop

.. code:: python

    # get all the unweighted expectations for all the sample wavefunctions
    occupations = list(range(angle_min, angle_max))
    indices = list(range(4))
    for offset in occupations:
        # set up the Program object, each time we have a new parameter
        program = build_wf_ansatz_prep(angle_min + offset * angle_step)
        bitstrings = qvm.run(program, indices, trials=shots)

        totals = [0, 0, 0, 0]
        for bitstring in bitstrings:
            for index in indices:
                totals[index] += bitstring[index]
        occupations[offset] = [t / shots for t in totals]

becomes

.. code:: python

    # set up the Program object, ONLY ONCE
    program = build_wf_ansatz_prep()
    program.wrap_in_numshots_loop(shots=shots)
    nq_program = qc.compiler.quil_to_native_quil(program)
    binary = qc.compiler.native_quil_to_executable(nq_program)
    qc.qam.load(binary)

    # get all the unweighted expectations for all the sample wavefunctions
    occupations = list(range(angle_min, angle_max))
    indices = list(range(4))
    for offset in occupations:
        qc.qam.write_memory(region_name='theta', value=angle_min + offset * angle_step)
        qc.qam.run()
        qc.qam.wait()
        bitstrings = qc.qam.read_from_memory_region(region_name="ro", offsets=True)

        totals = [0, 0, 0, 0]
        for bitstring in bitstrings:
            for index in indices:
                totals[index] += bitstring[index]
        occupations[offset] = [t / shots for t in totals]

Overall, the resulting program looks like this:

.. code:: python

    from pyquil.api import get_qc
    from pyquil.quil import Program


    def setup_forest_objects():
        qc = get_qc("9q-square-qvm")
        return qc


    def build_wf_ansatz_prep():
        program = Program("""
    # set up memory
    DECLARE ro BIT[4]
    DECLARE theta REAL

    # set up initial state
    X 0
    X 1

    # build the exponentiated operator
    RX(pi/2) 0
    H 1
    H 2
    H 3

    CNOT 0 1
    CNOT 1 2
    CNOT 2 3
    RZ(theta) 3
    CNOT 2 3
    CNOT 1 2
    CNOT 0 1

    RX(-pi/2) 0
    H 1
    H 2
    H 3

    # measure out the results
    MEASURE 0 ro[0]
    MEASURE 1 ro[1]
    MEASURE 2 ro[2]
    MEASURE 3 ro[3]""")
        return program


    # some constants
    bond_step, bond_min, bond_max = 0.05, 0, 200
    angle_step, angle_min, angle_max = 0.1, 0, 63
    convolution_coefficients = [0.1698845197777728, 0.16988451977777283, -0.2188630663199042,
                                -0.2188630663199042]
    shots = 1000

    # set up the Forest object
    qc = setup_forest_objects()

    # set up the Program object, ONLY ONCE
    program = build_wf_ansatz_prep()
    program.wrap_in_numshots_loop(shots=shots)
    nq_program = qc.compiler.quil_to_native_quil(program)
    binary = qc.compiler.native_quil_to_executable(nq_program)
    qc.qam.load(binary)

    # get all the unweighted expectations for all the sample wavefunctions
    occupations = list(range(angle_min, angle_max))
    indices = list(range(4))
    for offset in occupations:
        qc.qam.write_memory(region_name='theta', value=angle_min + offset * angle_step)
        qc.qam.run()
        qc.qam.wait()
        bitstrings = qc.qam.read_from_memory_region(region_name="ro", offsets=True)

        totals = [0, 0, 0, 0]
        for bitstring in bitstrings:
            for index in indices:
                totals[index] += bitstring[index]
        occupations[offset] = [t / shots for t in totals]

    # compute minimum energy as a function of bond length
    min_energies = list(range(bond_min, bond_max))
    for bond_length in min_energies:
        energies = []
        for offset in range(angle_min, angle_max):
            energy = 0
            for j in range(4):
                energy += occupations[offset][j] * convolution_coefficients[j]
            energies.append(energy)

        min_energies[bond_length] = min(energies)

    min_index = min_energies.index(min(min_energies))
    min_energy, relaxed_length = min_energies[min_index], min_index * bond_step


Miscellanea
-----------

Quil promises that a BIT is 1 bit and that an OCTET is 8 bits. Quil does not promise, however, the size or layout of
INTEGER or REAL. These are implementation-dependent.

On the QPU, ``INTEGER`` refers to an unsigned integer stored in a 48-bit wide little-endian word, and ``REAL`` refers to
a 48-bit wide little-endian fixed point number of type <0.48>. In general, these datatypes are implementation-dependent.
``OCTET`` always refers to an 8-bit wide unsigned integer and is independent of implementation.

Memory regions are all "global": ``DECLARE`` directives cannot appear in the body of a ``DEFCIRCUIT``.

On the QVM, INTEGER is a two's complement signed 64-bit integer. REAL is an IEEE-754 double-precision floating-point number.


Error reporting
~~~~~~~~~~~~~~~

Because the Forest 2.0 execution model is no longer asynchronous, our error reporting model has also changed. Rather
than writing to technical support with a job ID, users will need to provide all pertinent details to how they produced an
error.

PyQuil 2.0 makes this task easy with the function decorator ``@pyquil_protect``, found in the module
``pyquil.api``. By decorating a failing function (or a function that has the potential to fail), any
unhandled exceptions will cause an error log to be written to disk (at a user-specifiable location). For example, the
nonsense code block

::

    from pyquil.api import pyquil_protect

    ...

    @pyquil_protect
    def my_function():
        ...
        qc.qam.load(qc)
        ...

    my_function()

causes the following error to be printed:

::

    >>> PYQUIL_PROTECT <<<
    An uncaught exception was raised in a function wrapped in pyquil_protect.  We are writing out a
    log file to "/Users/your_name/Documents/pyquil/pyquil_error.log".

    Along with a description of what you were doing when the error occurred, send this file to Rigetti Computing
    support by email at support@rigetti.com for assistance.
    >>> PYQUIL_PROTECT <<<

as well as the following log file to be written to disk at the indicated
location:

::

    {
      "stack_trace": [
        {
          "name": "pyquil_protect_wrapper",
          "filename": "/Users/your_name/Documents/pyquil/pyquil/error_reporting.py",
          "line_number": 197,
          "locals": {
            "e": "TypeError('quil_binary argument must be a QVMExecutableResponse. This error is typically triggered by
                forgetting to pass (nativized) Quil to native_quil_to_executable or by using a compiler meant to be used
                for jobs bound for a QPU.',)",
            "old_filename": "'pyquil_error.log'",
            "kwargs": "{}",
            "args": "()",
            "log_filename": "'pyquil_error.log'",
            "func": "<function my_function at 0x106dc4510>"
          }
        },
        {
          "name": "my_function",
          "filename": "<stdin>",
          "line_number": 10,
          "locals": {
            "offset": "0",
            "occupations": "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]"
          }
        },
        {
          "name": "wrapper",
          "filename": "/Users/your_name/Documents/pyquil/pyquil/error_reporting.py",
          "line_number": 228,
          "locals": {
            "pre_entry": "CallLogValue(timestamp_in=datetime.datetime(2018, 9, 11, 18, 40, 19, 65538),
                timestamp_out=None, return_value=None)",
            "key": "run('<pyquil.api._qvm.QVM object at 0x1027e3940>', )",
            "kwargs": "{}",
            "args": "(<pyquil.api._qvm.QVM object at 0x1027e3940>,)",
            "func": "<function QVM.run at 0x106db4e18>"
          }
        },
        {
          "name": "run",
          "filename": "/Users/your_name/Documents/pyquil/pyquil/api/_qvm.py",
          "line_number": 376,
          "locals": {
            "self": "<pyquil.api._qvm.QVM object at 0x1027e3940>",
            "__class__": "<class 'pyquil.api._qvm.QVM'>"
          }
        }
      ],
      "timestamp": "2018-09-11T18:40:19.253286",
      "call_log": {
        "__init__('<pyquil.api._qvm.QVM object at 0x1027e3940>', '<pyquil.api._base_connection.ForestConnection object at
            0x1027e3588>', )": {
          "timestamp_in": "2018-09-11T18:40:18.967750",
          "timestamp_out": "2018-09-11T18:40:18.968170",
          "return_value": "None"
        },
        "run('<pyquil.api._qvm.QVM object at 0x1027e3940>', )": {
          "timestamp_in": "2018-09-11T18:40:19.065538",
          "timestamp_out": null,
          "return_value": null
        }
      },
      "exception": "TypeError('quil_binary argument must be a QVMExecutableResponse. This error is typically triggered
        by forgetting to pass (nativized) Quil to native_quil_to_executable or by using a compiler meant to be used for
        jobs bound for a QPU.',)",
      "system_info": {
        "python_version": "3.6.3 (default, Jan 25 2018, 13:55:02) \n[GCC 4.2.1 Compatible Apple LLVM 9.0.0
            (clang-900.0.39.2)]",
        "pyquil_version": "2.0.0-internal.1"
      }
    }

Please attach such a logfile to any request for support.


QPU-allowable Quil: "ProtoQuil"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apart from ``DECLARE`` and ``PRAGMA`` directives, a program must break into the following three regions, each optional:

1. A ``RESET`` command.
2. A sequence of quantum gate applications.
3. A sequence of ``MEASURE`` commands.

The only memory that is writeable is the region named ``ro``, and only through ``MEASURE`` instructions. All other
memory is read-only.

The keyword ``SHARING`` is disallowed.

Compilation is unavailable for invocations of ``DEFGATE``\ s with parameters read from classical memory.


QVM
~~~

.. note::

    The QVM uses a legacy HTTP interface, which will be replaced by a ``pidgin`` interface in a future release.
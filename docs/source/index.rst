.. pyQuil documentation master file, created by
   sphinx-quickstart on Mon Jun 13 17:59:05 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: red

Welcome to the Docs for the Forest SDK!
=======================================

The Rigetti Forest `Software Development Kit <http://rigetti.com/forest>`_ includes pyQuil, our Quil Compiler (quilc),
and our QVM.

**Longtime users of Rigetti Forest will notice a few changes.** First, the SDK now contains a downloadable compiler and a
QVM. Second, the SDK contains pyQuil 2.0, with significant updates to previous versions. As a result, programs written
using previous versions of the Forest toolkit will need to be updated to pyQuil 2.0 to be compatible with the QVM or compiler.

Scroll down to the :ref:`quickstart` to get caught up on what's new!

Quantum Cloud Services will provide users with a dedicated Quantum Machine Image, which will come prepackaged with the
Forest SDK. We’re releasing a Preview to the Forest SDK now, so current users can begin migrating code (and share feedback
with us early and often!). Longtime Forest users should start with the Migration Guide which outlines key changes in this
SDK Preview release.

If you’re new to Forest, we hope this documentation will provide everything you need to get up and running with the toolkit.
Once you’ve oriented yourself here, proceed to the section :ref:`start` to get started. Once you’ve installed the SDK,
take a look at our section on :ref:`intro`. There, you’ll learn the basic concepts needed to write quantum software. You
can also work through an introduction to quantum computing in a jupyter notebook; launch the notebook from the source folder
in pyquil's docs:

.. code::

    cd pyquil/docs/source
    jupyter notebook intro_to_qc.ipynb


**A few terms to orient you as you get started with Forest:**

- pyQuil is an open source Python library developed at Rigetti Computing that allows you to write programs for quantum computers.
  The source is hosted on `github <http://github.com/rigetticomputing/pyquil>`_.
- Quil, the Quantum Instruction Language, is the lower-level code that pyQuil gets compiled into. A full description of
  Quil can be found in our whitepaper, `A Practical Quantum Instruction Set Architecture <https://arxiv.org/abs/1608.03355>`_.
- quilc is the Quil Compiler that compiles pyQuil into Quil. The SDK includes quilc, which will enable you to compile your
  pyQuil programs into executable Quil code.
- The QVM is a simulator of our quantum computers. When you download the SDK, you’ll install the QVM and you will execute
  Quil programs against it.
- Forest is our software development kit, optimized for near-term quantum computers that operate as coprocessors, working in
  concert with traditional processors to run hybrid quantum-classical algorithms. For references on problems addressable
  with near-term quantum computers, see `Quantum Computing in the NISQ era and beyond <https://arxiv.org/abs/1801.00862>`_.

Our flagship product, `Quantum Cloud Services <http://rigetti.com/qcs>`_ offers users an on-premise, dedicated access
point to our quantum computers, and to a powerful 34-qubit Quantum Virtual Machine. This access point sits in a Virtual
Machine, which we call a Quantum Machine Image. A QMI is bundled with the same downloadable SDK mentioned above, and a
Command Line Interface (CLI), which is used for scheduling compute time on our quantum computers. To sign up for our
waitlist, please click the link above. If need access to our quantum computers for research, please email support@rigetti.com.


Contents
--------

.. toctree::
   :maxdepth: 3

   start
   basics
   advanced_usage
   exercises
   qvm
   compiler
   qubit-placeholder
   noise
   modules
   changes
   intro


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. _quickstart:


Forest 2.0: Quick-Start & Migration Guide
=========================================

In this section, we'll go over how to get set up, what's changed, and go through an example migration of a VQE program
from Forest 1.3 (pyQuil 1.9, Quil 1.0) to be able to run on the new Forest SDK (pyQuil 2.0, Quil 2).


Registration, Installation & Setup
----------------------------------
Download the Forest SDK `here <http://rigetti.com/forest>`_. The SDK will pre-package pyQuil v2.0/Quil 2, a compatible
downloadable QVM, and Quil Compiler.

You can also install pyQuil using package manager pip.  ``pip install --pre pyquil`` will install pyQuil; you can
install requirements directly by typing ``pip install -r requirements.txt`` in your pyquil folder.

For those of you that already have pyQuil, you can upgrade by typing ``pip install --upgrade --pre pyquil`` in your
pyquil folder.

.. note::

    pyQuil requires Python 3.6 or later.


What's changed
--------------

Local development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected locations of the QVM and Compiler endpoints are configurable in pyQuil. When running on a QMI, these
configuration values are automatically managed so as to point to the relevant Rigetti-internal endpoints.

When running locally, these default to values reasonable for a user running local instances of the Rigetti toolchain on
their laptop. Ideally, little-to-no work will be required for setting up this configuration environment locally or
remotely, or for transferring functioning code from one configured environment to another.

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


Support
~~~~~~~

For support issues, please email ``support@rigetti.com``.


Overview of updates to Quil and pyQuil
--------------------------------------

The primary differences in the programming language Quil 1.0 (as appearing in pyQuil 1.3) and Quil 2 (as appearing in
2.0) amount to an enhanced memory model. Whereas the classical memory model in Quil 1.0 amounted to an flat bit array of
indefinite size, the memory model in  Quil 2 is segmented into typed, sized, named regions.

In terms of compatibility with Quil 1.0, this primarily changes how ``MEASURE`` instructions are formulated, since their
classical address targets must be modified to fit the new framework. In terms of new functionality, this allows angle
values to be read in from classical memory.

Parametric programs
^^^^^^^^^^^^^^^^^^^

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
--------------------------

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

-  The unnamed memory references ``[n]`` and ``[n-m]`` have no direct equivalent in Quil 2 and must be replaced by named memory references. (This primarily affects ``MEASURE`` instructions.)

-  The classical memory manipulation instructions have been modified: the operands of ``AND`` have been reversed (so that
    in Quil 2, the left operand is the target address) and ``OR`` has been replaced by ``IOR`` and its operands reversed (so
    that, again, in Quil 2 the left operand is the target address).

In all other instances, Quil 1.0 will operate identically with Quil 2.

When confronted with program text conforming to Quil 1.0, pyQuil 2.0 will automatically rewrite ``MEASURE q [n]`` to
``MEASURE q ro[n]`` and insert a ``DECLARE`` statement which allocates a ``BIT``-array of the appropriate size named
``ro``.


Details of pyQuil and Forest updates
------------------------------------

Updates to Forest
~~~~~~~~~~~~~~~~~

-  In Forest 1.3, job submission to the QPU was done from your workstation and the ability was gated by on user ID. In
    Forest 2.0, job submission to the QPU must be done from your remote virtual machine, called a QMI (*Quantum Machine Image*).

-  In Forest 1.3, user data persisted indefinitely in cloud storage and could be accessed using the assigned job ID. In
    Forest 2.0, user data is stored only transiently, and it is the user's responsibility to handle long-term data storage
    on their QMI.


Updates to pyQuil
~~~~~~~~~~~~~~~~~

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By way of example, let's consider the following pyQuil 1.9 program,
which computes the natural bond distance in molecular hydrogen using a
VQE-type algorithm:

::


    from pyquil.quil import Program
    from pyquil.api import QVMConnection

    def setup_forest_objects():
        qvm = QVMConnection(sync_endpoint="http://localhost:5000")
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

    def get_convolution_coefficients(bond_length):
        return [0.1698845197777728, 0.16988451977777283, -0.2188630663199042, -0.2188630663199042]


    # some constants
    bond_step, bond_min, bond_max = 0.05, 0, 200
    angle_step, angle_min, angle_max = 0.1, 0, 63
    shots = 1000


    qvm = setup_forest_objects()


    # get all the unweighted expectations for all the sample wavefunctions.
    #
    # in a more elaborate example, you'd want to interleave this with the loop below
    # and intelligently query the QPU for wavefunctions along some path of descent.
    occupations = list(range(angle_min, angle_max))
    for offset in range(len(occupations)):
        program = build_wf_ansatz_prep(angle_min + offset*angle_step)
        bitstrings = qvm.run(program, [0,1,2,3])
        totals = [0,0,0,0]
        for array in bitstrings:
            totals[0] += array[0]
            totals[1] += array[1]
            totals[2] += array[2]
            totals[3] += array[3]
        occupations[offset] = [t/shots for t in totals]

    min_energies = list(range(bond_min, bond_max))
    for bond_length in range(len(min_energies)):
        coeffs = get_convolution_coefficients(bond_min + bond_length*bond_step)
        min_energies[bond_length] = min([sum([occupations[offset][j] * coeffs[j]
                                              for j in range(0, 4)])
                                         for offset in range(angle_min, angle_max)])

    min_index = min(range(len(min_energies)), key=lambda x: min_energies[x])

    min_energy, relaxed_length = min_energies[min_index], min_index * bond_step

In order to port this code to pyQuil 2.0, we need change only one thing: the part referencing ``QVMConnection`` should be replaced by an equivalent part referencing a ``QuantumComputer`` connected to a ``QVM``. Specifically, the following
snippet

::

    def setup_forest_objects():
        qvm = QVMConnection(sync_endpoint="http://localhost:5000")
        return qvm

can be changed to

::

    from pyquil.api import get_qc

    def setup_forest_objects():
        qc = get_qc("9q-generic-qvm")
        return qc

and the references to ``qvm`` in the main body are changed to ``qc`` instead. Since the ``QuantumComputer`` object also
exposes a ``run`` routine and pyQuil itself automatically rewrites 1.9-style ``MEASURE`` instructions into 2.0-style
instructions, this is all we need to do.

If we are willing to be more intrusive, we can also take advantage of pyQuil 2.0's classical memory and parametric
programs. The first piece to change is the Quil program itself: we remove the argument ``theta`` from the Python
function ``build_wf_ansatz_prep``, with the intention of letting the QPU fill it in later. In turn, we modify the Quil
program itself to have a ``REAL`` memory parameter named ``theta``. We also declare a few ``BIT``\ s for our ``MEASURE``
instructions to target.

::

    def build_wf_ansatz_prep():
        program = Program("""
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

::

    # get all the unweighted expectations for all the sample wavefunctions.
    #
    # in a more elaborate example, you'd want to interleave this with the loop below
    # and intelligently query the QPU for wavefunctions along some path of descent.
    occupations = list(range(angle_min, angle_max))
    for offset in range(len(occupations)):
        program = build_wf_ansatz_prep(angle_min + offset * angle_step)
        bitstrings = qvm.run(program, [0,1,2,3])
        totals = [0,0,0,0]
        for array in bitstrings:
            totals[0] += array[0]
            totals[1] += array[1]
            totals[2] += array[2]
            totals[3] += array[3]
        occupations[offset] = [t/shots for t in totals]

becomes

::

    program = build_wf_ansatz_prep()

    program.wrap_in_numshots_loop(shots=shots)
    nq_program = qc.compiler.quil_to_native_quil(program)
    binary = qc.compiler.native_quil_to_executable(nq_program)
    qc.qam.load(binary)

    # get all the unweighted expectations for all the sample wavefunctions.
    #
    # in a more elaborate example, you'd want to interleave this with the loop below
    # and intelligently query the QPU for wavefunctions along some path of descent.
    occupations = list(range(angle_min, angle_max))
    for offset in range(len(occupations)):
        qc.qam.write_memory(region_name='theta', value=angle_min + offset * angle_step)
        qc.qam.run()
        qc.qam.wait()
        totals = [0,0,0,0]
        for array in qc.qam.read_from_memory_region(region_name="ro", offsets=True):
            totals[0] += array[0]
            totals[1] += array[1]
            totals[2] += array[2]
            totals[3] += array[3]
        occupations[offset] = [t/shots for t in totals]

Overall, the resulting program looks like this:

::

    from pyquil.quil import Program
    from pyquil.api import get_qc

    def setup_forest_objects():
        qc = get_qc("9q-generic-qvm")
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

    def get_convolution_coefficients(bond_length):
        return [0.1698845197777728, 0.16988451977777283, -0.2188630663199042, -0.2188630663199042]


    # some constants
    bond_step, bond_min, bond_max = 0.05, 0, 200
    angle_step, angle_min, angle_max = 0.1, 0, 63
    shots = 1000

    # set up the Forest object
    qc = setup_forest_objects()

    # set up the Program object, once and for all
    program = build_wf_ansatz_prep()
    program.wrap_in_numshots_loop(shots=shots)
    nq_program = qc.compiler.quil_to_native_quil(program)
    binary = qc.compiler.native_quil_to_executable(nq_program)
    qc.qam.load(binary)

    # get all the unweighted expectations for all the sample wavefunctions.
    #
    # in a more elaborate example, you'd want to interleave this with the loop below
    # and intelligently query the QPU for wavefunctions along some path of descent.
    occupations = list(range(angle_min, angle_max))
    for offset in range(len(occupations)):
        qc.qam.write_memory(region_name='theta', value=angle_min + offset * angle_step)
        qc.qam.run()
        qc.qam.wait()
        totals = [0,0,0,0]
        for array in qc.qam.read_from_memory_region(region_name="ro", offsets=True):
            totals[0] += array[0]
            totals[1] += array[1]
            totals[2] += array[2]
            totals[3] += array[3]
        occupations[offset] = [t/shots for t in totals]

    min_energies = list(range(bond_min, bond_max))
    for bond_length in range(len(min_energies)):
        coeffs = get_convolution_coefficients(bond_min + bond_length*bond_step)
        min_energies[bond_length] = min([sum([occupations[offset][j] * coeffs[j]
                                              for j in range(0, 4)])
                                         for offset in range(angle_min, angle_max)])

    min_index = min(range(len(min_energies)), key=lambda x: min_energies[x])

    min_energy, relaxed_length = min_energies[min_index], min_index * bond_step

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


Miscellanea
^^^^^^^^^^^

On both the QVM and the QPU, ``INTEGER`` refers to an unsigned integer
stored in a 48-bit wide little-endian word, and ``REAL`` refers to a
48-bit wide little-endian fixed point number of type <0.48>. In general,
these datatypes are implementation-dependent. ``OCTET`` always refers to
an 8-bit wide unsigned integer and is independent of implementation.

Memory regions are all "global": ``DECLARE`` directives cannot appear in
the body of a ``DEFCIRCUIT``.


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
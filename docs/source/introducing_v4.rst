.. _introducing_v4:

Introducing pyQuil v4
=====================

The 4.0 major release of pyQuil moves the foundation of program parsing, manipulation, compilation, and execution into Rigetti's latest generation of SDKs written in Rust. This comes with improved performance, stronger type safety, better error messages, and access to exciting new features.

As a first step, read through the :doc:`changes` to get an overview of what's new. Pay special attention to the breaking changes you may need to accommodate. In the rest of this introduction, we'll expand on some of the key changes and new features.

.. note::

   pyQuil v4 is currently pre-release software. If using ``pip``, be sure to install it with the ``--pre`` flag: ``pip install --pre pyquil``

Parameters & Memory
-------------------

In order to provide more flexibility when executing parameterized Programs, the execution methods on ``QAM``, ``QVM``, ``QPU`` and the like now accept an optional ``memory_map`` keyword parameter. This parameter is defined as a mapping of a memory region's name to a sequence of values that will be used to initialize that memory region before executing the program. This replaces the ability to use the write_memory method on a Program.
Here is an example of how you might use a memory map in practice:

.. code:: python

    from pyquil.api import get_qc
    from pyquil.gates import RZ
    from pyquil.quil import Program

    qc = get_qc("Ankaa-1")
    program = Program()
    theta = program.declare("theta", "REAL")
    program += RZ(theta, 0)
    exe = qc.compile(program)

    # Previously, we would've used program.write_memory(region_name="theta", value=0.0)
    memory_map = {"theta": [0.0]}

    result = qc.run(exe, memory_map=memory_map)

The ``MemoryMap`` type is defined as ``Mapping[str, Union[Sequence[int], Sequence[float]]``. Note that the values mapped to a memory region must always be a sequence. This is different from ``write_memory`` which would allow writing an atomic value to a region of length 1.


QCS Gateway and Execution Options
---------------------------------

The QCS Gateway is a new service that provides on-demand access to a QPU. See the `QCS documentation`_ for more information on what it is and why you might find it useful.

.. _QCS documentation: https://docs.rigetti.com/qcs/guides/qcs-gateway

In pyQuil v4, Gateway is enabled by default and it is generally recommended to keep it on. However, if you have a use case for sending your job to the QPU directly, you can use the new ``ExecutionOptions`` and ``ConnectionStrategy`` classes to configure your request:

.. code:: python

    from pyquil.api import get_qc, ExecutionOptionsBuilder, ConnectionStrategy
    from pyquil.quil import Program

    qc = get_qc("Ankaa-1")
    program = Program()
    exe = qc.compile(program)

    # Use an ``ExecutionOptionsBuilder`` to build a custom ``ExecutionOptions``
    execution_options_builder = ExecutionOptionsBuilder()
    execution_options_builder.connection_strategy = ConnectionStrategy.direct_access()
    execution_options = execution_options_builder.build()

    # Option 1: Override execution options on a per-request basis.
    result = qc.run(exe, execution_options=execution_options)

    # Option 2: Sets the default options for all execution requests where no execution_options parameter is provided.
    result = qc.qam.execution_options = execution_options


Using the new QPU Compiler Backend
----------------------------------

Rigetti's next-generation QPU compiler is accessible through pyQuil v4. This new backend is still in development, so while it will eventually become the default, it is currently in limited access. If you have access, you can configure your compiler to use it using the new ``QPUCompilerAPIOptions`` class:

.. code:: python

    from pyquil.api import get_qc, QPUCompilerAPIOptions
    from pyquil.quil import Program

    program = Program()
    qc = get_qc("Ankaa-1")

    api_options = QPUCompilerAPIOptions()
    api_options.use_backend_v2()

    # Option 1: Apply to all compiled programs
    qc.compiler.api_options = api_options

    # Option 2: Apply to one specific compilation
    qc.compiler.native_quil_to_executable(program, api_options=api_options)

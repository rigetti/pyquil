.. _migration:

Migrating from pyQuil v2 to v3
==============================

To begin migrating your pyQuil v2 code, we recommend first reading the :doc:`changes` and making adjustments according
to the code affected. Most users should only need to make minimal changes.

If you've supplied ``PyquilConfig`` objects to functions (or used the ``QVM_URL`` and ``QUILC_URL`` environment variables)
to override configuration, see :ref:`pyquil_configuration`.

Authentication
--------------

pyQuil v3 relies on an updated authentication model. To get started, install the new `QCS CLI
<https://docs.rigetti.com/>`_  and
log in with it before using pyQuil v3 with QCS and live QPUs.


Parameters & Memory
-------------------

In order to give the user more control over and visibility into program execution, especially in
parallel, objects such as ``QuantumComputer``, ``QAM``, ``QPU``, and ``QVM`` are no longer stateful
with respect to individual programs. These objects are now safe to
share among different threads, so you can execute and retrieve results in parallel for even better
performance. (See :doc:`advanced_usage` for more information).

However, this required three small but important changes:

1. ``write_memory`` is no longer a method on ``QAM`` but rather on ``Program`` and ``EncryptedProgram``.
2. ``qc.run()`` no longer accepts a ``memory_map`` argument. All memory values must be set directly
  on the ``Program`` or ``EncryptedProgram`` using ``write_memory``.
3. ``QAM.load()``, ``QAM.wait()``, and ``QAM.reset()`` no longer exist, because the
  ``QAM`` no longer "stores" program state.

This means that you should now execute your programs using one of these options:

.. code:: python

   qc = get_qc("Aspen-X")
   program = Program()
   theta = program.declare('theta', 'REAL')
   program += RZ(theta, 0)
   exe = qc.compile(program)

   # Previously, we would have called ``qc.qam.write_memory`` instead
   exe.write_memory(region_name='theta', value=np.pi)

   # Option 1
   ro_bitstring = qc.run(exe)

   # Option 2
   result = qc.qam.run(exe)

   # Option 3
   job = qc.qam.execute(exe)
   result = qc.qam.get_result(job)

   # Run our program 10 times, enqueuing all the programs before retrieving results for any of them
   jobs = [qc.qam.execute(exe) for _ in range(10)]
   results = [qc.qam.get_result(job) for job in jobs]


Compatibility Utilities
-----------------------

We understand that the changes above regarding parameters might cause difficulty in migration,
especially for large projects and lengthy scripts. So, to ease the migration path, we've added
the following utility classes which allow you to upgrade your pyQuil projects without having to
change any code.

.. code:: python

   from pyquil.compatibility.v2 import get_qc, QuantumComputer
   from pyquil.compatibility.v2.api import QAM, QVM, QPU

You can use these imported objects exactly how you already use their counterparts in pyQuil v2.
Once you've verified that your scripts still work with v3, we recommend that you gradually convert
them to use the new versions of each object. This compatibility layer won't see any new
development, and without fully upgrading you'd miss out on all the new features to come in the
future.

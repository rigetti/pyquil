.. _migration:

Migrating from pyQuil v2 to v3
==============================

To begin migrating your pyQuil v2 code, we recommend first reading the :doc:`changes` and making adjustments according
to the code affected. Most users should only need to make minimal changes.

If you've supplied ``PyquilConfig`` objects to functions (or used the ``QVM_URL`` and ``QUILC_URL`` environment variables)
to override configuration, see :ref:`pyquil_configuration`.

Lastly, pyQuil v3 relies on an updated authentication model. To get going smoothly, you should install the new `QCS CLI
<https://docs.rigetti.com/en/command-line-interface/command-line-interface>`_ and log in with it before using pyQuil v3
against real QPUs.

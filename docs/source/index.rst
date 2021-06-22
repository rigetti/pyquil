.. pyQuil documentation master file, created by
   sphinx-quickstart on Mon Jun 13 17:59:05 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: red

Welcome to the Docs for pyQuil!
===============================

As a component of the Rigetti Forest SDK, pyQuil allows you to build and execute Quil programs using Python. pyQuil
requires installation of the other components of the Forest SDK, namely the Quil compiler (quilc) and the Quantum Virtual
Machine (QVM), used for simulating quantum computers. You can also use pyQuil to run programs against real quantum computers
using Rigetti's Quantum Cloud Services (QCS).

To learn more about Quil, the Forest SDK, and QCS, visit the `Rigetti docs site <https://docs.rigetti.com>`_.

If you’re new to pyQuil, we hope this documentation will serve as a helpful reference. Head to :ref:`start`
to get started.

.. note::

   If you've used pyQuil before, be sure to check out :ref:`migration` for help with moving to the newest pyQuil release.

.. note::

    To join our user community, connect to the `Rigetti Slack workspace <https://rigetti-forest.slack.com>`_ using `this invite <https://rigetti-forest.slack.com/join/shared_invite/enQtNTUyNTE1ODg3MzE2LWQwNzBlMjZlMmNlN2M5MzQyZDlmOGViODQ5ODI0NWMwNmYzODY4YTc2ZjdjOTNmNzhiYTk2YjVhNTE2NTRkODY>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   start
   basics
   qvm
   wavefunction_simulator
   compiler
   noise
   advanced_usage
   troubleshooting
   exercises
   migration
   changes

.. toctree::
   :maxdepth: 2
   :caption: Quil-T

   quilt
   quilt_getting_started
   quilt_waveforms
   quilt_parametric
   quilt_raw_capture

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   apidocs/modules

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

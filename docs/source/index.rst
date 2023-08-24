.. pyQuil documentation master file, created by
   sphinx-quickstart on Mon Jun 13 17:59:05 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: red

===============================
Welcome to the docs for pyQuil!
===============================

As a part of the Quil SDK, pyQuil allows you to build and execute Quil programs using Python. pyQuil
requires installation of the other components of the Quil SDK, namely the Quil compiler (quilc) and the Quantum Virtual
Machine (QVM), used for simulating quantum computers. You can also use pyQuil to run programs on real quantum computers
using `Rigetti's Quantum Cloud Services (QCS) <https://docs.rigetti.com/qcs/>`_.

To learn more about Quil, the Quil SDK, and QCS, see `Rigetti's documentation <https://docs.rigetti.com>`_.

If you’re new to pyQuil, head to the `getting started <getting_started>`_ guide to get setup and run your first program!

.. note::

   If you've used pyQuil before, be sure to check out :ref:`introducing_v4` to help get oriented on the key changes in v4.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   programs_and_gates
   qvm
   wavefunction_simulator
   compiler
   noise
   advanced_usage
   troubleshooting
   introducing_v4
   exercises
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

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. pyQuil documentation master file, created by
   sphinx-quickstart on Mon Jun 13 17:59:05 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Docs for Forest and pyQuil!
==========================================

Overview
--------

pyQuil is part of the Rigetti Forest `toolkit <http://forest.rigetti.com>`_ for
**quantum programming in the cloud**. If you are
interested in obtaining an API key for the beta, please reach out by signing up
`here <http://forest.rigetti.com>`_. We look forward to hearing from you.

pyQuil is an open source Python library developed at `Rigetti Computing <http://rigetti.com>`_
that constructs programs for quantum computers. The source is hosted on
`GitHub <https://github.com/rigetticomputing/pyquil>`_. More concretely,
pyQuil produces programs in the **Quantum Instruction Language** (Quil).
For a full description of Quil, please refer to the whitepaper
*A Practical Quantum Instruction Set Architecture*. [1]_  Quil is an opinionated quantum
instruction language --- its basic belief is that in the near term quantum computers
will operate as coprocessors, working in concert with traditional CPUs. This means that
Quil is designed to execute on a Quantum Abstract Machine that has a shared classical/quantum
architecture at its core.

Quil programs can be executed on a cloud-based **Quantum Virtual Machine** (QVM). This is a
classical simulation of a quantum processor that can simulate various qubit operations.
The default access key allows you to run simulations of up to 26 qubits. These simulations
can be run through either synchronous API calls, or through an asynchronous job queue for
larger programs. More information about the QVM can be found in the :ref:`qvm`.

In addition to the QVM, we offer the ability to run programs on our superconducting quantum processors,
or **Quantum Processing Units** (QPUs), at our lab in Berkeley, California. To request upgraded
access to our 19Q QPU, please fill out the `request form <https://www.rigetti.com/qpu-request>`_
with a brief summary of what you hope to use it for. For more information on QPUs, check out
:ref:`qpu`.

If you are already familiar with quantum computing, then feel free to proceed to
:ref:`start`. Otherwise, take a look at our :ref:`intro`, where we use Quil
introduce the basics of quantum computing and the Quantum Abstract Machine on which it runs.

.. [1] https://arxiv.org/abs/1608.03355

Contents
--------

.. toctree::
   :maxdepth: 3

   intro
   start
   basics
   advanced_usage
   exercises
   qvm
   qpu
   compiler
   qubit-placeholder
   noise
   modules
   changes


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

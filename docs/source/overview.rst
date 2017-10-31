
Overview
========

Welcome to pyQuil!

pyQuil is part of the Rigetti Forest `toolkit <http://forest.rigetti.com>`_ for
**quantum programming in the cloud**, which is currently in public beta.  If you are interested in
obtaining an API key for the beta, please reach out by signing up
`here <http://forest.rigetti.com>`_.  We look forward to hearing from you.

pyQuil is an open source Python library developed at
`Rigetti Quantum Computing <http://rigetti.com>`_ that constructs
programs for quantum computers.  The source is hosted on
`GitHub <https://github.com/rigetticomputing/pyquil>`_.

More concretely, pyQuil produces programs in the Quantum Instruction Language (Quil).  For a full
description of Quil, please refer to the whitepaper *A Practical Quantum Instruction Set Architecture*.
[1]_  Quil is an opinionated quantum instruction language: its basic
belief is that in the near term quantum computers will operate as coprocessors, working in
concert with traditional CPUs.  This means that Quil is designed to execute on a Quantum Abstract
Machine that has a shared classical/quantum architecture at its core.

Quil programs can be executed on a local or cloud-based Quantum Virtual Machine.  This is a
classical simulation of a quantum processor that can simulate up to 36 qubits.  The default access
key allows you to run simulations of up to 26 qubits.  These simulations can be run through either
synchronous API calls, or through an asynchronous job queue for larger programs.  More information
about the QVM can be found at `Overview of the Quantum Virtual Machine <qvm_overview.html>`_.

If you are already familiar with quantum computing, then feel free to proceed to
`Installation and Getting Started <getting_started.html>`_.

Otherwise, take a look at our `Brief Introduction to Quantum Computing <intro_to_qc.html>`_,
where the basics of quantum computing are introduced using Quil and the Quantum
Abstract Machine on which it runs.

.. [1] https://arxiv.org/abs/1608.03355

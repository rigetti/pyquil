
Overview
========

Welcome to pyQuil!

pyQuil is part of the Rigetti Forest `toolkit <http://forest.rigetti.com>`_ for
**quantum programming in the cloud**, which is currently in public beta. If you are
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
larger programs. More information about the QVM can be found in the
`Overview of the Quantum Virtual Machine <qvm_overview.html>`_.

In addition to the QVM, we offer the ability to run programs on the physical qubits,
or **Quantum Processing Units** (QPUs), at our lab in Berkeley, California. To request upgraded
access to our 19Q QPU, please fill out the `request form <https://www.rigetti.com/qpu-request>`_
with a brief summary of what you hope to use it for. For more information on the QPU, check out
the `QPU Overview <qpu_overview.html>`_ and `Using the QPU-based stack <qpu_usage.html>`_.

If you are already familiar with quantum computing, then feel free to proceed to
`Installation and Getting Started <getting_started.html>`_. Otherwise, take a look at our
`Introduction to Quantum Computing <intro_to_qc.html>`_, where we use Quil
introduce the basics of quantum computing and the Quantum Abstract Machine on which it runs.

.. [1] https://arxiv.org/abs/1608.03355

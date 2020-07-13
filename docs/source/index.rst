.. pyQuil documentation master file, created by
   sphinx-quickstart on Mon Jun 13 17:59:05 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: red

Welcome to the Docs for the Forest SDK!
=======================================

The Rigetti Forest `Software Development Kit <https://qcs.rigetti.com/sdk-downloads>`_ includes pyQuil, the Rigetti Quil Compiler
(quilc), and the Quantum Virtual Machine (qvm).

**Longtime users of Rigetti Forest will notice a few changes.** First, the SDK now contains a downloadable compiler and a
QVM. Second, the SDK contains pyQuil 2.0, with significant updates to previous versions. As a result, programs written
using previous versions of the Forest toolkit will need to be updated to pyQuil 2.0 to be compatible with the QVM or compiler.

After installing the SDK and updating pyQuil in :ref:`start`, see the `migration guide <migration.ipynb>`_
to get caught up on what's new!

Quantum Cloud Services will provide users with a dedicated Quantum Machine Image, which will come prepackaged with the
Forest SDK. We’re releasing a Preview to the Forest SDK now, so current users can begin migrating code (and share feedback
with us early and often!). Longtime Forest users should start with the Migration Guide which outlines key changes in this
SDK Preview release.

If you’re new to Forest, we hope this documentation will provide everything you need to get up and running with the toolkit.
Once you’ve oriented yourself here, proceed to the section :ref:`start` to get started. If you're new to quantum computing,
you also go to our section on :ref:`intro`. There, you’ll learn the basic concepts needed to write quantum software. You
can also work through an introduction to quantum computing in a jupyter notebook; launch the notebook from the source folder
in pyquil's docs:

.. code::

    cd pyquil/docs/source
    jupyter notebook intro_to_qc.ipynb


**A few terms to orient you as you get started with Forest:**

1. **pyQuil**: An open source Python library to help you write and run quantum programs.
   The source is hosted on `github <http://github.com/rigetti/pyquil>`_.
2. **Quil**: The Quantum Instruction Language standard. Instructions written in Quil can be executed on any
   implementation of a quantum abstract machine, such as the quantum virtual machine (QVM), or on a real quantum processing
   unit (QPU). More details regarding Quil can be found in the whitepaper,
   `A Practical Quantum Instruction Set Architecture <https://arxiv.org/abs/1608.03355>`__.
3. **QVM**: The `Quantum Virtual Machine <qvm.html>`_ is an open source implementation of a quantum abstract machine on
    classical hardware. The QVM lets you use a regular computer to simulate a small quantum computer and execute Quil
    programs. Find `QVM on GitHub <https://github.com/rigetti/qvm>`__.
4. **QPU**: Quantum processing unit. This refers to the physical hardware chip which we run quantum programs on.
5. **Quil Compiler**: The compiler, ``quilc``, compiles Quil written for one quantum abstract machine (QAM) to another. Our
   open source compiler will take arbitrary Quil and compile it for the given QAM, according to its supported instruction
   set architecture. Find `quilc on GitHub <https://github.com/rigetti/quilc>`__.
6. **Forest SDK**: Our software development kit, optimized for near-term quantum computers that operate as coprocessors, working in
   concert with traditional processors to run hybrid quantum-classical algorithms. For references on problems addressable
   with near-term quantum computers, see `Quantum Computing in the NISQ era and beyond <https://arxiv.org/abs/1801.00862>`_.

Our flagship product `Quantum Cloud Services <https://qcs.rigetti.com/request-access>`_ offers users an on-premise, dedicated access
point to our quantum computers. This access point is a fully-configured VM, which we call a Quantum Machine Image. A QMI
is bundled with the same downloadable SDK mentioned above, and a command line interface (CLI), which is used for
scheduling compute time on our quantum computers. To sign up for our waitlist, please click the link above. If you'd like
to access to our quantum computers for research, please email support@rigetti.com.

.. note::

    To join our user community, connect to the `Rigetti Slack workspace <https://rigetti-forest.slack.com>`_ using `this invite <https://rigetti-forest.slack.com/join/shared_invite/enQtNTUyNTE1ODg3MzE2LWQwNzBlMjZlMmNlN2M5MzQyZDlmOGViODQ5ODI0NWMwNmYzODY4YTc2ZjdjOTNmNzhiYTk2YjVhNTE2NTRkODY>`_.

Contents
--------

.. toctree::
   :maxdepth: 3

   start
   migration
   migration2-qc
   migration3-declare
   migration4
   basics
   qvm
   wavefunction_simulator
   compiler
   noise
   advanced_usage
   troubleshooting
   exercises
   changes
   intro

.. toctree::
   :maxdepth: 3
   :caption: Quilt Tutorials

   quilt_getting_started
   quilt_waveforms
   quilt_parametric
   quilt_raw_capture

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   apidocs/program
   apidocs/gates
   apidocs/pauli
   apidocs/quantum_computer
   apidocs/compilers
   apidocs/qam
   apidocs/devices
   apidocs/simulators
   apidocs/noise
   apidocs/operator_estimation
   apidocs/visualization
   apidocs/experiment

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. pyQuil documentation master file, created by
   sphinx-quickstart on Mon Jun 13 17:59:05 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: red

Welcome to the Docs for the Forest SDK!
=======================================

The Rigetti Forest `Software Development Kit <http://rigetti.com/forest>`_ includes pyQuil, the Rigetti Quil Compiler
(quilc), and the Quantum Virtual Machine (qvm).

**Longtime users of Rigetti Forest will notice a few changes.** First, the SDK now contains a downloadable compiler and a
QVM. Second, the SDK contains pyQuil 2.0, with significant updates to previous versions. As a result, programs written
using previous versions of the Forest toolkit will need to be updated to pyQuil 2.0 to be compatible with the QVM or compiler.

After installing the SDK and updating pyQuil in :ref:`start`, see :ref:`quickstart` to get caught up on what's new!

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

- pyQuil is an open source Python library developed at Rigetti Computing that allows you to write programs for quantum computers.
  The source is hosted on `github <http://github.com/rigetticomputing/pyquil>`_.
- Quil, the Quantum Instruction Language, is the lower-level code that pyQuil gets compiled into. A full description of
  Quil can be found in our whitepaper, `A Practical Quantum Instruction Set Architecture <https://arxiv.org/abs/1608.03355>`_.
- quilc is the Rigetti Quil Compiler that compiles pyQuil into Quil. The SDK includes quilc, which will enable you to
  compile your pyQuil programs into executable Quil code.
- The QVM is a simulator of our quantum computers. When you download the SDK, you’ll install the QVM and you will execute
  Quil programs against it.
- Forest is our software development kit, optimized for near-term quantum computers that operate as coprocessors, working in
  concert with traditional processors to run hybrid quantum-classical algorithms. For references on problems addressable
  with near-term quantum computers, see `Quantum Computing in the NISQ era and beyond <https://arxiv.org/abs/1801.00862>`_.

Our flagship product `Quantum Cloud Services <http://rigetti.com/qcs>`_ offers users an on-premise, dedicated access
point to our quantum computers. This access point is a fully-configured VM, which we call a Quantum Machine Image. A QMI
is bundled with the same downloadable SDK mentioned above, and a command line interface (CLI), which is used for
scheduling compute time on our quantum computers. To sign up for our waitlist, please click the link above. If you'd like
to access to our quantum computers for research, please email support@rigetti.com.

.. note::

    To join our user community, connect to the Rigetti Slack workspace at https://rigetti-forest.slack.com.

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


.. pyQuil documentation master file, created by
   sphinx-quickstart on Mon Jun 13 17:59:05 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: red

Welcome to the Docs for Forest and pyQuil!
==========================================


PLEASE READ: A NOTE FROM RIGETTI COMPUTING
==========================================
.. note::
      Our forthcoming flagship product, `Quantum Cloud Services <http://rigetti.com/qcs>`_, is an overhaul of pyQuil,
      Quil, compilation, access, and execution. Due to the significance of the changes, programs written in pyQuil v1.9
      will need refactoring before they will run on the new QVM, Quil Compiler or QPU. Our 1.3-based QPUs are now
      offline; the QPUs that replace them will only be compatible with pyQuil v2.0 and beyond. For now, you will still
      be able to run programs against our cloud-based QVM, but we encourage that you transition to our SDK, which
      includes a downloadable QVM and QUILC (thus eliminating the need for you to send jobs to a cloud queue just to run
      programs). To download our new SDK Preview, see the :ref:`sdk`. Current users: please refer to the :ref:`quickstart`
      after downloading to get caught up on changes.


Overview
--------

pyQuil is part of the Rigetti Forest `Software Development Kit <http://rigetti.com/forest>`_.

pyQuil is an open source Python library developed at `Rigetti Computing <http://rigetti.com>`_ that constructs programs
for quantum computers. The source is hosted on `GitHub <https://github.com/rigetticomputing/pyquil>`_. pyQuil produces
programs in the **Quantum Instruction Language** (Quil). For a full description of Quil, please refer to the whitepaper
*A Practical Quantum Instruction Set Architecture*. [1]_  Quil is an opinionated quantum instruction language --- its
basic belief is that in the near term quantum computers will operate as coprocessors, working in concert with traditional
CPUs. This means that Quil is designed to execute on a Quantum Abstract Machine that has a shared classical/quantum
architecture at its core.

Quil programs can be executed on a downloadable **Quantum Virtual Machine** (QVM). This is a classical simulation of a
quantum processor that can simulate various qubit operations. The Forest SDK comes with a QVM that you can run on your
local machine. More information about the QVM can be found in the :ref:`qvm`.

Our flagship product, `Quantum Cloud Services <http://rigetti.com/qcs>`_ offers users an on-premise, dedicated access
point to our quantum computers, and to a powerful 34-qubit Quantum Virtual Machine. This access point sits in a Virtual
Machine, which we call a Quantum Machine Image. A QMI is bundled with the same downloadable SDK mentioned above, and a
Command Line Interface (CLI), which is used for scheduling compute time on our quantum computers. To sign up for our
waitlist, please click the link above. If need access to our quantum computers for research, please email
support@rigetti.com.

If you are already familiar with quantum computing, then feel free to proceed to :ref:`start`. If you're just getting
started, try out our Introduction to Quantum Programming juypter notebook, and take a look at our :ref:`intro`, where we
use Quil, and introduce the basics of quantum computing and the Quantum Abstract Machine on which it runs. You can run
this notebook by going into your pyQuil directory and finding the source folder in docs. Then run

.. code::`jupyter notebook intro_to_qc.ipynb`.


Try it out!

.. [1] https://arxiv.org/abs/1608.03355

Contents
--------

.. toctree::
   :maxdepth: 3

   SDK_Download_Instructions
   2.0_quickstart_and_migration
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

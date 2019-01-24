Changelog
=========

v2.3 (January 30, 2019)
-----------------------

PyQuil 2.3 is the latest release of pyQuil, Rigetti's toolkit for constructing and running
quantum programs. A major new feature is the release of a new suite of simulators:

- We're proud to introduce the first iteration of a Python-based quantum virtual machine (QVM)
  called PyQVM. This QVM is completely contained within pyQuil and does not need any external
  dependencies. Try using it with ``get_qc("9q-square-pyqvm")`` or explore the
  ``pyquil.pyqvm.PyQVM`` object directly. Under-the-hood, there are three quantum simulator
  backends:
     - ``ReferenceWavefunctionSimulator`` uses standard matrix-vector multiplication to
       evolve a statevector. This includes a suite of tools in ``pyquil.unitary_tools`` for dealing
       with unitary matrices.
     - ``NumpyWavefunctionSimulator`` uses numpy's tensordot functionality to efficiently evolve
       a statevector. For most simulations, performance is quite good.
     - ``ReferenceDensitySimulator`` uses matrix-matrix multiplication to evolve a density
       matrix.
- Matrix representations of Quil standard gates are included in ``pyquil.gate_matrices`` (gh-552).
- The density simulator has extremely limited support for Kraus-operator based noise models.
  Let us know if you're interested in contributing more robust noise-model support.
- This functionality should be considered experimental and may undergo minor API changes.

Important changes to note:

- Quil math functions (like COS, SIN, ...) used to be ambiguous with respect to case sensitivity.
  They are now case-sensitive and should be uppercase (gh-774).
- Communication with quilc now happens via the rpcq protocol. ``LocalQVMCompiler``
  and ``LocalBenchmarkConnection`` have been removed in favor of a now-unified ``QVMCompiler``
  and ``BenchmarkConnection``. This change should be transparent if you were using ``get_qc`` and
  ``get_benchmarker``, respectively. This change requires that you upgrade your version of
  quilc to at least x.x (gh-730).
- When using a paramaterized gate, the QPU control electronics only allowed multiplying
  parameters by powers of two. If you only ever multiply a parameter by the same constant,
  this isn't too much of a problem because you can fold the multiplicative constant
  into the definition of the parameter. However, if you are multiplying the same variable
  (e.g. ``gamma`` in QAOA) by different constants (e.g. weighted maxcut edge weights) it doesn't
  work. PyQuil will now transparently handle the latter case by expanding to a vector of
  parameters with the constants folded in, allowing you to multiply variables by whatever you
  want (gh-707).

As always, this release contains bug fixes and improvements:

- The CZ gate fidelity metric available in the Specs object now has its associated standard
  error, which is accessible from the method ``Specs.fCZ_std_errs`` (gh-751).
- Operator estimation code now correctly handles identity terms with coefficients. Previously,
  it would always estimate these terms as 1.0 (gh-758).
- Operator estimation results include the total number of counts (shots) taken.
- Operator estimation JSON serialization uses utf-8. Please let us know if this
  causes problems (gh-769).
- The example quantum die program now can roll dice that are not powers of two (gh-749).
- When running on the QPU, you could get into trouble if the QPU name passed to ``get_qc`` did not
  match the lattice you booked. This is now validated (gh-771).


v2.2 (January 4, 2019)
----------------------

PyQuil 2.2 is the latest release of pyQuil, Rigetti's toolkit for constructing and running
quantum programs. Bug fixes and improvements include:

- ``pauli.is_zero`` and ``paulis.is_identity`` would sometimes return erroneous answers (gh-710).
- Parameter expressions involving addition and subtraction are now converted to Quil with spaces
  around the operators, e.g. ``theta + 2`` instead of ``theta+2``. This disambiguates subtracting
  two parameters, e.g. ``alpha - beta`` is not one variable named ``alpha-beta`` (gh-743).
- T1 is accounted for in T2 noise models (gh-745).
- Documentation improvements (gh-723, gh-719, gh-720, gh-728, gh-732, gh-742).
- Support for PNG generation of circuit diagrams via LaTeX (gh-745).
- We've started transitioning to using Gitlab as our continuous integration provider for
  pyQuil (gh-741, gh-752).

This release includes a new module for facilitating the estimation of quantum
observables/operators (gh-682). First-class support for estimating observables should
make it easier to express near-term algorithms. This release includes:

- data structures for expressing tomography-like experiments and their results
- grouping of experiment settings that can be simultaneously estimated
- functionality to executing a tomography-like experiment on a quantum computer

Please look forward to more features and polish in future releases. Don't hesitate to submit
feedback or suggestions as GitHub issues.

We extend thanks to community member petterwittek for their contribution to this release.

Bugfix release 2.2.1 was released January 11 to maintain compatibility with the latest version
of the quilc compiler (gh-759).


v2.1 (November 30, 2018)
------------------------

PyQuil 2.1 is an incremental release of pyQuil, Rigetti's toolkit for constructing and running
quantum programs. Changes include:

- Major documentation improvements.
- ``QuantumComputer.run()`` accepts an optional ``memory_map`` parameter to facilitate running
  parametric executables (gh-657).
- ``QuantumComputer.reset()`` will reset the state of a QAM to recover from an error condition
  (gh-703).
- Bug fixes (gh-674, gh-696).
- Quil parser improvements (gh-689, gh-685).
- Optional interleaver argument when generating RB sequences (gh-673).
- Our GitHub organization name has changed from ``rigetticomputing`` to ``rigetti`` (gh-713).


v2.0 (November 1, 2018)
-----------------------

PyQuil 2.0 is a major release of pyQuil, Rigetti's toolkit for constructing and running quantum
programs. This release contains many major changes including:

1. The introduction of `Quantum Cloud Services <https://www.rigetti.com/qcs>`_. Access
   Rigetti's QPUs from co-located classical compute resources for minimal latency. The
   web API for running QVM and QPU jobs has been deprecated and cannot be accessed with
   pyQuil 2.0
2. Advances in classical control systems and compilation allowing the pre-compilation of
   parametric binary executables for rapid hybrid algorithm iteration.
3. Changes to Quil---our quantum instruction language---to provide easier ways of
   interacting with classical memory.

The new QCS access model and features will allow you to execute hybrid quantum algorithms
several orders of magnitude (!) faster than the previous web endpoint. However, to fully
exploit these speed increases you must update your programs to use the latest pyQuil features
and APIs. Please read :ref:`quickstart` for a comprehensive migration guide.

An incomplete list of significant changes:

- Python 2 is no longer supported. Please use Python 3.6+
- Parametric gates are now normal functions. You can no longer write ``RX(pi/2)(0)`` to get a
  Quil ``RX(pi/2) 0`` instruction. Just use ``RX(pi/2, 0)``.
- Gates support keyword arguments, so you can write ``RX(angle=pi/2, qubit=0)``.
- All ``async`` methods have been removed from ``QVMConnection`` and ``QVMConnection`` is
  deprecated. ``QPUConnection`` has been removed in accordance with the QCS access model.
  Use :py:func:`pyquil.get_qc` as the primary means of interacting with the QVM or QPU.
- ``WavefunctionSimulator`` allows unfettered access to wavefunction properties and routines.
  These methods and properties previously lived on ``QVMConnection`` and have been deprecated
  there.
- Classical memory in Quil must be declared with a name and type. Please read :ref:`quickstart`
  for more.
- Compilation has changed. There are now different ``Compiler`` objects that target either the
  QPU or QVM. You **must** explicitly compile your programs to run on a QPU or a realistic QVM.

Version 2.0.1 was released on November 9, 2018 and includes documentation changes only. This
release is only available as a git tag. We have not pushed a new package to PyPI.


v1.9 (June 6, 2018)
-------------------

We’re happy to announce the release of pyQuil 1.9. PyQuil is Rigetti’s toolkit for constructing
and running quantum programs. This release is the latest in our series of regular releases,
and it’s filled with convenience features, enhancements, bug fixes, and documentation improvements.

Special thanks to community members sethuiyer, vtomole, rht, akarazeev, ejdanderson, markf94,
playadust, and kadora626 for contributing to this release!

Qubit placeholders
~~~~~~~~~~~~~~~~~~

One of the focuses of this release is a re-worked concept of "Qubit Placeholders". These are
logical qubits that can be used to construct programs. Now, a program containing qubit placeholders
must be "addressed" prior to running on a QPU or QVM. The addressing stage involves mapping
each qubit placeholder to a physical qubit (represented as an integer). For example, if you have
a 3 qubit circuit that you want to run on different sections of the Agave chip, you now can
prepare one Program and address it to many different subgraphs of the chip topology.
Check out the ``QubitPlaceholder`` example notebook for more.

To support this idea, we've refactored parts of Pyquil to remove the assumption that qubits
can be "sorted". While true for integer qubit labels, this probably isn't true in general.
A notable change can be found in the construction of a ``PauliSum``: now terms will stay in the
order they were constructed.

- :py:class:`PauliTerm` now remembers the order of its operations. ``sX(1)*sZ(2)`` will compile
  to different Quil code than ``sZ(2)*sX(1)``, although the terms will still be equal according
  to the ``__eq__`` method. During :py:class:`PauliSum` combination
  of like terms, a warning will be emitted if two terms are combined that have different orders
  of operation.
- :py:func:`PauliTerm.id()` takes an optional argument ``sort_ops`` which defaults to True for
  backwards compatibility. However, this function should not be used for comparing term-type like
  it has been used previously. Use :py:func:`PauliTerm.operations_as_set()` instead. In the future,
  ``sort_ops`` will default to False and will eventually be removed.
- :py:func:`Program.alloc()` has been deprecated. Please instantiate :py:class:`QubitPlaceholder()`
  directly or request a "register" (list) of ``n`` placeholders by using the class constructor
  :py:func:`QubitPlaceholder.register(n)`.
- Programs must contain either (1) all instantiated qubits with integer indexes or (2) all
  placeholder qubits of type :py:class:`QubitPlaceholder`. We have found that most users use
  (1) but (2) will become useful with larger and more diverse devices.
- Programs that contain qubit placeholders must be **explicitly addressed** prior to execution.
  Previously, qubits would be assigned "under the hood" to integers 0...N. Now, you must use
  :py:func:`address_qubits` which returns a new program with all qubits indexed depending
  on the ``qubit_mapping`` argument. The original program is unaffected and can be "readdressed"
  multiple times.
- :py:class:`PauliTerm` can now accept :py:class:`QubitPlaceholder` in addition to integers.
- :py:class:`QubitPlaceholder` is no longer a subclass of :py:class:`Qubit`.
  :py:class:`LabelPlaceholder` is no longer a subclass of :py:class:`Label`.
- :py:class:`QuilAtom` subclasses' hash functions have changed.

Randomized benchmarking sequence generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pyquil now includes support for performing a simple benchmarking routine - randomized
benchmarking. There is a new method in the :py:class:`CompilerConnection` that will return
sequences of pyquil programs, corresponding to elements of the Clifford group. These programs
are uniformly randomly sampled, and have the property that they compose to the identity. When
concatenated and run as one program, these programs can be used in a procedure called randomized
benchmarking to gain insight about the fidelity of operations on a QPU.

In addition, the :py:class:`CompilerConnection` has another new method,
:py:func:`apply_clifford_to_pauli` which conjugates :py:class:`PauliTerms` by
:py:class:`Program` that are composed of Clifford gates. That is to say, given a circuit C,
that contains only gates corresponding to elements of the Clifford group, and a tensor product of
elements P, from the Pauli group, this method will compute $PCP^{\dagger}$. Such a procedure can
be used in various ways. An example is predicting the effect a Clifford circuit will have on an
input state modeled as a density matrix, which can be written as a sum of Pauli matrices.


Ease of Use
~~~~~~~~~~~

This release includes some quality-of-life improvements such as the ability to initialize
programs with generator expressions, sensible defaults for :py:func:`Program.measure_all`,
and sensible defaults for ``classical_addresses`` in :py:func:`run` methods.


- :py:class:`Program` can be initiated with a generator expression.
- :py:func:`Program.measure_all` (with no arguments) will measure all qubits in a program.
- ``classical_addresses`` is now optional in QVM and QPU :py:func:`run` methods. By default,
  any classical addresses targeted by ``MEASURE`` will be returned.
- :py:func:`QVMConnection.pauli_expectation` accepts ``PauliSum`` as arguments. This offers
  a more sensible API compared to :py:func:`QVMConnection.expectation`.
- pyQuil will now retry jobs every 10 seconds if the QPU is re-tuning.
- :py:func:`CompilerConnection.compile` now takes an optional argument ``isa`` that allows
  per-compilation specification of the target ISA.
- An empty program will trigger an exception if you try to run it.

Supported versions of Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We strongly support using Python 3 with Pyquil. Although this release works with Python 2,
we are dropping official support for this legacy language and moving to community support for
Python 2. The next major release of Pyquil will introduce Python 3.5+ only features and will
no longer work without modification for Python 2.


Bug fixes
~~~~~~~~~

- ``shift_quantum_gates`` has been removed. Users who relied on this
  functionality should use :py:class:`QubitPlaceholder` and :py:func:`address_qubits` to
  achieve the same result. Users should also double-check data resulting from use of this function
  as there were several edge cases which would cause the shift to be applied incorrectly resulting
  in badly-addressed qubits.
- Slightly perturbed angles when performing RX gates under a Kraus noise model could result in
  incorrect behavior.
- The quantum die example returned incorrect values when ``n = 2^m``.

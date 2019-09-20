Changelog
=========

[v2.12](https://github.com/rigetti/pyquil/compare/v2.11.0...master) (in development)
----------------------------------------------------------------------------------------

### Announcements

-   There is now a [Contributing Guide](CONTRIBUTING.md) for those who would like
    to participate in the development of pyQuil. Check it out! (@karalekas, gh-996)
-   PyQuil now has a [Bug Report Template](.github/ISSUE_TEMPLATE/BUG_REPORT.md),
    and a [Feature Request Template](.github/ISSUE_TEMPLATE/FEATURE_REQUEST.md),
    which contain sections to fill out when filing a bug or suggesting an enhancement
    (@karalekas, gh-985, gh-986).
-   `PauliSum` objects can now be constructed from strings via
    `from_compact_str()` and `PauliTerm.from_compact_str()` supports multi-qubit
    strings (@jlbosse, gh-984).

### Improvements and Changes

-   The `local_qvm` context manager has been renamed to `local_forest_runtime`,
    which now checks if the designated ports are used before starting `qvm`/`quilc`.
    The original `local_qvm` has been deprecated (@sauercrowd, gh-976).
-   The test suite for pyQuil now runs against both Python 3.6 and 3.7 to ensure
    compatibility with the two most recent versions of Python (@karalekas, gh-987).
-   Add support for the `FORKED` gate modifier (@kilimanjaro, gh-989).
-   Deleted the deprecated modules `parameters.py` and `qpu.py` (@karalekas, gh-991).
-   The test suite for pyQuil now runs much faster, by setting the default value
    of the `--use-seed` option for `pytest` to `True` (@karalekas, gh-992).
-   Support non-gate instructions (e.g. `MEASURE`) in `to_latex()` (@notmgsk, gh-975).
-   Test suite has been updated to reduce the use of deprecated features
    (@kilimanjaro, gh-998, gh-1005).
-   Certain tests have been marked as "slow", and are skipped unless
    the `--runslow` option is specified for `pytest` (@kilimanjaro, gh-1001).

### Bugfixes

-   Strength two symmetrization was not correctly producing orthogonal 
    arrays due to erroneous truncation, which has been fixed 
    (@kylegulshen, gh-990).

[v2.11](https://github.com/rigetti/pyquil/compare/v2.10.0...v2.11.0) (September 3, 2019)
----------------------------------------------------------------------------------------

### Announcements

-   PyQuil's changelog has been overhauled and rewritten in Markdown instead of
    RST, and can be found in the top-level directory of the repository as the
    [CHANGELOG.md](CHANGELOG.md)  file (which is the standard for most GitHub
    repositories). However, during the build process, we use `pandoc` to convert
    it back to RST so that it can be included as part of the ReadTheDocs
    documentation [here](https://pyquil.readthedocs.io/en/stable/changes.html)
    (@karalekas, gh-945, gh-973).

### Improvements and Changes

-   Test suite attempts to retry specific tests that fail often. Tests are
    retried only a single time (@notmgsk, gh-951).
-   The `QuantumComputer.run_symmetrized_readout()` method has been
    revamped, and now has options for using more advanced forms of
    readout symmetrization (@joshcombes, gh-919).
-   The ProtoQuil restrictions built in to PyQVM have been removed
    (@ecpeterson, gh-874).
-   Add the ability to query for other memory regions after both QPU and QVM
    runs. This removes a previously unnecessary restriction on the QVM, although
    `ro` remains the only QPU-writeable memory region during Quil execution
    (@ecpeterson, gh-873).
-   Now, running `QuantumComputer.reset()` (and `QuantumComputer.compile()`
    when using the QPU) additionally resets the connection information for
    the underlying `QVM`/`QPU` and `QVMCompiler`/`QPUCompiler` objects,
    which should resolve bugs that arise due to stale clients/connections
    (@karalekas, gh-872).
-   In addition to the simultaneous 1Q RB fidelities contained in device 
    specs prior to this release, there are now 1Q RB fidelities for 
    non-simultaneous gate operation. The names of these fields have been
    changed for clarity, and standard errors for both fidelities have been
    added as well. Finally, deprecation warnings have been added regarding
    the `fCPHASE` and `fBellState` device spec fields, which are no longer
    routinely updated and will be removed in release v2.13 (@jvalery2, gh-968).
-   The NOTICE has been updated to accurately reflect the third-party software
    used in pyQuil (@karalekas, gh-979).
-   PyQuil now sends “modern” ISA payloads to quilc, which must be of version
    \>= `1.10.0`. Check out the details of `get_isa` for information on how to
    specify custom payloads (@ecpeterson, gh-961).

### Bugfixes

-   The `MemoryReference` warnings have been removed from the unit
    tests (@maxKenngott, gh-950).
-   The `merge_programs` function now supports merging programs with
    `DefPermutationGate`, instead of throwing an error, and avoids
    redundant readout declaration (@kylegulshen, gh-971).
-   Remove unsound logic to fill out non-"ro" memory regions when
    targeting a QPU (@notmgsk, gh-982).

[v2.10](https://github.com/rigetti/pyquil/compare/v2.9.1...v2.10.0) (July 31, 2019)
-----------------------------------------------------------------------------------

### Improvements and Changes

-   Rewrote the README, adding a more in-depth overview of the purpose
    of pyQuil as a library, as well as two badges \-- one for PyPI
    downloads and another for the Forest Slack workspace. Also, included
    an example section for how to get started with running a simple Bell
    state program on the QVM (@karalekas, gh-946, gh-949).
-   The test suite for `pyquil.operator_estimation` now has an
    (optional) faster version that uses fixed random seeds instead of
    averaging over several experiments. This can be enabled with the
    `--use-seed` command line option when running `pytest` (@msohaibalam,
    gh-928).
-   Deleted the deprecated modules `job_results.py` and `kraus.py`
    (@karalekas, gh-957).
-   Updated the examples README. Removed an outdated notebook. Updated
    remaining notebooks to use `MemoryReference`, and fix any parts that
    were broken (@notmgsk, gh-820).
-   The `AbstractCompiler.quil_to_native_quil()` function now accepts a
    `protoquil` keyword which tells the compiler to restrict both input
    and output to protoquil (i.e. Quil code executable on a QPU).
    Additionally, the compiler will return a metadata dictionary that
    contains statistics about the compiled program, e.g. its estimated
    QPU runtime. See the compiler docs for more information (@notmgsk,
    gh-940).
-   Updated the QCS and Slack invite links on the `index.rst` docs page
    (@starktech23, gh-965).
-   Provided example code for reading out the QPU runtime estimation for
    a program (@notmgsk, gh-963).

### Bugfixes

-   `unitary_tools.lifted_gate()` was not properly handling modifiers
    such as `DAGGER` and `CONTROLLED` (@kylegulshen, gh-931).
-   Fixed warnings raised by Sphinx when building the documentation
    (@appleby, gh-929).

[v2.9.1](https://github.com/rigetti/pyquil/compare/v2.9.0...v2.9.1) (June 28, 2019)
-----------------------------------------------------------------------------------

### Bugfixes

-   Relaxed the requirement for a quilc server to exist when users of
    the `QuantumComputer` object only want to do simulation work with a
    `QVM` or `pyQVM` backend (@karalekas, gh-934).

[v2.9](https://github.com/rigetti/pyquil/compare/v2.8.0...v2.9.0) (June 25, 2019)
---------------------------------------------------------------------------------

### Announcements

-   PyQuil now has a [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md),
    which contains a checklist of things that must be completed (if
    applicable) before a PR can be merged (@karalekas, gh-921).

### Improvements and Changes

-   Removed a bunch of logic around creating inverse gates from
    user-defined gates in `Program.dagger()` in favor of a simpler call
    to `Gate.dagger()` (@notmgsk, gh-887).
-   The `RESET` instruction now works correctly with `QubitPlaceholder`
    objects and the `address_qubits` function (@jclapis, gh-910).
-   `ReferenceDensitySimulator` can now have a state that is persistent
    between rounds of `run` or `run_and_measure` (@joshcombes, gh-920).

### Bugfixes

-   Small negative probabilities were causing
    `ReferenceDensitySimulator` to fail (@joshcombes, gh-908).
-   The `dagger` function was incorrectly dropping gate modifiers like
    `CONTROLLED` (@jclapis, gh-914).
-   Negative numbers in classical instruction arguments were not being
    parsed (@notmgsk, gh-917).
-   Inline math rendering was not working correctly in `intro.rst`
    (@appleby, gh-927).

Thanks to community member @jclapis for the contributions to this
release!

[v2.8](https://github.com/rigetti/pyquil/compare/v2.7.2...v2.8.0) (May 20, 2019)
--------------------------------------------------------------------------------

### Improvements and Changes

-   PyQuil now verifies that you are using the correct version of the
    QVM and quilc (@karalekas, gh-913).
-   Added support for defining permutation gates for use with the latest
    version of quilc (@notmgsk, gh-891).
-   The rpcq dependency requirement has been raised to v2.5.1 (@notmgsk,
    gh-911).
-   Added a note about the QVM's compilation mode to the documentation
    (@stylewarning, gh-900).
-   Some measure_observables params now have the `Optional` type
    specification (@msohaibalam, gh-903).

### Bugfixes

-   Preserve modifiers during `address_qubits` (@notmgsk, gh-907).

[v2.7.2](https://github.com/rigetti/pyquil/compare/v2.7.1...v2.7.2) (May 3, 2019)
---------------------------------------------------------------------------------

### Bugfixes

-   An additional backwards-incompatible change from gh-870 snuck
    through 2.7.1, and is addressed in this patch release
    (@karalekas, gh-901).

[v2.7.1](https://github.com/rigetti/pyquil/compare/v2.7.0...v2.7.1) (April 30, 2019)
------------------------------------------------------------------------------------

### Bugfixes

-   The changes to operator estimation (gh-870, gh-896) were not made in
    a backwards-compatible fashion, and therefore this patch release
    aims to remedy that. Going forward, there will be much more
    stringent requirements around backwards compatibility and
    deprecation (@karalekas, gh-899).

[v2.7](https://github.com/rigetti/pyquil/compare/v2.6.0...v2.7.0) (April 29, 2019)
----------------------------------------------------------------------------------

### Improvements and Changes

-   Standard deviation -\> standard error in operator estimation
    (@msohaibalam, gh-870).
-   Update what pyQuil expects from quilc in terms of rewiring pragmas
    \-- they are now comments rather than distinct instructions
    (@ecpeterson, gh-878).
-   Allow users to deprioritize QPU jobs \-- mostly a Rigetti-internal
    feature (@jvalery2, gh-877).
-   Remove the `qubits` field from the `TomographyExperiment` dataclass
    (@msohaibalam, gh-896).

### Bugfixes

-   Ensure that shots aren\'t lost when passing a `Program` through
    `address_qubits` (@notmgsk, gh-895).
-   Fixed the `conda` install command in the README (@seandiscovery,
    gh-890).

[v2.6](https://github.com/rigetti/pyquil/compare/v2.5.2...v2.6.0) (March 29, 2019)
----------------------------------------------------------------------------------

### Improvements and Changes

-   Added a CODEOWNERS file for default reviewers (@karalekas, gh-855).
-   Bifurcated the `QPUCompiler` endpoint parameter into two \--
    `quilc_endpoint` and `qpu_compiler_endpoint` \-- to reflect changes
    in Quantum Cloud Services (@karalekas, gh-856).
-   Clarified documentation around the DELAY pragma (@willzeng, gh-862).
-   Added information about the `local_qvm` context manager to the
    getting started documentation (@willzeng, gh-851).
-   Added strict version lower bounds on the rpcq and networkx
    dependencies (@notmgsk, gh-828).
-   A slice of a `Program` object now returns a `Program` object
    (@notmgsk, gh-848).

### Bugfixes

-   Added a non-None default timeout to the `QVMCompiler` object
    and the `get_benchmarker` function (@karalekas, gh-850, gh-854).
-   Fixed the docstring for the `apply_clifford_to_pauli` function
    (@kylegulshen, gh-836).
-   Allowed the `apply_clifford_to_pauli` function to now work with the
    Identity as input (@msohaibalam, gh-849).
-   Updated a stale link to the Rigetti Forest Slack workspace
    (@karalekas, gh-860).
-   Fixed a notation typo in the documentation for noise (@willzeng,
    gh-861).
-   An `IndexError` is now raised when trying to access an
    out-of-bounds entry in a `MemoryReference` (@notmgsk, gh-819).
-   Added a check to ensure that `measure_observables` takes as many
    shots as requested (@marcusps, gh-846).

Special thanks to @willzeng for all the contributions this release!

v2.5 (March 6, 2019)
--------------------

### Improvements and Changes

-   PyQuil\'s Gate objects now expose `.controlled(q)` and `.dagger()`
    modifiers, which turn a gate respectively into its controlled
    variant, conditional on the qubit `q`, or into its inverse.
-   The operator estimation suite\'s `measure_observables` method now
    exposes a `readout_symmetrize` argument, which helps mitigate a
    machine\'s fidelity asymmetry between recognizing a qubit in the
    ground state versus the excited state.
-   The `MEASURE` instruction in pyQuil now has a *mandatory* second
    argument. Previously, the second argument could be omitted to induce
    \"measurement for effect\", without storing the readout result to a
    classical register, but users found this to be a common source of
    accidental error and a generally rude surprise. To ensure the user
    really intends to measure only for effect, we now require that they
    supply an explicit `None` as the second argument.

### Bugfixes

-   Some stale tests have been brought into the modern era.

v2.4 (February 14, 2019)
------------------------

### Announcements

-   The Quil Compiler ([quilc](https://github.com/rigetti/quilc)) and
    the Quantum Virtual Machine
    ([QVM](https://github.com/rigetti/quilc)), which are part of the
    Forest SDK, have been open sourced! In addition to downloading the
    binaries, you can now build these applications locally from source,
    or run them via the Docker images
    [rigetti/quilc](https://hub.docker.com/r/rigetti/quilc) and
    [rigetti/qvm](https://hub.docker.com/r/rigetti/qvm). These Docker
    images are now used as the `services` in the GitLab CI build plan
    YAML (gh-792, gh-794, gh-795).

### Improvements and Changes

-   The `WavefunctionSimulator` now supports the use of parametric Quil
    programs, via the `memory_map` parameter for its various methods
    (gh-787).
-   Operator estimation data structures introduced in **v2.2** have
    changed. Previously, `ExperimentSettings` had two members:
    `in_operator` and `out_operator`. The `out_operator` is unchanged,
    but `in_operator` has been renamed to `in_state` and its data type
    is now `TensorProductState` instead of `PauliTerm`. It was always an
    abuse of notation to interpret pauli operators as defining initial
    states. Analogous to the Pauli helper functions sI, sX, sY, and sZ,
    `TensorProductState` objects are constructed by multiplying together
    terms generated by the helper functions plusX, minusX, plusY,
    minusY, plusZ, and minusZ. This functionality enables process
    tomography and process DFE (gh-770).
-   Operator estimation now offers a \"greedy\" method for grouping
    tomography-like experiments that share a natural tensor product
    basis (ntpb), as an alternative to the clique cover version
    (gh-754).
-   The `quilc` endpoint for rewriting Quil parameter arithmetic has
    been changed from `resolve_gate_parameter_arithmetic` to
    `rewrite_arithmetic` (gh-802).
-   The difference between ProtoQuil and QPU-supported Quil is now
    better defined (gh-798).

### Bugfixes

-   Resolved an issue with post-gate noise in the pyQVM (gh-801).
-   A `TypeError` with a useful error message is now raised when a
    `Program` object is run on a QPU-backed `QuantumComputer`, rather
    than a confusing `AttributeError` (gh-799).

v2.3 (January 28, 2019)
-----------------------

PyQuil 2.3 is the latest release of pyQuil, Rigetti\'s toolkit for
constructing and running quantum programs. A major new feature is the
release of a new suite of simulators:

-   We\'re proud to introduce the first iteration of a Python-based
    quantum virtual machine (QVM) called PyQVM. This QVM is completely
    contained within pyQuil and does not need any external dependencies.
    Try using it with `get_qc("9q-square-pyqvm")` or explore the
    `pyquil.pyqvm.PyQVM` object directly. Under-the-hood, there are
    three quantum simulator backends:
    -   `ReferenceWavefunctionSimulator` uses standard matrix-vector
        multiplication to evolve a statevector. This includes a suite of
        tools in `pyquil.unitary_tools` for dealing with unitary
        matrices.
    -   `NumpyWavefunctionSimulator` uses numpy\'s tensordot
        functionality to efficiently evolve a statevector. For most
        simulations, performance is quite good.
    -   `ReferenceDensitySimulator` uses matrix-matrix multiplication to
        evolve a density matrix.
-   Matrix representations of Quil standard gates are included in
    `pyquil.gate_matrices` (gh-552).
-   The density simulator has extremely limited support for
    Kraus-operator based noise models. Let us know if you\'re interested
    in contributing more robust noise-model support.
-   This functionality should be considered experimental and may undergo
    minor API changes.

### Important changes to note

-   Quil math functions (like COS, SIN, \...) used to be ambiguous with
    respect to case sensitivity. They are now case-sensitive and should
    be uppercase (gh-774).
-   In the next release of pyQuil, communication with quilc will happen
    exclusively via the rpcq protocol. `LocalQVMCompiler` and
    `LocalBenchmarkConnection` will be removed in favor of a unified
    `QVMCompiler` and `BenchmarkConnection`. This change should be
    transparent if you use `get_qc` and `get_benchmarker`, respectively.
    In anticipation of this change we recommend that you upgrade your
    version of quilc to 1.3, released Jan 30, 2019 (gh-730).
-   When using a paramaterized gate, the QPU control electronics only
    allowed multiplying parameters by powers of two. If you only ever
    multiply a parameter by the same constant, this isn\'t too much of a
    problem because you can fold the multiplicative constant into the
    definition of the parameter. However, if you are multiplying the
    same variable (e.g. `gamma` in QAOA) by different constants (e.g.
    weighted maxcut edge weights) it doesn\'t work. PyQuil will now
    transparently handle the latter case by expanding to a vector of
    parameters with the constants folded in, allowing you to multiply
    variables by whatever you want (gh-707).

### Bug fixes and improvements

-   The CZ gate fidelity metric available in the Specs object now has
    its associated standard error, which is accessible from the method
    `Specs.fCZ_std_errs` (gh-751).
-   Operator estimation code now correctly handles identity terms with
    coefficients. Previously, it would always estimate these terms as
    1.0 (gh-758).
-   Operator estimation results include the total number of counts
    (shots) taken.
-   Operator estimation JSON serialization uses utf-8. Please let us
    know if this causes problems (gh-769).
-   The example quantum die program now can roll dice that are not
    powers of two (gh-749).
-   The teleportation and Meyer penny game examples had a syntax error
    (gh-778, gh-772).
-   When running on the QPU, you could get into trouble if the QPU name
    passed to `get_qc` did not match the lattice you booked. This is now
    validated (gh-771).

We extend thanks to community member @estamm12 for their contribution to
this release.

v2.2 (January 4, 2019)
----------------------

PyQuil 2.2 is the latest release of pyQuil, Rigetti\'s toolkit for
constructing and running quantum programs. Bug fixes and improvements
include:

-   `pauli.is_zero` and `paulis.is_identity` would sometimes return
    erroneous answers (gh-710).
-   Parameter expressions involving addition and subtraction are now
    converted to Quil with spaces around the operators, e.g. `theta + 2`
    instead of `theta+2`. This disambiguates subtracting two parameters,
    e.g. `alpha - beta` is not one variable named `alpha-beta` (gh-743).
-   T1 is accounted for in T2 noise models (gh-745).
-   Documentation improvements (gh-723, gh-719, gh-720, gh-728, gh-732,
    gh-742).
-   Support for PNG generation of circuit diagrams via LaTeX (gh-745).
-   We\'ve started transitioning to using Gitlab as our continuous
    integration provider for pyQuil (gh-741, gh-752).

This release includes a new module for facilitating the estimation of
quantum observables/operators (gh-682). First-class support for
estimating observables should make it easier to express near-term
algorithms. This release includes:

-   data structures for expressing tomography-like experiments and their
    results
-   grouping of experiment settings that can be simultaneously estimated
-   functionality to executing a tomography-like experiment on a quantum
    computer

Please look forward to more features and polish in future releases.
Don\'t hesitate to submit feedback or suggestions as GitHub issues.

We extend thanks to community member @petterwittek for their contribution
to this release.

Bugfix release 2.2.1 was released January 11 to maintain compatibility
with the latest version of the quilc compiler (gh-759).

v2.1 (November 30, 2018)
------------------------

PyQuil 2.1 is an incremental release of pyQuil, Rigetti\'s toolkit for
constructing and running quantum programs. Changes include:

-   Major documentation improvements.
-   `QuantumComputer.run()` accepts an optional `memory_map` parameter
    to facilitate running parametric executables (gh-657).
-   `QuantumComputer.reset()` will reset the state of a QAM to recover
    from an error condition (gh-703).
-   Bug fixes (gh-674, gh-696).
-   Quil parser improvements (gh-689, gh-685).
-   Optional interleaver argument when generating RB sequences (gh-673).
-   Our GitHub organization name has changed from `rigetticomputing` to
    `rigetti` (gh-713).

v2.0 (November 1, 2018)
-----------------------

PyQuil 2.0 is a major release of pyQuil, Rigetti\'s toolkit for
constructing and running quantum programs. This release contains many
major changes including:

1.  The introduction of [Quantum Cloud
    Services](https://www.rigetti.com/qcs). Access Rigetti\'s QPUs from
    co-located classical compute resources for minimal latency. The web
    API for running QVM and QPU jobs has been deprecated and cannot be
    accessed with pyQuil 2.0
2.  Advances in classical control systems and compilation allowing the
    pre-compilation of parametric binary executables for rapid hybrid
    algorithm iteration.
3.  Changes to Quil\-\--our quantum instruction language\-\--to provide
    easier ways of interacting with classical memory.

The new QCS access model and features will allow you to execute hybrid
quantum algorithms several orders of magnitude (!) faster than the
previous web endpoint. However, to fully exploit these speed increases
you must update your programs to use the latest pyQuil features and
APIs. Please read the documentation on what is [New in Forest
2](https://pyquil.readthedocs.io/en/stable/migration4.html) for a
comprehensive migration guide.

An incomplete list of significant changes:

-   Python 2 is no longer supported. Please use Python 3.6+
-   Parametric gates are now normal functions. You can no longer write
    `RX(pi/2)(0)` to get a Quil `RX(pi/2) 0` instruction. Just use
    `RX(pi/2, 0)`.
-   Gates support keyword arguments, so you can write
    `RX(angle=pi/2, qubit=0)`.
-   All `async` methods have been removed from `QVMConnection` and
    `QVMConnection` is deprecated. `QPUConnection` has been removed in
    accordance with the QCS access model. Use `pyquil.get_qc` as the
    primary means of interacting with the QVM or QPU.
-   `WavefunctionSimulator` allows unfettered access to wavefunction
    properties and routines. These methods and properties previously
    lived on `QVMConnection` and have been deprecated there.
-   Classical memory in Quil must be declared with a name and type.
    Please read [New in Forest
    2](https://pyquil.readthedocs.io/en/stable/migration4.html) for
    more.
-   Compilation has changed. There are now different `Compiler` objects
    that target either the QPU or QVM. You **must** explicitly compile
    your programs to run on a QPU or a realistic QVM.

Version 2.0.1 was released on November 9, 2018 and includes
documentation changes only. This release is only available as a git tag.
We have not pushed a new package to PyPI.

v1.9 (June 6, 2018)
-------------------

We're happy to announce the release of pyQuil 1.9. PyQuil is Rigetti's
toolkit for constructing and running quantum programs. This release is
the latest in our series of regular releases, and it's filled with
convenience features, enhancements, bug fixes, and documentation
improvements.

Special thanks to community members @sethuiyer, @vtomole, @rht, @akarazeev,
@ejdanderson, @markf94, @playadust, and @kadora626 for contributing to this
release!

### Qubit placeholders

One of the focuses of this release is a re-worked concept of \"Qubit
Placeholders\". These are logical qubits that can be used to construct
programs. Now, a program containing qubit placeholders must be
\"addressed\" prior to running on a QPU or QVM. The addressing stage
involves mapping each qubit placeholder to a physical qubit (represented
as an integer). For example, if you have a 3 qubit circuit that you want
to run on different sections of the Agave chip, you now can prepare one
Program and address it to many different subgraphs of the chip topology.
Check out the `QubitPlaceholder` example notebook for more.

To support this idea, we\'ve refactored parts of Pyquil to remove the
assumption that qubits can be \"sorted\". While true for integer qubit
labels, this probably isn\'t true in general. A notable change can be
found in the construction of a `PauliSum`: now terms will stay in the
order they were constructed.

-   `PauliTerm` now remembers the order of its operations. `sX(1)*sZ(2)`
    will compile to different Quil code than `sZ(2)*sX(1)`, although the
    terms will still be equal according to the `__eq__` method. During
    `PauliSum` combination of like terms, a warning will be emitted if
    two terms are combined that have different orders of operation.
-   `PauliTerm.id()` takes an optional argument `sort_ops` which
    defaults to True for backwards compatibility. However, this function
    should not be used for comparing term-type like it has been used
    previously. Use `PauliTerm.operations_as_set()` instead. In the
    future, `sort_ops` will default to False and will eventually be
    removed.
-   `Program.alloc()` has been deprecated. Please instantiate
    `QubitPlaceholder()` directly or request a \"register\" (list) of
    `n` placeholders by using the class constructor
    `QubitPlaceholder.register(n)`.
-   Programs must contain either (1) all instantiated qubits with
    integer indexes or (2) all placeholder qubits of type
    `QubitPlaceholder`. We have found that most users use
    (1) but (2) will become useful with larger and more diverse devices.
-   Programs that contain qubit placeholders must be **explicitly
    addressed** prior to execution. Previously, qubits would be assigned
    \"under the hood\" to integers 0\...N. Now, you must use
    `address_qubits` which returns a new program with all qubits indexed
    depending on the `qubit_mapping` argument. The original program is
    unaffected and can be \"readdressed\" multiple times.
-   `PauliTerm` can now accept `QubitPlaceholder` in addition to
    integers.
-   `QubitPlaceholder` is no longer a subclass of `Qubit`.
    `LabelPlaceholder` is no longer a subclass of `Label`.
-   `QuilAtom` subclasses\' hash functions have changed.

### Randomized benchmarking sequence generation

Pyquil now includes support for performing a simple benchmarking routine
- randomized benchmarking. There is a new method in the
`CompilerConnection` that will return sequences of pyquil programs,
corresponding to elements of the Clifford group. These programs are
uniformly randomly sampled, and have the property that they compose to
the identity. When concatenated and run as one program, these programs
can be used in a procedure called randomized benchmarking to gain
insight about the fidelity of operations on a QPU.

In addition, the `CompilerConnection` has another new method,
`apply_clifford_to_pauli` which conjugates `PauliTerms` by `Program`
that are composed of Clifford gates. That is to say, given a circuit C,
that contains only gates corresponding to elements of the Clifford
group, and a tensor product of elements P, from the Pauli group, this
method will compute `$PCP^{dagger}$` Such a procedure can be used in
various ways. An example is predicting the effect a Clifford circuit
will have on an input state modeled as a density matrix, which can be
written as a sum of Pauli matrices.

### Ease of Use

This release includes some quality-of-life improvements such as the
ability to initialize programs with generator expressions, sensible
defaults for `Program.measure_all`, and sensible defaults for
`classical_addresses` in `run` methods.

-   `Program` can be initiated with a generator expression.
-   `Program.measure_all` (with no arguments) will measure all qubits in
    a program.
-   `classical_addresses` is now optional in QVM and QPU `run` methods.
    By default, any classical addresses targeted by `MEASURE` will be
    returned.
-   `QVMConnection.pauli_expectation` accepts `PauliSum` as arguments.
    This offers a more sensible API compared to
    `QVMConnection.expectation`.
-   pyQuil will now retry jobs every 10 seconds if the QPU is re-tuning.
-   `CompilerConnection.compile` now takes an optional argument `isa`
    that allows per-compilation specification of the target ISA.
-   An empty program will trigger an exception if you try to run it.

### Supported versions of Python

We strongly support using Python 3 with Pyquil. Although this release
works with Python 2, we are dropping official support for this legacy
language and moving to community support for Python 2. The next major
release of Pyquil will introduce Python 3.5+ only features and will no
longer work without modification for Python 2.

### Bug fixes

-   `shift_quantum_gates` has been removed. Users who relied on this
    functionality should use `QubitPlaceholder` and `address_qubits` to
    achieve the same result. Users should also double-check data
    resulting from use of this function as there were several edge cases
    which would cause the shift to be applied incorrectly resulting in
    badly-addressed qubits.
-   Slightly perturbed angles when performing RX gates under a Kraus
    noise model could result in incorrect behavior.
-   The quantum die example returned incorrect values when `n = 2^m`.

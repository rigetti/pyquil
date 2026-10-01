

## Known breaking changes 
* mutable types are now unhashable (technically a bugfix, but might break buggy code)
* hashable types are now immutable (in particular, impacts `Gate`)
* `Expression.__str__()`/`to_quil()` formatting changed cosmetically to match `quil-rs`'s native
  `Display` implementation: infix operators (`+`, `*`, `^`, `/`) are no longer surrounded by spaces
  (only `-` is, to disambiguate from hyphenated identifiers), extra disambiguating parentheses are
  now inserted around nested binary operations, and built-in function names (`SIN`, `COS`, `EXP`,
  etc.) are now rendered lowercase (`sin`, `cos`, `exp`). For example, `str(Parameter("x") +
  Parameter("y"))` is now `"%x+%y"` instead of the old `"%x + %y"`. This affects anything that
  calls `str()`/`to_quil()` on an `Expression` (e.g. displaying/serializing parametric gates). A
  faithful pyquil-side workaround was evaluated and rejected: a naive string-level transform
  (inserting spaces around a bare `+`) cannot safely distinguish an infix-expression `+` from a `+`
  that appears inside a complex-number literal (e.g. `"0.707+0.707i"`), so a correct fix would
  require fully reimplementing `quil-rs`'s recursive expression-formatting logic in Python. Decided
  to accept this as a breaking (cosmetic) change rather than carry that duplicated logic.
  (`test/unit/test_parameters.py::test_expression_to_string` updated to match.)
* `isinstance` checks to `Halt`, `Nop`, and `Wait` now fail because these are singleton instances
    of type `HaltType`, `NopType`, and `WaitType` respectively.
    `isinstance(some_instruction, {Halt | Nop | Wait})` should be replaced with
    `some_instruction is {Halt | Nop | Wait}` as appropriate
    (technically, this could be replaced with `isinstance(some_instruction, {Halt| Nop| Wait}Type)`, 
    but that's as clunky as `isinstance(foo, NoneType)` vs the recommended `foo is None`).

* `pyquil.external.rpcq.Operator._dict()` now always emits `duration`/`fidelity` keys (defaulting to
  `0.0` when unset) instead of omitting them when `None`. `qcs_sdk`'s
  `compiler.quilc.TargetDevice.from_json()` now requires both fields unconditionally on every
  per-gate ISA entry (they used to be optional), so omitting them caused
  `QuilcError: missing field 'duration'` whenever an `ISA`/`CompilerISA` built from
  `AbstractCompiler`/`QCSCompilerISA` was round-tripped through `TargetDevice.from_json`. Not
  considered user-facing breaking (the emitted JSON is strictly more complete than before; no pyquil
  test or code path asserted on the literal absence of these keys), just documented here as a
  schema-drift accommodation.
* `pyquil.quilatom.BinaryExp` was repointed from `InfixExpression` (the raw data payload used by
  `Expression.Infix`) to `Expression.Infix` itself (the enum variant), since every runtime binary
  expression value (e.g. the result of `MemoryReference("a") - MemoryReference("b")`) is actually an
  `Expression.Infix` instance, not a bare `InfixExpression` -- so `isinstance(x, BinaryExp)` always
  returned `False` for real values under the old alias. This was a latent bug (the alias existed but
  could never match anything), not an accepted breaking change, since nothing could have relied on
  the old (non-functional) behavior. Callers must now unwrap the underlying `InfixExpression` via
  `expr._0` to access `.left`/`.operator`/`.right` (related to the pending `quil-rs` ergonomics gap
  noted for `Function`, just above). `test/unit/test_quilbase.py::TestGate`'s `program` fixture was
  updated to match (it also needed to unwrap `Expression.Address` on each side of the infix
  expression to compare against `MemoryReference`, for the same reason).

* `CircuitDefinition.qubit_variables` is a `list[str]`, not `list[FormalArgument]`.
  Callers must now wrap each name in `Qubit(name)` or `FormalArgument` to get a variable object.

## Confirmed `quil-rs` Python-binding gaps (not pyquil-side issues; flagged for `quil-rs` work)

* `ExternParameterType.VariableLengthVector` is missing from the PyO3 bindings, so pyquil cannot
  construct/compare `DefExtern`/`ExternSignatureMap` parameter types representing variable-length
  vectors. One test in `test/unit/test_qpu_randomized_compiling.py` was skipped pending this.
* Several native instruction types (`Jump`, `JumpWhen`, `JumpUnless`, `Label`) do not support
  Python-side subclassing in this build (`class Foo(Jump): pass` silently discards the subclass and
  returns a plain `Jump` instance), unlike `quil-rs`'s `master`/future build, which clearly supports
  `#[pyclass(subclass)]` for these types. This blocked porting `origin/master`'s `pyquil/quilbase.py`
  patterns for these classes; pyquil instead keeps them as plain native re-exports and fixes
  individual call sites (e.g. unwrapping `.target`) or uses method-level monkeypatching where subclassing
  would otherwise have been used.

## Known out-of-scope / environment-dependent test failures (not pyquil bugs)

* `test/unit/test_api.py::test_quil_to_native_quil` is marked `xfail` (non-strict). The locally
  configured `quilc` (invoked via this development environment's local `qcs-sdk-rust` build, per
  `pyproject.toml`'s `qcs-sdk-python` path dependency) compiles the test's `BELL_STATE` program
  (`H(0); CNOT(0, 1)`) to a native-quil `RZ`/`RX`/`CZ` sequence that was verified, by direct
  simulation, to **not** actually reproduce a Bell state (its output on `|00>` is a separable state,
  not an entangled one) -- a different circuit entirely, not just a different-but-equivalent
  decomposition or a global/local phase difference. Every individual gate matrix involved (`RZ`,
  `RX`, `CZ`) was checked independently against `pyquil.simulation.matrices`'s standard definitions
  and matches; `program_unitary`/`lifted_gate` were independently confirmed correct by reproducing
  the expected Bell state for the original, uncompiled `BELL_STATE` program. This rules out a
  pyquil-side simulation bug, leaving the locally-built `quilc`/`qcs-sdk-rust` compilation pipeline
  as the suspect. Not investigated further per the user's request not to access the `qcs-sdk-rust`
  repository directly; left `xfail` so it surfaces again (and should be re-triaged) once run against
  a stable/released `qcs-sdk-python`.

* Several `test/unit/test_quantum_computer.py` tests that perform real round-trips against a live
  QVM server (`test_qc_calibration_1q`/`_2q`, `test_qc_joint_expectation`, `test_qc_joint_calibration`,
  `test_qc_expectation_on_qvm`, `test_undeclared_memory_region`, and the pre-existing `test_run`) are
  intermittently flaky when run as part of the full suite, failing with
  `qcs_sdk._qcs_sdk.qvm.QVMError: Could not communicate with QVM at http://127.0.0.1:5000`, even though
  the QVM server remains reachable throughout (confirmed via direct `curl`/manual `qvm.run()` calls,
  including repeated runs and multiple fresh `QVM`/`QVMClient` instances in a standalone script, all of
  which succeed). The failures only reproduce when multiple of these tests run together in the same
  pytest session using the session-scoped `qvm_client`/`client_configuration` fixtures (never in
  isolation), pointing to some shared async-runtime/connection state in the local `qcs-sdk-rust`
  dev build (loaded as a `file://` path dependency, not a released version -- see above) rather than a
  pyquil-side bug. Not investigated further per the user's request not to access the `qcs-sdk-rust`
  repository directly. These tests are left as-is (no `xfail`, matching how the pre-existing `test_run`
  flakiness has always been handled) since they do pass correctly in isolation and the failure mode is
  purely an artifact of this development environment.

* Restored `QuantumComputer.calibrate()` and `QuantumComputer.run_experiment()` (and their
  `test_qc_expectation*`/`test_qc_calibration*`/`test_qc_joint*` tests), which had been prematurely
  removed on this branch under the mistaken belief that they depended on the since-removed
  `pyquil.experiment` module. `pyquil.experiment` is only deprecated-for-v5-removal (per the user's
  policy), not yet removed, and `origin/master` still keeps these `@deprecated`-but-functional methods
  and their tests intact, so they were restored verbatim (using `TYPE_CHECKING`-guarded imports of
  `Experiment`/`ExperimentResult` for type hints) to match upstream. Not a breaking change relative to
  pyquil v4's intended deprecation timeline.

* Fixed several latent `isinstance(x, DeprecatedShimFunctionName)` bugs uncovered while restoring the
  tests above (same shim-vs-type pattern as the earlier `Halt`/`Nop`/`Wait`/`JumpTarget` fixes):
  `ResetQubit` (a deprecated function returning a plain `Reset`) was used as an `isinstance` target in
  `pyquil/experiment/_main.py` (2 call sites -- this one was an active, crashing bug blocking the
  restored `calibrate`/`run_experiment` tests above), `pyquil/noise/_noise_model.py` (type annotations
  only, not a runtime crash), and `pyquil/latex/_diagram.py` (a real but previously-untriggered crash in
  instruction-reordering logic). Likewise `DefGate` (a deprecated function returning a `GateDefinition`)
  was used as an `isinstance` target in `test/unit/test_quilbase.py`'s `TestDefGate.test_copy`/
  `test_pickle` (currently-skipped tests, so latent) and in `test/benchmarks/test_simulators.py`. All
  fixed to check against the real underlying type (`Reset`/`GateDefinition` respectively). Pure bug
  fixes, not breaking changes.

* Reduced internal use of deprecated shim functions in non-deprecated pyquil modules, so that
  pyquil's own code no longer emits `PyQuilDeprecationWarning`s against itself (users calling these
  non-deprecated public APIs should no longer see unrelated deprecation warnings bubble up):
  - `Program.defgate()` called the deprecated `DefGate(...)` function internally; factored the
    shared validation/construction logic out into a private `pyquil.quilbase._defgate_from_matrix`
    helper that both `DefGate` (deprecated, public) and `Program.defgate()` (non-deprecated, public)
    now call. Also fixed `Program.defined_gates`'s return-type annotation, which incorrectly
    referenced `DefGate` (a function) instead of the actual `GateDefinition` type it returns.
  - `pyquil/_qpu/randomized_compiling.py` called the deprecated `ClassicalAdd`/`ClassicalGreaterEqual`/
    `ClassicalShiftRight` functions internally; replaced with direct construction of their underlying
    types (`ArithmeticBinaryOp(ArithmeticOperator.ADD, ...)`, `Comparison(ComparisonOperator.
    GREATER_THAN_OR_EQUAL, ...)`, `BinaryLogic(BinaryOperator.SHR, ...)`).
  - `pyquil/_qpu/_classical_computations.py` called the deprecated `FenceAll()` function internally;
    replaced with `Fence([])` directly.
  All of these are internal-only call-site changes (no public API surface changed) and not breaking.

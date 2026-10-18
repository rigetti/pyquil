

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
  `expr._0` to access `.left`/`.operator`/`.right`.
  `test/unit/test_quilbase.py::TestGate`'s `program` fixture was updated to match (it also needed to
  unwrap `Expression.Address` on each side of the infix expression to compare against
  `MemoryReference`, for the same reason).

* `CircuitDefinition.qubit_variables` is a `list[str]`, not `list[FormalArgument]`.
  Callers must now wrap each name in `Qubit(name)` or `FormalArgument` to get a variable object.

## Deprecated names in `pyquil.quilbase` (backwards-compatibility layer)

Many pyQuil v4 names were thin aliases for, or "function classes" (`def`s that pretended to be
`class`es) around, native `quil.instructions` types. They are all kept importable, but now emit a
`DeprecationWarning` that names the replacement. They are all implemented with the
`DeprecatedSubclass` metaclass (plus `@deprecated`, so static tools see the deprecation) in
`pyquil/quilbase.py`:

* Calling the name warns and returns an instance of the native type (never an instance of the
  deprecated name itself), so `type(Declare(...)) is Declaration`.
* `isinstance(x, OldName)` and `issubclass(cls, OldName)` also warn, and are answered by checking
  against the native type.
* Names that were really partial applications of a native type (e.g. `ClassicalNeg`,
  `ClassicalAnd`, `ClassicalAdd`, `ClassicalEqual`, `DelayFrames`, `DelayQubits`, `FenceAll`,
  `ResetQubit`, `DefGate`, `DefPermutationGate`, `DefGateByPaulis`) additionally check the
  discriminating property in `isinstance` (the `operator`, `has_frames()`, `qubits`/`qubit`, or
  `specification` variant), so `isinstance(UnaryLogic(UnaryOperator.NOT, r), ClassicalNeg)` is
  `False`. This is done via the `matches=` argument of `DeprecatedSubclass`; a `__new__` defined on
  the deprecated class is used as the constructor.

| Deprecated name | Replacement |
|---|---|
| `ArithmeticBinaryOp` | `Arithmetic` |
| `LogicalBinaryOp` | `BinaryLogic` |
| `UnaryClassicalInstruction` | `UnaryLogic` |
| `ClassicalComparison` | `Comparison` |
| `ClassicalConvert` / `ClassicalExchange` / `ClasssicalExchange` (sic) | `Convert` / `Exchange` |
| `ClassicalLoad` / `ClassicalMove` / `ClassicalStore` | `Load` / `Move` / `Store` |
| `Declare` | `Declaration` |
| `DefCalibration` / `DefMeasureCalibration` | `CalibrationDefinition` / `MeasureCalibrationDefinition` |
| `DefCircuit` / `DefFrame` / `DefWaveform` | `CircuitDefinition` / `FrameDefinition` / `WaveformDefinition` |
| `Frame` | `FrameIdentifier` |
| `JumpTarget` | `Label` (the instruction; not `pyquil.quilatom.Label`, which is a target) |
| `ClassicalNeg` / `ClassicalNot` | `UnaryLogic(UnaryOperator.NEG / NOT, ...)` |
| `ClassicalAnd` / `ClassicalInclusiveOr` / `ClassicalExclusiveOr` / `ClassicalArithmeticShiftRight` / `ClassicalShiftLeft` / `ClassicalShiftRight` | `BinaryLogic(BinaryOperator.AND / IOR / XOR / ASHR / SHL / SHR, ...)` |
| `ClassicalAdd` / `ClassicalSub` / `ClassicalMul` / `ClassicalDiv` | `Arithmetic(ArithmeticOperator.ADD / SUBTRACT / MULTIPLY / DIVIDE, ...)` |
| `ClassicalEqual` / `ClassicalLessThan` / `ClassicalLessEqual` / `ClassicalGreaterThan` / `ClassicalGreaterEqual` | `Comparison(ComparisonOperator.EQUAL / LESS_THAN / LESS_THAN_OR_EQUAL / GREATER_THAN / GREATER_THAN_OR_EQUAL, ...)` |
| `DelayFrames` / `DelayQubits` | `Delay.from_frames(duration, frames)` / `Delay(duration, qubits)` |
| `FenceAll` | `Fence([])` |
| `ResetQubit` | `Reset(qubit)` |
| `DefGate` / `DefPermutationGate` / `DefGateByPaulis` | `GateDefinition(name, parameters, GateSpecification.Matrix / Permutation / PauliSum(...))` |

(`pyquil.quilatom.Frame` is a separate, still-supported alias for `FrameIdentifier`.)

### Internal use of the deprecated names removed

pyQuil's own code and unit tests no longer use any of these names, so the only way to see these
warnings is to use the names directly (running the unit tests with
`-W error::DeprecationWarning:pyquil` is a quick check). Call-site changes, all non-breaking:

* `pyquil/_pyqvm.py`, `_qpu/randomized_compiling.py`, `_qpu/_classical_computations.py`,
  `api/_compiler.py`, `gates.py`, `latex/_diagram.py`, `noise/_channels.py`,
  `noise/_legacy_noise.py`, `quil.py`, `simulation/_resolver.py`: switched to the native names in
  imports, annotations, constructors, and `isinstance` checks.
* `JumpTarget(...)` became the `Label` instruction. Where the module also uses
  `pyquil.quilatom.Label` (`quil.py`, `_pyqvm.py`), the instruction is imported as `LabelInstruction`.
  The redundant `Label(...)` wrapping in `randomized_compiling.py` was dropped.
* `FENCE()` returns `Fence([])`; stale docstrings mentioning `DelayFrames`/`FenceAll`/`ResetQubit`
  were corrected.
* Unit tests were migrated the same way, with `DefGate`/`DefPermutationGate` calls rewritten to
  `GateDefinition(..., GateSpecification.Matrix/Permutation(...))`.
  `TestReset::test_deprecated` intentionally still uses `ResetQubit`, since it tests the warning.

Earlier fixes in the same vein (all pure bug fixes): `isinstance` checks against the old function
shims `ResetQubit` (in `pyquil/experiment/_main.py`, `pyquil/noise/_noise_model.py`,
`pyquil/latex/_diagram.py`) and `DefGate` (tests and `test/benchmarks/test_simulators.py`) never
matched or crashed, and now check the real types (`Reset`/`GateDefinition`). `Program.defined_gates`'s
return annotation was corrected to `GateDefinition`.

Not yet migrated: `docs/` still shows `DefGate`, `DefPermutationGate` and `JumpTarget` in examples.

## `quil-rs` Python-binding gaps

Resolved upstream (verified against the current `quil` build):

* `ExternParameterType.VariableLengthVector` is now exposed, so the test in
  `test/unit/test_qpu_randomized_compiling.py` that was skipped for it no longer needs to be.
* `Jump`, `JumpWhen`, `JumpUnless` and `Label` can now be subclassed from Python (the subclass is
  preserved instead of being silently replaced by the base type). The earlier workaround, keeping
  them as plain native re-exports with call-site fixes and method-level monkeypatching, still
  works, so it has not been reworked; subclassing is available if `origin/master`-style patterns for
  these classes are wanted.

Still open:

* `qcs-sdk-python` 0.26.x pins `quil <0.37`, so testing against a `quil` 0.37 release candidate
  requires either a `qcs-sdk-python` build that relaxes the bound (e.g. a git/path dependency) or
  overriding the dependency after install; Poetry has no override mechanism. The local-build
  `poetry.lock`/`pyproject.toml` changes used for this are deliberately uncommitted.

## Known out-of-scope / environment-dependent test failures (not pyquil bugs)

* `test/unit/test_api.py::test_quil_to_native_quil` is marked `xfail` (non-strict) and still
  `xfail`s with the locally built `qcs-sdk-python`. The locally configured `quilc` compiles the
  test's `BELL_STATE` program (`H(0); CNOT(0, 1)`) to a native-quil `RZ`/`RX`/`CZ` sequence that was
  verified, by direct simulation, to **not** reproduce a Bell state (a different circuit, not just
  a different-but-equivalent decomposition). Every gate matrix involved matches
  `pyquil.simulation.matrices`, and `program_unitary`/`lifted_gate` correctly reproduce the Bell
  state for the uncompiled program, which rules out a pyquil-side simulation bug and leaves the
  locally built `quilc`/`qcs-sdk-rust` pipeline as the suspect. Re-triage against a released
  `qcs-sdk-python`.

* Several `test/unit/test_quantum_computer.py` and `test/unit/test_reference_wavefunction.py` tests
  that round-trip against a live QVM server are intermittently flaky when run as part of the full
  suite (especially under `pytest-xdist`), failing with
  `qcs_sdk._qcs_sdk.qvm.QVMError: Could not communicate with QVM at http://127.0.0.1:5000`, even
  though the QVM stays reachable (direct `curl`/standalone `qvm.run()` calls succeed). Failures
  vary from run to run (0 to ~20 tests) and the affected tests pass when run on their own, which
  points at shared async-runtime/connection state in the local `qcs-sdk-rust` dev build (a `file://`
  path dependency) rather than at pyquil. No `xfail` is applied, matching how the pre-existing
  `test_run` flakiness has been handled.

* `QuantumComputer.calibrate()` and `QuantumComputer.run_experiment()` (and their
  `test_qc_expectation*`/`test_qc_calibration*`/`test_qc_joint*` tests) were restored after being
  prematurely removed on this branch. `pyquil.experiment` is only deprecated-for-v5-removal, and
  `origin/master` still keeps these `@deprecated`-but-functional methods and their tests, so they
  were restored verbatim (with `TYPE_CHECKING`-guarded imports of `Experiment`/`ExperimentResult`).
  Not a breaking change relative to the intended deprecation timeline.

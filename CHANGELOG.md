# Changelog

## 4.6.2-rc.2 (2024-02-16)

### Fixes

#### bump qcs-sdk-python version to pull DEFCIRCUIT serialization fix (#1735)

## 4.6.2-rc.1 (2024-02-13)

### Fixes

#### bump qcs-sdk-python version to pull DEFCIRCUIT serialization fix (#1735)

## 4.6.2-rc.0 (2024-02-07)

### Fixes

#### bump qcs-sdk-python version to pull DEFCIRCUIT serialization fix (#1735)

## 4.6.1 (2024-02-06)

### Fixes

#### Compatibility layer will prefer to return DelayFrames, then DelayQubits, then Delay. (#1731)

## 4.6.1-rc.2 (2024-02-06)

### Fixes

#### Compatibility layer will prefer to return DelayFrames, then DelayQubits, then Delay. (#1731)

## 4.6.1-rc.1 (2024-02-06)

### Fixes

#### Compatibility layer will prefer to return DelayFrames, then DelayQubits, then Delay. (#1731)

## 4.6.1-rc.0 (2024-02-05)

### Fixes

#### Compatibility layer will prefer to return DelayFrames, then DelayQubits, then Delay. (#1731)

## 4.6.0 (2024-02-01)

### Features

#### qcs-sdk-python version for libquil support, and document libquil usage (#1698)

#### Add a `with_loop` method to `Program` (#1717)

### Fixes

#### Add deprecated property to DefFrame for CHANNEL-DELAY attribute (#1729)

#### ResetQubit instructions will not be returned as Reset after being inserted into a Program (#1727)

## 4.6.0-rc.4 (2024-02-01)

### Features

#### qcs-sdk-python version for libquil support, and document libquil usage (#1698)

#### Add a `with_loop` method to `Program` (#1717)

### Fixes

#### Add deprecated property to DefFrame for CHANNEL-DELAY attribute (#1729)

#### ResetQubit instructions will not be returned as Reset after being inserted into a Program (#1727)

## 4.6.0-rc.3 (2024-01-30)

### Features

#### qcs-sdk-python version for libquil support, and document libquil usage (#1698)

#### Add a `with_loop` method to `Program` (#1717)

### Fixes

#### Add deprecated property to DefFrame for CHANNEL-DELAY attribute (#1729)

## 4.6.0-rc.2 (2024-01-27)

### Features

#### qcs-sdk-python version for libquil support, and document libquil usage (#1698)

#### Add a `with_loop` method to `Program` (#1717)

## 4.6.0-rc.1 (2024-01-27)

### Features

#### qcs-sdk-python version for libquil support, and document libquil usage (#1698)

## 4.6.0-rc.0 (2024-01-26)

### Features

#### qcs-sdk-python version for libquil support, and document libquil usage (#1698)

## 4.5.0 (2024-01-24)

### Features

#### Add utility for filtering Programs and a method for removing Quil-T instructions (#1718)

### Fixes

#### Arithmetic instructions will not raise an error when used in Programs (#1723)

## 4.5.0-rc.0 (2024-01-18)

### Features

#### Add utility for filtering Programs and a method for removing Quil-T instructions (#1718)

### Fixes

#### Arithmetic instructions will not raise an error when used in Programs (#1723)

## 4.4.1-rc.0 (2024-01-16)

### Fixes

#### Arithmetic instructions will not raise an error when used in Programs (#1723)

## 4.4.0 (2023-12-15)

### Features

#### Add `get_attribute`, `set_attribute` `__getitem__` and `__setitem__` methods to `DefFrame` (#1714)

### Fixes

#### Relax numpy and scipy requirements (#1713)

#### DefFrame no longer attempts to serialize HARDWARE-OBJECT as json. (#1715)

#### Gate instructions specified as tuples no longer error when using a list of parameters. (#1716)

## 4.4.0-rc.1 (2023-12-15)

### Features

#### Add `get_attribute`, `set_attribute` `__getitem__` and `__setitem__` methods to `DefFrame` (#1714)

### Fixes

#### Relax numpy and scipy requirements (#1713)

#### DefFrame no longer attempts to serialize HARDWARE-OBJECT as json. (#1715)

#### Gate instructions specified as tuples no longer error when using a list of parameters. (#1716)

## 4.4.0-rc.0 (2023-12-15)

### Features

#### Add `get_attribute`, `set_attribute` `__getitem__` and `__setitem__` methods to `DefFrame` (#1714)

### Fixes

#### Relax numpy and scipy requirements (#1713)

#### DefFrame no longer attempts to serialize HARDWARE-OBJECT as json. (#1715)

## 4.3.1-rc.1 (2023-12-15)

### Fixes

#### Relax numpy and scipy requirements (#1713)

#### DefFrame no longer attempts to serialize HARDWARE-OBJECT as json. (#1715)

## 4.3.1-rc.0 (2023-12-14)

### Fixes

#### Relax numpy and scipy requirements (#1713)

## 4.3.0 (2023-12-08)

### Features

#### update qcs_sdk to add ability to modify TranslationOptions (#1706)

## 4.3.0-rc.1 (2023-12-08)

### Features

#### update qcs_sdk to add ability to modify TranslationOptions (#1706)

## 4.3.0-rc.0 (2023-12-06)

### Features

#### update qcs_sdk to add ability to modify TranslationOptions (#1706)

## 4.2.0 (2023-11-28)

### Features

#### Support Python 3.12 (#1696)

#### Final memory values are now available on QAMExecutionResults (#1703)

## 4.2.0-rc.1 (2023-11-28)

### Features

#### Support Python 3.12 (#1696)

#### Final memory values are now available on QAMExecutionResults (#1703)

## 4.2.0-rc.0 (2023-11-28)

### Features

#### Support Python 3.12 (#1696)

## 4.1.1 (2023-11-15)

### Fixes

#### The ``execution_options`` property is now used for retrieving results if no overriding options were provided to the ``execute`` method. (#1694)

## 4.1.1-rc.0 (2023-11-15)

### Fixes

#### The ``execution_options`` property is now used for retrieving results if no overriding options were provided to the ``execute`` method. (#1694)

## 4.1.0 (2023-11-13)

### Features

#### update qcs-sdk-rust (#1683)

### Fixes

#### The `DefGate.matrix` property will no longer raise an exception when the matrix contains a mix of atomic and object types. (#1685)

#### Instruction types no longer return a superclass instance when using `copy.deepcopy` (#1689)

#### DefGate's no longer appear in the instructions list (#1688)

## 4.1.0-rc.5 (2023-11-13)

### Features

#### update qcs-sdk-rust (#1683)

### Fixes

#### The `DefGate.matrix` property will no longer raise an exception when the matrix contains a mix of atomic and object types. (#1685)

#### Instruction types no longer return a superclass instance when using `copy.deepcopy` (#1689)

#### DefGate's no longer appear in the instructions list (#1688)

## 4.1.0-rc.4 (2023-11-09)

### Features

#### update qcs-sdk-rust (#1683)

### Fixes

#### The `DefGate.matrix` property will no longer raise an exception when the matrix contains a mix of atomic and object types. (#1685)

#### Instruction types no longer return a superclass instance when using `copy.deepcopy` (#1689)

#### DefGate's no longer appear in the instructions list (#1688)

## 4.1.0-rc.3 (2023-11-07)

### Features

#### update qcs-sdk-rust (#1683)

### Fixes

#### The `DefGate.matrix` property will no longer raise an exception when the matrix contains a mix of atomic and object types. (#1685)

#### Instruction types no longer return a superclass instance when using `copy.deepcopy` (#1689)

## 4.1.0-rc.2 (2023-11-01)

### Features

#### update qcs-sdk-rust (#1683)

### Fixes

#### The `DefGate.matrix` property will no longer raise an exception when the matrix contains a mix of atomic and object types. (#1685)

## 4.1.0-rc.1 (2023-10-31)

### Features

#### update qcs-sdk-rust (#1683)

### Fixes

#### The `DefGate.matrix` property will no longer raise an exception when the matrix contains a mix of atomic and object types. (#1685)

## 4.1.0-rc.0 (2023-10-27)

### Features

#### update qcs-sdk-rust (#1683)

## 4.0.3 (2023-10-18)

### Fixes

#### only rewrite arithmetic when targeting Aspen processors (#1679)

## 4.0.3-rc.0 (2023-10-18)

### Fixes

#### only rewrite arithmetic when targeting Aspen processors (#1679)

## 4.0.2 (2023-10-16)

### Fixes

#### update qcs-sdk-rust and quil-rs to pull in fixes (#1680)

## 4.0.2-rc.0 (2023-10-16)

### Fixes

#### update qcs-sdk-rust and quil-rs to pull in fixes (#1680)

## 4.0.1 (2023-09-27)

### Fixes

#### `Gate`s should no longer compare as equal and not equal. (#1671)

## 4.0.1-rc.0 (2023-09-27)

### Fixes

#### `Gate`s should no longer compare as equal and not equal. (#1671)

## 4.0.0

The 4.0 release of pyQuil migrates its core functionality into Rigetti's latest generation of Rust SDKs. With this comes access to new features, improved performance, stronger type safety, and better error messages. While this is a significant change for the internals of pyQuil, we've attempted to keep breaking changes to a minimum. Unless necessary, we've chosen to only remove redundant or lesser used features that aren't likely to bother most users.

### Breaking Changes

- Replaced the `qcs-api-client` dependency with `qcs-sdk-python`. Consequentially, the former's `QCSClientConfiguration` has been replaced by the latter’s `QCSClient`. The `QCSClient` class can be imported from the `api` module.
- Removed the `compatibility.v2` sub-package.
- Removed the `EngagementManager` class as RPCQ is no longer used.
- Python 3.7 is no longer supported.
- The environment variable overrides for `quilc` and `QVM` URLs have been renamed to `QCS_SETTINGS_APPLICATIONS_QUILC_URL` and `QCS_SETTINGS_APPLICATIONS_QVM_URL`, respectively.
- The `QuantumComputer`'s `run` method now takes an optional `memory_map` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the ability to use `write_memory` on `Program`s.
- `Program` and instructions have been re-written using the `quil` package. Much of the API remains the same, with the following exceptions:
	- `SwapPhase` has been renamed to `SwapPhases`
	- `TemplateWaveform` and its subclasses are no longer `@dataclass`es.
	- `DefFrame` and `Frame` are no longer `@dataclass`es.
	- The `pop` method has been removed from `Program`.
	- A `Program` that uses `QubitPlaceholder`s or `LabelPlaceholder`s can no longer be pickled
	- `DefMeasureCalibration` now requires a `MemoryReference`.
	- `fill_placeholders` has been removed since it is no longer needed to expand calibrations.
	- The `get_qubits` method on `Gate` now returns a `list` so that ordering is guaranteed.
	- Setting the `offsets` property on `Declare` will raise a `ValueError` if no `shared_region` is set.
    - When converting to Quil, a `Program` automatically places `DECLARE`s at the top of the program.
    - The `Program#calibrations` property no longer returns measure calibrations, instead use the new `measure_calibrations` property.
- The `parser` module has been removed. Parsing now happens by initializing a `Program` with the program string you want to be parsed.
- `PRAGMA` instructions can no longer have a directive that conflicts with a Quil keyword. If you were using directives like `DELAY` or `FENCE`, consider using the respective Quil-T instructions instead.
- `QubitPlaceholders` can no longer be used in `PRAGMA` instructions.
- `DefGate` and the other gate definition instructions will no longer accept names that conflict with Quil keywords.
- `Program#get_qubits()` will raise a `TypeError` if any of the qubits in the program are not a fixed index.
- A `Program`s `LabelPlaceholder`s are no longer resolved automatically when getting its instructions. Use the `resolve_label_placeholders` method to do it explicitly. Note that the `if_then` and `while_do` methods will add `LabelPlaceholder`s to your program.
- There may be some minor differences in how instructions are converted to a Quil string. These differences should only be cosmetic and should not affect the behavior of a program. However, they may break unit tests or other code that rely on specific formatting of programs.
- The `pyquil.quil.get_default_qubit_mapping` function for getting a mapping of `QubitPlaceholders` to resolved indices has been removed. Generating a default mapping is handled automatically by the placeholder resolving methods.
- The `JumpConditional` base class has been removed, use `JumpWhen` and/or `JumpUnless` directly instead.
- The `Program` class automatically sorts `DECLARE` instructions to the top of the Program when converting to Quil.
- `FenceAll` is now a subclass of `Fence`. This can be impactful if you are doing something like `isinstance(instruction, Fence)` since that will now match `Fence` and `FenceAll`. If the difference between the two is important, check for `FenceAll` first. You can also check if the `qubits` property is empty, which implies a `FenceAll` instruction.
- The `RawInstr` class has been removed. All Quil instructions should be supported by either parsing them with the `Program` class, or constructing them with an instruction class. If you were using `RawInstr` for QASM2.0 transpilation, use the new `transpile_qasm_2` method on `AbstractCompiler`.

### Features

- pyQuil now uses `qcs-sdk-python` (bindings to the [QCS Rust SDK](https://github.com/rigetti/qcs-sdk-rust/)) for compiling and executing programs.
- With the exception of requests to a `quilc` server, RPCQ has been removed in favor of OpenAPI and gRPC calls. This enables:
	- Better performance
	- Better error messages when a request fails
- The improved Rust backend allows on-demand access to a QPU.
- The new `QPUCompilerAPIOptions` class provides can now be used to customize how a program is compiled against a QPU.
- The `diagnostics` module has been introduced with a `get_report` function that will gather information on the currently running pyQuil
installation, perform diagnostics checks, and return a summary.
- `Program` has new methods for resolving Qubit and Label Placeholders in a program.
- `QubitPlaceholders` can now be used in programs that also use fixed or variable qubits.
- `QAMExecutionResult` now has a `raw_readout_data` property that can be used to get the raw form of readout data returned from the executor.
- `WaveformInvocation` has been added as a simpler, more flexible class for invoking waveforms.
- Added two new instruction classes:
   	- The `Include` class for `INCLUDE` instructions.
   	- The `DefCircuit` class `DEFCIRCUIT` instructions.
- The `Program.copy` method now performs a deep copy.
- The `AbstractCompiler` class now has a new `transpile_qasm_2` method for transpiling QASM2.0 programs to Quil.

### Deprecations

- The `QAMExecutionResult` `readout_data` property has been deprecated to avoid confusion with the new `raw_readout_data` property. Use the `register_map` property instead.
- The `indices` flag on the `get_qubits` method on `Program`s and instruction classes continues to work, but will be removed in future versions. A separate `get_qubit_indices` method has been added to get indices. In future versions, `get_qubits` will only return a list of `QubitDesignator`s.
- The `is_protoquil`, `is_supported_on_qpu` methods on `Program` and the `validate_supported_quil` function will always return `True`. These methods were never reliable as they were implemented as client-side checks that don't necessarily reflect the latest available features on Rigetti compilers or QPUs. It's safe to stop using these functions and rely on the API to tell you if a program isn't supported.
- `percolate_declares` is a no-op and will be removed in future versions. `Program` now “percolates” declares automatically.
- `merge_programs` continues to work, but will be removed in future versions, use `Program` addition instead.
- The `format_parameter` function continues to work, but will be removed in future versions.
- The `WaveformReference` class continues to work, but will be removed in future versions. The new `WaveformInvocation` should be used instead.

## 3.5.4

### Fixes

- Loosen `networkx` requirements (#1584)

## 3.5.3

### Fixes

- Correctly parse matrix gate definitions and support lower case function call expressions (#1588)

## 3.5.2

## 3.5.1

## 3.5.0

### Features

- Add CHANNEL-DELAY attribute to DefFrame (#1564)

### Fixes

- improve messaging on QPU and compiler timeout (#1397)

## 3.5.0-rc.0

### Features

- Add CHANNEL-DELAY attribute to DefFrame (#1564)

## 3.4.1

### Fixes

- regression on pyquil 3.4.0 regarding DEFCAL filtering (#1562)

## 3.4.0

### Features

- check and warn for existing gate when using defgate (#1512)
- add hash method to Program (#1527)

## 3.3.5

### Fixes

- Allow benchmarks to be missing for 1QRB; fallback to default RX fidelity (#1556)

## 3.3.4

## 3.3.3

### Fixes

- incorrect circuit rendering (#1520)

## 3.3.2

## 3.3.1

### Fixes

- report non-503 http status errors when creating engagements (#1479)
- Ensure adding programs doesn't mutate the first (#1477)

## 3.3.0

### Features

- gracefully handle error when QPU unavailable for engagement (#1457)

## 3.3.0-rc.0

### Features

- gracefully handle error when QPU unavailable for engagement (#1457)

## [v3.2.0](https://github.com/rigetti/pyquil/releases/tag/v3.2.0)

### Improvements and Changes

- `QAMExecutionResult` now includes `execution_duration_microseconds`, providing the amount of time
  a job held exclusive hardware access. (@randall-fulton, #1436)
- Upgrade `qcs-api-client` so that clients can specify a QCS account on their profile, which `qcs-api-client` will in turn use to set `X-QCS-ACCOUNT-{ID/TYPE}` headers on outgoing QCS requests, most notably during engagement creation. (@erichulburd, #1439)

- Upgrade `qcs-api-client` to address bug that occurs when the QCS profile and credentials name do not match. (@erichulburd, #1442)

- Allow newer versions of `qcs-api-client` (which allows newer versions of `iso8601` and `pyjwt`) to be used. (@vtomole, #1449)

## [v3.1.0](https://github.com/rigetti/pyquil/releases/tag/v3.1.0)

### Announcements

- `setup.py` has been removed and will no longer be generated as part of the automated release process.

### Improvements and Changes

- Function `pyquil.quilatom.substitute()` now supports substitution of classical `MemoryReference`
  objects such as `theta[4]` with their parameter values, enabling user-side parameter substitution.
- Versions of `qcs-api-client` up to 0.20.x are now supported.
- The CompilerISA of physical QPUs now assigns a fidelity of 1 to virtual RZs.

### Bugfixes

- Fix docs typo in `start.rst`, where an extra parentheses was present in a python code block (@ThomasMerkh).
- Fixed typo where `scale` was being used as the `phase` in generation of several waveforms.

## [v3.0.1](https://github.com/rigetti/pyquil/releases/tag/v3.0.1)

### Improvements and Changes

### Bugfixes

## [v3.1.0](https://github.com/rigetti/pyquil/releases/tag/v3.1.0)

### Announcements

- `setup.py` has been removed and will no longer be generated as part of the automated release process.

### Improvements and Changes

- Function `pyquil.quilatom.substitute()` now supports substitution of classical `MemoryReference`
  objects such as `theta[4]` with their parameter values, enabling user-side parameter substitution.
- Versions of `qcs-api-client` up to 0.20.x are now supported.

### Bugfixes

- Fix docs typo in `start.rst`, where an extra parentheses was present in a python code block (@ThomasMerkh).
- Fixed typo where `scale` was being used as the `phase` in generation of several waveforms.

## [v3.0.1](https://github.com/rigetti/pyquil/releases/tag/v3.0.1)

### Improvements and Changes

- Both `get_qc` and `QPU` now accept an `endpoint_id` argument which is used to engage
  against a specific QCS [quantum processor endpoint](https://docs.api.qcs.rigetti.com/#tag/endpoints).

### Bugfixes

- Allow `np.ndarray` when writing QAM memory. Disallow non-integer and non-float types.
- Fix typo where `qc.compiler.calibration_program` should be `qc.compiler.get_calibration_program()`.
- `DefFrame` string-valued fields that contain JSON strings now round trip to valid Quil and back to
  JSON via `DefFrame.out` and `parse`. Quil and JSON both claim `"` as their only string delimiter,
  so the JSON `"`s are escaped in the Quil.

## [v3.0.0](https://github.com/rigetti/pyquil/releases/tag/v3.0.0)

### Announcements

- pyQuil now directly supports the QCS API v1.0, offering you better performance and more
  granular data about QCS quantum processors.

- Python 3.6 is no longer supported. Python 3.7, 3.8, and 3.9 are supported.
- `pyquil.compatibility.v2` provides a number of classes/utilities which support the pyQuil v2 API, such as
  `get_qc`; `pyquil.compatibility.v2.api` offers `QuantumComputer`, `QPU`, and `QVM`. These may be
  used to incrementally migrate from v2 to v3, but should not be relied on indefinitely, as the
  underlying mechanics of these two versions will continue to diverge in the future.

### Improvements and Changes

- Added support and documentation for concurrent compilation and execution (see "Advanced Usage" in docs)

- `pyquil.version.__version__` has been moved to `pyquil.__version__`.

- `PyquilConfig` has been replaced by `api.QCSClientConfiguration`. As a result, the only supported configuration-related
  environment variables are:

  - `QCS_SETTINGS_APPLICATIONS_PYQUIL_QVM_URL` (replaces `QVM_URL`)
  - `QCS_SETTINGS_APPLICATIONS_PYQUIL_QUILC_URL` (replaces `QUILC_URL`)
  - `QCS_SETTINGS_FILE_PATH` (overrides location for `settings.toml`)
  - `QCS_SECRETS_FILE_PATH` (overrides location for `secrets.toml`)

- `ForestConnection` and `ForestSession` have been removed. Connection information is now managed via `api.QCSClientConfiguration`
  and `api.EngagementManager`.
- `QVMCompiler` now produces a `Program` instead of a `PyQuilExecutableResponse`.
- `QPU.get_version_info()` has been removed.

- `get_qc()` now accepts an `execution_timeout` parameter (in addition to the existing `compiler_timeout`) to specify
  a time limit on execution requests.
- `AbstractCompiler.set_timeout()` has been removed. Set timeouts via `get_qc()` instead (`execution_timeout`, `compiler_timeout` parameters).

- `QPUCompiler.refresh_calibration_program()` and `QPUCompiler.calibration_program` have been removed. Instead, use `QPUCompiler.get_calibration_program()`
  (with optional `force_refresh` argument).

- `QVMCompiler.get_calibration_program()`, `QVMCompiler.calibration_program`, and `QVMCompiler.refresh_calibration_program()` have been removed.
- `get_benchmarker()` has been removed in favor of calling `BenchmarkConnection` constructor directly.

- Moved compiler/ RPCQ models to `external/rpcq.py`, including `CompilerISA`. Eventually, we will move these into the RPCQ package.

- Replaced intermediary `Qubit.type` with an explicit list of gates that the client may pass to the compiler without further transformation.

- Dropped the intermediary `ISA` class. Rely exclusively on `CompilerISA` as a carrier of instruction set architecture information.

- Renamed package `device` to `quantum_processor`. Also renamed any symbols including `device` to include `quantum_processor` instead.

- Renamed `AbstractDevice` to `AbstractQuantumProcessor`, `CompilerDevice` to `CompilerQuantumProcessor`, `NxDevice` to `NxQuantumProcessor`, and `QCSDevice` to `QCSQuantumProcessor`.

- Support `AbstractQuantumProcessor`s derived from QCS `InstructionSetArchitecture`, `CompilerISA`, and `nx.Graph`.

- Dropped `api._quantum_processors`. Moved `get_device` to `pyquil.quantum_processor.qcs.get_qcs_quantum_processor`.

- Dropped `gates_in_isa` and refactored as an internal function for preparing a list of `pyquil.Gate`'s that the user may use to initialize a `NoiseModel` based on the underlying `CompilerISA`.

- `get_qc()` raises `ValueError` when the user passes a QCS quantum processor name and `noisy=True`.

- `QuantumComputer.run_and_measure()` has been removed. Instead, add explicit `MEASURE` instructions to programs and use
  `QuantumComputer.compile()` along with `QuantumComputer.run()` to compile and execute.
- The `ro` memory region is no longer implicitly declared. All memory regions must be declared explicitly.

- The `pyquil.magic` package has been removed in favor of writing programs more explicitly.

- Removed `TomographyExperiment` (deprecated). Use `Experiment` instead.

- Removed `Experiment.qubits` (deprecated).

- `ExperimentSetting` constructor no longer accepts a `PauliTerm` for its `in_state` parameter (deprecated). Supply a
  `TensorProductState` instead.

- Removed `ExperimentSetting.in_operator` (deprecated). Use `ExperimentSetting.in_state` instead.

- Removed the following `ExperimentResult` attributes (deprecated):

  - `stddev`: Use `std_err` instead.
  - `raw_stddev`: Use `raw_std_err` instead.
  - `calibration_stddev`: Use `calibration_std_err` instead.

- Removed deprecated `protoquil_positional` parameter from `QuantumComputer.compile()`. Use `protoquil` keyword
  parameter instead.

- `get_qc()` no longer accepts `"9q-generic"` (deprecated). Use `"9q-square"` instead.

- Removed `QAM.read_from_memory_region()` (deprecated). Use `QAMExecutionResult.readout_data.get(region_name)` instead.

- Removed `local_qvm()` (deprecated). Use `local_forest_runtime()` instead.

- Removed `Wavefunction.ground()` (deprecated). Use `Wavefunction.zeros()` instead.

- `WavefunctionSimulator`'s `run_and_measure()`, `wavefunction()`, and `expectation()` methods no longer accept a
  `Dict[MemoryReference, Any]` for the `memory_map` parameter (deprecated). Supply a
  `Dict[str, List[Union[int, float]]]` instead.
- `gates.MEASURE()` no longer accepts an integer for `classical_reg` (deprecated). Use a `MemoryReference` instead.

- Removed `gates.TRUE()`, `gates.FALSE()`, and classes `ClassicalTrue` and `ClassicalFalse` (deprecated).
  Use `gates.MOVE()` and class `ClassicalMove` instead.

- Removed `gates.OR()` and class `ClassicalOr` (deprecated). Use `gates.IOR()` and class `ClassicalInclusiveOr` instead.

- `measure_observables()` no longer accepts the following parameters (deprecated):
  - `n_shots`: Set on experiment's program with `Program.wrap_in_numshots_loop()` instead.
  - `active_reset`: Add `RESET` instruction to experiment's program instead.
  - `symmetrize_readout` & `readout_symmetrize`: Set `Experiment.symmetrization` on experiment instead.
- `PauliTerm.pauli_string()` now requires `qubits` instead of accepting them optionally (deprecated).

- Removed `Program.alloc()` (deprecated). Instantiate a `QubitPlaceholder` instead.

- Removed `Addr` (deprecated). Use `MemoryReference` instead.

- `QPUConnection` and `QVMConnection` have been removed in favor of using `QuantumComputer`, `QVM` or`QPU`
  (e.g. via `QuantumComputer.qam`), or `WavefunctionSimulator`.

- `WavefunctionSimulator` constructor now accepts optional `measurement_noise` and `gate_noise`. These noise parameters
  are passed to the QVM by `WavefunctionSimulator.run_and_measure()` and `WavefunctionSimulator.wavefunction()`.
- `noise.estimate_assignment_probs()` now accepts a `QuantumComputer` instead of `QVMConnection`.

- `QAM` and its subclasses (such as `QPU` and `QVM`) do not store any information specific to the state
  of execution requests, and thus are safe to be used concurrently by different requests. `QAM.run`
  is now composed of two intermediate calls:

  - `QAM.execute` starts execution of the provided executable, returning an opaque handle.
  - `QAM.get_result` uses the opaque handle returned by `execute` to retrieve the result values.

  These new calls can be used to enqueue multiple programs for execution prior to retrieving
  results for any of them. Note that this new pattern means that `QAM.load`, `QAM.reset`, and
  `QAM.wait` no longer exist.

- `QAM.run` no longer accepts a `memory_map` argument. Memory values must be written onto
  executable directly with `Program.write_memory()` and `EncryptedProgram.write_memory()` instead.

- `QuantumComputer`, `QAM`, `QPU`, and `QVM` are now safe to share across threads and processes,
  as they no longer store request-related state.

- `PyQVM.execute` has been renamed to `PyQVM.execute_once` to execute a single program from start
  to finish within the context of the existing `PyQVM` state. `PyQVM` is the only stateful `QAM`.
  `PyQVM.execute` now implements `QAM.execute` and resets the `PyQVM` state prior to program execution.

- `QuantumComputer.experiment` has been renamed to `QuantumComputer.run_experiment`.

- Results returned from execution are now referred to as `readout_data` rather than `memory`, reflecting the reality
  that the memory of the QAM is not currently exposed to the user. The exception to this rule is the stateful `PyQVM`,
  whose state is maintained within the pyQuil process and whose memory _may truly be inspected._ For that,
  `PyQVM.read_memory` remains available.

- `QuantumComputer.run` now returns a `QAMExecutionResult` rather than the readout data from the `ro` readout
  source. To access those same readout results, use `qc.run().readout_data.get('ro')`. This allows access to other
  execution-related information and other readout sources.
- Simultaneous, rather than independent, random benchmark scores are passed to quilc as the gate fidelity for RX and RZ operations.

## [v2.28.2](https://github.com/rigetti/pyquil/compare/v2.28.1..v2.28.2) (July 6, 2021)

### Announcements

### Improvements and Changes

### Bugfixes

- Fix parser bug that prevented calling a circuit without parameters, e.g.
  `BELL` (@notmgsk).

## [v2.28.1](https://github.com/rigetti/pyquil/compare/v2.28.0..v2.28.1) (May 5, 2021)

### Announcements

### Improvements and Changes

### Bugfixes

- Fix key error for unmeasured memory regions (@notmgsk, @ameyer-rigetti, #1156)
- Remove extraneous debug prints from `def_gate_matrix()` (@notmgsk)

## [v2.28.0](https://github.com/rigetti/pyquil/compare/v2.27.0..v2.28.0) (January 26, 2021)

### Announcements

### Improvements and Changes

### Bugfixes

- Fix parsing error for parameterized `DEFCIRUCIT`s (@ameyer-rigetti, #1295)

## [v2.27.0](https://github.com/rigetti/pyquil/compare/v2.26.0..v2.27.0) (December 30, 2020)

### Announcements

- Switched to Github Actions.

### Improvements and Changes

- Bump RPCQ dependency to 3.6.0 (@notmgsk, #1286).
- Tests can be run in parallel (@notmgsk, #1289).

### Bugfixes

- Fix hanging test due to ZMQ bug (@notmgsk).
- Fix unitary comparison in Quil compilation test (@notmgsk).
- Fix parsing comments in Lark grammar (@notmgsk, #1290).

## [v2.26.0](https://github.com/rigetti/pyquil/compare/v2.25.0..v2.26.0) (December 10, 2020)

### Announcements

- Quil-T brings the dimension of time to your quantum programs! Quil-T is an extension of
  Quil which allows one to develop quantum programs at the level of pulses and waveforms
  and brings an unprecedented level of fine-grained control over the QPU.

### Improvements and Changes

- Unpacking bitstrings is significantly faster (@mhodson-rigetti, @notmgsk, #1276).
- Parsing is now performed using Lark rather than ANTLR, often allowing a 10x improvement
  in parsing large and complex programs (@notmgsk, #1278).
- Gates now generally allow a "formal" qubit label as in `DEFCIRCUIT`, rather than
  requiring a numeric index (#1257).
- `Program` objects come with additional Quil-T related properties, such as
  `calibrations`, `waveforms`, and `frames` (#1257).
- The `AbstractCompiler` classes come with tools for performing calibration of
  programs. Namely, `get_calibration_program` provides a program for calibrating against
  recent QPU settings (#1257).
- `rewrite_arithmetic` now converts phase angle from radians to revolutions (#1257).
- Readout is more permissive, and does not require the destination to be named `"ro"`
  (#1257).
- The default value for `QPU_COMPILER_URL` has been updated to point to Rigetti's
  translation service. This changes allows one to use the translation service to translate
  a Quil-T program and receive the binary payload without having a QPU reservation
  (#1257).

### Bugfixes

## [v2.25.0](https://github.com/rigetti/pyquil/compare/v2.24.0..v2.25.0) (November 17, 2020)

### Announcements

### Improvements and Changes

- Timeout configuration has been revamped. `get_qc` now accepts a `compiler_timeout`
  option, and `QVMCompiler` and `QPUCompiler` provide a `set_timeout` method, which should
  greatly simplify the task of changing the default timeout. `QVMCompiler` also provides a
  `quilc_client` property so that it shares the same interface as
  `QPUCompiler`. Documentation has been updated to reflect these changes (@notmgsk,
  @kalzoo, #1273).

### Bugfixes

## [v2.24.0](https://github.com/rigetti/pyquil/compare/v2.23.1..v2.24.0) (November 5, 2020)

### Announcements

### Improvements and Changes

- `run_and_measure` now only measures the qubits that are used in a program
  (rather than all qubits on the device) when the target QAM is a QVM without
  noise. This prevents the QVM from exhausting memory when it tries to allocate
  for e.g. 32 qubits when only e.g. 2 qubits are used in the program (@notmgsk,
  #1252).

- Include a `py.typed` so that libraries that depend on pyquil can
  validate their typing against it (@notmgsk, #1256).

- Removed warnings expected in normal workflows that cannot be avoided
  programmatically. This included the warning about passing native Quil to
  `native_quil_to_executable`. Documentation has been updated to clarify
  expected behavior (@mhodson-rigetti, gh-1267).

### Bugfixes

- Fixed incorrect return type hint for the `exponential_map` function, which
  now accepts both `float` and `MemoryReference` types for exponentiation
  (@mhodson-rigetti, gh-1243).

## [v2.23.1](https://github.com/rigetti/pyquil/compare/v2.23.0..v2.23.1) (September 9, 2020)

### Announcements

### Improvements and Changes

- Push new pyquil versions to pypi as part of CI/CD pipelines (@notmgsk, gh-1249)

### Bugfixes

- Allow `np.ndarray` in `DefPermutationGate` (@notmgsk, gh-1248)

## [v2.23.0](https://github.com/rigetti/pyquil/compare/v2.22.0..v2.23.0) (September 7, 2020)

### Announcements

### Improvements and Changes

- Compiler connection timeouts are now entirely user-configurable (@kalzoo, gh-1246)

### Bugfixes

- Do not issue a warning if OAuth2 token returns a string (@erichulburd, gh-1244)

## [v2.22.0](https://github.com/rigetti/pyquil/compare/v2.21.1..v2.22.0) (August 3, 2020)

### Announcements

### Improvements and Changes

- Various improvements and updates to the documentation.

### Bugfixes

## [v2.21.1](https://github.com/rigetti/pyquil/compare/v2.21.0..v2.21.1) (July 15, 2020)

### Announcements

- This is just a cosmetic update, to trigger a new docker build.

### Improvements and Changes

### Bugfixes

- Fix type hinting (@notmgsk, gh-1230)

## [v2.21.0](https://github.com/rigetti/pyquil/compare/v2.20.0..v2.21.0) (July 14, 2020)

### Announcements

### Improvements and Changes

- Documentation for Compiler, Advanced Usage, and Troubleshooting sections updated
  (@notmgsk, gh-1220).
- Use numeric abstract base classes for type checking (@kilimanjaro, gh-1219).
- Add XY to docs (@notmgsk, gh-1226).

### Bugfixes

- Fix damping after dephasing noise model (@max-radin, gh-1217).

## [v2.20](https://github.com/rigetti/pyquil/compare/v2.19.0..v2.20.0) (June 5, 2020)

### Announcements

### Improvements and Changes

- Added a PyQuil only `rewrite_arithmetic` handler, deprecating the previous
  RPC call to `quilc` in `native_quil_to_executable` (@kilimanjaro, gh-1210).

### Bugfixes

- Fix link in documentation (@notmgsk, gh-1204).
- Add `RX(0) _` to the native gates of a N-q qvm (@notmgsk, gh-1211).

## [v2.19](https://github.com/rigetti/pyquil/compare/v2.18.0...v2.19.0) (March 26, 2020)

### Announcements

### Improvements and Changes

- Add a section to `CONTRIBUTING.md` about publishing packages to conda-forge
  (@appleby, gh-1186).
- Correctly insert state preparation code in `Experiment`s _before_ main program code
  (@notmgsk, gh-1189).
- `controlled` modifier now accepts either a Sequence of control qubits or a single control qubit. Previously, only a single control qubit was supported (@adamglos92, gh-1196).

### Bugfixes

- Fix flakiness in `test_run` in `pyquil/test/test_quantum_computer.py`
  (@appleby, gh-1190).
- Fix a bug in QuantumComputer.experiment that resulted in a TypeError being
  raised when called multiple times on the same experiment when the underlying QAM
  was a QVM based on a physical device (@appleby, gh-1188).

## [v2.18](https://github.com/rigetti/pyquil/compare/v2.17.0...v2.18.0) (March 3, 2020)

### Announcements

### Improvements and Changes

### Bugfixes

- Fixed the QCS access request link in the README (@amyfbrown, gh-1171).
- Fix the SDK download link and instructions in the docs (@amyfbrown, gh-1173).
- Fix broken link to example now in forest-tutorials (@jlapeyre, gh-1181).
- Removed HALT from valid Protoquil / supported Quil. (@kilimanjaro, gh-1176).
- Fix error in comment in Noise and Quantum Computation page (@jlapeyre gh-1180)

## [v2.17](https://github.com/rigetti/pyquil/compare/v2.16.0...v2.17.0) (January 30, 2020)

### Announcements

- In order to make the pyQuil examples more accessible, we recently made a new
  repository, [rigetti/forest-tutorials][forest-tutorials], which is set up so
  that the example notebooks can be run via a web browser in a preconfigured
  execution environment on [Binder][mybinder]. The pyQuil README now has a
  "launch binder" badge for running these tutorial notebooks, as well as a
  "Quickstart" section explaining how they work. To run the tutorial notebooks,
  click the badge in the README or the link [here][binder] (@karalekas, gh-1167).

[binder]: https://mybinder.org/v2/gh/rigetti/forest-tutorials/master?urlpath=lab/tree/Welcome.ipynb
[forest-tutorials]: https://github.com/rigetti/forest-tutorials
[mybinder]: https://mybinder.org

### Improvements and Changes

- Pin the `antlr4-python3-runtime` package to below `v4.8` (@karalekas, gh-1163).
- Expand upon the [acknowledgements](ACKNOWLEDGEMENTS.md) file to mention
  contributions from pre-QCS and list previous maintainers (@karalekas, gh-1165).
- Use the [rigetti/gitlab-pipelines](https://github.com/rigetti/gitlab-pipelines)
  repository's template YAMLs in the `.gitlab-ci.yml`, and add a section to
  `CONTRIBUTING.md` about the CI/CD pipelines (@karalekas, gh-1166).
- Add another round of improvements to the README (@karalekas, gh-1168).

### Bugfixes

- Replace references to non-existent `endpoint` init arg when constructing
  `QPUCompiler`s in `test_qpu.py` (@appleby, gh-1164).
- Preserve program metadata when constructing and manipulating `Experiment`
  objects (@kilimanjaro, gh-1160).

## [v2.16](https://github.com/rigetti/pyquil/compare/v2.15.0...v2.16.0) (January 10, 2020)

### Announcements

- The `TomographyExperiment` class has been renamed to `Experiment`. In addition,
  there is a new `QuantumComputer.calibration` method for performing readout
  calibration on a provided `Experiment`, and utilities for applying the results
  of the calibration to correct for symmetrized readout error. `ExperimentSetting`
  objects now also have an `additional_expectations` attribute for extracting
  simultaneously measurable expectation values from a single setting when using
  `QuantumComputer.experiment` (@karalekas, gh-1152, gh-1158).

### Improvements and Changes

- Type hints have been added to the `quil.py` file (@rht, gh-1115, gh-1134).
- Use [Black](https://black.readthedocs.io/en/stable/index.html) for code style
  and enforce it (along with a line length of 100) via the `style` (`flake8`)
  and `formatcheck` (`black --check`) CI jobs (@karalekas, gh-1132).
- Ignore fewer `flake8` style rules, add the `flake8-bugbear` plugin, and
  rename the style-related `Makefile` targets and CI jobs so that they have
  a uniform naming convention: `check-all`, `check-format`, `check-style`,
  and `check-types` (@karalekas, gh-1133).
- Added type hints to `noise.py`, began verifying in the CI (@rht, gh-1136).
- Improved reStructuredText markup in docstrings (@peterjc, gh-1141).
- Add helper to separate `ExperimentResults` by groups of qubits on
  which their operator acts (@kylegulshen, gh-1078).
- Added typing to the `pyquil/latex` module and added the module to the
  `check-types` CI job (@karalekas, gh-1142).
- Add helper to merge `TomographyExperiment`s in the `experiment` module's
  `_group.py` file. Move `group_experiments` from `operator_estimation.py`
  to `_group.py` and rename to `group_settings` but maintain backwards
  compatibility (@kylegulshen, gh-1077).
- The code in `gate_matrices.py`, `numpy_simulator.py`, `reference_simulator.py`,
  and `unitary_tools.py` has been typed and reorganized into a new `simulation`
  subdirectory, maintaining backwards compatibility (@karalekas, gh-1143).
- Added a `.travis.yml` file to enable Travis CI for external-contributor builds,
  and upgraded GitLab CI style checks to py37 (@karalekas, gh-1145).
- Delete `api/_job.py`, `JobConnection`, and `SyncConnection`, which have been
  deprecated for over a year and a half (@karalekas, gh-1144).
- Added typing to the `pyquil/experiment` module and added the module to the
  `check-types` CI job (@karalekas, gh-1146).
- Use `dataclasses` instead of `namedtuples` in the `pyquil/device` module, and
  add type annotations to the entire module (@karalekas, gh-1149).
- Reduced the number of `mypy` errors in `paulis.py` (@rht, gh-1147).
- Compile to XY gates as well as CZ gates on dummy QVMs (@ecpeterson, gh-1151).
- `QAM.write_memory` now accepts either a `Sequence` of values or a single
  value (@tommy-moffat, gh-1114).
- Added type hints for all remaining top-level files (@karalekas, gh-1150).
- Added type annotations to the whole `pyquil.api` module (@karalekas, gh-1157).

### Bugfixes

- Don't attach pipes to stdout/stderr when starting quilc and qvm processes in
  `local_forest_runtime`. This prevents the pipe buffers from getting full and
  causing hung quilc/qvm for long running processes (@appleby, gh-1122).
- Pass a sequence to `np.vstack` to avoid a `FutureWarning`, and add a protoquil
  keyword argument to `MyLazyCompiler.quil_to_native_quil` to avoid a `TypeError`
  in the `migration2-qc.ipynb` notebook (@appleby, gh-1138).
- Removed unused method `Program._out()` in `quil.py` (@rht, gh-1137).
- Fixed string concatenation style, caused by `black` (@peterjc, gh-1139).

## [v2.15](https://github.com/rigetti/pyquil/compare/v2.14.0...v2.15.0) (December 20, 2019)

### Announcements

- PyQuil now supports encryption for communication with the QPU. It does so
  by requesting an `Engagement` from Forest Dispatch, which includes the keys
  necessary for encryption along with the endpoints to use. This workflow is
  managed by the new `ForestSession` class, and in the general case is
  transparent to the user (@kalzoo, gh-1123).

### Improvements and Changes

- LaTeX circuit output now ignores `RESET` instructions by default, rendering
  instead the (equivalent) program with `RESET` omitted (@kilimanjaro, gh-1118)
- Broadened the scope of `flake8` compliance to the include the `examples` and
  `docs` directories, and thus the whole repository (@tommy-moffat, gh-1113).
- `DEFGATE ... AS PAULI-SUM` is now supported (@ecpeterson, gh-1125).
- Add unit test for validating Trotterization order (@jmbr, gh-1120).
- Updated the authentication mechanism to Forest server. Preferentially use
  credentials found at `~/.qcs/user_auth_credentials` and fall back to
  `~/.qcs/qmi_auth_credentials` (@erichulburd, gh-1123).
- The log level can now be controlled with the `LOG_LEVEL` environment variable,
  set to `LOG_LEVEL=DEBUG` to help diagnose problems. In addition, certain errors
  will no longer print their entire stack trace outside of `DEBUG` mode, for a
  cleaner console and better user experience. This is only true for errors where
  the cause is well known (@kalzoo, gh-1123).
- Connection to the QPU compiler now supports both ZeroMQ and HTTP(S)
  (@kalzoo, gh-1127).
- Bump quilc / qvm parent Docker images to v1.15.1 (@karalekas, gh-1128).

### Bugfixes

- Pinned the `mypy` version to work around issue with nested types causing the
  `make typecheck` CI job to fail (@erichulburd, gh-1119).
- Minor fixes for `examples/1.3_vqe_demo.py` and `examples/quantum_walk.ipynb`
  (@appleby, gh-1116).
- Only request engagement from Forest Dispatch when QPU and QPU Compiler addresses
  are not provided by other configuration sources (@kalzoo, gh-1130).

## [v2.14](https://github.com/rigetti/pyquil/compare/v2.13.0...v2.14.0) (November 25, 2019)

### Announcements

- There is a new `QuantumComputer.experiment` method for running a collection of
  quantum programs as defined by a `TomographyExperiment`. These objects have a
  main program body and a collection of state preparation and measurement
  specifications, which capture the structure of many near-term applications
  and algorithms like the variational quantum eigensolver (VQE). In addition,
  the `TomographyExperiment` encodes information about symmetrization, active
  qubit reset, and the number of shots to perform on the quantum backend (e.g.
  the QVM or QPU). For more information check out the API documentation sections
  on the [Quantum Computer](docs/source/apidocs/quantum_computer.rst) and on the
  [Experiment Module](docs/source/apidocs/experiment.rst) (@karalekas, gh-1100).

### Improvements and Changes

- Type hints have been added to the `PauliTerm` class (@rht, gh-1075).
- The `rigetti/forest` Docker image now has less noisy output due to stdout and
  stderr redirection to log files `entrypoint.sh` (@karalekas, gh-1105).
- Added a `make typecheck` target to run `mypy` over a subset of the pyquil
  sources, and enabled typechecks in the GitLab CI pipeline (@appleby, gh-1098).
- Added support for the `XY` (parameterized `iSWAP`) gate family in `Program`s
  and in `ISA`s (@ecpeterson, gh-1096, gh-1107, gh-1111).
- Removed the `tox.ini` and `readthedocs.yml` files (@karalekas, gh-1108).
- Type hints have been added to the `PauliSum` class (@rht, gh-1104).

### Bugfixes

- Fixed a bug in the LaTeX output of controlled unitary operations (@kilimanjaro,
  gh-1103).
- Fixed an example of using the `qc.run` method in the docs to correctly declare
  the size of a memory register (@appleby, gh-1099).
- Specify UTF-8 encoding when opening files that might contain non-ascii characters,
  such as when reading the pyquil README.md file in setup.py or when serializing /
  deserializing pyquil.experiment objects to/from JSON (@appleby, gh-1102).

## [v2.13](https://github.com/rigetti/pyquil/compare/v2.12.0...v2.13.0) (November 7, 2019)

### Announcements

- Rather than installing pyQuil from PyPI, conda-forge, or the source directly,
  users with [Docker](https://www.docker.com/) installed can pull and run the
  [`rigetti/forest`](https://hub.docker.com/r/rigetti/forest) Docker image
  to quickly get started with compiling and simulating quantum programs! When
  running the image, a user will be dropped into an `ipython` REPL that has
  pyQuil and its requirements preinstalled, along with quilc and qvm servers
  running in the background (@karalekas, gh-1035, gh-1039).
- Circuit diagram generation has had a makeover! In particular, the
  `pyquil.latex` module provides two mechanisms for generating diagrams from
  pyQuil programs: `pyquil.latex.to_latex` generates human-readable LaTeX
  output expressing a `Program` as a circuit diagram, and
  `pyquil.latex.display` renders a `Program` as an `IPython.display.Image` for
  inline viewing in Jupyter Notebooks. Learn more about these features in the
  [new example notebook](examples/LaTeXQuilCircuits.ipynb) (@kilimanjaro, gh-1074).

### Improvements and Changes

- Added a `Makefile` with some simple targets for performing common build
  operations like creating and uploading a package (@karalekas, gh-1032).
- Replaced symmetrization in `operator_estimation` with functionality contained
  within `QuantumComputer.run_symmetrized_readout` (@kylegulshen, gh-1047).
- As part of the CI, we now package and push to TestPyPI on every commit, which
  de-risks breaking the `setup.py` and aids with testing (@karalekas, gh-1017).
- We now calculate code coverage as part of the CI pipeline (@karalekas, gh-1052).
- Moved the program generation from `measure_observables` into its own private
  function (@kylegulshen, gh-1043).
- All uses of `__future__` and `six` have been dropped (@karalekas, gh-1060).
- The `conftest.py` has been moved to the project root dir (@karalekas, gh-1064).
- Using `protoquil` as a positional argument to `qc.compile` has been deprecated,
  and it is now a keyword-only argument (@karalekas, gh-1071).
- `PauliSum` objects are now hashable (@ecpeterson, gh-1073).
- The code in `device.py` as been reorganized into a new `device` subdirectory
  in a completely backwards-compatible fashion (@karalekas, gh-1066, gh-1094).
- `PauliTerm` and `PauliSum` now have `__repr__` methods (@karalekas, gh-1080).
- The experiment-schema-related code in `operator_estimation.py` has been moved
  into a new `experiment` subdirectory (@karalekas, gh-1084, gh-1094).
- The keyword arguments to `measure_observables` are now captured as part of
  the `TomographyExperiment` class (@karalekas, gh-1090).
- Type hints have been added to the `pyquil.gates`, `pyquil.quilatom`, and
  `pyquil.quilbase` modules (@appleby, gh-999).
- We now support Python 3.8 and it is tested in the CI (@karalekas, gh-1093).

### Bugfixes

- Updated `examples/meyer_penny_game.py` with the correct path to the Meyer Penny
  game exercise in `docs/source/exercises.rst` (@appleby, gh-1045).
- Fixed the Slack Workspace invite link in the README (@amyfbrown, gh-1042).
- `QPU.reset()` now checks whether `pyquil_config.qpu_url` exists before updating
  the endpoint so as not to break custom connections (@kylegulshen, gh-1072).
- Fixed pretty printing of parameter expressions where π is involved
  (@notmgsk, gh-1076).
- Fixed a regression in `PyQVM.execute` that prevented it from running programs
  containing user-defined gates (@appleby, gh-1067).
- Remove some stale code for pulling quilc version info (@notmgsk, gh-1089).

## [v2.12](https://github.com/rigetti/pyquil/compare/v2.11.0...v2.12.0) (September 28, 2019)

### Announcements

- There is now a [Contributing Guide](CONTRIBUTING.md) for those who would like
  to participate in the development of pyQuil. Check it out! In addition, pyQuil
  now has a [Bug Report Template](.github/ISSUE_TEMPLATE/BUG_REPORT.md),
  and a [Feature Request Template](.github/ISSUE_TEMPLATE/FEATURE_REQUEST.md),
  which contain sections to fill out when filing a bug or suggesting an enhancement
  (@karalekas, gh-985, gh-986, gh-996).

### Improvements and Changes

- The `local_qvm` context manager has been renamed to `local_forest_runtime`,
  which now checks if the designated ports are used before starting `qvm`/`quilc`.
  The original `local_qvm` has been deprecated (@sauercrowd, gh-976).
- The test suite for pyQuil now runs against both Python 3.6 and 3.7 to ensure
  compatibility with the two most recent versions of Python (@karalekas, gh-987).
- Add support for the `FORKED` gate modifier (@kilimanjaro, gh-989).
- Deleted the deprecated modules `parameters.py` and `qpu.py` (@karalekas, gh-991).
- The test suite for pyQuil now runs much faster, by setting the default value
  of the `--use-seed` option for `pytest` to `True` (@karalekas, gh-992).
- Support non-gate instructions (e.g. `MEASURE`) in `to_latex()` (@notmgsk, gh-975).
- Test suite has been updated to reduce the use of deprecated features
  (@kilimanjaro, gh-998, gh-1005).
- Certain tests have been marked as "slow", and are skipped unless
  the `--runslow` option is specified for `pytest` (@kilimanjaro, gh-1001).
- `PauliSum` objects can now be constructed from strings via `from_compact_str()`
  and `PauliTerm.from_compact_str()` supports multi-qubit strings (@jlbosse, gh-984).

### Bugfixes

- Strength two symmetrization was not correctly producing orthogonal arrays due to
  erroneous truncation, which has been fixed (@kylegulshen, gh-990).
- The `STORE` instruction now accepts `int` or `float` in addition to `MemoryReference`
  as its `source` argument. As a result, you can now `STORE` an immediate value into a
  memory register. Also, the `EQ`, `LT`, `LE`, `GT`, and `GE` instructions now all
  accept `float` in addition to `int` or `MemoryReference` as their third and final
  argument. As a result, you can now perform classical comparisons against an
  immediate `float` value. Finally, the `CONVERT` instruction now accepts any valid
  memory reference designator (a `MemoryReference`, a string, or a tuple of type
  `(str, int)`) for both its arguments (@appleby, gh-1010).
- Raise an error if a gate with non-constant parameters is provided to `lifted_gate`
  (@notmgsk, gh-1012).

## [v2.11](https://github.com/rigetti/pyquil/compare/v2.10.0...v2.11.0) (September 3, 2019)

### Announcements

- PyQuil's changelog has been overhauled and rewritten in Markdown instead of
  RST, and can be found in the top-level directory of the repository as the
  [CHANGELOG.md](CHANGELOG.md) file (which is the standard for most GitHub
  repositories). However, during the build process, we use `pandoc` to convert
  it back to RST so that it can be included as part of the ReadTheDocs
  documentation [here](https://pyquil.readthedocs.io/en/stable/changes.html)
  (@karalekas, gh-945, gh-973).

### Improvements and Changes

- Test suite attempts to retry specific tests that fail often. Tests are
  retried only a single time (@notmgsk, gh-951).
- The `QuantumComputer.run_symmetrized_readout()` method has been
  revamped, and now has options for using more advanced forms of
  readout symmetrization (@joshcombes, gh-919).
- The ProtoQuil restrictions built in to PyQVM have been removed
  (@ecpeterson, gh-874).
- Add the ability to query for other memory regions after both QPU and QVM
  runs. This removes a previously unnecessary restriction on the QVM, although
  `ro` remains the only QPU-writeable memory region during Quil execution
  (@ecpeterson, gh-873).
- Now, running `QuantumComputer.reset()` (and `QuantumComputer.compile()`
  when using the QPU) additionally resets the connection information for
  the underlying `QVM`/`QPU` and `QVMCompiler`/`QPUCompiler` objects,
  which should resolve bugs that arise due to stale clients/connections
  (@karalekas, gh-872).
- In addition to the simultaneous 1Q RB fidelities contained in device
  specs prior to this release, there are now 1Q RB fidelities for
  non-simultaneous gate operation. The names of these fields have been
  changed for clarity, and standard errors for both fidelities have been
  added as well. Finally, deprecation warnings have been added regarding
  the `fCPHASE` and `fBellState` device spec fields, which are no longer
  routinely updated and will be removed in release v2.13 (@jvalery2, gh-968).
- The NOTICE has been updated to accurately reflect the third-party software
  used in pyQuil (@karalekas, gh-979).
- PyQuil now sends “modern” ISA payloads to quilc, which must be of version
  \>= `1.10.0`. Check out the details of `get_isa` for information on how to
  specify custom payloads (@ecpeterson, gh-961).

### Bugfixes

- The `MemoryReference` warnings have been removed from the unit
  tests (@maxKenngott, gh-950).
- The `merge_programs` function now supports merging programs with
  `DefPermutationGate`, instead of throwing an error, and avoids
  redundant readout declaration (@kylegulshen, gh-971).
- Remove unsound logic to fill out non-"ro" memory regions when
  targeting a QPU (@notmgsk, gh-982).

## [v2.10](https://github.com/rigetti/pyquil/compare/v2.9.1...v2.10.0) (July 31, 2019)

### Improvements and Changes

- Rewrote the README, adding a more in-depth overview of the purpose
  of pyQuil as a library, as well as two badges \-- one for PyPI
  downloads and another for the Forest Slack workspace. Also, included
  an example section for how to get started with running a simple Bell
  state program on the QVM (@karalekas, gh-946, gh-949).
- The test suite for `pyquil.operator_estimation` now has an
  (optional) faster version that uses fixed random seeds instead of
  averaging over several experiments. This can be enabled with the
  `--use-seed` command line option when running `pytest` (@msohaibalam,
  gh-928).
- Deleted the deprecated modules `job_results.py` and `kraus.py`
  (@karalekas, gh-957).
- Updated the examples README. Removed an outdated notebook. Updated
  remaining notebooks to use `MemoryReference`, and fix any parts that
  were broken (@notmgsk, gh-820).
- The `AbstractCompiler.quil_to_native_quil()` function now accepts a
  `protoquil` keyword which tells the compiler to restrict both input
  and output to protoquil (i.e. Quil code executable on a QPU).
  Additionally, the compiler will return a metadata dictionary that
  contains statistics about the compiled program, e.g. its estimated
  QPU runtime. See the compiler docs for more information (@notmgsk,
  gh-940).
- Updated the QCS and Slack invite links on the `index.rst` docs page
  (@starktech23, gh-965).
- Provided example code for reading out the QPU runtime estimation for
  a program (@notmgsk, gh-963).

### Bugfixes

- `unitary_tools.lifted_gate()` was not properly handling modifiers
  such as `DAGGER` and `CONTROLLED` (@kylegulshen, gh-931).
- Fixed warnings raised by Sphinx when building the documentation
  (@appleby, gh-929).

## [v2.9.1](https://github.com/rigetti/pyquil/compare/v2.9.0...v2.9.1) (June 28, 2019)

### Bugfixes

- Relaxed the requirement for a quilc server to exist when users of
  the `QuantumComputer` object only want to do simulation work with a
  `QVM` or `pyQVM` backend (@karalekas, gh-934).

## [v2.9](https://github.com/rigetti/pyquil/compare/v2.8.0...v2.9.0) (June 25, 2019)

### Announcements

- PyQuil now has a [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md),
  which contains a checklist of things that must be completed (if
  applicable) before a PR can be merged (@karalekas, gh-921).

### Improvements and Changes

- Removed a bunch of logic around creating inverse gates from
  user-defined gates in `Program.dagger()` in favor of a simpler call
  to `Gate.dagger()` (@notmgsk, gh-887).
- The `RESET` instruction now works correctly with `QubitPlaceholder`
  objects and the `address_qubits` function (@jclapis, gh-910).
- `ReferenceDensitySimulator` can now have a state that is persistent
  between rounds of `run` or `run_and_measure` (@joshcombes, gh-920).

### Bugfixes

- Small negative probabilities were causing
  `ReferenceDensitySimulator` to fail (@joshcombes, gh-908).
- The `dagger` function was incorrectly dropping gate modifiers like
  `CONTROLLED` (@jclapis, gh-914).
- Negative numbers in classical instruction arguments were not being
  parsed (@notmgsk, gh-917).
- Inline math rendering was not working correctly in `intro.rst`
  (@appleby, gh-927).

Thanks to community member @jclapis for the contributions to this
release!

## [v2.8](https://github.com/rigetti/pyquil/compare/v2.7.2...v2.8.0) (May 20, 2019)

### Improvements and Changes

- PyQuil now verifies that you are using the correct version of the
  QVM and quilc (@karalekas, gh-913).
- Added support for defining permutation gates for use with the latest
  version of quilc (@notmgsk, gh-891).
- The rpcq dependency requirement has been raised to v2.5.1 (@notmgsk,
  gh-911).
- Added a note about the QVM's compilation mode to the documentation
  (@stylewarning, gh-900).
- Some measure_observables params now have the `Optional` type
  specification (@msohaibalam, gh-903).

### Bugfixes

- Preserve modifiers during `address_qubits` (@notmgsk, gh-907).

## [v2.7.2](https://github.com/rigetti/pyquil/compare/v2.7.1...v2.7.2) (May 3, 2019)

### Bugfixes

- An additional backwards-incompatible change from gh-870 snuck
  through 2.7.1, and is addressed in this patch release
  (@karalekas, gh-901).

## [v2.7.1](https://github.com/rigetti/pyquil/compare/v2.7.0...v2.7.1) (April 30, 2019)

### Bugfixes

- The changes to operator estimation (gh-870, gh-896) were not made in
  a backwards-compatible fashion, and therefore this patch release
  aims to remedy that. Going forward, there will be much more
  stringent requirements around backwards compatibility and
  deprecation (@karalekas, gh-899).

## [v2.7](https://github.com/rigetti/pyquil/compare/v2.6.0...v2.7.0) (April 29, 2019)

### Improvements and Changes

- Standard deviation -\> standard error in operator estimation
  (@msohaibalam, gh-870).
- Update what pyQuil expects from quilc in terms of rewiring pragmas
  \-- they are now comments rather than distinct instructions
  (@ecpeterson, gh-878).
- Allow users to deprioritize QPU jobs \-- mostly a Rigetti-internal
  feature (@jvalery2, gh-877).
- Remove the `qubits` field from the `TomographyExperiment` dataclass
  (@msohaibalam, gh-896).

### Bugfixes

- Ensure that shots aren\'t lost when passing a `Program` through
  `address_qubits` (@notmgsk, gh-895).
- Fixed the `conda` install command in the README (@seandiscovery,
  gh-890).

## [v2.6](https://github.com/rigetti/pyquil/compare/v2.5.2...v2.6.0) (March 29, 2019)

### Improvements and Changes

- Added a CODEOWNERS file for default reviewers (@karalekas, gh-855).
- Bifurcated the `QPUCompiler` endpoint parameter into two \--
  `quilc_endpoint` and `qpu_compiler_endpoint` \-- to reflect changes
  in Quantum Cloud Services (@karalekas, gh-856).
- Clarified documentation around the DELAY pragma (@willzeng, gh-862).
- Added information about the `local_qvm` context manager to the
  getting started documentation (@willzeng, gh-851).
- Added strict version lower bounds on the rpcq and networkx
  dependencies (@notmgsk, gh-828).
- A slice of a `Program` object now returns a `Program` object
  (@notmgsk, gh-848).

### Bugfixes

- Added a non-None default timeout to the `QVMCompiler` object
  and the `get_benchmarker` function (@karalekas, gh-850, gh-854).
- Fixed the docstring for the `apply_clifford_to_pauli` function
  (@kylegulshen, gh-836).
- Allowed the `apply_clifford_to_pauli` function to now work with the
  Identity as input (@msohaibalam, gh-849).
- Updated a stale link to the Rigetti Forest Slack workspace
  (@karalekas, gh-860).
- Fixed a notation typo in the documentation for noise (@willzeng,
  gh-861).
- An `IndexError` is now raised when trying to access an
  out-of-bounds entry in a `MemoryReference` (@notmgsk, gh-819).
- Added a check to ensure that `measure_observables` takes as many
  shots as requested (@marcusps, gh-846).

Special thanks to @willzeng for all the contributions this release!

## v2.5 (March 6, 2019)

### Improvements and Changes

- PyQuil\'s Gate objects now expose `.controlled(q)` and `.dagger()`
  modifiers, which turn a gate respectively into its controlled
  variant, conditional on the qubit `q`, or into its inverse.
- The operator estimation suite\'s `measure_observables` method now
  exposes a `readout_symmetrize` argument, which helps mitigate a
  machine\'s fidelity asymmetry between recognizing a qubit in the
  ground state versus the excited state.
- The `MEASURE` instruction in pyQuil now has a _mandatory_ second
  argument. Previously, the second argument could be omitted to induce
  \"measurement for effect\", without storing the readout result to a
  classical register, but users found this to be a common source of
  accidental error and a generally rude surprise. To ensure the user
  really intends to measure only for effect, we now require that they
  supply an explicit `None` as the second argument.

### Bugfixes

- Some stale tests have been brought into the modern era.

## v2.4 (February 14, 2019)

### Announcements

- The Quil Compiler ([quilc](https://github.com/rigetti/quilc)) and
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

- The `WavefunctionSimulator` now supports the use of parametric Quil
  programs, via the `memory_map` parameter for its various methods
  (gh-787).
- Operator estimation data structures introduced in **v2.2** have
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
- Operator estimation now offers a \"greedy\" method for grouping
  tomography-like experiments that share a natural tensor product
  basis (ntpb), as an alternative to the clique cover version
  (gh-754).
- The `quilc` endpoint for rewriting Quil parameter arithmetic has
  been changed from `resolve_gate_parameter_arithmetic` to
  `rewrite_arithmetic` (gh-802).
- The difference between ProtoQuil and QPU-supported Quil is now
  better defined (gh-798).

### Bugfixes

- Resolved an issue with post-gate noise in the pyQVM (gh-801).
- A `TypeError` with a useful error message is now raised when a
  `Program` object is run on a QPU-backed `QuantumComputer`, rather
  than a confusing `AttributeError` (gh-799).

## v2.3 (January 28, 2019)

PyQuil 2.3 is the latest release of pyQuil, Rigetti\'s toolkit for
constructing and running quantum programs. A major new feature is the
release of a new suite of simulators:

- We\'re proud to introduce the first iteration of a Python-based
  quantum virtual machine (QVM) called PyQVM. This QVM is completely
  contained within pyQuil and does not need any external dependencies.
  Try using it with `get_qc("9q-square-pyqvm")` or explore the
  `pyquil.pyqvm.PyQVM` object directly. Under-the-hood, there are
  three quantum simulator backends:
  - `ReferenceWavefunctionSimulator` uses standard matrix-vector
    multiplication to evolve a statevector. This includes a suite of
    tools in `pyquil.unitary_tools` for dealing with unitary
    matrices.
  - `NumpyWavefunctionSimulator` uses numpy\'s tensordot
    functionality to efficiently evolve a statevector. For most
    simulations, performance is quite good.
  - `ReferenceDensitySimulator` uses matrix-matrix multiplication to
    evolve a density matrix.
- Matrix representations of Quil standard gates are included in
  `pyquil.gate_matrices` (gh-552).
- The density simulator has extremely limited support for
  Kraus-operator based noise models. Let us know if you\'re interested
  in contributing more robust noise-model support.
- This functionality should be considered experimental and may undergo
  minor API changes.

### Important changes to note

- Quil math functions (like COS, SIN, \...) used to be ambiguous with
  respect to case sensitivity. They are now case-sensitive and should
  be uppercase (gh-774).
- In the next release of pyQuil, communication with quilc will happen
  exclusively via the rpcq protocol. `LocalQVMCompiler` and
  `LocalBenchmarkConnection` will be removed in favor of a unified
  `QVMCompiler` and `BenchmarkConnection`. This change should be
  transparent if you use `get_qc` and `get_benchmarker`, respectively.
  In anticipation of this change we recommend that you upgrade your
  version of quilc to 1.3, released Jan 30, 2019 (gh-730).
- When using a paramaterized gate, the QPU control electronics only
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

- The CZ gate fidelity metric available in the Specs object now has
  its associated standard error, which is accessible from the method
  `Specs.fCZ_std_errs` (gh-751).
- Operator estimation code now correctly handles identity terms with
  coefficients. Previously, it would always estimate these terms as
  1.0 (gh-758).
- Operator estimation results include the total number of counts
  (shots) taken.
- Operator estimation JSON serialization uses utf-8. Please let us
  know if this causes problems (gh-769).
- The example quantum die program now can roll dice that are not
  powers of two (gh-749).
- The teleportation and Meyer penny game examples had a syntax error
  (gh-778, gh-772).
- When running on the QPU, you could get into trouble if the QPU name
  passed to `get_qc` did not match the lattice you booked. This is now
  validated (gh-771).

We extend thanks to community member @estamm12 for their contribution to
this release.

## v2.2 (January 4, 2019)

PyQuil 2.2 is the latest release of pyQuil, Rigetti\'s toolkit for
constructing and running quantum programs. Bug fixes and improvements
include:

- `pauli.is_zero` and `paulis.is_identity` would sometimes return
  erroneous answers (gh-710).
- Parameter expressions involving addition and subtraction are now
  converted to Quil with spaces around the operators, e.g. `theta + 2`
  instead of `theta+2`. This disambiguates subtracting two parameters,
  e.g. `alpha - beta` is not one variable named `alpha-beta` (gh-743).
- T1 is accounted for in T2 noise models (gh-745).
- Documentation improvements (gh-723, gh-719, gh-720, gh-728, gh-732,
  gh-742).
- Support for PNG generation of circuit diagrams via LaTeX (gh-745).
- We\'ve started transitioning to using Gitlab as our continuous
  integration provider for pyQuil (gh-741, gh-752).

This release includes a new module for facilitating the estimation of
quantum observables/operators (gh-682). First-class support for
estimating observables should make it easier to express near-term
algorithms. This release includes:

- data structures for expressing tomography-like experiments and their
  results
- grouping of experiment settings that can be simultaneously estimated
- functionality to executing a tomography-like experiment on a quantum
  computer

Please look forward to more features and polish in future releases.
Don\'t hesitate to submit feedback or suggestions as GitHub issues.

We extend thanks to community member @petterwittek for their contribution
to this release.

Bugfix release 2.2.1 was released January 11 to maintain compatibility
with the latest version of the quilc compiler (gh-759).

## v2.1 (November 30, 2018)

PyQuil 2.1 is an incremental release of pyQuil, Rigetti\'s toolkit for
constructing and running quantum programs. Changes include:

- Major documentation improvements.
- `QuantumComputer.run()` accepts an optional `memory_map` parameter
  to facilitate running parametric executables (gh-657).
- `QuantumComputer.reset()` will reset the state of a QAM to recover
  from an error condition (gh-703).
- Bug fixes (gh-674, gh-696).
- Quil parser improvements (gh-689, gh-685).
- Optional interleaver argument when generating RB sequences (gh-673).
- Our GitHub organization name has changed from `rigetticomputing` to
  `rigetti` (gh-713).

## v2.0 (November 1, 2018)

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

- Python 2 is no longer supported. Please use Python 3.6+
- Parametric gates are now normal functions. You can no longer write
  `RX(pi/2)(0)` to get a Quil `RX(pi/2) 0` instruction. Just use
  `RX(pi/2, 0)`.
- Gates support keyword arguments, so you can write
  `RX(angle=pi/2, qubit=0)`.
- All `async` methods have been removed from `QVMConnection` and
  `QVMConnection` is deprecated. `QPUConnection` has been removed in
  accordance with the QCS access model. Use `pyquil.get_qc` as the
  primary means of interacting with the QVM or QPU.
- `WavefunctionSimulator` allows unfettered access to wavefunction
  properties and routines. These methods and properties previously
  lived on `QVMConnection` and have been deprecated there.
- Classical memory in Quil must be declared with a name and type.
  Please read [New in Forest
  2](https://pyquil.readthedocs.io/en/stable/migration4.html) for
  more.
- Compilation has changed. There are now different `Compiler` objects
  that target either the QPU or QVM. You **must** explicitly compile
  your programs to run on a QPU or a realistic QVM.

Version 2.0.1 was released on November 9, 2018 and includes
documentation changes only. This release is only available as a git tag.
We have not pushed a new package to PyPI.

## v1.9 (June 6, 2018)

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

- `PauliTerm` now remembers the order of its operations. `sX(1)*sZ(2)`
  will compile to different Quil code than `sZ(2)*sX(1)`, although the
  terms will still be equal according to the `__eq__` method. During
  `PauliSum` combination of like terms, a warning will be emitted if
  two terms are combined that have different orders of operation.
- `PauliTerm.id()` takes an optional argument `sort_ops` which
  defaults to True for backwards compatibility. However, this function
  should not be used for comparing term-type like it has been used
  previously. Use `PauliTerm.operations_as_set()` instead. In the
  future, `sort_ops` will default to False and will eventually be
  removed.
- `Program.alloc()` has been deprecated. Please instantiate
  `QubitPlaceholder()` directly or request a \"register\" (list) of
  `n` placeholders by using the class constructor
  `QubitPlaceholder.register(n)`.
- Programs must contain either (1) all instantiated qubits with
  integer indexes or (2) all placeholder qubits of type
  `QubitPlaceholder`. We have found that most users use
  (1) but (2) will become useful with larger and more diverse devices.
- Programs that contain qubit placeholders must be **explicitly
  addressed** prior to execution. Previously, qubits would be assigned
  \"under the hood\" to integers 0\...N. Now, you must use
  `address_qubits` which returns a new program with all qubits indexed
  depending on the `qubit_mapping` argument. The original program is
  unaffected and can be \"readdressed\" multiple times.
- `PauliTerm` can now accept `QubitPlaceholder` in addition to
  integers.
- `QubitPlaceholder` is no longer a subclass of `Qubit`.
  `LabelPlaceholder` is no longer a subclass of `Label`.
- `QuilAtom` subclasses\' hash functions have changed.

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

- `Program` can be initiated with a generator expression.
- `Program.measure_all` (with no arguments) will measure all qubits in
  a program.
- `classical_addresses` is now optional in QVM and QPU `run` methods.
  By default, any classical addresses targeted by `MEASURE` will be
  returned.
- `QVMConnection.pauli_expectation` accepts `PauliSum` as arguments.
  This offers a more sensible API compared to
  `QVMConnection.expectation`.
- pyQuil will now retry jobs every 10 seconds if the QPU is re-tuning.
- `CompilerConnection.compile` now takes an optional argument `isa`
  that allows per-compilation specification of the target ISA.
- An empty program will trigger an exception if you try to run it.

### Supported versions of Python

We strongly support using Python 3 with Pyquil. Although this release
works with Python 2, we are dropping official support for this legacy
language and moving to community support for Python 2. The next major
release of Pyquil will introduce Python 3.5+ only features and will no
longer work without modification for Python 2.

### Bug fixes

- `shift_quantum_gates` has been removed. Users who relied on this
  functionality should use `QubitPlaceholder` and `address_qubits` to
  achieve the same result. Users should also double-check data
  resulting from use of this function as there were several edge cases
  which would cause the shift to be applied incorrectly resulting in
  badly-addressed qubits.
- Slightly perturbed angles when performing RX gates under a Kraus
  noise model could result in incorrect behavior.
- The quantum die example returned incorrect values when `n = 2^m`.

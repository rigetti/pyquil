# Changelog

## 4.0.0-rc.57

### Breaking Changes

- merging master reset version

### Fixes

- Corrected the ryy gate definition (#1603)
- Return `FenceAll` when appropriate, `TemplateWaveform`s should no longer raise `ValueError`s when being constructed from certain `quil` instructions.  (#1654)
- Program declarations should be empty after copy_everything_except_instructions (#1614)

## 3.5.5-rc.2

### Fixes

- Corrected the ryy gate definition (#1603)
- Return `FenceAll` when appropriate, `TemplateWaveform`s should no longer raise `ValueError`s when being constructed from certain `quil` instructions.  (#1654)
- Program declarations should be empty after copy_everything_except_instructions (#1614)

## 3.5.5-rc.1

### Fixes

- Corrected the ryy gate definition (#1603)
- Return `FenceAll` when appropriate, `TemplateWaveform`s should no longer raise `ValueError`s when being constructed from certain `quil` instructions.  (#1654)
- Program declarations should be empty after copy_everything_except_instructions (#1614)

## 4.0.0-rc.56

### Breaking Changes

- Revamp the QuantumComputer and the WavefunctionSimulator sections. (#1643)

### Features

- Quilc clients support (#1638)
- Implement `__iadd__` on `Program` (#1648)

### Fixes

- Return `FenceAll` when appropriate, `TemplateWaveform`s should no longer raise `ValueError`s when being constructed from certain `quil` instructions.  (#1654)
- Attempt to reconstruct `TemplateWaveform`s from `quil_rs.WaveformInvocation`s (#1650)
- Program#calibrate now returns the original instruction if there was no match (#1646)
- Program declarations should be empty after copy_everything_except_instructions (#1614)

## 4.0.0-rc.55

### Breaking Changes

- Revamp the QuantumComputer and the WavefunctionSimulator sections. (#1643)

### Features

- Quilc clients support (#1638)
- Implement `__iadd__` on `Program` (#1648)

### Fixes

- Attempt to reconstruct `TemplateWaveform`s from `quil_rs.WaveformInvocation`s (#1650)
- Program#calibrate now returns the original instruction if there was no match (#1646)
- Program declarations should be empty after copy_everything_except_instructions (#1614)

## 4.0.0-rc.54

### Breaking Changes

- Revamp the QuantumComputer and the WavefunctionSimulator sections. (#1643)

### Features

- Quilc clients support (#1638)
- Implement `__iadd__` on `Program` (#1648)

### Fixes

- Program#calibrate now returns the original instruction if there was no match (#1646)
- Program declarations should be empty after copy_everything_except_instructions (#1614)

## 4.0.0-rc.53

### Breaking Changes

- Revamp the QuantumComputer and the WavefunctionSimulator sections. (#1643)

### Features

- Implement `__iadd__` on `Program` (#1648)

### Fixes

- Program#calibrate now returns the original instruction if there was no match (#1646)
- Program declarations should be empty after copy_everything_except_instructions (#1614)

## 4.0.0-rc.52

### Breaking Changes

- Revamp the QuantumComputer and the WavefunctionSimulator sections. (#1643)

### Fixes

- Program#calibrate now returns the original instruction if there was no match (#1646)
- Program declarations should be empty after copy_everything_except_instructions (#1614)

## 4.0.0-rc.51

### Breaking Changes

- Revamp the QuantumComputer and the WavefunctionSimulator sections. (#1643)

### Fixes

- Program declarations should be empty after copy_everything_except_instructions (#1614)

## 4.0.0-rc.50

### Breaking Changes

- Program and Instruction APIs are backed by quil-rs (#1639)
- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- `QAMExecutionResult` now has a `raw_readout_data` property (#1631)
- release docker images using github ci, not gitlab (#1636)
- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Improve readout_data deprecation warning
- Improve readout_data deprecation warning
- broken action pt 2 (vars is not env context) (#1642)
- broken action (#1641)
- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.49

### Breaking Changes

- Program and Instruction APIs are backed by quil-rs (#1639)
- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- `QAMExecutionResult` now has a `raw_readout_data` property (#1631)
- release docker images using github ci, not gitlab (#1636)
- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Improve readout_data deprecation warning
- Improve readout_data deprecation warning
- broken action pt 2 (vars is not env context) (#1642)
- broken action (#1641)
- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.48

### Breaking Changes

- Program and Instruction APIs are backed by quil-rs (#1639)
- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- `QAMExecutionResult` now has a `raw_readout_data` property (#1631)
- release docker images using github ci, not gitlab (#1636)
- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Improve readout_data deprecation warning
- Improve readout_data deprecation warning
- broken action pt 2 (vars is not env context) (#1642)
- broken action (#1641)
- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.47

### Breaking Changes

- Program and Instruction APIs are backed by quil-rs (#1639)
- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- `QAMExecutionResult` now has a `raw_readout_data` property (#1631)
- release docker images using github ci, not gitlab (#1636)
- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- broken action pt 2 (vars is not env context) (#1642)
- broken action (#1641)
- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.46

### Breaking Changes

- Program and Instruction APIs are backed by quil-rs (#1639)
- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- `QAMExecutionResult` now has a `raw_readout_data` property (#1631)
- release docker images using github ci, not gitlab (#1636)
- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- broken action (#1641)
- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.45

### Breaking Changes

- Program and Instruction APIs are backed by quil-rs (#1639)
- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- `QAMExecutionResult` now has a `raw_readout_data` property (#1631)
- release docker images using github ci, not gitlab (#1636)
- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.44

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- `QAMExecutionResult` now has a `raw_readout_data` property (#1631)
- release docker images using github ci, not gitlab (#1636)
- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.43

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- release docker images using github ci, not gitlab (#1636)
- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.42

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.41

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.40

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.39

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.38

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Getting a QuantumComputer based on a generic QVM will no longer hang if quilc isn't running (#1624)
- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.37

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- The timeout parameter on the `QPU` class is now respected. The default has been increased to 30.0 seconds (#1615)
- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.36

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.35

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.34

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.33

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.32

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.31

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Initializing a QPU with an `endpoint_id` should no longer raise an exception
- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.30

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.29

### Breaking Changes

- Use `ExecutionOptions` parameter to configure how jobs are submitted and retrieved from a QPU. This replaces the `use_gateway` flag on `QCSClient.load()` has been removed. (#1598)
- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.28

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- `Program.copy_everything_except_instructions()` no longer adds declarations to the instructions property (#1612)
- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.27

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- increase gRPC message size limit (#1610)
- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.26

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- specify `quil` as a dependency
- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.25

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.24

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.23

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- `copy_everything_but_instructions` now correctly copies `DECLARE` statements (#1600)
- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.22

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- get_qc will use the given client_configuration
- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.21

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.20

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Remove calibrations from program before sending them to a QVM (#1592)
- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.19

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- support translation options for QPUCompiler (#1590)
- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.18

### Breaking Changes

- The `QuantumComputer`'s `run` method now takes an optional `MemoryMap` parameter. This mapping takes memory region names to a list of values to use for a run. This replaces the need to use `write_memory` on `Program`s between runs.
- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.17

### Breaking Changes

- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Replace `retry`, loosen `networkx` requirements, ensure adding programs don't mutate the first
- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.16

### Breaking Changes

- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- Parametric DefGates and upper case function call expressions will no longer fail to parse. (#1589)
- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.15

### Breaking Changes

- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- native_quil_to_executable will no longer block indefinitely (#1585)
- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.14

### Breaking Changes

- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- The default QCSClient will now load without having QCS credentials (#1582)
- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.13

### Breaking Changes

- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.12

### Breaking Changes

- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.11

### Breaking Changes

- use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- remove qcs-api-client dependency (#1550)
- Removes the compatilbility.v2 subpackage

### Features

- remove v2 compatibility layer (#1475)
- gracefully handle error when QPU unavailable for engagement (#1457)

### Fixes

- bump qcs-sdk-python to fix waveforms (#1507)
- Remove pyi type annotations causing runtime errors (#1506)
- use highest priority Gateway (#1504)
- use properly packaged qcs-sdk types
- Docs Theme
- Docker qvm/quilc in `rigetti/forest` image. (#1437)
- DefFrame to/from Quil with JSON values (#1419)
- DefFrame to/from Quil with JSON values (#1419)
- Correctly apply the phase in quilwaveforms (#1408)
- allow spaces in DEFFRAME specs
- Correctly apply the phase in quilwaveforms (#1408)
- Changed pad_left to pad_right
- allow spaces in DEFFRAME specs
- Changed pad_left to pad_right
- update Quil-T docs to use `get_calibration_program()` name (#1389)
- allow np.ndarray in write_memory and disallow non-int and non-fl… (#1365)
- document error on noisy qcs qpu request
- Fix bug in QPU workflow
- Fix execution of parametric programs (#1353)
- sphinx>=3.0.0,<4.0.0
- support instructions with no qubits or parameters
- remove extraneous debug prints
- Remove test RPCQ server to improve CI consistency. (#1350)
- lock port test fixture
- provide default client configuration on get qcs qpu (#1333)
- raise error on noisy qcs qpu (#1332)
- ignore node modules and commit npm lock
- Fix contiguous engagement handling (#1325)
- Re-add `QPUCompiler.refresh_calibration_program()` (#1323)
- add git commit messge check (#1318)
- include dead attributes when no gates present (#1317)
- Fix RC publising to PyPI
- GitHub checks for PRs to rc branch

## 4.0.0-rc.10

### Breaking Changes

- Use qcs-sdk-python implementation of conjugate_pauli_by_clifford and generate_randomized_benchmarking_sequence (#1557)
- Replace `qcs-api-client` with `qcs-sdk-python` (#1550)
- Removed the `compatilbility.v2` subpackage (#1475)
- Removed the `EngagementManager` class as RPCQ is no longer used.
- Python 3.7 is no longer supported.

### Features

- pyQuil now uses `qcs-sdk-python` (bindings to the QCS Rust SDK) for compiling and executing programs.
- RPCQ has been removed in favor of OpenAPI and GRPC calls. This enables:
    - Better performance
    - Better error messages when a request fails
- The improved Rust backend allows on-demand access to a QPU.

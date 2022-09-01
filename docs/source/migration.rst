.. _migration:

Migrating from pyQuil v3 to v4
==============================

`QPU.execute` and `QPU.get_result` are now async.

New `LocalCompiler` for compiling and translating via the qcs-rusk-sdk.
`quil_to_native_quil` and `native_quil_to_executable` on this compiler are both
async.

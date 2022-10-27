.. _migration:

Migrating from pyQuil v3 to v4
==============================

Internally, `asyncio` is now used for compilation, execution, and result collection. If using pyQuil in an existing `asyncio` context,
it may be necessary to use [nest-asyncio](https://pypi.org/project/nest-asyncio/) to allow nesting of async contexts.
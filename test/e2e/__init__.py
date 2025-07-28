import os

import pytest

e2e = pytest.mark.skipif(
    not os.environ.get("TEST_QUANTUM_PROCESSOR"),
    reason="'TEST_QUANTUM_PROCESSOR' env var required for e2e tests."
)


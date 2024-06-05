"""Diagnostic utilities for pyQuil and QCS."""

from qcs_sdk import diagnostics

import pyquil


def get_report() -> str:
    """Get a report of the current state of the pyQuil installation, Python environment, and supporting packages.

    Note: this format is not stable and its content may change between versions.
    """
    return f"""pyQuil version: {pyquil.__version__}
{diagnostics.get_report()}
"""

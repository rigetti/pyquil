"""Functions and classes for running simulations.

.. deprecated:: 4.22.0
    The simulators exported here -- :py:class:`NumpyWavefunctionSimulator`,
    :py:class:`ReferenceWavefunctionSimulator` and :py:class:`ReferenceDensitySimulator` -- and the
    helper functions exported alongside them are deprecated. They will be removed in pyQuil v5 in
    favor of the Quax-based simulators. Those simulators are available in pyQuil v4
    (``pyquil.simulation._simulator``), but are private, experimental and subject to change, so we
    recommend continuing to use the deprecated simulators until you upgrade to pyQuil v5.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "get_measure_probabilities",
    "NumpyWavefunctionSimulator",
    "ReferenceDensitySimulator",
    "ReferenceWavefunctionSimulator",
    "targeted_einsum",
    "targeted_tensordot",
    "zero_state_matrix",
]

if TYPE_CHECKING:
    from pyquil.simulation._numpy import (
        NumpyWavefunctionSimulator,
        get_measure_probabilities,
        targeted_einsum,
        targeted_tensordot,
    )
    from pyquil.simulation._reference import (
        ReferenceDensitySimulator,
        ReferenceWavefunctionSimulator,
        zero_state_matrix,
    )

# The exports are all deprecated, and their modules import the deprecated ``pyquil.pyqvm``. They are
# loaded on first access, so that importing ``pyquil.simulation`` -- e.g. for ``matrices`` or
# ``tools`` -- does not import ``pyquil.pyqvm`` (see ``pyquil._deprecation``).
_LAZY_EXPORTS = {
    "NumpyWavefunctionSimulator": "pyquil.simulation._numpy",
    "get_measure_probabilities": "pyquil.simulation._numpy",
    "targeted_einsum": "pyquil.simulation._numpy",
    "targeted_tensordot": "pyquil.simulation._numpy",
    "ReferenceDensitySimulator": "pyquil.simulation._reference",
    "ReferenceWavefunctionSimulator": "pyquil.simulation._reference",
    "zero_state_matrix": "pyquil.simulation._reference",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        value = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

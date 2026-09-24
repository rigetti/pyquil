"""Functions and classes for running simulations.

.. deprecated:: 4.22.0
    The simulators exported here -- :py:class:`NumpyWavefunctionSimulator`,
    :py:class:`ReferenceWavefunctionSimulator` and :py:class:`ReferenceDensitySimulator` -- and the
    helper functions exported alongside them are deprecated. They will be removed in pyQuil v5 in
    favor of the Quax-based simulators. Those simulators are available in pyQuil v4
    (``pyquil.simulation._simulator``), but are private, experimental and subject to change, so we
    recommend continuing to use the deprecated simulators until you upgrade to pyQuil v5.
"""

__all__ = [
    "get_measure_probabilities",
    "NumpyWavefunctionSimulator",
    "ReferenceDensitySimulator",
    "ReferenceWavefunctionSimulator",
    "targeted_einsum",
    "targeted_tensordot",
    "zero_state_matrix",
]

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

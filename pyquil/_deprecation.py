##############################################################################
# Copyright 2026 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
"""Shared constants for deprecating pyQuil APIs ahead of their removal in pyQuil v5.

Classes, functions and methods are marked with ``deprecated.sphinx.deprecated``, passing
:data:`DEPRECATED_IN_VERSION`, one of the ``*_REASON`` strings below and
``category=PyQuilDeprecationWarning``::

    @deprecated(version=DEPRECATED_IN_VERSION, reason=SIMULATOR_REASON, category=PyQuilDeprecationWarning)
    class QVM: ...

A deprecated module warns at import with ``warnings.warn(..., PyQuilDeprecationWarning,
stacklevel=2)`` at its top level, which attributes the warning to the importing line.

Warning on import relies on pyQuil not having imported the module first: once a module is in
``sys.modules``, importing it again does not run it. Non-deprecated pyQuil modules therefore import
deprecated modules lazily, from the code that needs them, or only under ``TYPE_CHECKING``.
"""

__all__ = [
    "DEPRECATED_IN_VERSION",
    "EXPERIMENT_REASON",
    "LATEX_REASON",
    "NOISE_REASON",
    "OPERATOR_ESTIMATION_REASON",
    "PyQuilDeprecationWarning",
    "SIMULATOR_REASON",
]

# Any additional deprecations added after the 4.22 release should convert this to an str enum for
# the different "deprecated in" versions.
DEPRECATED_IN_VERSION = "4.22.0"
"""The pyQuil release in which the APIs slated for removal in v5 were deprecated."""

SIMULATOR_REASON = (
    "To be removed in pyQuil v5 in favor of the Quax-based simulators. Those simulators are "
    "available in pyQuil v4 (pyquil.simulation._simulator), but are private, experimental and "
    "subject to change, so we recommend continuing to use this API until you upgrade to pyQuil v5."
)

NOISE_REASON = (
    "To be removed in pyQuil v5 in favor of the Quax-based noise model. That noise model is "
    "available in pyQuil v4 (pyquil.noise._noise_model), but is private, experimental and subject "
    "to change, so we recommend continuing to use this API until you upgrade to pyQuil v5."
)

EXPERIMENT_REASON = "To be removed in pyQuil v5 in favor of rigetti-qpu-hybrid-benchmark."

OPERATOR_ESTIMATION_REASON = (
    "To be removed in pyQuil v5 in favor of the Estimator interface in rigetti-qpu-hybrid-benchmark."
)

LATEX_REASON = "To be removed in pyQuil v5, with no replacement in pyQuil."


class PyQuilDeprecationWarning(FutureWarning):
    """Warns that a pyQuil API is deprecated and will be removed in pyQuil v5.

    This is a :class:`FutureWarning` rather than a :class:`DeprecationWarning` because Python's default
    filters show ``DeprecationWarning`` only when it is attributed to ``__main__``, which would hide it
    from most users. To silence it, filter on this class, e.g.
    ``warnings.filterwarnings("ignore", category=PyQuilDeprecationWarning)``.
    """

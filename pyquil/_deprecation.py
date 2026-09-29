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
"""Shared pieces of pyQuil's deprecation notices.

Deprecate a class, function or method with ``deprecated.sphinx.deprecated``, writing its reason inline::

    @deprecated(
        version=PENDING_DEPRECATION_RELEASE,
        line_length=0,
        reason=f"To be removed in pyQuil v5, where TrajectorySimulator ... instead. {QUAX_REPLACEMENT_NOTE}",
        category=PyQuilDeprecationWarning,
    )
    class QVM: ...

``line_length=0`` stops the decorator from wrapping the reason into the docstring, which would break a
URL at a hyphen (``pyquil-docs``).

The decorator's warning names the class or function ("Call to deprecated class QVM. (...)"), so the
reason need not. Anything else -- a module, or only some uses of a function -- warns with
``warnings.warn(message, PyQuilDeprecationWarning, stacklevel=...)``, whose message must say what is
deprecated. See "Deprecating an API" in CONTRIBUTING.md.
"""

from pyquil._version import DOCS_URL, pyquil_version

__all__ = ["PENDING_DEPRECATION_RELEASE", "QUAX_REPLACEMENT_NOTE", "SIMULATION_DOCS", "PyQuilDeprecationWarning"]

PENDING_DEPRECATION_RELEASE = pyquil_version
"""The version for deprecations that have not been released yet.

It defaults to the installed pyQuil version, which is right once the release that includes them is
installed. Each release pull request replaces every reference to it with that release's version (see
"Release Process" in CONTRIBUTING.md), so released deprecations keep their version.
"""

SIMULATION_DOCS = f"{DOCS_URL}/simulation_architecture.html"
"""The documentation of the Quax-based simulators and noise model that replace pyQuil's older ones."""

QUAX_REPLACEMENT_NOTE = (
    "pyQuil v4 already includes the Quax-based simulators and noise model that replace it, as private, "
    f"experimental modules (see {SIMULATION_DOCS}), so we recommend continuing to use this API until you "
    "upgrade to pyQuil v5."
)
"""A sentence for deprecations whose replacement is the Quax-based simulators or noise model."""


class PyQuilDeprecationWarning(FutureWarning):
    """Warns that a pyQuil API is deprecated and will be removed in pyQuil v5.

    This is a :class:`FutureWarning` rather than a :class:`DeprecationWarning` because Python's default
    filters show ``DeprecationWarning`` only when it is attributed to ``__main__``, which would hide it
    from most users. To silence it, filter on this class, e.g.
    ``warnings.filterwarnings("ignore", category=PyQuilDeprecationWarning)``.
    """

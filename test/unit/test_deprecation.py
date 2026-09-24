"""Tests for the deprecation of APIs slated for removal in pyQuil v5."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from pyquil._deprecation import DEPRECATED_IN_VERSION, PyQuilDeprecationWarning


def _pyquil_deprecations(record) -> list:
    """Filter a ``pytest.warns`` record down to pyQuil v5 deprecations (it records every warning)."""
    return [w for w in record if issubclass(w.category, PyQuilDeprecationWarning)]


def _run_python(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a fresh interpreter with Python's default warning filters."""
    command = [sys.executable, "-c", textwrap.dedent(code)]
    return subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603 - our own code


def test_deprecated_class_warns_at_the_callers_line():
    from pyquil.wavefunction import Wavefunction

    with pytest.warns(PyQuilDeprecationWarning, match="class Wavefunction") as record:
        Wavefunction(np.array([1.0, 0.0]))
    (warning,) = _pyquil_deprecations(record)
    assert warning.filename == __file__


def test_docstring_carries_the_sphinx_directive():
    from pyquil.api import QVM

    assert f".. deprecated:: {DEPRECATED_IN_VERSION}" in (QVM.__doc__ or "")


def test_legacy_noise_warns():
    from pyquil.noise import pauli_kraus_map

    with pytest.warns(PyQuilDeprecationWarning, match="pauli_kraus_map"):
        pauli_kraus_map([1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("method", "arguments"),
    [("define_noisy_gate", ("X", (0,), [np.eye(2)])), ("define_noisy_readout", (0, 0.9, 0.8))],
)
def test_program_noise_methods_warn(method: str, arguments: tuple):
    from pyquil import Program

    with pytest.warns(PyQuilDeprecationWarning, match=method) as record:
        getattr(Program(), method)(*arguments)
    assert any(w.filename == __file__ for w in _pyquil_deprecations(record))


def test_merge_with_pauli_noise_warns():
    from pyquil.gates import X
    from pyquil.quil import Program, merge_with_pauli_noise

    with pytest.warns(PyQuilDeprecationWarning, match="merge_with_pauli_noise"):
        merge_with_pauli_noise([Program(X(0))], [1.0, 0.0, 0.0, 0.0], [0])


@pytest.mark.parametrize(
    ("name", "helper"),
    [
        ("9q-square-qvm", "_get_9q_square_qvm"),
        ("9q-square-pyqvm", "_get_9q_square_qvm"),
        ("5q-qvm", "_get_unrestricted_qvm"),
        ("5q", "_get_unrestricted_qvm"),
    ],
)
def test_get_qc_warns_when_returning_a_qvm(mocker, name: str, helper: str):
    from pyquil.api import get_qc

    mocker.patch(f"pyquil.api._quantum_computer.{helper}", return_value="qc")
    with pytest.warns(PyQuilDeprecationWarning, match="QVM-backed quantum computer") as record:
        assert get_qc(name, as_qvm=True, client_configuration=mocker.Mock()) == "qc"
    (warning,) = _pyquil_deprecations(record)
    assert warning.filename == __file__


def test_local_forest_runtime_warns_on_entry(mocker):
    from pyquil.api import local_forest_runtime

    # Pretend both ports are taken, so no server is started.
    mocker.patch("pyquil.api._quantum_computer._port_used", return_value=True)
    with pytest.warns(PyQuilDeprecationWarning, match="start only quilc") as record:
        with local_forest_runtime() as (qvm, quilc):
            assert qvm is None and quilc is None
    (warning,) = _pyquil_deprecations(record)
    assert warning.filename == __file__


def test_importing_pyquil_does_not_warn_or_import_deprecated_modules():
    result = _run_python(
        """
        import sys, warnings
        warnings.simplefilter("error", FutureWarning)
        import pyquil, pyquil.api, pyquil.noise, pyquil.simulation, pyquil.simulation.tools
        from pyquil.api import QVM, QVMCompiler, WavefunctionSimulator, get_qc
        deprecated = ["pyquil.experiment", "pyquil.latex", "pyquil.operator_estimation", "pyquil.pyqvm",
                      "pyquil.wavefunction"]
        print(sorted(m for m in deprecated if m in sys.modules))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize(
    "module",
    ["pyquil.experiment", "pyquil.latex", "pyquil.operator_estimation", "pyquil.pyqvm", "pyquil.wavefunction"],
)
def test_importing_a_deprecated_module_warns_by_default(module: str):
    result = _run_python(f"import {module}\n")
    assert result.returncode == 0, result.stderr
    # Shown under Python's default filters and attributed to the import statement.
    assert f"<string>:1: PyQuilDeprecationWarning: The module {module} is deprecated." in result.stderr

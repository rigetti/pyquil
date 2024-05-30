import pyquil
from pyquil.diagnostics import get_report


def test_report():
    report = get_report()
    assert report.split("\n").pop(0) == f"pyQuil version: {pyquil.__version__}"

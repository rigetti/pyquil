from pyquil.experiment import ExperimentResult
from pyquil.paulis import sZ, sX
from pyquil.experiment._setting import (
    ExperimentSetting,
    plusX,
)
from pyquil.experiment._group import get_results_by_qubit_groups


def test_results_by_qubit_groups():
    er1 = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)), expectation=0.0, std_err=0.0, total_counts=1,
    )

    er2 = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(1)), expectation=0.0, std_err=0.0, total_counts=1,
    )

    er3 = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sX(0) * sZ(1)),
        expectation=0.0,
        std_err=0.0,
        total_counts=1,
    )

    er4 = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sX(0) * sZ(2)),
        expectation=0.0,
        std_err=0.0,
        total_counts=1,
    )
    groups = [(0,), (1,), (2, 0)]
    res_by_group = get_results_by_qubit_groups([er1, er2, er3, er4], groups)

    assert res_by_group == {(0,): [er1], (1,): [er2], (0, 2): [er1, er4]}

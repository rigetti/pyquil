from pyquil.experiment._main import TomographyExperiment
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import (
    ExperimentSetting,
    plusX,
    minusX,
    plusZ,
    TensorProductState
)
from pyquil.experiment._group import get_results_by_qubit_groups, merge_disjoint_experiments
from pyquil.gates import X, Z
from pyquil.paulis import sZ, sX, sY
from pyquil import Program


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


def test_merge_disjoint_experiments():
    sett1 = ExperimentSetting(TensorProductState(), sX(0) * sY(1))
    sett2 = ExperimentSetting(plusZ(1), sY(1))
    sett3 = ExperimentSetting(plusZ(0), sX(0))
    sett4 = ExperimentSetting(minusX(1), sY(1))
    sett5 = ExperimentSetting(TensorProductState(), sZ(2))

    expt1 = TomographyExperiment(settings=[sett1, sett2], program=Program(X(1)))
    expt2 = TomographyExperiment(settings=[sett3, sett4], program=Program(Z(0)))
    expt3 = TomographyExperiment(settings=[sett5], program=Program())

    merged_expt = merge_disjoint_experiments([expt1, expt2, expt3])
    assert len(merged_expt) == 2

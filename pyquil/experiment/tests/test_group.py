import itertools
import pytest

from pyquil.experiment._main import Experiment
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import (
    ExperimentSetting,
    plusX,
    minusX,
    plusZ,
    minusZ,
    plusY,
    TensorProductState,
    zeros_state,
    _pauli_to_product_state,
)
from pyquil.experiment._group import (
    get_results_by_qubit_groups,
    merge_disjoint_experiments,
    group_settings,
    _max_weight_operator,
    _max_weight_state,
    _max_tpb_overlap,
)
from pyquil.gates import X, Z
from pyquil.paulis import sZ, sX, sY, sI, PauliTerm
from pyquil import Program
from pyquil.gates import H


def test_results_by_qubit_groups():
    er1 = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)),
        expectation=0.0,
        std_err=0.0,
        total_counts=1,
    )

    er2 = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(1)),
        expectation=0.0,
        std_err=0.0,
        total_counts=1,
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

    expt1 = Experiment(settings=[sett1, sett2], program=Program(X(1)))
    expt2 = Experiment(settings=[sett3, sett4], program=Program(Z(0)))
    expt3 = Experiment(settings=[sett5], program=Program())

    merged_expt = merge_disjoint_experiments([expt1, expt2, expt3])
    assert len(merged_expt) == 2


@pytest.fixture(params=["clique-removal", "greedy"])
def grouping_method(request):
    return request.param


def test_expt_settings_share_ntpb():
    expts = [
        [
            ExperimentSetting(zeros_state([0, 1]), sX(0) * sI(1)),
            ExperimentSetting(zeros_state([0, 1]), sI(0) * sX(1)),
        ],
        [
            ExperimentSetting(zeros_state([0, 1]), sZ(0) * sI(1)),
            ExperimentSetting(zeros_state([0, 1]), sI(0) * sZ(1)),
        ],
    ]
    for group in expts:
        for e1, e2 in itertools.combinations(group, 2):
            assert _max_weight_state([e1.in_state, e2.in_state]) is not None
            assert _max_weight_operator([e1.out_operator, e2.out_operator]) is not None


def test_group_experiments(grouping_method):
    expts = [  # cf above, I removed the inner nesting. Still grouped visually
        ExperimentSetting(TensorProductState(), sX(0) * sI(1)),
        ExperimentSetting(TensorProductState(), sI(0) * sX(1)),
        ExperimentSetting(TensorProductState(), sZ(0) * sI(1)),
        ExperimentSetting(TensorProductState(), sI(0) * sZ(1)),
    ]
    suite = Experiment(expts, Program())
    grouped_suite = group_settings(suite, method=grouping_method)
    assert len(suite) == 4
    assert len(grouped_suite) == 2


def test_max_weight_operator_1():
    pauli_terms = [sZ(0), sX(1) * sZ(0), sY(2) * sX(1)]
    assert _max_weight_operator(pauli_terms) == sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_2():
    pauli_terms = [sZ(0), sX(1) * sZ(0), sY(2) * sX(1), sZ(5) * sI(3)]
    assert _max_weight_operator(pauli_terms) == sZ(5) * sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_3():
    pauli_terms = [sZ(0) * sX(5), sX(1) * sZ(0), sY(2) * sX(1), sZ(5) * sI(3)]
    assert _max_weight_operator(pauli_terms) is None


def test_max_weight_operator_misc():
    assert _max_weight_operator([sZ(0), sZ(0) * sZ(1)]) is not None
    assert _max_weight_operator([sX(5), sZ(4)]) is not None
    assert _max_weight_operator([sX(0), sY(0) * sZ(2)]) is None

    x_term = sX(0) * sX(1)
    z1_term = sZ(1)
    z0_term = sZ(0)
    z0z1_term = sZ(0) * sZ(1)
    assert _max_weight_operator([x_term, z1_term]) is None
    assert _max_weight_operator([z0z1_term, x_term]) is None

    assert _max_weight_operator([z1_term, z0_term]) is not None
    assert _max_weight_operator([z0z1_term, z0_term]) is not None
    assert _max_weight_operator([z0z1_term, z1_term]) is not None
    assert _max_weight_operator([z0z1_term, sI(1)]) is not None
    assert _max_weight_operator([z0z1_term, sI(2)]) is not None
    assert _max_weight_operator([z0z1_term, sX(5) * sZ(7)]) is not None

    xxxx_terms = (
        sX(1) * sX(2) + sX(2) + sX(3) * sX(4) + sX(4) + sX(1) * sX(3) * sX(4) + sX(1) * sX(4) + sX(1) * sX(2) * sX(3)
    )
    true_term = sX(1) * sX(2) * sX(3) * sX(4)
    assert _max_weight_operator(xxxx_terms.terms) == true_term

    zzzz_terms = sZ(1) * sZ(2) + sZ(3) * sZ(4) + sZ(1) * sZ(3) + sZ(1) * sZ(3) * sZ(4)
    assert _max_weight_operator(zzzz_terms.terms) == sZ(1) * sZ(2) * sZ(3) * sZ(4)

    pauli_terms = [sZ(0), sX(1) * sZ(0), sY(2) * sX(1), sZ(5) * sI(3)]
    assert _max_weight_operator(pauli_terms) == sZ(5) * sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_4():
    # this last example illustrates that a pair of commuting operators
    # need not be diagonal in the same tpb
    assert _max_weight_operator([sX(1) * sZ(0), sZ(1) * sX(0)]) is None


def test_max_weight_state_1():
    states = [plusX(0) * plusZ(1), plusX(0), plusZ(1)]
    assert _max_weight_state(states) == states[0]


def test_max_weight_state_2():
    states = [plusX(1) * plusZ(0), plusX(0), plusZ(1)]
    assert _max_weight_state(states) is None


def test_max_weight_state_3():
    states = [plusX(0) * minusZ(1), plusX(0), minusZ(1)]
    assert _max_weight_state(states) == states[0]


def test_max_weight_state_4():
    states = [plusX(1) * minusZ(0), plusX(0), minusZ(1)]
    assert _max_weight_state(states) is None


def test_max_tpb_overlap_1():
    tomo_expt_settings = [
        ExperimentSetting(plusZ(1) * plusX(0), sY(2) * sY(1)),
        ExperimentSetting(plusX(2) * plusZ(1), sY(2) * sZ(0)),
    ]
    tomo_expt_program = Program(H(0), H(1), H(2))
    tomo_expt = Experiment(tomo_expt_settings, tomo_expt_program)
    expected_dict = {
        ExperimentSetting(plusX(0) * plusZ(1) * plusX(2), sZ(0) * sY(1) * sY(2)): [
            ExperimentSetting(plusZ(1) * plusX(0), sY(2) * sY(1)),
            ExperimentSetting(plusX(2) * plusZ(1), sY(2) * sZ(0)),
        ]
    }
    assert expected_dict == _max_tpb_overlap(tomo_expt)


def test_max_tpb_overlap_2():
    expt_setting = ExperimentSetting(
        _pauli_to_product_state(PauliTerm.from_compact_str("(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6")),
        PauliTerm.from_compact_str("(1+0j)*Z4X8Y5X3Y7Y1"),
    )
    p = Program(H(0), H(1), H(2))
    tomo_expt = Experiment([expt_setting], p)
    expected_dict = {expt_setting: [expt_setting]}
    assert expected_dict == _max_tpb_overlap(tomo_expt)


def test_max_tpb_overlap_3():
    # add another ExperimentSetting to the above
    expt_setting = ExperimentSetting(
        _pauli_to_product_state(PauliTerm.from_compact_str("(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6")),
        PauliTerm.from_compact_str("(1+0j)*Z4X8Y5X3Y7Y1"),
    )
    expt_setting2 = ExperimentSetting(plusZ(7), sY(1))
    p = Program(H(0), H(1), H(2))
    tomo_expt2 = Experiment([expt_setting, expt_setting2], p)
    expected_dict2 = {expt_setting: [expt_setting, expt_setting2]}
    assert expected_dict2 == _max_tpb_overlap(tomo_expt2)


def test_group_experiments_greedy():
    ungrouped_tomo_expt = Experiment(
        [
            [
                ExperimentSetting(
                    _pauli_to_product_state(PauliTerm.from_compact_str("(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6")),
                    PauliTerm.from_compact_str("(1+0j)*Z4X8Y5X3Y7Y1"),
                )
            ],
            [ExperimentSetting(plusZ(7), sY(1))],
        ],
        program=Program(H(0), H(1), H(2)),
    )
    grouped_tomo_expt = group_settings(ungrouped_tomo_expt, method="greedy")
    expected_grouped_tomo_expt = Experiment(
        [
            [
                ExperimentSetting(
                    TensorProductState.from_str("Z0_7 * Y0_8 * Z0_1 * Y0_4 * Z0_2 * Y0_5 * Y0_0 * X0_6"),
                    PauliTerm.from_compact_str("(1+0j)*Z4X8Y5X3Y7Y1"),
                ),
                ExperimentSetting(plusZ(7), sY(1)),
            ]
        ],
        program=Program(H(0), H(1), H(2)),
    )
    assert grouped_tomo_expt == expected_grouped_tomo_expt


def test_expt_settings_diagonal_in_tpb():
    def _expt_settings_diagonal_in_tpb(es1: ExperimentSetting, es2: ExperimentSetting):
        """
        Extends the concept of being diagonal in the same tpb to ExperimentSettings, by
        determining if the pairs of in_states and out_operators are separately diagonal in the same
        tpb
        """
        max_weight_in = _max_weight_state([es1.in_state, es2.in_state])
        max_weight_out = _max_weight_operator([es1.out_operator, es2.out_operator])
        return max_weight_in is not None and max_weight_out is not None

    expt_setting1 = ExperimentSetting(plusZ(1) * plusX(0), sY(1) * sZ(0))
    expt_setting2 = ExperimentSetting(plusY(2) * plusZ(1), sZ(2) * sY(1))
    assert _expt_settings_diagonal_in_tpb(expt_setting1, expt_setting2)
    expt_setting3 = ExperimentSetting(plusX(2) * plusZ(1), sZ(2) * sY(1))
    expt_setting4 = ExperimentSetting(plusY(2) * plusZ(1), sX(2) * sY(1))
    assert not _expt_settings_diagonal_in_tpb(expt_setting2, expt_setting3)
    assert not _expt_settings_diagonal_in_tpb(expt_setting2, expt_setting4)

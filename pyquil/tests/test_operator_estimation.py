import functools
import itertools
import random
from math import pi
from unittest.mock import Mock
import numpy as np
import functools
from operator import mul

import numpy as np
import pytest

from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.api import WavefunctionSimulator, QVMConnection
from pyquil.operator_estimation import ExperimentSetting, TomographyExperiment, to_json, read_json, \
    group_experiments, ExperimentResult, measure_observables, SIC0, SIC1, SIC2, SIC3, \
    plusX, minusX, plusY, minusY, plusZ, minusZ, \
    _max_tpb_overlap, _max_weight_operator, _max_weight_state, _max_tpb_overlap, \
    TensorProductState, zeros_state, \
    group_experiments, group_experiments_greedy, ExperimentResult, measure_observables, \
    _ops_bool_to_prog, _stack_dicts, _stats_from_measurements, \
    ratio_variance
from pyquil.paulis import sI, sX, sY, sZ, PauliSum, PauliTerm


def _generate_random_states(n_qubits, n_terms):
    oneq_states = [SIC0, SIC1, SIC2, SIC3, plusX, minusX, plusY, minusY, plusZ, minusZ]
    all_s_inds = np.random.randint(len(oneq_states), size=(n_terms, n_qubits))
    states = []
    for s_inds in all_s_inds:
        state = functools.reduce(mul, (oneq_states[pi](i) for i, pi in enumerate(s_inds)),
                                 TensorProductState([]))
        states += [state]
    return states


def _generate_random_paulis(n_qubits, n_terms):
    paulis = [sI, sX, sY, sZ]
    all_op_inds = np.random.randint(len(paulis), size=(n_terms, n_qubits))
    operators = []
    for op_inds in all_op_inds:
        op = functools.reduce(mul, (paulis[pi](i) for i, pi in enumerate(op_inds)), sI(0))
        op *= np.random.uniform(-1, 1)
        operators += [op]
    return operators


def test_experiment_setting():
    in_states = _generate_random_states(n_qubits=4, n_terms=7)
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for ist, oop in zip(in_states, out_ops):
        expt = ExperimentSetting(ist, oop)
        assert str(expt) == expt.serializable()
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_state == ist
        assert expt2.out_operator == oop


@pytest.mark.filterwarnings("ignore:ExperimentSetting")
def test_setting_no_in_back_compat():
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for oop in out_ops:
        expt = ExperimentSetting(sI(), oop)
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_operator == sI()
        assert expt2.out_operator == oop


@pytest.mark.filterwarnings("ignore:ExperimentSetting")
def test_setting_no_in():
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for oop in out_ops:
        expt = ExperimentSetting(zeros_state(oop.get_qubits()), oop)
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_operator == functools.reduce(mul, [sZ(q) for q in oop.get_qubits()], sI())
        assert expt2.out_operator == oop


def test_tomo_experiment():
    expts = [
        ExperimentSetting(sI(), sX(0) * sY(1)),
        ExperimentSetting(sZ(0), sZ(0)),
    ]

    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1)),
        qubits=[0, 1]
    )
    assert len(suite) == 2
    for e1, e2 in zip(expts, suite):
        # experiment suite puts in groups of length 1
        assert len(e2) == 1
        e2 = e2[0]
        assert e1 == e2
    prog_str = str(suite).splitlines()[0]
    assert prog_str == 'X 0; Y 1'


def test_tomo_experiment_pre_grouped():
    expts = [
        [ExperimentSetting(sI(), sX(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sX(1))],
        [ExperimentSetting(sI(), sZ(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sZ(1))],
    ]

    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1)),
        qubits=[0, 1]
    )
    assert len(suite) == 2  # number of groups
    for es1, es2 in zip(expts, suite):
        for e1, e2 in zip(es1, es2):
            assert e1 == e2
    prog_str = str(suite).splitlines()[0]
    assert prog_str == 'X 0; Y 1'


def test_tomo_experiment_empty():
    suite = TomographyExperiment([], program=Program(X(0)), qubits=[0])
    assert len(suite) == 0
    assert str(suite.program) == 'X 0\n'


def test_experiment_deser(tmpdir):
    expts = [
        [ExperimentSetting(sI(), sX(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sX(1))],
        [ExperimentSetting(sI(), sZ(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sZ(1))],
    ]

    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1)),
        qubits=[0, 1]
    )
    to_json(f'{tmpdir}/suite.json', suite)
    suite2 = read_json(f'{tmpdir}/suite.json')
    assert suite == suite2


@pytest.fixture(params=['clique-removal', 'greedy'])
def grouping_method(request):
    return request.param


def test_expt_settings_share_ntpb():
    expts = [[ExperimentSetting(zeros_state([0, 1]), sX(0) * sI(1)), ExperimentSetting(zeros_state([0, 1]), sI(0) * sX(1))],
             [ExperimentSetting(zeros_state([0, 1]), sZ(0) * sI(1)), ExperimentSetting(zeros_state([0, 1]), sI(0) * sZ(1))]]
    for group in expts:
        for e1, e2 in itertools.combinations(group, 2):
            assert _max_weight_state([e1.in_state, e2.in_state]) is not None
            assert _max_weight_operator([e1.out_operator, e2.out_operator]) is not None


def test_group_experiments(grouping_method):
    expts = [  # cf above, I removed the inner nesting. Still grouped visually
        ExperimentSetting(sI(), sX(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sX(1)),
        ExperimentSetting(sI(), sZ(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sZ(1)),
    ]
    suite = TomographyExperiment(expts, Program(), qubits=[0, 1])
    grouped_suite = group_experiments(suite, method=grouping_method)
    assert len(suite) == 4
    assert len(grouped_suite) == 2


def test_experiment_result_compat():
    er = ExperimentResult(
        setting=ExperimentSetting(sX(0), sZ(0)),
        expectation=0.9,
        stddev=0.05,
        total_counts=100,
    )
    assert str(er) == 'X0_0→(1+0j)*Z0: 0.9 +- 0.05'


def test_experiment_result():
    er = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)),
        expectation=0.9,
        stddev=0.05,
        total_counts=100,
    )
    assert str(er) == 'X0_0→(1+0j)*Z0: 0.9 +- 0.05'


def test_measure_observables(forest):
    expts = [
        ExperimentSetting(sI(), o1 * o2)
        for o1, o2 in itertools.product([sI(0), sX(0), sY(0), sZ(0)], [sI(1), sX(1), sY(1), sZ(1)])
    ]
    suite = TomographyExperiment(expts, program=Program(X(0), CNOT(0, 1)), qubits=[0, 1])
    assert len(suite) == 4 * 4
    gsuite = group_experiments(suite)
    assert len(gsuite) == 3 * 3  # can get all the terms with I for free in this case

    qc = get_qc('2q-qvm')
    for res in measure_observables(qc, gsuite, n_shots=10_000):
        if res.setting.out_operator in [sI(), sZ(0), sZ(1), sZ(0) * sZ(1)]:
            assert np.abs(res.expectation) > 0.9
        else:
            assert np.abs(res.expectation) < 0.1


def _random_2q_programs(n_progs=10):
    """Generate random programs that consist of single qubit rotations, a CZ, and single
    qubit rotations.
    """
    r = random.Random(52)

    def RI(qubit, angle):
        # throw away angle so we can randomly choose the identity
        return I(qubit)

    def _random_1q_gate(qubit):
        return r.choice([RI, RX, RY, RZ])(qubit=qubit, angle=r.uniform(0, 2 * pi))

    for _ in range(n_progs):
        prog = Program()
        prog += _random_1q_gate(0)
        prog += _random_1q_gate(1)
        prog += CZ(0, 1)
        prog += _random_1q_gate(0)
        prog += _random_1q_gate(1)
        yield prog


def test_measure_observables_many_progs(forest):
    expts = [
        ExperimentSetting(sI(), o1 * o2)
        for o1, o2 in itertools.product([sI(0), sX(0), sY(0), sZ(0)], [sI(1), sX(1), sY(1), sZ(1)])
    ]

    qc = get_qc('2q-qvm')
    qc.qam.random_seed = 51
    for prog in _random_2q_programs():
        suite = TomographyExperiment(expts, program=prog, qubits=[0, 1])
        assert len(suite) == 4 * 4
        gsuite = group_experiments(suite)
        assert len(gsuite) == 3 * 3  # can get all the terms with I for free in this case

        wfn = WavefunctionSimulator()
        wfn_exps = {}
        for expt in expts:
            wfn_exps[expt] = wfn.expectation(gsuite.program, PauliSum([expt.out_operator]))

        for res in measure_observables(qc, gsuite, n_shots=1_000):
            np.testing.assert_allclose(wfn_exps[res.setting], res.expectation, atol=0.1)


def test_append():
    expts = [
        [ExperimentSetting(sI(), sX(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sX(1))],
        [ExperimentSetting(sI(), sZ(0) * sI(1)), ExperimentSetting(sI(), sI(0) * sZ(1))],
    ]
    suite = TomographyExperiment(
        settings=expts,
        program=Program(X(0), Y(1)),
        qubits=[0, 1]
    )
    suite.append(ExperimentSetting(sI(), sY(0) * sX(1)))
    assert (len(str(suite))) > 0


def test_no_complex_coeffs(forest):
    qc = get_qc('2q-qvm')
    suite = TomographyExperiment([ExperimentSetting(sI(), 1.j * sY(0))], program=Program(X(0)),
                                 qubits=[0])
    with pytest.raises(ValueError):
        res = list(measure_observables(qc, suite))


def test_max_weight_operator_1():
    pauli_terms = [sZ(0),
                   sX(1) * sZ(0),
                   sY(2) * sX(1)]
    assert _max_weight_operator(pauli_terms) == sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_2():
    pauli_terms = [sZ(0),
                   sX(1) * sZ(0),
                   sY(2) * sX(1),
                   sZ(5) * sI(3)]
    assert _max_weight_operator(pauli_terms) == sZ(5) * sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_3():
    pauli_terms = [sZ(0) * sX(5),
                   sX(1) * sZ(0),
                   sY(2) * sX(1),
                   sZ(5) * sI(3)]
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

    xxxx_terms = sX(1) * sX(2) + sX(2) + sX(3) * sX(4) + sX(4) + \
        sX(1) * sX(3) * sX(4) + sX(1) * sX(4) + sX(1) * sX(2) * sX(3)
    true_term = sX(1) * sX(2) * sX(3) * sX(4)
    assert _max_weight_operator(xxxx_terms.terms) == true_term

    zzzz_terms = sZ(1) * sZ(2) + sZ(3) * sZ(4) + \
        sZ(1) * sZ(3) + sZ(1) * sZ(3) * sZ(4)
    assert _max_weight_operator(zzzz_terms.terms) == sZ(1) * sZ(2) * \
        sZ(3) * sZ(4)

    pauli_terms = [sZ(0), sX(1) * sZ(0), sY(2) * sX(1), sZ(5) * sI(3)]
    assert _max_weight_operator(pauli_terms) == sZ(5) * sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_4():
    # this last example illustrates that a pair of commuting operators
    # need not be diagonal in the same tpb
    assert _max_weight_operator([sX(1) * sZ(0), sZ(1) * sX(0)]) is None


def test_max_weight_state_1():
    states = [plusX(0) * plusZ(1),
              plusX(0),
              plusZ(1),
              ]
    assert _max_weight_state(states) == states[0]


def test_max_weight_state_2():
    states = [plusX(1) * plusZ(0),
              plusX(0),
              plusZ(1),
              ]
    assert _max_weight_state(states) is None


def test_max_weight_state_3():
    states = [plusX(0) * minusZ(1),
              plusX(0),
              minusZ(1),
              ]
    assert _max_weight_state(states) == states[0]


def test_max_weight_state_4():
    states = [plusX(1) * minusZ(0),
              plusX(0),
              minusZ(1),
              ]
    assert _max_weight_state(states) is None


def test_max_tpb_overlap_1():
    tomo_expt_settings = [ExperimentSetting(sZ(1) * sX(0), sY(2) * sY(1)),
                          ExperimentSetting(sX(2) * sZ(1), sY(2) * sZ(0))]
    tomo_expt_program = Program(H(0), H(1), H(2))
    tomo_expt_qubits = [0, 1, 2]
    tomo_expt = TomographyExperiment(tomo_expt_settings, tomo_expt_program, tomo_expt_qubits)
    expected_dict = {
        ExperimentSetting(plusX(0) * plusZ(1) * plusX(2), sZ(0) * sY(1) * sY(2)): [
            ExperimentSetting(plusZ(1) * plusX(0), sY(2) * sY(1)),
            ExperimentSetting(plusX(2) * plusZ(1), sY(2) * sZ(0))
        ]
    }
    assert expected_dict == _max_tpb_overlap(tomo_expt)


def test_max_tpb_overlap_2():
    expt_setting = ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                                     PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))
    p = Program(H(0), H(1), H(2))
    qubits = [0, 1, 2]
    tomo_expt = TomographyExperiment([expt_setting], p, qubits)
    expected_dict = {expt_setting: [expt_setting]}
    assert expected_dict == _max_tpb_overlap(tomo_expt)


def test_max_tpb_overlap_3():
    # add another ExperimentSetting to the above
    expt_setting = ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                                     PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))
    expt_setting2 = ExperimentSetting(sZ(7), sY(1))
    p = Program(H(0), H(1), H(2))
    qubits = [0, 1, 2]
    tomo_expt2 = TomographyExperiment([expt_setting, expt_setting2], p, qubits)
    expected_dict2 = {expt_setting: [expt_setting, expt_setting2]}
    assert expected_dict2 == _max_tpb_overlap(tomo_expt2)


def test_group_experiments_greedy():
    ungrouped_tomo_expt = TomographyExperiment(
        [[ExperimentSetting(PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'),
                            PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))],
         [ExperimentSetting(sZ(7), sY(1))]], program=Program(H(0), H(1), H(2)),
        qubits=[0, 1, 2])
    grouped_tomo_expt = group_experiments(ungrouped_tomo_expt, method='greedy')
    expected_grouped_tomo_expt = TomographyExperiment(
        [[
            ExperimentSetting(TensorProductState.from_str('Z0_7 * Y0_8 * Z0_1 * Y0_4 * '
                                                          'Z0_2 * Y0_5 * Y0_0 * X0_6'),
                              PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1')),
            ExperimentSetting(plusZ(7), sY(1))
        ]],
        program=Program(H(0), H(1), H(2)),
        qubits=[0, 1, 2])
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


def test_identity(forest):
    qc = get_qc('2q-qvm')
    suite = TomographyExperiment([ExperimentSetting(plusZ(0), 0.123 * sI(0))],
                                 program=Program(X(0)), qubits=[0])
    result = list(measure_observables(qc, suite))[0]
    assert result.expectation == 0.123


def test_sic_process_tomo(forest):
    qc = get_qc('2q-qvm')
    process = Program(X(0))
    settings = []
    for in_state in [SIC0, SIC1, SIC2, SIC3]:
        for out_op in [sI, sX, sY, sZ]:
            settings += [ExperimentSetting(
                in_state=in_state(q=0),
                out_operator=out_op(q=0)
            )]

    experiment = TomographyExperiment(settings=settings, program=process, qubits=[0])
    results = list(measure_observables(qc, experiment))
    assert len(results) == 4 * 4


def test_measure_observables_symmetrize(forest):
    """
    Symmetrization alone should not change the outcome on the QVM
    """
    expts = [
        ExperimentSetting(sI(), o1 * o2)
        for o1, o2 in itertools.product([sI(0), sX(0), sY(0), sZ(0)], [sI(1), sX(1), sY(1), sZ(1)])
    ]
    suite = TomographyExperiment(expts, program=Program(X(0), CNOT(0, 1)), qubits=[0, 1])
    assert len(suite) == 4 * 4
    gsuite = group_experiments(suite)
    assert len(gsuite) == 3 * 3  # can get all the terms with I for free in this case

    qc = get_qc('2q-qvm')
    for res in measure_observables(qc, gsuite, n_shots=10_000, readout_symmetrize='exhaustive'):
        if res.setting.out_operator in [sI(), sZ(0), sZ(1), sZ(0) * sZ(1)]:
            assert np.abs(res.expectation) > 0.9
        else:
            assert np.abs(res.expectation) < 0.1


def test_measure_observables_symmetrize_calibrate(forest):
    """
    Symmetrization + calibration should not change the outcome on the QVM
    """
    expts = [
        ExperimentSetting(sI(), o1 * o2)
        for o1, o2 in itertools.product([sI(0), sX(0), sY(0), sZ(0)], [sI(1), sX(1), sY(1), sZ(1)])
    ]
    suite = TomographyExperiment(expts, program=Program(X(0), CNOT(0, 1)), qubits=[0, 1])
    assert len(suite) == 4 * 4
    gsuite = group_experiments(suite)
    assert len(gsuite) == 3 * 3  # can get all the terms with I for free in this case

    qc = get_qc('2q-qvm')
    for res in measure_observables(qc, gsuite, n_shots=10_000,
                                   readout_symmetrize='exhaustive', calibrate_readout='plus-eig'):
        if res.setting.out_operator in [sI(), sZ(0), sZ(1), sZ(0) * sZ(1)]:
            assert np.abs(res.expectation) > 0.9
        else:
            assert np.abs(res.expectation) < 0.1


def test_measure_observables_zero_expectation(forest):
    """
    Testing case when expectation value of observable should be close to zero
    """
    qc = get_qc('2q-qvm')
    exptsetting = ExperimentSetting(plusZ(0), sX(0))
    suite = TomographyExperiment([exptsetting],
                                 program=Program(I(0)), qubits=[0])
    result = list(measure_observables(qc, suite, n_shots=10000, readout_symmetrize='exhaustive',
                                      calibrate_readout='plus-eig'))[0]
    np.testing.assert_almost_equal(result.expectation, 0.0, decimal=1)


def test_measure_observables_no_symm_calibr_raises_error(forest):
    qc = get_qc('2q-qvm')
    exptsetting = ExperimentSetting(plusZ(0), sX(0))
    suite = TomographyExperiment([exptsetting],
                                 program=Program(I(0)), qubits=[0])
    with pytest.raises(ValueError):
        result = list(measure_observables(qc, suite, n_shots=1000,
                                          readout_symmetrize=None, calibrate_readout='plus-eig'))


def test_ops_bool_to_prog():
    qubits = [0, 2, 3]
    ops_strings = list(itertools.product([0, 1], repeat=len(qubits)))
    d_expected = {(0, 0, 0): '', (0, 0, 1): 'X 3\n', (0, 1, 0): 'X 2\n', (0, 1, 1): 'X 2\nX 3\n',
                  (1, 0, 0): 'X 0\n', (1, 0, 1): 'X 0\nX 3\n', (1, 1, 0): 'X 0\nX 2\n',
                  (1, 1, 1): 'X 0\nX 2\nX 3\n'}
    for op_str in ops_strings:
        p = _ops_bool_to_prog(op_str, qubits)
        assert str(p) == d_expected[op_str]


def test_stack_dicts():
    d1 = {0: np.array([0] * 10), 1: np.array([1] * 10)}
    d2 = {0: np.array([10] * 10), 1: np.array([20] * 10)}

    d12 = _stack_dicts(d1, d2)
    d12_expected = {0: np.array([0] * 10 + [10] * 10), 1: np.array([1] * 10 + [20] * 10)}
    assert d12.keys() == d12_expected.keys()
    for k, v in d12.items():
        np.testing.assert_allclose(v, d12_expected[k])


def test_stack_multiple_dicts():
    d1 = {0: np.array([0] * 10), 1: np.array([1] * 10)}
    d2 = {0: np.array([10] * 10), 1: np.array([20] * 10)}
    d3 = {0: np.array([100] * 10), 1: np.array([200] * 10)}
    list_dicts = [d1, d2, d3]

    d123 = functools.reduce(lambda d1, d2: _stack_dicts(d1, d2), list_dicts)
    d123_expected = {0: np.array([0] * 10 + [10] * 10 + [100] * 10),
                     1: np.array([1] * 10 + [20] * 10 + [200] * 10)}
    assert d123.keys() == d123_expected.keys()
    for k, v in d123.items():
        np.testing.assert_allclose(v, d123_expected[k])


def test_stats_from_measurements():
    d_results = {0: np.array([0] * 10), 1: np.array([1] * 10)}
    setting = ExperimentSetting(TensorProductState(), sZ(0) * sX(1))
    n_shots = 1000

    obs_mean, obs_var = _stats_from_measurements(d_results, setting, n_shots)
    assert obs_mean == -1.0
    assert obs_var == 0.0


def test_ratio_variance_float():
    a, b, var_a, var_b = 1.0, 2.0, 0.1, 0.05
    ab_ratio_var = ratio_variance(a, var_a, b, var_b)
    assert ab_ratio_var == 0.028125


def test_ratio_variance_numerator_zero():
    # denominator can't be zero, but numerator can be
    a, b, var_a, var_b = 0.0, 2.0, 0.1, 0.05
    ab_ratio_var = ratio_variance(a, var_a, b, var_b)
    assert ab_ratio_var == 0.025


def test_ratio_variance_array():
    a = np.array([1.0, 10.0, 100.0])
    b = np.array([2.0, 20.0, 200.0])
    var_a = np.array([0.1, 1.0, 10.0])
    var_b = np.array([0.05, 0.5, 5.0])
    ab_ratio_var = ratio_variance(a, var_a, b, var_b)
    np.testing.assert_allclose(ab_ratio_var, np.array([0.028125, 0.0028125, 0.00028125]))

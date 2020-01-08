import numpy as np

from pyquil import Program
from pyquil.experiment._main import _remove_reset_from_program
from pyquil.experiment._program import (
    parameterized_readout_symmetrization,
    parameterized_single_qubit_measurement_basis,
    parameterized_single_qubit_state_preparation,
)
from pyquil.experiment import (
    ExperimentSetting,
    TensorProductState,
    Experiment,
    plusZ,
    read_json,
    to_json,
)
from pyquil.gates import MEASURE, RESET, X, Y
from pyquil.paulis import sI, sX, sY, sZ


EXPERIMENT_REPR = """
shots: 1
active reset: disabled
symmetrization: -1 (exhaustive)
calibration: 1 (plus_eigenstate)
program:
   X 0
   Y 1
"""


def test_tomo_experiment():
    expts = [
        ExperimentSetting(TensorProductState(), sX(0) * sY(1)),
        ExperimentSetting(plusZ(0), sZ(0)),
    ]

    suite = Experiment(settings=expts, program=Program(X(0), Y(1)))
    assert len(suite) == 2
    for e1, e2 in zip(expts, suite):
        # experiment suite puts in groups of length 1
        assert len(e2) == 1
        e2 = e2[0]
        assert e1 == e2
    prog_str = str(suite).splitlines()[3:5]
    assert prog_str == EXPERIMENT_REPR.splitlines()[4:6]


def test_tomo_experiment_pre_grouped():
    expts = [
        [
            ExperimentSetting(TensorProductState(), sX(0) * sI(1)),
            ExperimentSetting(TensorProductState(), sI(0) * sX(1)),
        ],
        [
            ExperimentSetting(TensorProductState(), sZ(0) * sI(1)),
            ExperimentSetting(TensorProductState(), sI(0) * sZ(1)),
        ],
    ]

    suite = Experiment(settings=expts, program=Program(X(0), Y(1)))
    assert len(suite) == 2  # number of groups
    for es1, es2 in zip(expts, suite):
        for e1, e2 in zip(es1, es2):
            assert e1 == e2
    prog_str = str(suite).splitlines()[3:5]
    assert prog_str == EXPERIMENT_REPR.splitlines()[4:6]


def test_tomo_experiment_empty():
    suite = Experiment([], program=Program(X(0)))
    assert len(suite) == 0
    assert str(suite.program) == "X 0\n"


def test_experiment_deser(tmpdir):
    expts = [
        [
            ExperimentSetting(TensorProductState(), sX(0) * sI(1)),
            ExperimentSetting(TensorProductState(), sI(0) * sX(1)),
        ],
        [
            ExperimentSetting(TensorProductState(), sZ(0) * sI(1)),
            ExperimentSetting(TensorProductState(), sI(0) * sZ(1)),
        ],
    ]

    suite = Experiment(settings=expts, program=Program(X(0), Y(1)))
    to_json(f"{tmpdir}/suite.json", suite)
    suite2 = read_json(f"{tmpdir}/suite.json")
    assert suite == suite2


DEFGATE_X = """
DEFGATE XGATE:
    0, 1
    1, 0
"""


TRIMMED_PROG = """
DEFGATE XGATE:
    0, 1
    1, 0

X 0
"""


def test_remove_reset_from_program():
    p = Program(DEFGATE_X)
    p += RESET()
    p += X(0)
    new_p = _remove_reset_from_program(p)
    assert "\n" + new_p.out() == TRIMMED_PROG


def test_generate_experiment_program():
    # simplest example
    p = Program()
    s = ExperimentSetting(in_state=sZ(0), out_operator=sZ(0))
    e = Experiment(settings=[s], program=p, symmetrization=0)
    exp = e.generate_experiment_program()
    test_exp = Program()
    ro = test_exp.declare("ro", "BIT")
    test_exp += MEASURE(0, ro[0])
    assert exp.out() == test_exp.out()
    assert exp.num_shots == 1

    # 2Q exhaustive symmetrization
    p = Program()
    s = ExperimentSetting(in_state=sZ(0) * sZ(1), out_operator=sZ(0) * sZ(1))
    e = Experiment(settings=[s], program=p)
    exp = e.generate_experiment_program()
    test_exp = Program()
    test_exp += parameterized_readout_symmetrization([0, 1])
    ro = test_exp.declare("ro", "BIT", 2)
    test_exp += MEASURE(0, ro[0])
    test_exp += MEASURE(1, ro[1])
    assert exp.out() == test_exp.out()
    assert exp.num_shots == 1

    # add shots
    p = Program()
    p.wrap_in_numshots_loop(1000)
    s = ExperimentSetting(in_state=sZ(0), out_operator=sZ(0))
    e = Experiment(settings=[s], program=p, symmetrization=0)
    exp = e.generate_experiment_program()
    test_exp = Program()
    ro = test_exp.declare("ro", "BIT")
    test_exp += MEASURE(0, ro[0])
    assert exp.out() == test_exp.out()
    assert exp.num_shots == 1000

    # active reset
    p = Program()
    p += RESET()
    s = ExperimentSetting(in_state=sZ(0), out_operator=sZ(0))
    e = Experiment(settings=[s], program=p, symmetrization=0)
    exp = e.generate_experiment_program()
    test_exp = Program()
    test_exp += RESET()
    ro = test_exp.declare("ro", "BIT")
    test_exp += MEASURE(0, ro[0])
    assert exp.out() == test_exp.out()
    assert exp.num_shots == 1

    # state preparation and measurement
    p = Program()
    s = ExperimentSetting(in_state=sY(0), out_operator=sX(0))
    e = Experiment(settings=[s], program=p, symmetrization=0)
    exp = e.generate_experiment_program()
    test_exp = Program()
    test_exp += parameterized_single_qubit_state_preparation([0])
    test_exp += parameterized_single_qubit_measurement_basis([0])
    ro = test_exp.declare("ro", "BIT")
    test_exp += MEASURE(0, ro[0])
    assert exp.out() == test_exp.out()
    assert exp.num_shots == 1

    # multi-qubit state preparation and measurement
    p = Program()
    s = ExperimentSetting(in_state=sZ(0) * sY(1), out_operator=sZ(0) * sX(1))
    e = Experiment(settings=[s], program=p, symmetrization=0)
    exp = e.generate_experiment_program()
    test_exp = Program()
    test_exp += parameterized_single_qubit_state_preparation([0, 1])
    test_exp += parameterized_single_qubit_measurement_basis([0, 1])
    ro = test_exp.declare("ro", "BIT", 2)
    test_exp += MEASURE(0, ro[0])
    test_exp += MEASURE(1, ro[1])
    assert exp.out() == test_exp.out()
    assert exp.num_shots == 1


def test_build_experiment_setting_memory_map():
    p = Program()
    s = ExperimentSetting(in_state=sX(0), out_operator=sZ(0) * sY(1))
    e = Experiment(settings=[s], program=p)
    memory_map = e.build_setting_memory_map(s)
    assert memory_map == {
        "preparation_alpha": [0.0],
        "preparation_beta": [np.pi / 2],
        "preparation_gamma": [0.0],
        "measurement_alpha": [0.0, np.pi / 2],
        "measurement_beta": [0.0, np.pi / 2],
        "measurement_gamma": [0.0, -np.pi / 2],
    }


def test_build_symmetrization_memory_maps():
    p = Program()
    s = ExperimentSetting(in_state=sZ(0) * sZ(1), out_operator=sZ(0) * sZ(1))
    e = Experiment(settings=[s], program=p)
    memory_maps = [
        {"symmetrization": [0.0, 0.0]},
        {"symmetrization": [0.0, np.pi]},
        {"symmetrization": [np.pi, 0.0]},
        {"symmetrization": [np.pi, np.pi]},
    ]
    assert e.build_symmetrization_memory_maps([0, 1]) == memory_maps

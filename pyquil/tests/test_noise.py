import numpy as np
from unittest.mock import Mock
import pytest

from pyquil.noise import (
    pauli_kraus_map,
    damping_kraus_map,
    dephasing_kraus_map,
    tensor_kraus_maps,
    _get_program_gates,
    _decoherence_noise_model,
    add_decoherence_noise,
    combine_kraus_maps,
    damping_after_dephasing,
    INFINITY,
    apply_noise_model,
    _noise_model_program_header,
    KrausModel,
    NoiseModel,
    corrupt_bitstring_probs,
    correct_bitstring_probs,
    estimate_bitstring_probs,
    bitstring_probs_to_z_moments,
    estimate_assignment_probs,
    NO_NOISE,
)
from pyquil.gates import RZ, RX, I, CZ
from collections import OrderedDict

from pyquil.quil import Pragma, Program
from pyquil.quilbase import DefGate, Gate
from pyquil.api import QVMConnection


def test_pauli_kraus_map():
    probabilities = [0.1, 0.2, 0.3, 0.4]
    k1, k2, k3, k4 = pauli_kraus_map(probabilities)
    assert np.allclose(k1, np.sqrt(0.1) * np.eye(2), atol=1 * 10 ** -8)
    assert np.allclose(k2, np.sqrt(0.2) * np.array([[0, 1.0], [1.0, 0]]), atol=1 * 10 ** -8)
    assert np.allclose(k3, np.sqrt(0.3) * np.array([[0, -1.0j], [1.0j, 0]]), atol=1 * 10 ** -8)
    assert np.allclose(k4, np.sqrt(0.4) * np.array([[1, 0], [0, -1]]), atol=1 * 10 ** -8)

    two_q_pauli_kmaps = pauli_kraus_map(np.kron(probabilities, list(reversed(probabilities))))
    q1_pauli_kmaps = [k1, k2, k3, k4]
    q2_pauli_kmaps = pauli_kraus_map(list(reversed(probabilities)))
    tensor_kmaps = tensor_kraus_maps(q1_pauli_kmaps, q2_pauli_kmaps)
    assert np.allclose(two_q_pauli_kmaps, tensor_kmaps)


def test_damping_kraus_map():
    p = 0.05
    k1, k2 = damping_kraus_map(p=p)
    assert k1[1, 1] == np.sqrt(1 - p)
    assert k2[0, 1] == np.sqrt(p)


def test_dephasing_kraus_map():
    p = 0.05
    k1, k2 = dephasing_kraus_map(p=p)
    np.testing.assert_allclose(np.diag(k1), [np.sqrt(1 - p)] * 2)
    np.testing.assert_allclose(np.abs(np.diag(k2)), [np.sqrt(p)] * 2)


def test_tensor_kraus_maps():
    damping = damping_kraus_map()
    k1, k2, k3, k4 = tensor_kraus_maps(damping, damping)
    assert k1.shape == (4, 4)
    assert k2.shape == (4, 4)
    assert k3.shape == (4, 4)
    assert k4.shape == (4, 4)
    np.testing.assert_allclose(k1[-1, -1], 1 - 0.1)


def test_combine_kraus_maps():
    damping = damping_kraus_map()
    dephasing = dephasing_kraus_map()
    k1, k2, k3, k4 = combine_kraus_maps(damping, dephasing)
    assert k1.shape == (2, 2)
    assert k2.shape == (2, 2)
    assert k3.shape == (2, 2)
    assert k4.shape == (2, 2)


def test_damping_after_dephasing():
    gate_time = 1
    T1 = 10
    T2 = 3
    kraus_map = damping_after_dephasing(T1, T2, gate_time)

    # Density matrix for the |+> state
    rho = 0.5 * np.ones((2, 2))

    # See Eq. 7.144 of Nielsen and Chuang
    target_rho = [
        [1 - 0.5 * np.exp(-gate_time / T1), 0.5 * np.exp(-gate_time / T2)],
        [0.5 * np.exp(-gate_time / T2), 0.5 * np.exp(-gate_time / T1)],
    ]

    noisy_rho = np.zeros((2, 2))
    for op in kraus_map:
        noisy_rho += op @ rho @ (op.T.conj())

    np.testing.assert_allclose(noisy_rho, target_rho)


def test_noise_helpers():
    gates = RX(np.pi / 2, 0), RX(-np.pi / 2, 1), I(1), CZ(0, 1)
    prog = Program(*gates)
    inferred_gates = _get_program_gates(prog)
    assert set(inferred_gates) == set(gates)


def test_decoherence_noise():
    prog = Program(RX(np.pi / 2, 0), CZ(0, 1), RZ(np.pi, 0))
    gates = _get_program_gates(prog)
    m1 = _decoherence_noise_model(gates, T1=INFINITY, T2=INFINITY, ro_fidelity=1.0)

    # with no readout error, assignment_probs = identity matrix
    assert np.allclose(m1.assignment_probs[0], np.eye(2))
    assert np.allclose(m1.assignment_probs[1], np.eye(2))
    for g in m1.gates:
        # with infinite coherence time all kraus maps should only have a single, unitary kraus op
        assert len(g.kraus_ops) == 1
        (k0,) = g.kraus_ops
        # check unitarity
        k0dk0 = k0.dot(k0.conjugate().transpose())
        assert np.allclose(k0dk0, np.eye(k0dk0.shape[0]))

    # verify that selective (by qubit) dephasing and readout infidelity is working
    m2 = _decoherence_noise_model(gates, T1=INFINITY, T2={0: 30e-6}, ro_fidelity={0: 0.95, 1: 1.0})
    assert np.allclose(m2.assignment_probs[0], [[0.95, 0.05], [0.05, 0.95]])
    assert np.allclose(m2.assignment_probs[1], np.eye(2))
    for g in m2.gates:
        if 0 in g.targets:
            # single dephasing (no damping) channel on qc 0, no noise on qc1 -> 2 Kraus ops
            assert len(g.kraus_ops) == 2
        else:
            assert len(g.kraus_ops) == 1

    # verify that combined T1 and T2 will lead to 4 outcome Kraus map.
    m3 = _decoherence_noise_model(gates, T1={0: 30e-6}, T2={0: 30e-6})
    for g in m3.gates:
        if 0 in g.targets:
            # damping (implies dephasing) channel on qc 0, no noise on qc1 -> 4 Kraus ops
            assert len(g.kraus_ops) == 4
        else:
            assert len(g.kraus_ops) == 1

    # verify that gate names are translated
    new_prog = apply_noise_model(prog, m3)

    # check that headers have been embedded
    headers = _noise_model_program_header(m3)
    assert all(
        (isinstance(i, Pragma) and i.command in ["ADD-KRAUS", "READOUT-POVM"]) or isinstance(i, DefGate)
        for i in headers
    )
    assert headers.out() in new_prog.out()

    # verify that high-level add_decoherence_noise reproduces new_prog
    new_prog2 = add_decoherence_noise(prog, T1={0: 30e-6}, T2={0: 30e-6})
    assert new_prog == new_prog2


def test_kraus_model_1():
    km = KrausModel("I", (5.0,), (0, 1), [np.array([[1 + 1j]])], 1.0)
    d = km.to_dict()
    assert d == OrderedDict(
        [
            ("gate", km.gate),
            ("params", km.params),
            ("targets", (0, 1)),
            ("kraus_ops", [[[[1.0]], [[1.0]]]]),
            ("fidelity", 1.0),
        ]
    )
    assert KrausModel.from_dict(d) == km


@pytest.fixture
def kraus_model_I_dict():
    return {
        "gate": "I",
        "fidelity": 1.0,
        "kraus_ops": [[[[1.0]], [[1.0]]]],
        "targets": (0, 1),
        "params": (5.0,),
    }


def test_kraus_model_2(kraus_model_I_dict):
    km = KrausModel.from_dict(kraus_model_I_dict)
    assert km == KrausModel(
        gate=kraus_model_I_dict["gate"],
        params=kraus_model_I_dict["params"],
        targets=kraus_model_I_dict["targets"],
        kraus_ops=[KrausModel.unpack_kraus_matrix(kraus_op) for kraus_op in kraus_model_I_dict["kraus_ops"]],
        fidelity=kraus_model_I_dict["fidelity"],
    )
    d = km.to_dict()
    assert d == OrderedDict(
        [
            ("gate", km.gate),
            ("params", km.params),
            ("targets", (0, 1)),
            ("kraus_ops", [[[[1.0]], [[1.0]]]]),
            ("fidelity", 1.0),
        ]
    )


def test_noise_model_1():
    km1 = KrausModel("I", (5.0,), (0, 1), [np.array([[1 + 1j]])], 1.0)
    km2 = KrausModel("RX", (np.pi / 2,), (0,), [np.array([[1 + 1j]])], 1.0)
    nm = NoiseModel([km1, km2], {0: np.eye(2), 1: np.eye(2)})

    assert nm == NoiseModel.from_dict(nm.to_dict())
    assert nm.gates_by_name("I") == [km1]
    assert nm.gates_by_name("RX") == [km2]


@pytest.fixture
def kraus_model_RX90_dict():
    return {
        "gate": "RX",
        "fidelity": 1.0,
        "kraus_ops": [[[[1.0]], [[1.0]]]],
        "targets": (0,),
        "params": (np.pi / 2.0,),
    }


def test_noise_model_2(kraus_model_I_dict, kraus_model_RX90_dict):
    noise_model_dict = {
        "gates": [kraus_model_I_dict, kraus_model_RX90_dict],
        "assignment_probs": {"1": [[1.0, 0.0], [0.0, 1.0]], "0": [[1.0, 0.0], [0.0, 1.0]]},
    }

    nm = NoiseModel.from_dict(noise_model_dict)
    km1 = KrausModel.from_dict(kraus_model_I_dict)
    km2 = KrausModel.from_dict(kraus_model_RX90_dict)
    assert nm == NoiseModel(gates=[km1, km2], assignment_probs={0: np.eye(2), 1: np.eye(2)})
    assert nm.gates_by_name("I") == [km1]
    assert nm.gates_by_name("RX") == [km2]
    assert nm.to_dict() == noise_model_dict


def test_readout_compensation():
    np.random.seed(1234124)
    p = np.random.rand(2, 2, 2, 2, 2, 2)
    p /= p.sum()

    aps = [np.eye(2) + 0.2 * (np.random.rand(2, 2) - 1) for _ in range(p.ndim)]
    for ap in aps:
        ap.flat[ap.flat < 0] = 0.0
        ap /= ap.sum()
        assert np.alltrue(ap >= 0)

    assert np.alltrue(p >= 0)

    p_corrupted = corrupt_bitstring_probs(p, aps)
    p_restored = correct_bitstring_probs(p_corrupted, aps)
    assert np.allclose(p, p_restored)

    results = [[0, 0, 0]] * 100 + [[0, 1, 1]] * 200
    p1 = estimate_bitstring_probs(results)
    assert np.isclose(p1[0, 0, 0], 1.0 / 3.0)
    assert np.isclose(p1[0, 1, 1], 2.0 / 3.0)
    assert np.isclose(p1.sum(), 1.0)

    zm = bitstring_probs_to_z_moments(p1)
    assert np.isclose(zm[0, 0, 0], 1)
    assert np.isclose(zm[1, 0, 0], 1)
    assert np.isclose(zm[0, 1, 0], -1.0 / 3)
    assert np.isclose(zm[0, 0, 1], -1.0 / 3)
    assert np.isclose(zm[0, 1, 1], 1.0)
    assert np.isclose(zm[1, 1, 0], -1.0 / 3)
    assert np.isclose(zm[1, 0, 1], -1.0 / 3)
    assert np.isclose(zm[1, 1, 1], 1.0)


def test_estimate_assignment_probs():
    cxn = Mock(spec=QVMConnection)
    trials = 100
    p00 = 0.8
    p11 = 0.75
    cxn.run.side_effect = [
        [[0]] * int(round(p00 * trials)) + [[1]] * int(round((1 - p00) * trials)),
        [[1]] * int(round(p11 * trials)) + [[0]] * int(round((1 - p11) * trials)),
    ]
    ap_target = np.array([[p00, 1 - p11], [1 - p00, p11]])

    povm_pragma = Pragma("READOUT-POVM", (0, "({} {} {} {})".format(*ap_target.flatten())))
    ap = estimate_assignment_probs(0, trials, cxn, Program(povm_pragma))
    assert np.allclose(ap, ap_target)
    for call in cxn.run.call_args_list:
        args, kwargs = call
        prog = args[0]
        assert prog._instructions[0] == povm_pragma


def test_apply_noise_model():
    p = Program(RX(np.pi / 2, 0), RX(np.pi / 2, 1), CZ(0, 1), RX(np.pi / 2, 1))
    noise_model = _decoherence_noise_model(_get_program_gates(p))
    pnoisy = apply_noise_model(p, noise_model)
    for i in pnoisy:
        if isinstance(i, DefGate):
            pass
        elif isinstance(i, Pragma):
            assert i.command in ["ADD-KRAUS", "READOUT-POVM"]
        elif isinstance(i, Gate):
            assert i.name in NO_NOISE or not i.params


def test_apply_noise_model_perturbed_angles():
    eps = 1e-15
    p = Program(RX(np.pi / 2 + eps, 0), RX(np.pi / 2 - eps, 1), CZ(0, 1), RX(np.pi / 2 + eps, 1))
    noise_model = _decoherence_noise_model(_get_program_gates(p))
    pnoisy = apply_noise_model(p, noise_model)
    for i in pnoisy:
        if isinstance(i, DefGate):
            pass
        elif isinstance(i, Pragma):
            assert i.command in ["ADD-KRAUS", "READOUT-POVM"]
        elif isinstance(i, Gate):
            assert i.name in NO_NOISE or not i.params

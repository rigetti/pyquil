"""Unit tests for the rewritten local QVM (density-matrix-backed)."""

import numpy as np
import pytest
import networkx as nx
from qcs_sdk import QCSClient

from pyquil import Program
from pyquil.api._quantum_computer import QuantumComputer
from pyquil.api._qvm import QVM
from pyquil.gates import CNOT, H, MEASURE, RX, X
from pyquil.quantum_processor import NxQuantumProcessor
from pyquil.quilatom import MemoryReference
from pyquil.quilbase import Declare
from test.unit.utils import DummyCompiler


def _make_qc(client_configuration, noise_model=None, random_seed=None):
    quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
    return QuantumComputer(
        name="test-qvm",
        qam=QVM(noise_model=noise_model, random_seed=random_seed),
        compiler=DummyCompiler(
            quantum_processor=quantum_processor,
            client_configuration=client_configuration,
        ),
    )


class TestQVMBasic:
    def test_noiseless_bell_state(self, client_configuration: QCSClient):
        qc = _make_qc(client_configuration, random_seed=42)
        p = Program(
            Declare("ro", "BIT", 2),
            H(0),
            CNOT(0, 1),
            MEASURE(0, MemoryReference("ro", 0)),
            MEASURE(1, MemoryReference("ro", 1)),
        ).wrap_in_numshots_loop(100)
        result = qc.run(p)
        bitstrings = result.readout_data.get("ro")
        assert bitstrings.shape == (100, 2)
        # Bell state: only |00> and |11>
        for row in bitstrings:
            assert tuple(row) in [(0, 0), (1, 1)]

    def test_x_gate_deterministic(self, client_configuration: QCSClient):
        qc = _make_qc(client_configuration)
        p = Program(
            Declare("ro", "BIT", 1),
            X(0),
            MEASURE(0, MemoryReference("ro", 0)),
        ).wrap_in_numshots_loop(10)
        result = qc.run(p)
        bitstrings = result.readout_data.get("ro")
        assert bitstrings.shape == (10, 1)
        assert all(bit == 1 for bit in bitstrings.flatten())

    def test_identity_deterministic(self, client_configuration: QCSClient):
        qc = _make_qc(client_configuration)
        p = Program(
            Declare("ro", "BIT", 1),
            MEASURE(0, MemoryReference("ro", 0)),
        ).wrap_in_numshots_loop(10)
        result = qc.run(p)
        bitstrings = result.readout_data.get("ro")
        assert all(bit == 0 for bit in bitstrings.flatten())

    def test_parametric_program(self, client_configuration: QCSClient):
        qc = _make_qc(client_configuration)
        p = Program(
            Declare("theta", "REAL"),
            Declare("ro", "BIT", 1),
            RX(MemoryReference("theta"), 0),
            MEASURE(0, MemoryReference("ro", 0)),
        ).wrap_in_numshots_loop(100)
        result = qc.run(p, {"theta": [np.pi]})
        bitstrings = result.readout_data.get("ro")
        assert all(bit == 1 for bit in bitstrings.flatten())


class TestQVMNoisy:
    def test_noisy_simulation(self, client_configuration: QCSClient):
        from pyquil.noise import Channel, MeasurementChannel, NoiseModel
        from pyquil.quilbase import Gate as QuilGate, Measurement as QuilMeasurement
        from pyquil.quilatom import Qubit as QuilQubit

        # Build a simple noise model with depolarizing gate noise
        x_inst = QuilGate("X", [], [QuilQubit(0)])
        m_inst = QuilMeasurement(qubit=QuilQubit(0), classical_reg=None)
        noise_model = NoiseModel(
            channels=frozenset([
                Channel.from_gate_fidelity(inst=x_inst, fidelity=0.95),
                MeasurementChannel.from_readout_fidelity(inst=m_inst, fidelity=0.95),
            ])
        )
        quantum_processor = NxQuantumProcessor(nx.complete_graph(3))
        qc = QuantumComputer(
            name="noisy-test",
            qam=QVM(noise_model=noise_model, random_seed=42),
            compiler=DummyCompiler(
                quantum_processor=quantum_processor,
                client_configuration=client_configuration,
            ),
        )
        p = Program(
            Declare("ro", "BIT", 1),
            X(0),
            MEASURE(0, MemoryReference("ro", 0)),
        ).wrap_in_numshots_loop(1000)
        result = qc.run(p)
        bitstrings = result.readout_data.get("ro")
        # With noise, most but not all should be 1
        mean = np.mean(bitstrings)
        assert 0.8 < mean < 1.0  # noisy but mostly correct


class TestQVMReproducibility:
    def test_random_seed_reproducibility(self, client_configuration: QCSClient):
        p = Program(
            Declare("ro", "BIT", 2),
            H(0),
            CNOT(0, 1),
            MEASURE(0, MemoryReference("ro", 0)),
            MEASURE(1, MemoryReference("ro", 1)),
        ).wrap_in_numshots_loop(50)

        qc1 = _make_qc(client_configuration, random_seed=123)
        qc2 = _make_qc(client_configuration, random_seed=123)

        r1 = qc1.run(p).readout_data.get("ro")
        r2 = qc2.run(p).readout_data.get("ro")
        np.testing.assert_array_equal(r1, r2)

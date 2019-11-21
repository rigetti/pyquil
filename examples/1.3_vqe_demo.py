"""
This is a demo of VQE through the forest stack. We will do the H2 binding from the Google paper using
OpenFermion to generate Hamiltonians and Forest to simulate the system


"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize  # for real runs I recommend using ADAM optimizer because momentum helps with noise
from openfermionpsi4 import run_psi4
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import symmetry_conserving_bravyi_kitaev, get_fermion_operator
from openfermion.utils import uccsd_singlet_get_packed_amplitudes

from forestopenfermion import qubitop_to_pyquilpauli
from referenceqvm.unitary_generator import tensor_up

from pyquil.quil import Program
from pyquil.paulis import sX, sY, exponentiate, PauliSum
from pyquil.gates import X, I
from pyquil.api import QVMConnection

from grove.measurements.estimation import estimate_locally_commuting_operator


def get_h2_dimer(bond_length):
    # Set molecule parameters.
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    geometry = [('H', [0.0, 0.0, 0.0]), ('H', [0.0, 0.0, bond_length])]
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule.filename = "./" + molecule.filename.split('/')[-1]
    # Run Psi4.
    molecule = run_psi4(molecule,
                        run_scf=True,
                        run_mp2=False,
                        run_cisd=False,
                        run_ccsd=True,
                        run_fci=True)
    return molecule


def ucc_circuit(theta):
    """
    Implements

    exp(-i theta X_{0}Y_{1})

    :param theta: rotation parameter
    :return: pyquil.Program
    """
    generator = sX(0) * sY(1)
    initial_prog = Program().inst(X(1), X(0))

    # compiled program
    program = initial_prog + exponentiate(float(theta) * generator)  # float is required because pyquil has weird casting behavior
    return program


def objective_fun(theta, hamiltonian=None,
                  quantum_resource=QVMConnection(sync_endpoint='http://localhost:5000')):
    """
    Evaluate the Hamiltonian bny operator averaging

    :param theta:
    :param hamiltonian:
    :return:
    """
    if hamiltonian is None:
        # Hamiltonian is Identity
        return 1.0

    if isinstance(hamiltonian, PauliSum):
        result = estimate_locally_commuting_operator(ucc_circuit(theta), hamiltonian,
                                                     1.0E-6, quantum_resource=quantum_resource)
        result = result[0][0].real  # first output is expected value, second is variance, third is shots
    elif isinstance(hamiltonian, np.ndarray) and isinstance(quantum_resource, QVMConnection):
        wf = quantum_resource.wavefunction(ucc_circuit(theta))
        wf = wf.amplitudes.reshape((-1, 1))
        result = np.conj(wf).T.dot(hamiltonian).dot(wf)[0, 0].real
        print(result)
    else:
        raise TypeError("type of hamiltonian or qvm is unrecognized")

    return result


if __name__ == "__main__":
    qvm = QVMConnection(sync_endpoint='http://localhost:5000')
    bond_length = np.linspace(0.25, 3, 30)
    ucc_energy = []
    fci_energy = []
    hf_energy = []
    for rr in bond_length:
        molecule = get_h2_dimer(rr)
        hamiltonian = molecule.get_molecular_hamiltonian()
        bk_hamiltonian = symmetry_conserving_bravyi_kitaev(get_fermion_operator(hamiltonian), 4, 2)

        # generate the spin-adapted classical coupled-cluster amplitude to use as the input for the
        # circuit
        packed_amps = uccsd_singlet_get_packed_amplitudes(molecule.ccsd_single_amps, molecule.ccsd_double_amps,
                                                          molecule.n_qubits, molecule.n_electrons)

        theta = packed_amps[-1]  # always take the doubles amplitude

        # now that we're done setting up the Hamiltonian and grabbing initial opt parameters
        # we can switch over to how to run things
        ucc_program = ucc_circuit(theta)

        paulis_bk_hamiltonian = qubitop_to_pyquilpauli(bk_hamiltonian)
        bk_mat = tensor_up(paulis_bk_hamiltonian, 2)

        w, v = np.linalg.eigh(bk_mat)

        wf = qvm.wavefunction(ucc_program)
        wf = wf.amplitudes.reshape((-1, 1))

        tenergy = np.conj(wf).T.dot(bk_mat).dot(wf)[0, 0].real

        # observable = objective_fun(theta, hamiltonian=paulis_bk_hamiltonian, quantum_resource=qvm)
        observable = objective_fun(theta, hamiltonian=bk_mat, quantum_resource=qvm)

        result = minimize(objective_fun, x0=theta, args=(bk_mat, qvm), method='CG',
                          options={'disp':True})
        ucc_energy.append(result.fun)
        fci_energy.append(molecule.fci_energy)
        hf_energy.append(molecule.hf_energy)
        print(w[0], molecule.fci_energy, tenergy, result.fun)


    plt.plot(bond_length, hf_energy, 'C1o-', label='HF')
    plt.plot(bond_length, ucc_energy, 'C0o-', label='UCC-VQE')
    plt.plot(bond_length, fci_energy, 'k-', label='FCI')
    plt.xlabel(r'Bond Distance [$\AA$]', fontsize=14)
    plt.ylabel('Energy [Hartree]', fontsize=14)
    plt.legend(loc='upper right', fontsize=13)
    plt.tight_layout()
    plt.show()

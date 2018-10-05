##############################################################################
# Copyright 2018 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
from abc import ABC, abstractmethod
from typing import List

from pyquil import Program
from pyquil.quil import Gate
from pyquil.paulis import PauliTerm


class AbstractCompiler(ABC):
    @abstractmethod
    def quil_to_native_quil(self, program: Program) -> Program:
        """
        Compile an arbitrary quil program according to the ISA of target_device.

        :param program: Arbitrary quil to compile
        :return: Native quil and compiler metadata
        """

    @abstractmethod
    def native_quil_to_executable(self, nq_program: Program):
        """
        Compile a native quil program to a binary executable.

        :param nq_program: Native quil to compile
        :return: An (opaque) binary executable
        """


class AbstractBenchmarker(ABC):
    @abstractmethod
    def apply_clifford_to_pauli(self, clifford: Program, pauli_in: PauliTerm) -> PauliTerm:
        """
        Given a circuit that consists only of elements of the Clifford group,
        return its action on a PauliTerm.

        In particular, for Clifford C, and Pauli P, this returns the PauliTerm
        representing PCP^{\dagger}.

        :param clifford: A Program that consists only of Clifford operations.
        :param pauli_in: A PauliTerm to be acted on by clifford via conjugation.
        :return: A PauliTerm corresponding to pauli_in * clifford * pauli_in^{\dagger}
        """

    @abstractmethod
    def generate_rb_sequence(self, depth: int, gateset: List[Gate],
                             seed: int = None) -> List[Program]:
        """
        Construct a randomized benchmarking experiment on the given qubits, decomposing into
        gateset.

        The JSON payload that is parsed is a list of lists of indices, or Nones. In the
        former case, they are the index of the gate in the gateset.

        :param depth: The number of Clifford gates to include in the randomized benchmarking
         experiment. This is different than the number of gates in the resulting experiment.
        :param gateset: A list of pyquil gates to decompose the Clifford elements into. These
         must generate the clifford group on the qubits of interest. e.g. for one qubit
         [RZ(np.pi/2), RX(np.pi/2)].
        :param seed: A positive integer that seeds the random generation of the gate sequence.
        :return: A list of pyquil programs. Each pyquil program is a circuit that represents an
         element of the Clifford group. When these programs are composed, the resulting Program
         will be the randomized benchmarking experiment of the desired depth. e.g. if the return
         programs are called cliffords then `sum(cliffords, Program())` will give the randomized
         benchmarking experiment, which will compose to the identity program.
        """

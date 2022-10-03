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
from typing import List, Optional, Sequence, cast

from qcs_api_client.client import QCSClientConfiguration

from pyquil.api._abstract_compiler import AbstractBenchmarker
from pyquil.api._compiler_client import (
    GenerateRandomizedBenchmarkingSequenceRequest,
    ConjugatePauliByCliffordRequest,
    CompilerClient,
)

from pyquil.paulis import PauliTerm, is_identity
from pyquil.quil import address_qubits, Program
from pyquil.quilbase import Gate


class BenchmarkConnection(AbstractBenchmarker):
    """
    Represents a connection to a server that generates benchmarking data.
    """

    def __init__(self, *, timeout: float = 10.0, client_configuration: Optional[QCSClientConfiguration] = None):
        """
        Client to communicate with the benchmarking data endpoint.

        :param timeout: Time limit for requests, in seconds.
        :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
        """

        self._compiler_client = CompilerClient(
            client_configuration=client_configuration or QCSClientConfiguration.load(),
            request_timeout=timeout,
        )

    def apply_clifford_to_pauli(self, clifford: Program, pauli_in: PauliTerm) -> PauliTerm:
        r"""
        Given a circuit that consists only of elements of the Clifford group,
        return its action on a PauliTerm.

        In particular, for Clifford C, and Pauli P, this returns the PauliTerm
        representing CPC^{\dagger}.

        :param clifford: A Program that consists only of Clifford operations.
        :param pauli_in: A PauliTerm to be acted on by clifford via conjugation.
        :return: A PauliTerm corresponding to clifford * pauli_in * clifford^{\dagger}
        """
        # do nothing if `pauli_in` is the identity
        if is_identity(pauli_in):
            return pauli_in

        indices_and_terms = list(zip(*list(pauli_in.operations_as_set())))

        request = ConjugatePauliByCliffordRequest(
            pauli_indices=list(indices_and_terms[0]),
            pauli_symbols=list(indices_and_terms[1]),
            clifford=clifford.out(calibrations=False),
        )
        response = self._compiler_client.conjugate_pauli_by_clifford(request)

        phase_factor, paulis = response.phase_factor, response.pauli

        pauli_out = PauliTerm("I", 0, 1.0j**phase_factor)
        clifford_qubits = clifford.get_qubits()
        pauli_qubits = pauli_in.get_qubits()
        all_qubits = sorted(set(cast(List[int], pauli_qubits)).union(set(cast(List[int], clifford_qubits))))
        # The returned pauli will have specified its value on all_qubits, sorted by index.
        #  This is maximal set of qubits that can be affected by this conjugation.
        for i, pauli in enumerate(paulis):
            pauli_out = cast(PauliTerm, pauli_out * PauliTerm(pauli, all_qubits[i]))
        return cast(PauliTerm, pauli_out * pauli_in.coefficient)

    def generate_rb_sequence(
        self,
        depth: int,
        gateset: Sequence[Gate],
        seed: Optional[int] = None,
        interleaver: Optional[Program] = None,
    ) -> List[Program]:
        """
        Construct a randomized benchmarking experiment on the given qubits, decomposing into
        gateset. If interleaver is not provided, the returned sequence will have the form

            C_1 C_2 ... C_(depth-1) C_inv ,

        where each C is a Clifford element drawn from gateset, C_{< depth} are randomly selected,
        and C_inv is selected so that the entire sequence composes to the identity.  If an
        interleaver G (which must be a Clifford, and which will be decomposed into the native
        gateset) is provided, then the sequence instead takes the form

            C_1 G C_2 G ... C_(depth-1) G C_inv .

        The JSON response is a list of lists of indices, or Nones. In the former case, they are the
        index of the gate in the gateset.

        :param depth: The number of Clifford gates to include in the randomized benchmarking
         experiment. This is different than the number of gates in the resulting experiment.
        :param gateset: A list of pyquil gates to decompose the Clifford elements into. These
         must generate the clifford group on the qubits of interest. e.g. for one qubit
         [RZ(np.pi/2), RX(np.pi/2)].
        :param seed: A positive integer used to seed the PRNG.
        :param interleaver: A Program object that encodes a Clifford element.
        :return: A list of pyquil programs. Each pyquil program is a circuit that represents an
         element of the Clifford group. When these programs are composed, the resulting Program
         will be the randomized benchmarking experiment of the desired depth. e.g. if the return
         programs are called cliffords then `sum(cliffords, Program())` will give the randomized
         benchmarking experiment, which will compose to the identity program.
        """

        # Support QubitPlaceholders: we temporarily index to arbitrary integers.
        # `generate_rb_sequence` handles mapping back to the original gateset gates.
        gateset_as_program = address_qubits(sum(gateset, Program()))  # type: ignore
        qubits = len(gateset_as_program.get_qubits())
        gateset_for_api = gateset_as_program.out().splitlines()
        interleaver_out: Optional[str] = None
        if interleaver:
            assert isinstance(interleaver, Program)
            interleaver_out = interleaver.out(calibrations=False)

        depth = int(depth)  # needs to be jsonable, no np.int64 please!

        request = GenerateRandomizedBenchmarkingSequenceRequest(
            depth=depth,
            num_qubits=qubits,
            gateset=gateset_for_api,
            seed=seed,
            interleaver=interleaver_out,
        )
        response = self._compiler_client.generate_randomized_benchmarking_sequence(request)

        programs = []
        for clifford in response.sequence:
            clifford_program = Program()
            if interleaver:
                clifford_program._calibrations = interleaver.calibrations
            # Like below, we reversed the order because the API currently hands back the Clifford
            # decomposition right-to-left.
            for index in reversed(clifford):
                clifford_program.inst(gateset[index])
            programs.append(clifford_program)
        # The programs are returned in "textbook style" right-to-left order. To compose them into
        #  the correct pyquil program, we reverse the order.
        return list(reversed(programs))

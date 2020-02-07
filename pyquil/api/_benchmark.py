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

import rpcq
from rpcq._client import Client
from rpcq.messages import (
    RandomizedBenchmarkingRequest,
    RandomizedBenchmarkingResponse,
    ConjugateByCliffordRequest,
    ConjugateByCliffordResponse,
)

from pyquil.api._config import PyquilConfig
from pyquil.api._error_reporting import _record_call
from pyquil.api._qac import AbstractBenchmarker
from pyquil.paulis import PauliTerm, is_identity
from pyquil.quil import address_qubits, Program
from pyquil.quilbase import Gate


class BenchmarkConnection(AbstractBenchmarker):
    """
    Represents a connection to a server that generates benchmarking data.
    """

    @_record_call
    def __init__(self, endpoint: str, timeout: Optional[float] = None):
        """
        Client to communicate with the benchmarking data endpoint.

        :param endpoint: TCP or IPC endpoint of the Compiler Server
        """

        self.client = Client(endpoint, timeout=timeout)

    @_record_call
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

        payload = ConjugateByCliffordRequest(
            clifford=clifford.out(),
            pauli=rpcq.messages.PauliTerm(
                indices=list(indices_and_terms[0]), symbols=list(indices_and_terms[1])
            ),
        )
        response: ConjugateByCliffordResponse = self.client.call(
            "conjugate_pauli_by_clifford", payload
        )
        phase_factor, paulis = response.phase, response.pauli

        pauli_out = PauliTerm("I", 0, 1.0j ** phase_factor)
        clifford_qubits = clifford.get_qubits()
        pauli_qubits = pauli_in.get_qubits()
        all_qubits = sorted(
            set(cast(List[int], pauli_qubits)).union(set(cast(List[int], clifford_qubits)))
        )
        # The returned pauli will have specified its value on all_qubits, sorted by index.
        #  This is maximal set of qubits that can be affected by this conjugation.
        for i, pauli in enumerate(paulis):
            pauli_out = cast(PauliTerm, pauli_out * PauliTerm(pauli, all_qubits[i]))
        return cast(PauliTerm, pauli_out * pauli_in.coefficient)

    @_record_call
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
        gateset_as_program = address_qubits(sum(gateset, Program()))
        qubits = len(gateset_as_program.get_qubits())
        gateset_for_api = gateset_as_program.out().splitlines()
        interleaver_out: Optional[str] = None
        if interleaver:
            assert isinstance(interleaver, Program)
            interleaver_out = interleaver.out()

        depth = int(depth)  # needs to be jsonable, no np.int64 please!

        payload = RandomizedBenchmarkingRequest(
            depth=depth,
            qubits=qubits,
            gateset=gateset_for_api,
            seed=seed,
            interleaver=interleaver_out,
        )
        response = self.client.call(
            "generate_rb_sequence", payload
        )  # type: RandomizedBenchmarkingResponse

        programs = []
        for clifford in response.sequence:
            clifford_program = Program()
            # Like below, we reversed the order because the API currently hands back the Clifford
            # decomposition right-to-left.
            for index in reversed(clifford):
                clifford_program.inst(gateset[index])
            programs.append(clifford_program)
        # The programs are returned in "textbook style" right-to-left order. To compose them into
        #  the correct pyquil program, we reverse the order.
        return list(reversed(programs))


def get_benchmarker(endpoint: Optional[str] = None, timeout: float = 10) -> BenchmarkConnection:
    """
    Retrieve an instance of the appropriate AbstractBenchmarker subclass for a given endpoint.

    :param endpoint: Benchmarking sequence server address. Defaults to the setting in the user's
                     pyQuil config.
    :param timeout: Number of seconds to wait before giving up on a call.
    :return: Instance of an AbstractBenchmarker subclass, connected to the given endpoint.
    """
    if endpoint is None:
        config = PyquilConfig()
        endpoint = config.quilc_url

    return BenchmarkConnection(endpoint=endpoint, timeout=timeout)

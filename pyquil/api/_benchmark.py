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
import rpcq
from rpcq import Client
from rpcq.messages import (RandomizedBenchmarkingRequest, RandomizedBenchmarkingResponse,
                           ConjugateByCliffordRequest, ConjugateByCliffordResponse)

from pyquil.api._base_connection import get_session, post_json
from pyquil.api._config import PyquilConfig
from pyquil.api._error_reporting import _record_call
from pyquil.api._qac import AbstractBenchmarker
from pyquil.paulis import PauliTerm
from pyquil.quil import address_qubits, Program


class BenchmarkConnection(AbstractBenchmarker):
    """
    Represents a connection to a server that generates benchmarking data.
    """

    @_record_call
    def __init__(self, endpoint=None):
        """
        Client to communicate with the benchmarking data endpoint.

        :param endpoint: TCP or IPC endpoint of the Compiler Server
        """

        self.client = Client(endpoint)

    @_record_call
    def apply_clifford_to_pauli(self, clifford, pauli_in):
        r"""
        Given a circuit that consists only of elements of the Clifford group,
        return its action on a PauliTerm.

        In particular, for Clifford C, and Pauli P, this returns the PauliTerm
        representing PCP^{\dagger}.

        :param Program clifford: A Program that consists only of Clifford operations.
        :param PauliTerm pauli_in: A PauliTerm to be acted on by clifford via conjugation.
        :return: A PauliTerm corresponding to pauli_in * clifford * pauli_in^{\dagger}
        """

        indices_and_terms = list(zip(*list(pauli_in.operations_as_set())))

        payload = ConjugateByCliffordRequest(
            clifford=clifford.out(),
            pauli=rpcq.messages.PauliTerm(
                indices=list(indices_and_terms[0]), symbols=list(indices_and_terms[1])))
        response: ConjugateByCliffordResponse = self.client.call(
            'conjugate_pauli_by_clifford', payload)
        phase_factor, paulis = response.phase, response.pauli

        pauli_out = PauliTerm("I", 0, 1.j ** phase_factor)
        clifford_qubits = clifford.get_qubits()
        pauli_qubits = pauli_in.get_qubits()
        all_qubits = sorted(set(pauli_qubits).union(set(clifford_qubits)))
        # The returned pauli will have specified its value on all_qubits, sorted by index.
        #  This is maximal set of qubits that can be affected by this conjugation.
        for i, pauli in enumerate(paulis):
            pauli_out *= PauliTerm(pauli, all_qubits[i])
        return pauli_out * pauli_in.coefficient

    @_record_call
    def generate_rb_sequence(self, depth, gateset, seed=None, interleaver=None):
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

        :param int depth: The number of Clifford gates to include in the randomized benchmarking
         experiment. This is different than the number of gates in the resulting experiment.
        :param list gateset: A list of pyquil gates to decompose the Clifford elements into. These
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
        if interleaver:
            assert(isinstance(interleaver, Program))
            interleaver = interleaver.out()

        depth = int(depth)  # needs to be jsonable, no np.int64 please!

        payload = RandomizedBenchmarkingRequest(depth=depth,
                                                qubits=qubits,
                                                gateset=gateset_for_api,
                                                seed=seed,
                                                interleaver=interleaver)
        response = self.client.call('generate_rb_sequence', payload)  # type: RandomizedBenchmarkingResponse

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


class LocalBenchmarkConnection(AbstractBenchmarker):
    """
    Represents a connection to a locally-running server that generates randomized benchmarking data.
    """

    @_record_call
    def __init__(self, endpoint=None):
        super(LocalBenchmarkConnection, self).__init__()
        self.endpoint = endpoint
        self.session = get_session()

    @staticmethod
    def _clifford_application_payload(clifford, pauli):
        """
        Prepares a JSON payload for conjugating a Pauli by a Clifford.

         See :py:func:`apply_clifford_to_pauli`.

        :param Program clifford: A Program that consists only of Clifford operations.
        :param PauliTerm pauli: A PauliTerm to be acted on by clifford via conjugation.
        :return: The JSON payload, with keys "clifford" and "pauli".
        """
        indices_and_terms = zip(*list(pauli.operations_as_set()))
        return {"clifford": clifford.out(),
                "pauli": list(indices_and_terms)}

    @_record_call
    def apply_clifford_to_pauli(self, clifford, pauli_in):
        r"""
        Given a circuit that consists only of elements of the Clifford group,
        return its action on a PauliTerm.

        In particular, for Clifford C, and Pauli P, this returns the PauliTerm
        representing PCP^{\dagger}.

        :param Program clifford: A Program that consists only of Clifford operations.
        :param PauliTerm pauli_in: A PauliTerm to be acted on by clifford via conjugation.
        :return: A PauliTerm corresponding to pauli_in * clifford * pauli_in^{\dagger}
        """
        payload = self._clifford_application_payload(clifford, pauli_in)
        phase_factor, paulis = post_json(self.session, self.endpoint + "/apply-clifford",
                                         payload).json()
        pauli_out = PauliTerm("I", 0, 1.j ** phase_factor)
        clifford_qubits = clifford.get_qubits()
        pauli_qubits = pauli_in.get_qubits()
        all_qubits = sorted(set(pauli_qubits).union(set(clifford_qubits)))
        # The returned pauli will have specified its value on all_qubits, sorted by index.
        #  This is maximal set of qubits that can be affected by this conjugation.
        for i, pauli in enumerate(paulis):
            pauli_out *= PauliTerm(pauli, all_qubits[i])
        return pauli_out * pauli_in.coefficient

    @staticmethod
    def _rb_sequence_payload(depth, gateset, seed=None, interleaver=None):
        """
        Prepares a JSON payload for generating a randomized benchmarking sequence.

        See :py:func:`generate_rb_sequence`.

        :param int depth: The number of cliffords per rb sequences to generate.
        :param list gateset: A list of Gate objects that make up the gateset to decompose
            the Cliffords into.
        :return: The JSON payload, with keys "depth", "qubits", and "gateset".
        """
        # Support QubitPlaceholders: we temporarily index to arbitrary integers.
        # `generate_rb_sequence` handles mapping back to the original gateset gates.
        gateset_as_program = address_qubits(sum(gateset, Program()))
        n_qubits = len(gateset_as_program.get_qubits())
        gateset_for_api = gateset_as_program.out().splitlines()
        payload = {"depth": depth,
                   "qubits": n_qubits,
                   "gateset": gateset_for_api,
                   "seed": seed}

        if interleaver:
            assert(isinstance(interleaver, Program))
            payload["interleaver"] = interleaver.out()

        return payload

    @_record_call
    def generate_rb_sequence(self, depth, gateset, seed=None, interleaver=None):
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

        :param int depth: The number of Clifford gates to include in the randomized benchmarking
         experiment. This is different than the number of gates in the resulting experiment.
        :param list gateset: A list of pyquil gates to decompose the Clifford elements into. These
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
        depth = int(depth)  # needs to be jsonable, no np.int64 please!
        payload = self._rb_sequence_payload(depth, gateset, seed=seed, interleaver=interleaver)
        response = post_json(self.session, self.endpoint + "/rb", payload).json()
        programs = []
        for clifford in response:
            clifford_program = Program()
            # Like below, we reversed the order because the API currently hands back the Clifford
            # decomposition right-to-left.
            for index in reversed(clifford):
                clifford_program.inst(gateset[index])
            programs.append(clifford_program)
        # The programs are returned in "textbook style" right-to-left order. To compose them into
        #  the correct pyquil program, we reverse the order.
        return list(reversed(programs))


def get_benchmarker(endpoint: str = None):
    """
    Retrieve an instance of the appropriate AbstractBenchmarker subclass for a given endpoint.

    :param endpoint: Benchmarking sequence server address. Defaults to the setting in the user's
                     pyQuil config.
    :return: Instance of an AbstractBenchmarker subclass, connected to the given endpoint.
    """
    if endpoint is None:
        config = PyquilConfig()
        endpoint = config.compiler_url

    if endpoint.startswith("http"):
        return LocalBenchmarkConnection(endpoint=endpoint)
    elif endpoint.startswith("tcp"):
        return BenchmarkConnection(endpoint=endpoint)
    else:
        raise ValueError("Protocol for RB endpoint must be HTTP or TCP.")

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
from typing import Dict, List, Union, Optional, Any, Set, cast
from warnings import warn

import numpy as np

from pyquil.api._base_connection import ForestConnection
from pyquil.api._error_reporting import _record_call
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program, percolate_declares
from pyquil.quilatom import MemoryReference
from pyquil.gates import MOVE
from pyquil.wavefunction import Wavefunction


class WavefunctionSimulator:
    @_record_call
    def __init__(
        self, connection: Optional[ForestConnection] = None, random_seed: Optional[int] = None
    ) -> None:
        """
        A simulator that propagates a wavefunction representation of a quantum state.

        :param connection: A connection to the Forest web API.
        :param random_seed: A seed for the simulator's random number generators. Either None (for
            an automatically generated seed) or a non-negative integer.
        """
        if connection is None:
            connection = ForestConnection()

        self.connection = connection

        if random_seed is None:
            self.random_seed = None
        elif isinstance(random_seed, int) and random_seed >= 0:
            self.random_seed = random_seed
        else:
            raise TypeError("random_seed should be None or a non-negative int")

    @_record_call
    def wavefunction(self, quil_program: Program, memory_map: Any = None) -> Wavefunction:
        """
        Simulate a Quil program and return the wavefunction.

        .. note:: If your program contains measurements or noisy gates, this method may not do what
            you want. If the execution of ``quil_program`` is **non-deterministic** then the
            final wavefunction only represents a stochastically generated sample and the
            wavefunctions returned by *different* ``wavefunction`` calls *will generally be
            different*.

        :param quil_program: A Quil program.
        :param memory_map: An assignment of classical registers to values, representing an initial
                           state for the QAM's classical memory.

                           This is expected to be of type Dict[str, List[Union[int, float]]],
                           where the keys are memory region names and the values are arrays of
                           initialization data.

                           For now, we also support input of type Dict[MemoryReference, Any],
                           but this is deprecated and will be removed in a future release.
        :return: A Wavefunction object representing the state of the QVM.
        """

        if memory_map is not None:
            quil_program = self.augment_program_with_memory_values(quil_program, memory_map)

        return cast(
            Wavefunction,
            self.connection._wavefunction(quil_program=quil_program, random_seed=self.random_seed),
        )

    @_record_call
    def expectation(
        self,
        prep_prog: Program,
        pauli_terms: Union[PauliSum, List[PauliTerm]],
        memory_map: Any = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate the expectation value of Pauli operators given a state prepared by prep_program.

        If ``pauli_terms`` is a ``PauliSum`` then the returned value is a single ``float``,
        otherwise the returned value is an array of values, one for each ``PauliTerm`` in the
        list.

        .. note:: If your program contains measurements or noisy gates, this method may not do what
            you want. If the execution of ``quil_program`` is **non-deterministic** then the
            final wavefunction from which the expectation value is calculated only represents
            a stochastically generated sample and the wavefunctions returned by *different*
            ``wavefunction`` calls *will generally be different*.

        :param prep_prog: A program that prepares the state on which we measure the expectation.
        :param pauli_terms: A Pauli representation of a quantum operator.
        :param memory_map: An assignment of classical registers to values, representing an initial
                           state for the QAM's classical memory.

                           This is expected to be of type Dict[str, List[Union[int, float]]],
                           where the keys are memory region names and the values are arrays of
                           initialization data.

                           For now, we also support input of type Dict[MemoryReference, Any],
                           but this is deprecated and will be removed in a future release.
        :return: Either a float or array floats depending on ``pauli_terms``.
        """

        is_pauli_sum = False
        if isinstance(pauli_terms, PauliSum):
            progs, coeffs = pauli_terms.get_programs()
            is_pauli_sum = True
        else:
            coeffs = np.array([pt.coefficient for pt in pauli_terms])
            progs = [pt.program for pt in pauli_terms]

        if memory_map is not None:
            prep_prog = self.augment_program_with_memory_values(prep_prog, memory_map)

        bare_results = self.connection._expectation(prep_prog, progs, random_seed=self.random_seed)
        results = coeffs * bare_results
        if is_pauli_sum:
            return np.sum(results)
        return results

    @_record_call
    def run_and_measure(
        self,
        quil_program: Program,
        qubits: Optional[List[int]] = None,
        trials: int = 1,
        memory_map: Optional[
            Union[Dict[str, List[Union[int, float]]], Dict[MemoryReference, Any]]
        ] = None,
    ) -> np.ndarray:
        """
        Run a Quil program once to determine the final wavefunction, and measure multiple times.

        Alternatively, consider using ``wavefunction`` and calling ``sample_bitstrings`` on the
        resulting object.

        For a large wavefunction and a low-medium number of trials, use this function.
        On the other hand, if you're sampling a small system many times you might want to
        use ``Wavefunction.sample_bitstrings``.

        .. note:: If your program contains measurements or noisy gates, this method may not do what
            you want. If the execution of ``quil_program`` is **non-deterministic** then the
            final wavefunction from which the returned bitstrings are sampled itself only
            represents a stochastically generated sample and the outcomes sampled from
            *different* ``run_and_measure`` calls *generally sample different bitstring
            distributions*.

        :param quil_program: The program to run and measure
        :param qubits: An optional list of qubits to measure. The order of this list is
            respected in the returned bitstrings. If not provided, all qubits used in
            the program will be measured and returned in their sorted order.
        :param int trials: Number of times to sample from the prepared wavefunction.
        :param memory_map: An assignment of classical registers to values, representing an initial
                           state for the QAM's classical memory.

                           This is expected to be of type Dict[str, List[Union[int, float]]],
                           where the keys are memory region names and the values are arrays of
                           initialization data.

                           For now, we also support input of type Dict[MemoryReference, Any],
                           but this is deprecated and will be removed in a future release.
        :return: An array of measurement results (0 or 1) of shape (trials, len(qubits))
        """
        if qubits is None:
            qubits = sorted(cast(Set[int], quil_program.get_qubits(indices=True)))

        if memory_map is not None:
            quil_program = self.augment_program_with_memory_values(quil_program, memory_map)

        return self.connection._run_and_measure(
            quil_program=quil_program, qubits=qubits, trials=trials, random_seed=self.random_seed
        )

    @staticmethod
    def augment_program_with_memory_values(
        quil_program: Program,
        memory_map: Union[Dict[str, List[Union[int, float]]], Dict[MemoryReference, Any]],
    ) -> Program:
        p = Program()

        # we stupidly allowed memory_map to be of type Dict[MemoryReference, Any], whereas qc.run
        # takes a memory initialization argument of type Dict[str, List[Union[int, float]]. until
        # we are in a position to remove this, we support both styles of input.

        if len(memory_map.keys()) == 0:
            return quil_program
        elif isinstance(list(memory_map.keys())[0], MemoryReference):
            warn(
                "Use of memory_map values of type Dict[MemoryReference, Any] have been "
                "deprecated.  Please use Dict[str, List[Union[int, float]]], as with "
                "QuantumComputer.run ."
            )
            for k, v in memory_map.items():
                p += MOVE(k, v)
        elif isinstance(list(memory_map.keys())[0], str):
            for name, arr in memory_map.items():
                for index, value in enumerate(arr):
                    p += MOVE(MemoryReference(cast(str, name), offset=index), value)
        else:
            raise TypeError("Bad memory_map type; expected Dict[str, List[Union[int, float]]].")

        p += quil_program

        return percolate_declares(p)

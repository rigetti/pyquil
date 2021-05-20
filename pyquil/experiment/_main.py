##############################################################################
# Copyright 2016-2019 Rigetti Computing
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
"""
Schema definition of a Experiment, which is a collection of ExperimentSetting
objects and a main program body (or ansatz). This schema is widely useful for defining
and executing many common types of algorithms / applications, including state and process
tomography, and the variational quantum eigensolver.
"""
import json
import logging
import warnings
from json import JSONEncoder
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

from pyquil.experiment._calibration import CalibrationMethod
from pyquil.experiment._memory import (
    pauli_term_to_measurement_memory_map,
    pauli_term_to_preparation_memory_map,
)
from pyquil.experiment._program import (
    parameterized_single_qubit_measurement_basis,
    parameterized_single_qubit_state_preparation,
    parameterized_readout_symmetrization,
    measure_qubits,
)
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, _OneQState, TensorProductState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.gates import RESET
from pyquil.paulis import PauliTerm, is_identity
from pyquil.quil import Program
from pyquil.quilbase import Reset, ResetQubit


log = logging.getLogger(__name__)


def _abbrev_program(program: Program, max_len: int = 10) -> str:
    """Create an abbreviated string representation of a Program.

    This will join all instructions onto a single line joined by '; '. If the number of
    instructions exceeds ``max_len``, some will be excluded from the string representation.
    """
    program_lines = program.out().splitlines()
    if max_len is not None and len(program_lines) > max_len:
        first_n = max_len // 2
        last_n = max_len - first_n
        excluded = len(program_lines) - max_len
        program_lines = program_lines[:first_n] + [f"... {excluded} instrs not shown ..."] + program_lines[-last_n:]

    return "   " + "\n   ".join(program_lines)


def _remove_reset_from_program(program: Program) -> Program:
    """
    Trim the RESET from a program because in measure_observables it is re-added.

    :param program: Program to remove RESET(s) from.
    :return: Trimmed Program.
    """
    p = program.copy_everything_except_instructions()

    for inst in program:
        if not isinstance(inst, (Reset, ResetQubit)):
            p.inst(inst)

    return p


class Experiment:
    """
    A tomography-like experiment.

    Many near-term quantum algorithms involve:

     - some limited state preparation
     - enacting a quantum process (like in tomography) or preparing a variational ansatz state
       (like in VQE)
     - measuring observables of the state.

    Where we typically use a large number of (state_prep, measure) pairs but keep the ansatz
    program consistent. This class stores the ansatz program as a :py:class:`~pyquil.Program`
    and maintains a list of :py:class:`ExperimentSetting` objects which each represent a
    (state_prep, measure) pair.

    Settings diagonalized by a shared tensor product basis (TPB) can (optionally) be estimated
    simultaneously. Therefore, this class is backed by a list of list of ExperimentSettings.
    Settings sharing an inner list will be estimated simultaneously. If you don't want this,
    provide a list of length-1-lists. As a convenience, if you pass a 1D list to the constructor
    will expand it to a list of length-1-lists.

    This class will not group settings for you. Please see :py:func:`group_experiments` for
    a function that will automatically process a Experiment to group Experiments sharing
    a TPB.

    :ivar settings: The collection of ExperimentSetting objects that define this experiment.
    :ivar program: The main program body of this experiment. Also determines the ``shots``
        and ``reset`` instance variables. The ``shots`` instance variable is the number of
        shots to take per ExperimentSetting. The ``reset`` instance variable is whether to
        actively reset qubits instead of waiting several times the coherence length for qubits
        to decay to ``|0>`` naturally. Setting this to True is much faster but there is a ~1%
        error per qubit in the reset operation. Thermal noise from "traditional" reset is not
        routinely characterized but is of the same order.
    :ivar symmetrization: the level of readout symmetrization to perform for the estimation
        and optional calibration of each observable. The following integer levels, encapsulated in
        the ``SymmetrizationLevel`` integer enum, are currently supported:

        * -1 -- exhaustive symmetrization uses every possible combination of flips
        * 0 -- no symmetrization
        * 1 -- symmetrization using an orthogonal array (OA) with strength 1
        * 2 -- symmetrization using an orthogonal array (OA) with strength 2
        * 3 -- symmetrization using an orthogonal array (OA) with strength 3

        Note that (default) exhaustive symmetrization requires a number of QPU calls exponential in
        the number of qubits in the union of the support of the observables in any group of settings
        in ``tomo_experiment``; the number of shots may need to be increased to accommodate this.
        see :py:func:`run_symmetrized_readout` in api._quantum_computer for more information.
    """

    def __init__(
        self,
        settings: Union[List[ExperimentSetting], List[List[ExperimentSetting]]],
        program: Program,
        *,
        symmetrization: int = SymmetrizationLevel.EXHAUSTIVE,
        calibration: int = CalibrationMethod.PLUS_EIGENSTATE,
    ):
        if len(settings) == 0:
            s: List[List[ExperimentSetting]] = []
        else:
            if isinstance(settings[0], ExperimentSetting):
                # convenience wrapping in lists of length 1
                s = [[expt] for expt in cast(List[ExperimentSetting], settings)]
            else:
                s = cast(List[List[ExperimentSetting]], settings)

        self._settings = s
        self.program = program
        self.symmetrization = SymmetrizationLevel(symmetrization)
        if self.symmetrization != SymmetrizationLevel.EXHAUSTIVE:
            if type(calibration) == int and calibration != 0:
                warnings.warn(
                    "Calibration is only supported for exhaustive symmetrization, "
                    "thus setting self.calibration = 0 (CalibrationMethod.NONE)."
                )
            self.calibration = CalibrationMethod.NONE
        else:
            self.calibration = CalibrationMethod(calibration)
        self.shots = self.program.num_shots

        if "RESET" in self.program.out():
            self.reset = True
            self.program = _remove_reset_from_program(self.program)
        else:
            self.reset = False

    def __len__(self) -> int:
        return len(self._settings)

    def __getitem__(self, item: int) -> List[ExperimentSetting]:
        return self._settings[item]

    def __setitem__(self, key: int, value: List[ExperimentSetting]) -> None:
        self._settings[key] = value

    def __delitem__(self, key: int) -> None:
        self._settings.__delitem__(key)

    def __iter__(self) -> Generator[List[ExperimentSetting], None, None]:
        yield from self._settings

    def __reversed__(self) -> Generator[List[ExperimentSetting], None, None]:
        yield from reversed(self._settings)

    def __contains__(self, item: List[ExperimentSetting]) -> bool:
        return item in self._settings

    def append(self, expts: Union[ExperimentSetting, List[ExperimentSetting]]) -> None:
        if not isinstance(expts, list):
            expts = [expts]
        self._settings.append(expts)

    def count(self, expt: List[ExperimentSetting]) -> int:
        return self._settings.count(expt)

    def index(self, expt: List[ExperimentSetting], start: int = 0, stop: int = 0) -> int:
        return self._settings.index(expt, start, stop)

    def extend(self, expts: List[List[ExperimentSetting]]) -> None:
        self._settings.extend(expts)

    def insert(self, index: int, expt: List[ExperimentSetting]) -> None:
        self._settings.insert(index, expt)

    def pop(self, index: int = 0) -> List[ExperimentSetting]:
        return self._settings.pop(index)

    def remove(self, expt: List[ExperimentSetting]) -> None:
        self._settings.remove(expt)

    def reverse(self) -> None:
        self._settings.reverse()

    def sort(self, key: Optional[Callable[[List[ExperimentSetting]], Any]] = None, reverse: bool = False) -> None:
        return self._settings.sort(key=key, reverse=reverse)

    def setting_strings(self) -> Generator[str, None, None]:
        yield from (
            "{i}: {st_str}".format(i=i, st_str=", ".join(str(setting) for setting in settings))
            for i, settings in enumerate(self._settings)
        )

    def settings_string(self, abbrev_after: Optional[int] = None) -> str:
        setting_strs = list(self.setting_strings())
        if abbrev_after is not None and len(setting_strs) > abbrev_after:
            first_n = abbrev_after // 2
            last_n = abbrev_after - first_n
            excluded = len(setting_strs) - abbrev_after
            setting_strs = setting_strs[:first_n] + [f"... {excluded} settings not shown ..."] + setting_strs[-last_n:]
        return "   " + "\n   ".join(setting_strs)

    def __repr__(self) -> str:
        string = f"shots: {self.shots}\n"
        if self.reset:
            string += "active reset: enabled\n"
        else:
            string += "active reset: disabled\n"
        string += f"symmetrization: {self.symmetrization} ({self.symmetrization.name.lower()})\n"
        string += f"calibration: {self.calibration} ({self.calibration.name.lower()})\n"
        string += f"program:\n{_abbrev_program(self.program)}\n"
        string += f"settings:\n{self.settings_string(abbrev_after=20)}"
        return string

    def serializable(self) -> Dict[str, Any]:
        return {
            "type": "Experiment",
            "settings": self._settings,
            "program": self.program.out(),
            "symmetrization": self.symmetrization,
            "shots": self.shots,
            "reset": self.reset,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Experiment):
            return False
        return self.serializable() == other.serializable()

    def get_meas_qubits(self) -> List[int]:
        """
        Return the sorted list of qubits that are involved in the all the out_operators of the
        settings for this ``Experiment`` object.
        """
        meas_qubits: Set[int] = set()
        for settings in self:
            assert len(settings) == 1
            meas_qubits.update(cast(List[int], settings[0].out_operator.get_qubits()))
        return sorted(meas_qubits)

    def get_meas_registers(self, qubits: Optional[Sequence[int]] = None) -> List[int]:
        """
        Return the sorted list of memory registers corresponding to the list of qubits provided.
        If no qubits are provided, just returns the list of numbers from 0 to n-1 where n is the
        number of qubits resulting from the ``get_meas_qubits`` method.
        """
        meas_qubits = self.get_meas_qubits()

        if qubits is None:
            return list(range(len(meas_qubits)))

        meas_registers = []
        for q in qubits:
            meas_registers.append(meas_qubits.index(q))
        return sorted(meas_registers)

    def generate_experiment_program(self) -> Program:
        """
        Generate a parameterized program containing the main body program along with some additions
        to support the various state preparation, measurement, and symmetrization specifications of
        this ``Experiment``.

        State preparation and measurement are achieved via ZXZXZ-decomposed single-qubit gates,
        where the angles of each ``RZ`` rotation are declared parameters that can be assigned at
        runtime. Symmetrization is achieved by putting an ``RX`` gate (also parameterized by a
        declared value) before each ``MEASURE`` operation. In addition, a ``RESET`` operation
        is prepended to the ``Program`` if the experiment has active qubit reset enabled. Finally,
        each qubit specified in the settings is measured, and the number of shots is added.

        :return: Parameterized ``Program`` that is capable of collecting statistics for every
            ``ExperimentSetting`` in this ``Experiment``.
        """
        meas_qubits = self.get_meas_qubits()

        p = Program()

        if self.reset:
            if any(isinstance(instr, (Reset, ResetQubit)) for instr in self.program):
                raise ValueError("RESET already added to program")
            p += RESET()

        for settings in self:
            assert len(settings) == 1
            if ("X" in str(settings[0].in_state)) or ("Y" in str(settings[0].in_state)):
                if "DECLARE preparation_alpha" in self.program.out():
                    raise ValueError('Memory "preparation_alpha" has been declared already.')
                if "DECLARE preparation_beta" in self.program.out():
                    raise ValueError('Memory "preparation_beta" has been declared already.')
                if "DECLARE preparation_gamma" in self.program.out():
                    raise ValueError('Memory "preparation_gamma" has been declared already.')
                p += parameterized_single_qubit_state_preparation(meas_qubits)
                break

        p += self.program

        for settings in self:
            assert len(settings) == 1
            if ("X" in str(settings[0].out_operator)) or ("Y" in str(settings[0].out_operator)):
                if "DECLARE measurement_alpha" in self.program.out():
                    raise ValueError('Memory "measurement_alpha" has been declared already.')
                if "DECLARE measurement_beta" in self.program.out():
                    raise ValueError('Memory "measurement_beta" has been declared already.')
                if "DECLARE measurement_gamma" in self.program.out():
                    raise ValueError('Memory "measurement_gamma" has been declared already.')
                p += parameterized_single_qubit_measurement_basis(meas_qubits)
                break

        if self.symmetrization != 0:
            if "DECLARE symmetrization" in self.program.out():
                raise ValueError('Memory "symmetrization" has been declared already.')
            p += parameterized_readout_symmetrization(meas_qubits)

        if "DECLARE ro" in self.program.out():
            raise ValueError('Memory "ro" has already been declared for this program.')
        p += measure_qubits(meas_qubits)

        p.wrap_in_numshots_loop(self.shots)

        return p

    def build_setting_memory_map(self, setting: ExperimentSetting) -> Dict[str, List[float]]:
        """
        Build the memory map corresponding to the state preparation and measurement specifications
        encoded in the provided ``ExperimentSetting``, taking into account the full set of qubits
        that are present in the ``Experiment`` object.

        :return: Memory map for state prep and measurement.
        """
        meas_qubits = self.get_meas_qubits()

        in_pt = PauliTerm.from_list([(op, meas_qubits.index(cast(int, q))) for q, op in setting._in_operator()])
        out_pt = PauliTerm.from_list([(op, meas_qubits.index(cast(int, q))) for q, op in setting.out_operator])

        preparation_map = pauli_term_to_preparation_memory_map(in_pt)
        measurement_map = pauli_term_to_measurement_memory_map(out_pt)

        return {**preparation_map, **measurement_map}

    def build_symmetrization_memory_maps(
        self, qubits: Sequence[int], label: str = "symmetrization"
    ) -> List[Dict[str, List[float]]]:
        """
        Build a list of memory maps to be used in a program that is trying to perform readout
        symmetrization via parametric compilation. For example, if we have the following program:

            RX(symmetrization[0]) 0
            RX(symmetrization[1]) 1
            MEASURE 0 ro[0]
            MEASURE 1 ro[1]

        We can perform exhaustive readout symmetrization on our two qubits by providing the four
        following memory maps, and then appropriately flipping the resultant bitstrings:

            {'symmetrization': [0.0, 0.0]} -> XOR results with [0,0]
            {'symmetrization': [0.0, pi]}  -> XOR results with [0,1]
            {'symmetrization': [pi, 0.0]}  -> XOR results with [1,0]
            {'symmetrization': [pi, pi]}   -> XOR results with [1,1]

        :param qubits: List of qubits to symmetrize readout for.
        :param label: Name of the declared memory region. Defaults to "symmetrization".
        :return: List of memory maps that performs the desired level of symmetrization.
        """
        num_meas_registers = len(self.get_meas_qubits())
        symm_registers = self.get_meas_registers(qubits)

        if self.symmetrization == SymmetrizationLevel.NONE:
            return [{}]

        # TODO: add support for orthogonal arrays
        if self.symmetrization != SymmetrizationLevel.EXHAUSTIVE:
            raise ValueError("We only support exhaustive symmetrization for now.")

        import numpy as np
        import itertools

        assignments = itertools.product(np.array([0, np.pi]), repeat=len(symm_registers))
        memory_maps = []
        for a in assignments:
            zeros = np.zeros(num_meas_registers)
            for idx, r in enumerate(symm_registers):
                zeros[r] = a[idx]
            memory_maps.append({f"{label}": list(zeros)})
        return memory_maps

    def generate_calibration_experiment(self) -> "Experiment":
        """
        Generate another ``Experiment`` object that can be used to calibrate the various multi-qubit
        observables involved in this ``Experiment``. This is achieved by preparing the plus-one
        (minus-one) eigenstate of each ``out_operator``, and measuring the resulting expectation
        value of the same ``out_operator``. Ideally, this would always give +1 (-1), but when
        symmetric readout error is present the effect is to scale the resultant expectations by some
        constant factor. Determining this scale factor is what we call *readout calibration*, and
        then the readout error in subsequent measurements can then be mitigated by simply dividing
        by the scale factor.

        :return: A new ``Experiment`` that can calibrate the readout error of all the
            observables involved in this experiment.
        """
        if self.calibration != CalibrationMethod.PLUS_EIGENSTATE:
            raise ValueError('We currently only support the "plus eigenstate" calibration method.')

        calibration_settings = []
        for settings in self:
            assert len(settings) == 1
            calibration_settings.append(
                ExperimentSetting(
                    in_state=_pauli_to_product_state(settings[0].out_operator),
                    out_operator=settings[0].out_operator,
                    additional_expectations=settings[0].additional_expectations,
                )
            )

        calibration_program = Program()
        if self.reset:
            calibration_program += RESET()
        calibration_program.wrap_in_numshots_loop(self.shots)

        if self.symmetrization != SymmetrizationLevel.EXHAUSTIVE:
            raise ValueError("We currently only support calibration for exhaustive symmetrization")

        return Experiment(
            settings=calibration_settings,
            program=calibration_program,
            symmetrization=SymmetrizationLevel.EXHAUSTIVE,
            calibration=CalibrationMethod.NONE,
        )


def _pauli_to_product_state(in_state: PauliTerm) -> TensorProductState:
    """
    Convert a Pauli term to a TensorProductState.
    """
    if is_identity(in_state):
        return TensorProductState()
    else:
        return TensorProductState(
            [
                _OneQState(label=pauli_label, index=0, qubit=cast(int, qubit))
                for qubit, pauli_label in in_state._ops.items()
            ]
        )


class OperatorEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, ExperimentSetting):
            return o.serializable()
        if isinstance(o, Experiment):
            return o.serializable()
        if isinstance(o, ExperimentResult):
            return o.serializable()
        return o


def to_json(fn: str, obj: Any) -> str:
    """
    Convenience method to save pyquil.experiment objects as a JSON file.

    See :py:func:`read_json`.
    """
    # Specify UTF-8 to guard against systems that default to an ASCII locale.
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=OperatorEncoder, indent=2, ensure_ascii=False)
    return fn


def _operator_object_hook(obj: Mapping[str, Any]) -> Union[Mapping[str, Any], Experiment]:
    if "type" in obj and obj["type"] in ["Experiment", "TomographyExperiment"]:
        # I bet this doesn't work for grouped experiment settings
        settings = [[ExperimentSetting.from_str(s) for s in stt] for stt in obj["settings"]]
        p = Program(obj["program"])
        p.wrap_in_numshots_loop(obj["shots"])
        ex = Experiment(settings=settings, program=p, symmetrization=obj["symmetrization"])
        ex.reset = obj["reset"]
        return ex
    return obj


def read_json(fn: str) -> Any:
    """
    Convenience method to read pyquil.experiment objects from a JSON file.

    See :py:func:`to_json`.
    """
    # Specify UTF-8 to guard against systems that default to an ASCII locale.
    with open(fn, encoding="utf-8") as f:
        return json.load(f, object_hook=_operator_object_hook)

import typing as ty
from itertools import chain

from quil import instructions as inst

from pyquil import quilbase as qb
from pyquil.quil import Program

_QUIL_BLOCK_DELIMITERS = (qb.JumpTarget, qb.Jump, qb.JumpUnless, qb.JumpWhen, qb.Label)
_CLASSICAL_INSTRUCTION = (
    qb.ClassicalStore,
    qb.ClassicalMove,
    qb.ClassicalLoad,
    qb.ClassicalComparison,
    qb.LogicalBinaryOp,
    qb.Call,
    qb.ArithmeticBinaryOp,
)
_GATE_OR_PULSE_INSTRUCTION = (qb.Gate, qb.Measurement, qb.Reset, qb.Pulse, qb.Capture, qb.RawCapture)


def _into_blocks(instructions: ty.Sequence[qb.AbstractInstruction]) -> tuple[tuple[qb.AbstractInstruction, ...], ...]:
    blocks: list[list[qb.AbstractInstruction]] = [[]]
    for instruction in instructions:
        if isinstance(instruction, _QUIL_BLOCK_DELIMITERS):
            blocks.append([])
        blocks[-1].append(instruction)
    return tuple(tuple(block) for block in blocks)


@ty.overload
def add_leading_delay_to_first_pulse_block(program: Program, /, *, leading_delay_seconds: float) -> None: ...


@ty.overload
def add_leading_delay_to_first_pulse_block(program: Program, /, *, delays: ty.Sequence[qb.Delay]) -> None: ...


def add_leading_delay_to_first_pulse_block(
    program: Program, /, *, leading_delay_seconds: float | None = None, delays: ty.Sequence[qb.Delay] | None = None
) -> None:
    """Add a leading delay to the first Quil instruction block that includes pulse or gate instructions.

    This delay will be the first instruction in that Quil instruction block; any classical instructions before
    the first gate or pulse instruction will be followed by a global `Fence`.

    This function considers gates, measurements, resets, and captures to be "pulse" instructions. It does
    does not consider scale, frequence, or phase sets or shifts as pulse instructions.

    :param program: The program to which a leading delay will be added.
    :param leading_delay_seconds: The length of the delay to add to for each qubit returned from
        `Program.get_qubit_indices`.
    :param delays: The leading delays to add to the program.

    Exactly one of `leading_delay_seconds` and `delays` must be specified.

    :return: None. The program instructions are mutated.
    """
    if leading_delay_seconds is not None:
        if delays is not None:
            raise ValueError("must specify only one of `leading_delay_seconds` and `delays`")
        delays = [qb.Delay(leading_delay_seconds, [q], []) for q in program.get_qubit_indices()]
    elif delays is None:
        raise ValueError("must specify at least one of `leading_delay_seconds` and `delays`")

    instructions = program.instructions
    # note that we explicitly sort declarations to ensure determinism.
    declarations = [instruction for instruction in instructions if isinstance(instruction, qb.Declare)]
    declarations.sort(key=lambda declare: declare.name)
    instructions = [instruction for instruction in instructions if not isinstance(instruction, qb.Declare)]
    blocks = _into_blocks(instructions)

    first_gate_or_pulse_block_index = None
    for i, block in enumerate(blocks):
        if any(isinstance(instruction, _GATE_OR_PULSE_INSTRUCTION) for instruction in block):
            first_gate_or_pulse_block_index = i
            break
    if first_gate_or_pulse_block_index is None:
        return

    block = blocks[first_gate_or_pulse_block_index]
    fence_index = next(i for i in range(len(block)) if isinstance(block[i], _GATE_OR_PULSE_INSTRUCTION))
    if len(block) > 0 and isinstance(block[0], _QUIL_BLOCK_DELIMITERS):
        block = (block[0],) + tuple(delays) + block[1:fence_index] + (qb.Fence([]),) + block[fence_index:]
    else:
        block = tuple(delays) + block[0:fence_index] + (qb.Fence([]),) + block[fence_index:]

    blocks = list(blocks)
    blocks[first_gate_or_pulse_block_index] = block
    program.instructions = declarations + list(chain.from_iterable(blocks))


_NUMBER_PAULI_PAIRS = 16


def build_extern_function_signatures() -> dict[str, qb.Pragma]:
    """Build and return a map from extern function name to the corresponding EXTERN pragma.

    These signtures reflect extern function signatures publicly supported by various pyQuil features, such as
    but not limited to randomized compiling.
    """
    destination = inst.ExternParameter(
        "destination",
        True,
        inst.ExternParameterType.VariableLengthVector(inst.ScalarType.REAL),
    )
    unitary_angles = inst.ExternParameter(
        "unitary_angles",
        False,
        inst.ExternParameterType.VariableLengthVector(inst.ScalarType.REAL),
    )
    angle_offset = inst.ExternParameter(
        "angle_offset",
        False,
        inst.ExternParameterType.Scalar(inst.ScalarType.INTEGER),
    )
    pauli_set = inst.ExternParameterType.Scalar(inst.ScalarType.INTEGER)
    pauli_index = inst.ExternParameterType.Scalar(inst.ScalarType.INTEGER)
    is_pauli_left = inst.ExternParameter("is_pauli_left", False, inst.ExternParameterType.Scalar(inst.ScalarType.BIT))
    pauli_conjugates_map = inst.ExternParameter(
        "pauli_conjugates_map",
        False,
        inst.ExternParameterType.FixedLengthVector(inst.Vector(inst.ScalarType.INTEGER, _NUMBER_PAULI_PAIRS)),
    )
    pauli_literal = inst.ExternParameterType.Scalar(inst.ScalarType.INTEGER)
    signatures = [
        (
            "merge_zxzxz_unitary_with_paulis_reference_conjugate",
            inst.ExternSignature(
                parameters=[
                    destination,
                    unitary_angles,
                    angle_offset,
                    inst.ExternParameter("next_paulis", False, pauli_set),
                    inst.ExternParameter("next_pauli_index", False, pauli_index),
                    inst.ExternParameter("previous_paulis_left", False, pauli_set),
                    inst.ExternParameter("previous_pauli_left_index", False, pauli_index),
                    inst.ExternParameter("previous_paulis_right", False, pauli_set),
                    inst.ExternParameter("previous_pauli_right_index", False, pauli_index),
                    is_pauli_left,
                    pauli_conjugates_map,
                ],
                return_type=None,
            ).to_quil(),
        ),
        (
            "merge_zxzxz_unitary_with_paulis_literal_literal",
            inst.ExternSignature(
                parameters=[
                    destination,
                    unitary_angles,
                    angle_offset,
                    inst.ExternParameter("next_pauli", False, pauli_literal),
                    inst.ExternParameter("conjugate_pauli", False, pauli_literal),
                ],
                return_type=None,
            ).to_quil(),
        ),
        (
            "merge_zxzxz_unitary_with_paulis_literal_conjugate",
            inst.ExternSignature(
                parameters=[
                    destination,
                    unitary_angles,
                    angle_offset,
                    inst.ExternParameter("next_pauli", False, pauli_literal),
                    inst.ExternParameter("previous_paulis_left", False, pauli_set),
                    inst.ExternParameter("previous_pauli_left_index", False, pauli_index),
                    inst.ExternParameter("previous_paulis_right", False, pauli_set),
                    inst.ExternParameter("previous_pauli_right_index", False, pauli_index),
                    is_pauli_left,
                    pauli_conjugates_map,
                ],
                return_type=None,
            ).to_quil(),
        ),
        (
            "merge_zxzxz_unitary_with_paulis_reference_literal",
            inst.ExternSignature(
                parameters=[
                    destination,
                    unitary_angles,
                    angle_offset,
                    inst.ExternParameter("next_paulis", False, pauli_set),
                    inst.ExternParameter("next_pauli_index", False, pauli_index),
                    inst.ExternParameter("conjugate_pauli", False, pauli_literal),
                ],
                return_type=None,
            ).to_quil(),
        ),
        (
            "merge_zxzxz_unitary_with_paulis_literal_reference",
            inst.ExternSignature(
                parameters=[
                    destination,
                    unitary_angles,
                    angle_offset,
                    inst.ExternParameter("next_pauli", False, pauli_literal),
                    inst.ExternParameter("previous_paulis", False, pauli_set),
                    inst.ExternParameter("previous_pauli_index", False, pauli_index),
                ],
                return_type=None,
            ).to_quil(),
        ),
        (
            "merge_zxzxz_unitary_with_paulis_reference_reference",
            inst.ExternSignature(
                parameters=[
                    destination,
                    unitary_angles,
                    angle_offset,
                    inst.ExternParameter("next_paulis", False, pauli_set),
                    inst.ExternParameter("next_pauli_index", False, pauli_index),
                    inst.ExternParameter("previous_paulis", False, pauli_set),
                    inst.ExternParameter("previous_pauli_index", False, pauli_index),
                ],
                return_type=None,
            ).to_quil(),
        ),
        (
            "prng_set_seed_and_step",
            inst.ExternSignature(
                parameters=[
                    inst.ExternParameter("seed", False, inst.ExternParameterType.Scalar(inst.ScalarType.INTEGER))
                ],
                return_type=inst.ScalarType.INTEGER,
            ).to_quil(),
        ),
        (
            "prng_step",
            inst.ExternSignature(
                parameters=[],
                return_type=inst.ScalarType.INTEGER,
            ).to_quil(),
        ),
        (
            "if_then_else_integer",
            inst.ExternSignature(
                parameters=[
                    inst.ExternParameter("condition", False, inst.ExternParameterType.Scalar(inst.ScalarType.BIT)),
                    inst.ExternParameter("true_value", False, inst.ExternParameterType.Scalar(inst.ScalarType.INTEGER)),
                    inst.ExternParameter(
                        "false_value", False, inst.ExternParameterType.Scalar(inst.ScalarType.INTEGER)
                    ),
                ],
                return_type=inst.ScalarType.INTEGER,
            ).to_quil(),
        ),
        (
            "if_then_else_real",
            inst.ExternSignature(
                parameters=[
                    inst.ExternParameter("condition", False, inst.ExternParameterType.Scalar(inst.ScalarType.BIT)),
                    inst.ExternParameter("true_value", False, inst.ExternParameterType.Scalar(inst.ScalarType.REAL)),
                    inst.ExternParameter("false_value", False, inst.ExternParameterType.Scalar(inst.ScalarType.REAL)),
                ],
                return_type=inst.ScalarType.REAL,
            ).to_quil(),
        ),
        (
            "choose_random_real_sub_regions",
            inst.ExternSignature(
                parameters=[
                    inst.ExternParameter(
                        "destination", True, inst.ExternParameterType.VariableLengthVector(inst.ScalarType.REAL)
                    ),
                    inst.ExternParameter(
                        "source", False, inst.ExternParameterType.VariableLengthVector(inst.ScalarType.REAL)
                    ),
                    inst.ExternParameter(
                        "sub_region_size", False, inst.ExternParameterType.Scalar(inst.ScalarType.INTEGER)
                    ),
                    inst.ExternParameter("seed", True, inst.ExternParameterType.Scalar(inst.ScalarType.INTEGER)),
                ],
                return_type=None,
            ).to_quil(),
        ),
    ]
    return {name: qb.Pragma("EXTERN", [name], signature) for name, signature in signatures}

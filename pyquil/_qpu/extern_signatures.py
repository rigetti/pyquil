from quil import instructions as inst

from pyquil.quilbase import Pragma

_NUMBER_PAULI_PAIRS = 16


def build_extern_function_signatures() -> dict[str, Pragma]:
    """Build and return a map from extern function name to the corresponding EXTERN pragma.

    These signtures reflect extern function signatures publicly supported by various pyQuil features, such as
    but not limited to randomized compiling.
    """
    destination = inst.ExternParameter(
        "destination",
        True,
        inst.ExternParameterType.from_variable_length_vector(inst.ScalarType.Real),
    )
    unitary_angles = inst.ExternParameter(
        "unitary_angles",
        False,
        inst.ExternParameterType.from_variable_length_vector(inst.ScalarType.Real),
    )
    angle_offset = inst.ExternParameter(
        "angle_offset",
        False,
        inst.ExternParameterType.from_scalar(inst.ScalarType.Integer),
    )
    pauli_set = inst.ExternParameterType.from_scalar(inst.ScalarType.Integer)
    pauli_index = inst.ExternParameterType.from_scalar(inst.ScalarType.Integer)
    is_pauli_left = inst.ExternParameter(
        "is_pauli_left", False, inst.ExternParameterType.from_scalar(inst.ScalarType.Bit)
    )
    pauli_conjugates_map = inst.ExternParameter(
        "pauli_conjugates_map",
        False,
        inst.ExternParameterType.from_fixed_length_vector(inst.Vector(inst.ScalarType.Integer, _NUMBER_PAULI_PAIRS)),
    )
    pauli_literal = inst.ExternParameterType.from_scalar(inst.ScalarType.Integer)
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
                    inst.ExternParameter("seed", False, inst.ExternParameterType.from_scalar(inst.ScalarType.Integer))
                ],
                return_type=inst.ScalarType.Integer,
            ).to_quil(),
        ),
        (
            "prng_step",
            inst.ExternSignature(
                parameters=[],
                return_type=inst.ScalarType.Integer,
            ).to_quil(),
        ),
        (
            "if_then_else_integer",
            inst.ExternSignature(
                parameters=[
                    inst.ExternParameter("condition", False, inst.ExternParameterType.from_scalar(inst.ScalarType.Bit)),
                    inst.ExternParameter(
                        "true_value", False, inst.ExternParameterType.from_scalar(inst.ScalarType.Integer)
                    ),
                    inst.ExternParameter(
                        "false_value", False, inst.ExternParameterType.from_scalar(inst.ScalarType.Integer)
                    ),
                ],
                return_type=inst.ScalarType.Integer,
            ).to_quil(),
        ),
        (
            "if_then_else_real",
            inst.ExternSignature(
                parameters=[
                    inst.ExternParameter("condition", False, inst.ExternParameterType.from_scalar(inst.ScalarType.Bit)),
                    inst.ExternParameter(
                        "true_value", False, inst.ExternParameterType.from_scalar(inst.ScalarType.Real)
                    ),
                    inst.ExternParameter(
                        "false_value", False, inst.ExternParameterType.from_scalar(inst.ScalarType.Real)
                    ),
                ],
                return_type=inst.ScalarType.Real,
            ).to_quil(),
        ),
        (
            "choose_random_real_sub_regions",
            inst.ExternSignature(
                parameters=[
                    inst.ExternParameter(
                        "destination", True, inst.ExternParameterType.from_variable_length_vector(inst.ScalarType.Real)
                    ),
                    inst.ExternParameter(
                        "source", False, inst.ExternParameterType.from_variable_length_vector(inst.ScalarType.Real)
                    ),
                    inst.ExternParameter(
                        "sub_region_size", False, inst.ExternParameterType.from_scalar(inst.ScalarType.Integer)
                    ),
                    inst.ExternParameter("seed", True, inst.ExternParameterType.from_scalar(inst.ScalarType.Integer)),
                ],
                return_type=None,
            ).to_quil(),
        ),
    ]
    return {name: Pragma("EXTERN", [name], signature) for name, signature in signatures}

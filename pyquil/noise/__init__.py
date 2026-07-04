"""pyquil.noise — Noise modeling for quantum simulators.

This package provides:

- **Noise model** (``_legacy_noise``): Kraus-map based noise construction
  for the QVM, including ``KrausModel``, ``NoiseModel``, and decoherence
  helpers.

- **Channel classes** (``_channels``): quax-backed ``Channel``,
  ``MeasurementChannel``, ``ResetChannel``, and ``CycleChannel`` dataclasses
  for fine-grained noise modeling.  These are private and not re-exported.

- **New noise model** (``_noise_model``): The quax-based ``NoiseModel``
  container and program-level fidelity estimation utilities.  These are
  private and not re-exported; they will become the public API in the next
  major version.
"""

# ── Noise model (Kraus-map based) ───────────────────────────────────────
from pyquil.noise._legacy_noise import (
    ANGLE_TOLERANCE,
    INFINITY,
    NO_NOISE,
    KrausModel,
    NoiseModel,
    NoisyGateUndefined,
    _bitstring_probs_by_qubit,  # noqa: F401
    _check_kraus_ops,  # noqa: F401
    _create_kraus_pragmas,  # noqa: F401
    _decoherence_noise_model,  # noqa: F401
    _get_program_gates,  # noqa: F401
    _noise_model_program_header,  # noqa: F401
    _run,  # noqa: F401
    add_decoherence_noise,
    append_kraus_to_gate,
    apply_noise_model,
    bitstring_probs_to_z_moments,
    combine_kraus_maps,
    correct_bitstring_probs,
    corrupt_bitstring_probs,
    damping_after_dephasing,
    damping_kraus_map,
    decoherence_noise_with_asymmetric_ro,
    dephasing_kraus_map,
    estimate_assignment_probs,
    estimate_bitstring_probs,
    get_noisy_gate,
    pauli_kraus_map,
    tensor_kraus_maps,
)

__all__ = [
    # Noise model
    "ANGLE_TOLERANCE",
    "INFINITY",
    "KrausModel",
    "NO_NOISE",
    "NoiseModel",
    "NoisyGateUndefined",
    "add_decoherence_noise",
    "append_kraus_to_gate",
    "apply_noise_model",
    "bitstring_probs_to_z_moments",
    "combine_kraus_maps",
    "correct_bitstring_probs",
    "corrupt_bitstring_probs",
    "damping_after_dephasing",
    "damping_kraus_map",
    "decoherence_noise_with_asymmetric_ro",
    "dephasing_kraus_map",
    "estimate_assignment_probs",
    "estimate_bitstring_probs",
    "get_noisy_gate",
    "pauli_kraus_map",
    "tensor_kraus_maps",
]

.. currentmodule:: pyquil.quil

Program
=======


.. autoclass:: pyquil.quil.Program

    .. rubric:: Attributes
    .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~Program.instructions
        ~Program.defined_gates
        ~Program.calibrations
        ~Program.waveforms
        ~Program.frames
        ~Program.out
        ~Program.get_qubits
        ~Program.is_protoquil


    .. rubric:: Program Construction
    .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~Program.__iadd__
        ~Program.__add__
        ~Program.inst
        ~Program.gate
        ~Program.defgate
        ~Program.define_noisy_gate
        ~Program.define_noisy_readout
        ~Program.no_noise
        ~Program.measure
        ~Program.reset
        ~Program.measure_all
        ~Program.alloc
        ~Program.declare
        ~Program.wrap_in_numshots_loop


    .. rubric:: Control Flow
    .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~Program.while_do
        ~Program.if_then

    .. rubric:: Quilt Routines
    .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~Program.get_calibration
        ~Program.match_calibrations
        ~Program.calibrate

    .. rubric:: Utility Methods
    .. autosummary::
        :toctree: autogen
        :template: autosumm.rst

        ~Program.copy
        ~Program.pop
        ~Program.dagger
        ~Program.__getitem__


Utility Functions
-----------------

.. autofunction:: get_default_qubit_mapping
.. autofunction:: address_qubits
.. autofunction:: instantiate_labels
.. autofunction:: implicitly_declare_ro
.. autofunction:: merge_with_pauli_noise
.. autofunction:: merge_programs
.. autofunction:: get_classical_addresses_from_program
.. autofunction:: percolate_declares
.. autofunction:: validate_protoquil
.. autofunction:: pyquil.parser.parse
.. autofunction:: pyquil.parser.parse_program



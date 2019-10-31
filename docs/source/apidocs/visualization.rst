Visualization
=============

pyQuil programs may be converted to LaTeX circuit diagrams, or even rendered immediately in a Jupyter Notebook. The main entry point of LaTeX generation is :py:func:`~pyquil.latex.to_latex`. For inline rendering in a notebook, the main entry point is :py:func:`~pyquil.latex.display`.


Both of these functions take an optional :py:class:`~pyquil.latex.DiagramSettings`, which may be used to control some aspects of circuit layout and appearance.

.. currentmodule:: pyquil.latex
.. autoclass:: DiagramSettings

    .. rubric:: Attributes
    .. autosummary::
        :template: autosumm.rst

        ~DiagramSettings.texify_numerical_constants
        ~DiagramSettings.impute_missing_qubits
        ~DiagramSettings.label_qubit_lines
        ~DiagramSettings.abbreviate_controlled_rotations
        ~DiagramSettings.qubit_line_open_wire_length
        ~DiagramSettings.right_align_terminal_measurements

.. autofunction:: to_latex
.. autofunction:: display

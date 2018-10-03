
# IMPORTANT NOTE: Use of this file has been deprecated.  Its contents can now be found in quilatom.py, and users should
# reference that file directly.

import warnings

from pyquil.quilatom import Parameter, Expression, quil_sin, quil_cos, quil_sqrt, quil_exp, quil_cis, \
    _contained_parameters, format_parameter, Add, Sub, Mul, Div, substitute, substitute_array

__all__ = ['Parameter', 'Expression', 'quil_sin', 'quil_cos', 'quil_sqrt', 'quil_exp', 'quil_cis',
           '_contained_parameters', 'format_parameter', 'Add', 'Sub', 'Mul', 'Div', 'substitute',
           'substitute_array']

warnings.warn(ImportWarning("Use of this pyquil/parameters.py has been deprecated.\n"
                            "\n"
                            "Its contents can now be found in quilatom.py, and users should"
                            "reference that file directly."))

##############################################################################
# Copyright 2019 Rigetti Computing
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

import os
import shutil
import subprocess
import tempfile
from typing import Any, Optional

from IPython.display import Image

from pyquil.latex._main import to_latex
from pyquil.latex._diagram import DiagramSettings
from pyquil.quil import Program


def display(
    circuit: Program, settings: Optional[DiagramSettings] = None, **image_options: Any
) -> Image:
    """
    Displays a PyQuil circuit as an IPython image object.

    .. note::

       For this to work, you need two external programs, ``pdflatex`` and ``convert``,
       to be installed and accessible via your shell path.

       Further, your LaTeX installation should include class and style files for ``standalone``,
       ``geometry``, ``tikz``, and ``quantikz``. If it does not, you need to install
       these yourself.

    :param circuit: The circuit to be drawn, represented as a pyquil program.
    :param settings: An optional object of settings controlling diagram rendering and layout.
    :return: PNG image render of the circuit.
    """
    # The conversion process relies on two passes: first, 'pdflatex' is called to
    # render the tikz to a pdf. Second, Imagemagick's 'convert' is called to translate
    # this to a png image.
    pdflatex_path = shutil.which("pdflatex")
    convert_path = shutil.which("convert")

    if pdflatex_path is None:
        raise FileNotFoundError("Unable to locate 'pdflatex'.")
    if convert_path is None:
        raise FileNotFoundError("Unable to locate 'convert'.")

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "diagram.tex"), "w") as texfile:
            texfile.write(to_latex(circuit, settings))

        result = subprocess.run(
            [pdflatex_path, "-halt-on-error", "-output-directory", tmpdir, texfile.name],
            stdout=subprocess.PIPE,
        )
        if result.returncode != 0:
            msg = "'pdflatex' terminated with return code {}.".format(result.returncode)
            # NOTE: pdflatex writes all error messages to stdout
            if result.stdout:
                msg += " Transcript:\n\n" + result.stdout.decode("utf-8")
            raise RuntimeError(msg)

        png = os.path.join(tmpdir, "diagram.png")
        pdf = os.path.join(tmpdir, "diagram.pdf")

        result = subprocess.run([convert_path, "-density", "300", pdf, png], stderr=subprocess.PIPE)
        if result.returncode != 0:
            msg = "'convert' terminated with return code {}.".format(result.returncode)
            if result.stderr:
                msg += " Transcript:\n\n" + result.stderr.decode("utf-8")
            raise RuntimeError(msg)

        return Image(filename=png, **image_options)

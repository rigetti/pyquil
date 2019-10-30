import os
import subprocess
import shutil
import tempfile
from typing import Optional
from IPython.display import Image

from pyquil import Program
from pyquil.latex.latex_generation import to_latex, DiagramSettings


def display(circuit: Program, settings: Optional[DiagramSettings] = None,
            **image_options) -> Image:
    """
    Displays a PyQuil circuit as an IPython image object.

    For this to work, you need 'pdflatex' and 'convert' to be installed and
    accessible via your shell path, as seen by Python.

    Further, your LaTeX installation should include class and style files for 'standalone',
    'geometry', 'tikz', and 'quantikz'. If it does not, you may need to install these yourself.

    The conversion process relies on two passes: first, 'pdflatex' is called to
    render the tikz to a pdf. Second, Imagemagick's 'convert' is called to translate
    this to a png image.

    :param Program circuit: The circuit to be drawn, represented as a pyquil program.
    :param DiagramSettings settings: An optional object of settings controlling diagram rendering and layout.
    :return: PNG image render of the circuit.
    :rtype: Image
    """
    pdflatex_path = shutil.which("pdflatex")
    convert_path = shutil.which("convert")

    if pdflatex_path is None:
        raise FileNotFoundError("Unable to locate 'pdflatex'.")
    if convert_path is None:
        raise FileNotFoundError("Unable to locate 'convert'.")

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "diagram.tex"), "w") as texfile:
            texfile.write(to_latex(circuit, settings))

        result = subprocess.run([pdflatex_path, "-halt-on-error",
                                 "-output-directory", tmpdir,
                                 texfile.name],
                                stdout=subprocess.PIPE)
        if result.returncode != 0:
            msg = "'pdflatex' terminated with return code {}.".format(result.returncode)
            # NOTE: pdflatex writes all error messages to stdout
            if result.stdout:
                msg += " Transcript:\n\n" + result.stdout.decode("utf-8")
            raise RuntimeError(msg)

        png = os.path.join(tmpdir, "diagram.png")
        pdf = os.path.join(tmpdir, "diagram.pdf")

        result = subprocess.run([convert_path, "-density", "300", pdf, png],
                                 stderr=subprocess.PIPE)
        if result.returncode != 0:
            msg = "'convert' terminated with return code {}.".format(result.returncode)
            if result.stderr:
                msg += " Transcript:\n\n" + result.stderr.decode("utf-8")
            raise RuntimeError(msg)

        return Image(filename=png, **image_options)

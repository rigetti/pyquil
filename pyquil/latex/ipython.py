import os
import subprocess
import shutil
import tempfile
from IPython.display import Image
from pyquil import Program
from .latex_generation import to_latex


def display(circuit: Program, settings=None, **image_options) -> Image:
    """
    Displays a PyQuil circuit as an IPython image object.

    For this to work, you need 'pdflatex' and 'convert' to be installed and
    accessible via your shell path, as seen by Python. Further, your LaTeX
    installation should include class and style files for 'standalone', 'geometry',
    'tikz', and 'quantikz'.

    The conversion process relies on two passes: first, 'pdflatex' is called to
    render the tikz to a pdf. Second, Imagemagick's 'convert' is called to translate
    this to a png image.

    :param Program circuit: The circuit to be drawn, represented as a pyquil program.
    :param DiagramSettings settings: An optional object of settings controlling diagram rendering and layout.
    :return: PNG image render of the circuit.
    :rtype: Image
    """
    pdflatex_path = shutil.which('pdflatex')
    convert_path = shutil.which('convert')

    if pdflatex_path is None:
        raise FileNotFoundError("Unable to locate 'pdflatex'.")
    if convert_path is None:
        raise FileNotFoundError("Unable to locate 'convert'.")

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'diagram.tex'), 'w') as texfile:
            texfile.write(to_latex(circuit, settings))

        result = subprocess.call([pdflatex_path, "-output-directory", tmpdir, texfile.name])
        if result != 0:
            raise RuntimeError("pdflatex error")

        png = os.path.join(tmpdir, 'diagram.png')
        pdf = os.path.join(tmpdir, 'diagram.pdf')

        result = subprocess.call([convert_path, '-density', '300', pdf, png])
        if result != 0:
            raise RuntimeError("convert error")

        return Image(filename=png, **image_options)

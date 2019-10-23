import os
import subprocess
import shutil
import tempfile
from IPython.display import Image
from pyquil import Program
from .latex_generation import to_latex


def display(circuit: Program, settings=None, **image_options) -> Image:
    """
    Renders a PyQuil circuit as an IPython image object.

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

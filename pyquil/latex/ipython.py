import os
import subprocess
import tempfile
from IPython.display import Image
from pyquil import Program
from .latex_generation import to_latex

def display(circuit: Program, settings=None, **image_options):
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'diagram.tex'), 'w') as texfile:
            texfile.write(to_latex(circuit, settings))

        result = subprocess.call(['pdflatex', "-output-directory", tmpdir, texfile.name])
        if result != 0:
            # TODO better error message here
            raise ValueError("pdflatex error")

        png = os.path.join(tmpdir, 'diagram.png')
        pdf = os.path.join(tmpdir, 'diagram.pdf')
        result = subprocess.call(['convert', '-density', '300', pdf, png])
        if result != 0:
            raise ValueError("convert error")

        return Image(filename=png, **image_options)

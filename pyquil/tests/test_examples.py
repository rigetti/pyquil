
import os
import subprocess
from subprocess import Popen, PIPE, STDOUT

import pytest


@pytest.fixture
def examples_dir(scope="module"):
    path = os.path.dirname(__file__)
    path = os.path.join(path, '..', '..', 'examples')
    path = os.path.abspath(path)
    return path


def _call_script(path, script, *args):
    # As bare minimum test that all example scripts run without error
    rval = subprocess.call([os.path.join(path, script), *args])
    return rval


def test_meyer_penny_game(forest, examples_dir):
    assert 0 == _call_script(examples_dir, 'meyer_penny_game.py')


def test_run_quil(forest, examples_dir):
    assert 0 == _call_script(examples_dir, 'run_quil.py', 
                             os.path.join(examples_dir, 'hello_world.quil'))


def test_forest2_simple_prog(forest, examples_dir):
    assert 0 == _call_script(examples_dir, 'forest2-simple-prog.py')


def test_pointer(examples_dir):
    assert 0 == _call_script(examples_dir, 'pointer.py')


def test_qaoa_ansatz(forest, examples_dir):
    assert 0 == _call_script(examples_dir, 'qaoa_ansatz.py')


def test_quantum_die(forest, examples_dir):
    p = Popen(['examples/quantum_die.py'],
              stdout=PIPE, stdin=PIPE, stderr=STDOUT,
              universal_newlines=True,
              )
    p.communicate('6')
    assert p.returncode == 0


def test_teleportation(forest, examples_dir):
    assert 0 == _call_script(examples_dir, 'teleportation.py')


def test_website_old(forest, examples_dir):
    assert 0 == _call_script(examples_dir, 'old-website-script.py')


def test_website(forest, examples_dir):
    assert 0 == _call_script(examples_dir, 'website-script.py')

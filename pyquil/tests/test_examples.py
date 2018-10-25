
import subprocess
from subprocess import Popen, PIPE, STDOUT


# As bare minimum test that all example scripts run without error

def test_meyer_penny_game():
    rval = subprocess.call(['examples/meyer_penny_game.py'])
    assert rval == 0


def test_run_quil():
    rval = subprocess.call(['examples/run_quil.py',
                            'examples/hello_world.quil'])
    assert rval == 0


def test_forest2_simple_prog():
    rval = subprocess.call(['examples/forest2-simple-prog.py'])
    assert rval == 0


def test_pointer():
    rval = subprocess.call(['examples/pointer.py'])
    assert rval == 0


def test_qaoa_ansatz():
    rval = subprocess.call(['examples/qaoa_ansatz.py'])
    assert rval == 0


def test_quantum_die():
    p = Popen(['examples/quantum_die.py'],
              stdout=PIPE, stdin=PIPE, stderr=STDOUT,
              universal_newlines=True)
    p.communicate('6')
    assert p.returncode == 0


def test_teleportation():
    rval = subprocess.call(['examples/teleportation.py'])
    assert rval == 0


def test_website():
    rval = subprocess.call(['examples/website-script.py'])
    assert rval == 0


def test_website2():
    rval = subprocess.call(['examples/website-script2.py'])
    assert rval == 0

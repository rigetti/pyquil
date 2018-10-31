.. _wavefunction_simulator:

The Wavefunction Simulator
==========================

Formerly a part of the QVM object in pyQuil, the Wavefunction Simulator allows you to directly inspect the wavefunction
of a quantum state prepared by your program. Because of the probabilistic nature of quantum information, the programs you'll
be running on the QPU can give a distribution of outputs. When running on the QPU or QVM, you would aggregate results
(anywhere from tens of trials to 100k+!) that you can sample to get back a distribution.

With the Wavefunction Simulator, you can look at the distribution without having to collect samples from your program.
This can save a lot of time for small programs. Let's walk through a basic example of using WavefunctionSimulator:

.. code:: python

    from pyquil import Program
    from pyquil.gates import *
    from pyquil.api import WavefunctionSimulator
    wf_sim = WavefunctionSimulator()
    coin_flip = Program((H(0))
    wf_sim.wavefunction(coin_flip)

.. parsed-literal::

    <pyquil.wavefunction.Wavefunction at 0x1088a2c10>

The return value is a Wavefunction object that stores the amplitudes of the quantum state. We can print this object

.. code:: python

    coin_flip = Program(H(0))
    wavefunction = wf_sim.wavefunction(coin_flip)
    print(wavefunction)

.. parsed-literal::

  (0.7071067812+0j)|0> + (0.7071067812+0j)|1>

to see the amplitudes listed as a sum of computational basis states. We can index into those
amplitudes directly or look at a dictionary of associated outcome probabilities.

.. code:: python

  assert wavefunction[0] == 1 / np.sqrt(2)
  # The amplitudes are stored as a numpy array on the Wavefunction object
  print(wavefunction.amplitudes)
  prob_dict = wavefunction.get_outcome_probs() # extracts the probabilities of outcomes as a dict
  print(prob_dict)
  prob_dict.keys() # these store the bitstring outcomes
  assert len(wavefunction) == 1 # gives the number of qubits

.. parsed-literal::

  [ 0.70710678+0.j  0.70710678+0.j]
  {'1': 0.49999999999999989, '0': 0.49999999999999989}


It is important to remember that this ``wavefunction`` method is a useful debugging tool for small quantum systems, and
cannot be feasibly obtained on a quantum processor.

Meyer-Penny Game
----------------

A conceptually simple example that falls into this class is the Meyer-Penny Game. The game goes as follows:

The Starship Enterprise, during one of its deep-space missions, is facing an immediate calamity, when a powerful alien
suddenly appears on the bridge. The alien, named Q, offers to help Picard, the captain of the Enterprise, under the
condition that Picard beats Q in a simple game of penny flips.

The rules:
~~~~~~~~~~
Picard is to place a penny Heads up into an opaque box. Then Picard and Q take turns to flip or not flip the penny without
being able to see it; first Q then P then Q again. After this the penny is revealed; Q wins if it shows Heads (H), while
Tails (T) makes Picard the winner.


Picard vs. Q
~~~~~~~~~~~~

Picard estimates that his chance of winning is 50% and agrees to play the game. He loses the first round and insists on
playing again. To his surprise Q agrees, and they continue playing several rounds more, each of which Picard loses. How
is that possible?

What Picard did not anticipate is that Q has access to quantum tools. Instead of flipping the penny, Q puts the penny into
a superposition of Heads and Tails proportional to the quantum state |H⟩+|T⟩. Then no matter whether Picard flips the penny
or not, it will stay in a superposition (though the relative sign might change). In the third step Q undoes the superposition
and always finds the penny to shows Heads.

Let's see how this works!

To simulate the game, we first construct the corresponding quantum circuit, which takes two qubits – one to simulate
Picard’s choice whether or not to flip the penny and the other to represent the penny. The initial state for all Qubits
is |0⟩(=|T⟩). To simulate Picard’s decision, we assume that he chooses randomly whether or not to flip the coin, in
agreement with the optimal strategy for the classic penny-flip game. This random choice can be created by putting one
qubit into an equal superposition, e.g. with the Hadamard gate H, and then measure its state. The measurement will show
Heads or Tails with equal probability p=0.5.

To simulate the penny flip game we take the second qubit and put it into its excited state |1⟩(=|H⟩) by applying the X
(or NOT) gate. Q’s first move is to apply the Hadamard gate H. Picard’s decision about the flip is simulated as a CNOT
operation where the control bit is the outcome of the random number generator described above. Finally Q applies a Hadamard
gate again, before we measure the outcome.

We first import the necessary tools

.. code:: python

    from pyquil import Program
    from pyquil.api import WavefunctionSimulator
    from pyquil.gates import *

    wf_sim = WavefunctionSimulator()
    prog = Program()
    ro = prog.declare('ro', 'BIT', 2)

Then we need to define two registers that will be used for the measurement of Picard’s decision bit and the final answer
of the penny tossing game.

.. code:: python

    picard_register = ro[1]
    answer_register = ro[0]

We need to encode the two different actions of Picard, which conceptually is equivalent to an if-else control flow as:

.. code:: python

    then_branch = Program(X(0))
    else_branch = Program(I(0))

and then wire it all up into the overall measurement circuit:

.. code:: python

    prog.inst(X(0), H(1))
    prog.inst(H(0))
    prog.measure(1, picard_register)
    prog.if_then(picard_register, then_branch, else_branch)
    prog.inst(H(0))
    prog.measure(0, answer_register)
    print(prog)

Finally we play the game several times

.. code:: python

    wf_sim.run_and_measure(prog, [0, 1], 10)

Remember that the first number is the outcome of the game (value of the answer_register) whereas the second number is the
outcome of Picard’s decision (value of the picard_register).

No matter what Picard does, Q will always win!



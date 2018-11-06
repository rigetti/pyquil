.. _exercises:

Exercises
=========

Exercise 1: Quantum Dice
------------------------

Write a quantum program to simulate throwing an 8-sided die. The Python
function you should produce is:

::

    def throw_octahedral_die():
        # return the result of throwing an 8 sided die, an int between 1 and 8, by running a quantum program

Next, extend the program to work for any kind of fair die:

::

    def throw_polyhedral_die(num_sides):
        # return the result of throwing a num_sides sided die by running a quantum program

Exercise 2: Controlled Gates
----------------------------

We can use the full generality of NumPy to construct new gate matrices.

1. Write a function ``controlled`` which takes a :math:`2\times 2`
   matrix :math:`U` representing a single qubit operator, and makes a
   :math:`4\times 4` matrix which is a controlled variant of :math:`U`,
   with the first argument being the *control qubit*.

2. Write a Quil program to define a controlled-\ :math:`Y` gate in this
   manner. Find the wavefunction when applying this gate to qubit 1
   controlled by qubit 0.

Exercise 3: Grover's Algorithm
------------------------------

Write a quantum program for the single-shot Grover's algorithm. The
Python function you should produce is:

::

    # data is an array of 0's and 1's such that there are exactly three times as many
    # 0's as 1's
    def single_shot_grovers(data):
        # return an index that contains the value 1

As an example: ``single_shot_grovers([0,0,1,0])`` should return 2.

**HINT** - Remember that the Grover's diffusion operator is:

.. math::

   \begin{pmatrix}
   2/N - 1 & 2/N & \cdots & 2/N \\
   2/N &  & &\\
   \vdots & & \ddots & \\
   2/N & & & 2/N-1
   \end{pmatrix}


Exercise 4: Prisoner's Dilemma
------------------------------

A classic strategy game is the `prisoner's dilemma <https://en.wikipedia.org/wiki/Prisoner%27s_dilemma>`_ where two
prisoners get the minimal penalty if they collaborate and stay silent, get zero penalty if one of them defects and the
other collaborates (incurring maximum penalty) and get intermediate penalty if they both defect. This game has an
equilibrium where both defect and incur intermediate penalty.

However, things change dramatically when we allow for quantum strategies leading to the
`Quantum Prisoner's Dilemma <https://arxiv.org/abs/quant-ph/9806088>`_.

Can you design a program that simulates this game?

Examples of Quantum Programs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create intuition for a new class of algorithms, that will run on the Quantum Virtual Machine (QVM), it is useful (and
fun) to play with the abstraction that the software provides.

A broad class of programs that can easily be implemented on a QVM are generalizations of
`Game Theory to incorporate Quantum Strategies <https://arxiv.org/abs/quant-ph/0611234>`_.

Meyer-Penny Game
----------------

A conceptually simple example that falls into this class is the
`Meyer-Penny Game <https://link.aps.org/doi/10.1103/PhysRevLett.82.1052>`_. The game goes as follows:

The Starship Enterprise, during one of its deep-space missions, is facing an immediate calamity, when a powerful alien
suddenly appears on the bridge. The alien, named Q, offers to help Picard, the captain of the Enterprise, under the
condition that Picard beats Q in a simple game of penny flips.

The rules:
----------
Picard is to place a penny Heads up into an opaque box. Then Picard and Q take turns to flip or not flip the penny without
being able to see it; first Q then P then Q again. After this the penny is revealed; Q wins if it shows Heads (H), while
Tails (T) makes Picard the winner.

Picard vs. Q
~~~~~~~~~~~~

Picard quickly estimates that his chance of winning is 50% and agrees to play the game. He loses the first round and
insists on playing again. To his surprise Q agrees, and they continue playing several rounds more, each of which Picard
loses. How is that possible?

What Picard did not anticipate is that Q has access to quantum tools. Instead of flipping the penny, Q puts the penny
into a superposition of Heads and Tails proportional to the quantum state :math:`|H\rangle+|T\rangle`. Then no matter
whether Picard flips the penny or not, it will stay in a superposition (though the relative sign might change). In the
third step Q undoes the superposition and always finds the penny to shows Heads.

Let's see how this works!

To simulate the game we first construct the corresponding quantum circuit, which takes two qubits -- one to simulate
Picard's choice whether or not to flip the penny and the other to represent the penny. The initial state for all Qubits
is :math:`|0\rangle` (which is mapped to :math:`|T\rangle`, tails). To simulate Picard's decision, we assume that he
chooses randomly whether or not to flip the coin, in agreement with the optimal strategy for the classic penny-flip
game. This random choice can be created by putting one qubit into an equal superposition, e.g. with the Hadamard gate
:math:`H`, and then measure its state. The measurement will show heads or tails with equal probability p=0.5.

To simulate the penny flip game we take the second qubit and put it into its excited state
:math:`|1\rangle` (which is mapped to :math:`|H\rangle)`, heads) by applying the X (or NOT) gate. Q's first move is to
apply the Hadamard gate H. Picard's decision about the flip is simulated as a CNOT operation where the control bit is
the outcome of the random number generator described above. Finally Q applies a Hadamard gate again, before we measure
the outcome. The full circuit is shown in the figure below.

.. figure:: images/MeyerPennyGame.png
    :align: center
    :figwidth: 65%

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

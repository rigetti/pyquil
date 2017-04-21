Examples of Quantum Programs on a QVM
=====================================

To create intuition for a new class of algorithms, that will run on Quantum Virtual Machines (QVM), it is useful (and
fun) to play with the abstraction that the software provides.

A broad class of programs that can easily be implemented on a QVM are generalizations of
`Game Theory to incorporate Quantum Strategies <https://arxiv.org/abs/quant-ph/0611234>`_.


Meyer-Penny Game
----------------

A conceptually simple example that falls into this class is the
`Meyer-Penny Game <https://link.aps.org/doi/10.1103/PhysRevLett.82.1052>`_. The game goes as follows: The Starship
Enterprise, during one of its deep-space missions, is facing an immediate calamity, when a powerful alien suddenly
appears on the bridge. The alien, named Q, offers to help Picard, the captain of the Enterprise, under the condition
that Picard beats Q in a simple game of penny flips.

The rules: Picards is to place a penny Heads up into an opaque box. Then Picard and Q take turns to flip or not flip the
penny without being able to see it; first Q then P then Q again. After this the penny is revealed; Q wins if it shows
Heads (H), while Tails (T) makes Picard the winner.

Picard quickly estimates that his chance of winning is 50% and agrees to play the game. He loses the first round and
insists on playing again. To his surprise Q agrees, and they continue playing several rounds more, each of which Picard
loses. How is that possible?

What Picard did not anticipate is that Q has access to quantum tools. Instead of flipping the penny, Q puts the penny
into a superposition of Heads and Tails proportional to the quantum state :math:`|H\rangle+|T\rangle`. Then no matter
whether Picard flips the penny or not, it will stay in a superposition (though the relative sign might change). In the
third step Q undoes the superposition and always finds the penny to shows Heads.

To simulate the game we first construct the corresponding quantum circuit, which takes two qubits -- one to simulate
Picard's choice whether or not to flip the penny and the other to represent the penny. The initial state for all Qubits
is :math:`|0\rangle (= |T\rangle)`. To simulate Picard's decision, we assume that he chooses randomly whether or not to
flip the coin, in agreement with the optimal strategy for the classic penny-flip game. This random choice can be created
by putting one qubit into an equal superposition, e.g. with the Hadamard gate H, and then measure its state. The
measurement will show Heads or Tails with equal probability p=0.5.

To simulate the penny flip game we take the second qubit and put it into its excited state
:math:`|1\rangle (= |H\rangle)` by applying the X (or NOT) gate. Q's first move is to apply the Hadamard gate H.
Picard's decision about the flip is simulated as a CNOT operation where the control bit is the outcome of the random
number generator described above. Finally Q applies a Hadamard gate again, before we measure the outcome. The full
circuit is shown in the figure below.

.. figure:: images/MeyerPennyGame.png
    :align: center
    :figwidth: 65%

First we import all the necessary tools:

.. code-block:: python

    import pyquil.quil as pq
    from pyquil import forest

    from pyquil.gates import I, H, X
    qvm = forest.Connection()

Then we need to define two registers that will be used for the measurement of Picard's decision bit and the final answer
of the penny tossing game.

.. code-block:: python

    picard_register = 1
    answer_register = 0

Moreover we need to encode the two different actions of Picard, which conceptually is equivalent to an `if-else` control
flow as:

.. code-block:: python

    then_branch = pq.Program(X(0))
    else_branch = pq.Program(I(0))


and then wire it all up into the overall measurement circuit:

.. code-block:: python

   prog = (pq.Program()
       # Prepare Qubits in Heads state or superposition, respectively
       .inst(X(0), H(1))
       # Q puts the penny into a superposition
       .inst(H(0))
       # Picard makes a decision and acts accordingly
       .measure(1, picard_register)
       .if_then(picard_register, then_branch, else_branch)
       # Q undoes his superposition operation
       .inst(H(0))
       # The outcome is recorded into the answer register
       .measure(0, answer_register))


Finally we play the game several times

.. code-block:: python

   qvm.run(prog, [0, 1], trials=10)


and record the register outputs as

.. code-block:: python

   [[1, 1],
    [1, 1],
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 1],
    [1, 1],
    [1, 0],
    [1, 0]]

Remember that the first number is the outcome of the game (value of the `answer_register`) whereas the second number is
the outcome of Picard’s decision (value of the `picard_register`).

Indeed, no matter what Picard does, Q will always win!

Exercises
---------

Prisoner's Dilemma
++++++++++++++++++

A classic strategy game is the `prisoner's dilemma <https://en.wikipedia.org/wiki/Prisoner%27s_dilemma>`_ where two
prisoners get the minimal penalty if they collaborate and stay silent, get zero penalty if one of them defects and the
other collaborates (incurring maximum penalty) and get intermediate penalty if they both defect. This game has an
equilibrium where both defect and incur intermediate penalty.

However, things change dramatically when we allow for quantum strategies leading to the
`Quantum Prisoner's Dilemma <https://arxiv.org/abs/quant-ph/9806088>`_.

Can you design a program that simulates this game?

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

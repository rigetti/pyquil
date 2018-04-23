
Using Qubit Placeholders
========================

In PyQuil, we typically use integers to identify qubits

.. code:: ipython3

    from pyquil.quil import Program
    from pyquil.gates import CNOT, H
    print(Program(H(0), CNOT(0, 1)))


.. parsed-literal::

    H 0
    CNOT 0 1
    


However, when running on real, near-term QPUs we care about what
particular physical qubits our program will run on. In fact, we may want
to run the same program on an assortment of different qubits. This is
where using ``QubitPlaceholder``\ s comes in.

.. code:: ipython3

    from pyquil.quilatom import QubitPlaceholder
    q0 = QubitPlaceholder()
    q1 = QubitPlaceholder()
    prog = Program(H(q0), CNOT(q0, q1))
    print(prog)


.. parsed-literal::

    H <QubitPlaceholder 4590706304>
    CNOT <QubitPlaceholder 4590706304> <QubitPlaceholder 4590705912>
    


If you try to use this program directly, it will not work

.. code:: ipython3

    print(prog.out())


::


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-4-da474d3af403> in <module>()
    ----> 1 print(prog.out())
    
    ...

    pyquil/pyquil/quilatom.py in out(self)
         53 class QubitPlaceholder(QuilAtom):
         54     def out(self):
    ---> 55         raise RuntimeError("Qubit {} has not been assigned an index".format(self))
         56 
         57     def __str__(self):


    RuntimeError: Qubit <QubitPlaceholder 4590706304> has not been assigned an index


Instead, you must explicitly map the placeholders to physical qubits. By
default, the function ``address_qubits`` will address qubits from 0 to
N.

.. code:: ipython3

    from pyquil.quil import address_qubits
    print(address_qubits(prog))


.. parsed-literal::

    H 0
    CNOT 0 1
    


The real power comes into play when you provide an explicity mapping

.. code:: ipython3

    print(address_qubits(prog, qubit_mapping={
        q0: 14,
        q1: 19,
    }))


.. parsed-literal::

    H 14
    CNOT 14 19
    


Register
--------

Usually, your algorithm will use an assortment of qubits. You can use
the convenience function ``QubitPlaceholder.register()`` to request a
list of qubits to build your program.

.. code:: ipython3

    qbyte = QubitPlaceholder.register(8)
    prog2 = Program(H(q) for q in qbyte)
    print(address_qubits(prog2, {q: i*2 for i, q in enumerate(qbyte)}))


.. parsed-literal::

    H 0
    H 2
    H 4
    H 6
    H 8
    H 10
    H 12
    H 14
    


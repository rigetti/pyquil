.. _device_class:

The Device class
================

The pyQuil ``Device`` class provides useful information for learning about, and working with,
Rigetti's available QPUs. One may query for available devices using the ``get_devices`` function:

.. code:: python

    from pyquil.api import get_devices

    devices = get_devices(as_dict=True)
    # E.g. {'19Q-Acorn': <Device 19Q-Acorn online>, '8Q-Agave': <Device 8Q-Agave offline>}

    acorn = devices['19Q-Acorn']


The variable ``acorn`` points to a ``Device`` object that holds useful information regarding the
QPU, including:

1. Connectivity via its instruction set architecture (``acorn.isa`` of class ``ISA``).
2. Hardware specifications such as coherence times and fidelities (``acorn.specs`` of class ``Specs``).
3. Noise model information (``acorn.noise_model`` of class ``NoiseModel``).

These 3 attributes are accessed in the following ways:


.. code:: python

    print(acorn.specs)
    # Specs(qubits_specs=..., edges_specs=...)

    print(acorn.specs.qubits_specs)
    """
    [_QubitSpecs(id=0, fRO=0.938, f1QRB=0.9815, T1=1.52e-05, T2=7.2e-06),
     _QubitSpecs(id=1, fRO=0.958, f1QRB=0.9907, T1=1.76e-05, T2=7.7e-06),
     _QubitSpecs(id=2, fRO=0.97, f1QRB=0.9813, T1=1.82e-05, T2=1.08e-05),
     _QubitSpecs(id=3, fRO=0.886, f1QRB=0.9908, T1=3.1e-05, T2=1.68e-05),
     _QubitSpecs(id=4, fRO=0.953, f1QRB=0.9887, T1=2.3e-05, T2=5.2e-06),
     _QubitSpecs(id=5, fRO=0.965, f1QRB=0.9645, T1=2.22e-05, T2=1.11e-05),
     _QubitSpecs(id=6, fRO=0.84, f1QRB=0.9905, T1=2.68e-05, T2=2.68e-05),
     _QubitSpecs(id=7, fRO=0.925, f1QRB=0.9916, T1=2.94e-05, T2=1.3e-05),
     _QubitSpecs(id=8, fRO=0.947, f1QRB=0.9869, T1=2.45e-05, T2=1.38e-05),
     _QubitSpecs(id=9, fRO=0.927, f1QRB=0.9934, T1=2.08e-05, T2=1.11e-05),
     _QubitSpecs(id=10, fRO=0.942, f1QRB=0.9916, T1=1.71e-05, T2=1.06e-05),
     _QubitSpecs(id=11, fRO=0.9, f1QRB=0.9901, T1=1.69e-05, T2=4.9e-06),
     _QubitSpecs(id=12, fRO=0.942, f1QRB=0.9902, T1=8.2e-06, T2=1.09e-05),
     _QubitSpecs(id=13, fRO=0.921, f1QRB=0.9933, T1=1.87e-05, T2=1.27e-05),
     _QubitSpecs(id=14, fRO=0.947, f1QRB=0.9916, T1=1.39e-05, T2=9.4e-06),
     _QubitSpecs(id=16, fRO=0.948, f1QRB=0.9906, T1=1.67e-05, T2=7.5e-06),
     _QubitSpecs(id=17, fRO=0.921, f1QRB=0.9895, T1=2.4e-05, T2=8.4e-06),
     _QubitSpecs(id=18, fRO=0.93, f1QRB=0.9496, T1=1.69e-05, T2=1.29e-05),
     _QubitSpecs(id=19, fRO=0.93, f1QRB=0.9942, T1=2.47e-05, T2=9.8e-06)]
    """

    print(acorn.isa)
    # ISA(qubits=..., edges=...)

    print(acorn.isa.edges)
    """
    [Edge(targets=[0, 5], type='CZ', dead=False),
     Edge(targets=[0, 6], type='CZ', dead=False),
     Edge(targets=[1, 6], type='CZ', dead=False),
     Edge(targets=[1, 7], type='CZ', dead=False),
     Edge(targets=[2, 7], type='CZ', dead=False),
     Edge(targets=[2, 8], type='CZ', dead=False),
     Edge(targets=[4, 9], type='CZ', dead=False),
     Edge(targets=[5, 10], type='CZ', dead=False),
     Edge(targets=[6, 11], type='CZ', dead=False),
     Edge(targets=[7, 12], type='CZ', dead=False),
     Edge(targets=[8, 13], type='CZ', dead=False),
     Edge(targets=[9, 14], type='CZ', dead=False),
     Edge(targets=[10, 15], type='CZ', dead=False),
     Edge(targets=[10, 16], type='CZ', dead=False),
     Edge(targets=[11, 16], type='CZ', dead=False),
     Edge(targets=[11, 17], type='CZ', dead=False),
     Edge(targets=[12, 17], type='CZ', dead=False),
     Edge(targets=[12, 18], type='CZ', dead=False),
     Edge(targets=[13, 18], type='CZ', dead=False),
     Edge(targets=[13, 19], type='CZ', dead=False),
     Edge(targets=[14, 19], type='CZ', dead=False)]
    """

    print(acorn.noise_model)
    # NoiseModel(gates=[KrausModel(...) ...] ...)


Additionally, the ``Specs`` class provides methods for access specs info across the chip in a more
succinct manner:


.. code:: python

	acorn.specs.T1s()
	# {0: 1.52e-05, 1: 1.76e-05, 2: 1.82e-05, 3: 3.1e-05, ...}

	acorn.specs.fCZs()
	# {(0, 5): 0.888, (0, 6): 0.8, (1, 6): 0.837, (1, 7): 0.87, ...}


With these tools provided by the ``Device`` class, users may learn more about Rigetti hardware, and
construct programs tailored specifically to that hardware. In addition, the ``Device`` class serves
as a powerful tool for seeding a QVM with characteristics of the device. For more information on
this, see here: :ref:`qvm_with_device`.

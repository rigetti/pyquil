.. _troubleshooting:

Troubleshooting
===============

If you're having any trouble running your pyQuil programs locally or on the QPU, please check the
following things before sending a support request. It will save you time and make it easier for us
to help!

.. _timeouts:

Timeout in compilation
----------------------

This may occur due to the size of your program, problems with your local ``quilc`` service, or problems
with the remote QCS API.

1. Ensure that ``quilc`` is running, per the compiler documentation: :ref:`compiler`.
2. Try compiling and running a smaller program, or one that previously worked.
3. If you have a larger program and expect it to take a long time to compile, set a higher ``compiler_timeout``
   value per the instructions here: :ref:`compiler`.

Timeout in execution
--------------------

This may occur due to one of several different problems. Often, it's because you don't have network access
to the execution endpoint.

If you're running against the QVM, ensure that it's running: :ref:`server`. If you're using docker,
you can check this using ``docker ps``.

If you're running against the QPU, ensure that you are running your program from a supported environment.
Live Rigetti QPUs are **only accessible from Rigetti-provided environments**, such as
`JupyterHub <https://jupyterhub.qcs.rigetti.com>`_. If you are running from anywhere else, such as a
script on your local computer or a public cloud virtual machine,
**your program won't be able to reach a QPU and will time out**.

Collect debug information
-------------------------

1. Ensure that your pyQuil version is up to date. If you're using ``pip``, you can do this with
   ``pip show pyquil``. Within your script, you can use ``__version__``:

   .. testcode:: version

    import pyquil
    print(pyquil.__version__)

   .. testoutput:: version
      :hide:

      ...

   You can update pyQuil with ``pip`` using ``pip install pyquil --upgrade``. You can find
   the latest version available at
   `the releases page <https://github.com/rigetti/pyquil/releases>`_ or
   `on PyPi <https://pypi.org/project/pyquil/>`_.


2. pyQuil exposes a diagnostics report that prints helpful debugging information, including
   whether you have connectivity to ``quilc``, ``QVM`` and access to QCS services. You can
   use it by importing a function from the ``diagnostics`` module:

   .. testcode:: version

    from pyquil.diagnostics import get_report
    print(get_report())

   .. testoutput:: version
      :hide:
  
      ...

   Use this report to confirm you have connectivity to the services you need and that your
   environment matches what you expect.


3. Run your script with debug logging enabled by adding the following to the top of your script:

   .. testcode:: debug

    import logging
    logging.basicConfig(level=logging.DEBUG)

   .. note:: For information on how to filter the logs, see the `qcs-sdk-python logging documentation <https://github.com/rigetti/qcs-sdk-rust/tree/main/crates/python#enabling-debug-logging>`_

   .. testcode:: debug
   :hide:
    
    # Disable deubg logging, otherwise doctests will run with
    # debug logging enabled.
    logging.basicConfig(level=logging.INFO)

If the problem still isn't clear, then we can help! Please file an issue
on the `GitHub repo <https://github.com/rigetti/pyquil>`_ if it's an issue with pyQuil itself,
or contact us at our `support page <https://rigetti.zendesk.com>`_ for problems with QCS. If applicable,
be sure to include the diagnostics report and debug logs from above as they will help us better
diagnose the problem.

Thanks for using pyQuil!

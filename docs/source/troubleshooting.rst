.. _troubleshooting:

Troubleshooting
===============

If you're having any trouble running your pyQuil programs locally or on the QPU, please check the
following things before sending a support request. It will save you time and make it easier for us
to help!

1. Ensure that your pyQuil version is up to date. If you're using ``pip``, you can do this with
   ``pip freeze``. Within your script, you can use ``__version__``:

   .. code:: python

    import pyquil
    print(pyquil.__version__)

   You can update pyQuil with ``pip`` using ``pip install pyquil --upgrade``. You can find
   the latest version available at
   `our releases page <https://github.com/rigetti/pyquil/releases>`_ or
   `on PyPi <https://pypi.org/project/pyquil/>`_.

2. If the error appears to be authentication-related, refer to the `QCS CLI documentation
<https://docs.rigetti.com/en/command-line-interface/command-line-interface>`_.

3. Run your script with debug logging enabled. If you're running a script, you can enable that
   using an environment variable:

   .. code::

    LOG_LEVEL=DEBUG pyquil my_script.py

   .. code:: python

    import logging
    from pyquil.api._logger import logger

    logger.setLevel(logging.DEBUG)

If the problem still isn't clear, then we can help! Please contact us at our
`support page <https://rigetti.zendesk.com>`_. Thanks for using pyQuil!

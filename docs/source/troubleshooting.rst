.. _troubleshooting:

Troubleshooting
===============

If you're having any trouble running your Pyquil programs locally or on the QPU, please check the
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

2. If the error appears to be authentication-related, or makes any mention of your
   ``user_auth_token``, then please update your token following the directions at
   https://qcs.rigetti.com/auth/token.

3. Run your script with debug logging enabled. If you're running a script, you can enable that
   using an environment variable:

   .. code:: python

    LOG_LEVEL=DEBUG pyquil my_script.py

   If you're running a notebook, then you can change the log level within your code:

   .. code:: python

    import logging
    from pyquil.api._logger import logger

    logger.setLevel(logging.DEBUG)

   If the problem still isn't clear, then we can help! Please send your debug log to us, 
   along with the contents of your ``~/.qcs_config`` and ``~/.forest_config`` files, at our
   `support page <https://rigetti.zendesk.com>`_. Thanks for using pyQuil!

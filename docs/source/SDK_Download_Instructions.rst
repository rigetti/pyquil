.. _sdk:

The Rigetti Forest SDK 2.0 Preview: Download Instructions
=========================================================

Introduction
------------

The Rigetti Forest SDK 2.0 Preview is currently supported on AMD64
processors running UNIX-like operating systems, specifically macOS
10.12+ and Linux. Microsoft Windows is *not* currently supported.

The Rigetti Forest SDK 2.0 Preview currently contains:

-  The Rigetti Quantum Virtual Machine (``qvm``) which allows
   high-performance simulation of Quil programs,
-  The Rigetti Quil Compiler (``quilc``) which allows compilation and
   optimization of Quil programs to native gate sets

The SDK also includes PyQuil 2.0 Preview, though that can be found on
the usual `GitHub <https://github.com/rigetticomputing/pyquil>`__
page. PyQuil makes writing and executing Quil programs against
Rigetti's services easier.

The QVM and the compiler are packed as program binaries that are
accessed through the command line. Both of them provide support for
direct command-line interaction, as well as a server mode. The server
mode is required for use with PyQuil.

Download
--------

Determine your platform and download the necessary tarball from
`here <https://www.rigetti.com/forest>`__.

-  `macOS <https://downloads.rigetti.com/qcs-sdk/forest-sdk.dmg>`__: ``forest-sdk.dmg``
-  `Linux (Debian, Ubuntu, etc.) <https://downloads.rigetti.com/qcs-sdk/forest-sdk-linux-deb.tar.bz2>`__: ``forest-sdk-linux-deb.tar.bz2``
-  `Linux (Red Hat, CentOS, Fedora, etc.) <https://downloads.rigetti.com/qcs-sdk/forest-sdk-linux-rpm.tar.bz2>`__: ``forest-sdk-linux-rpm.tar.bz2``
-  `Linux (Bare-bones) <https://downloads.rigetti.com/qcs-sdk/forest-sdk-linux-barebones.tar.bz2>`__:  ``forest-sdk-linux-barebones.tar.bz2``

Install
-------

Below are installation instructions for the Rigetti Forest SDK 2.0
Preview. The main goals of installation are

1. to install the ``qvm`` and ``quilc`` command line programs, and
2. to install the Python library PyQuil 2.0 Preview.

All installation mechanisms, except the bare-bones package, require
administrative privileges to install. The bare-bones package just
contains binaries and documentation without any of the prerequisites.

Installing the QVM and Compiler on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mount the file ``forest-sdk.dmg`` by double clicking on it. From
there, open ``forest-sdk.pkg`` by double-clicking on it. Follow the
installation instructions.

Upon successful installation, one should be able to open a new terminal
window and run the following two commands:

::

    qvm --version
    quilc --version

To uninstall, delete the following files:

::

    /usr/local/bin/qvm
    /usr/local/bin/quilc
    /usr/local/share/man/man1/qvm.1
    /usr/local/share/man/man1/quilc.1

Installing the QVM and Compiler on Linux (deb)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, unpack the tarball and change to that directory by doing:

::

    tar -xf forest-sdk-linux-deb.tar.bz2
    cd forest-sdk-2.0rc2-linux-deb

From here, run the following command:

::

    sudo ./forest-sdk-2.0rc2-linux-deb.run

Upon successful installation, one should be able to run the following
two commands:

::

    qvm --version
    quilc --version

To uninstall, type:

::

    sudo apt remove forest-sdk

Installing the QVM and Compiler on Linux (rpm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, unpack the tarball and change to that directory by doing:

::

    tar -xf forest-sdk-linux-rpm.tar.bz2
    cd forest-sdk-2.0rc2-linux-rpm

From here, run the following command:

::

    sudo ./forest-sdk-2.0rc2-linux-rpm.run

Upon successful installation, one should be able to run the following
two commands:

::

    qvm --version
    quilc --version

To uninstall, type:

::

    sudo rpm -e forest-sdk
    # or
    sudo yum uninstall forest-sdk

Installing the QVM and Compiler on Linux (bare-bones)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The barebones installation only contains the executable binaries and
manual pages, and doesn't contain any of the requisite dynamic
libraries. As such, installation doesn't require administrative or
``sudo`` privileges.

First, unpack the tarball and change to that directory by doing:

::

    tar -xf forest-sdk-linux-barebones.tar.bz2
    cd forest-sdk-2.0rc2-linux-barebones

From here, run the following command:

::

    ./forest-sdk-2.0rc2-linux-barebones.run

Upon successful installation, this will have created a new directory
``rigetti`` in your home directory that contains all of the binary and
documentation artifacts.

This method of installation requires one, through whatever means, to
install shared libraries for BLAS, LAPACK, and libffi. On a Debian-derivative system, this could be accomplished with

::

   sudo apt-get install liblapack-dev libblas-dev libffi-dev

To uninstall, remove the directory ``~/rigetti``.


Installing PyQuil 2.0 Preview on All Platforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ATTENTION::

   This section will change once PyQuil 2.0 is released to GitHub.

On a Mac, change to the mounted directory

::

    cd /Volumes/ForestSDK

and on Linux change to the extracted directory

::

    cd forest-sdk-2.0rc2-linux-{deb, rpm, barebones}

Next, run the command:

::

    pip install pidgin-external-1.7.0.tar.gz
    pip install pyquil-2.0.0b1.tar.gz

Note that both now require at least Python 3.0.

Using the SDK
-------------

The SDK may either be used directly from the command line, or through
PyQuil.

Using the QVM and Compiler Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refer to the manual pages for the QVM and compiler for information on
how to use them directly. One can read the manual pages by open a new
terminal window and typing ``man qvm`` (for the QVM) or ``man quilc``
(for the compiler). One can quit out of the manual page by typing ``q``.

Using PyQuil 2.0 Preview with the SDK Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyQuil as provided through Rigetti Forest SDK 2.0 Preview is
automatically configured to connect to the QVM and quantum compiler
server, also available as part of that same local development
environment.

.. NOTE::

    In case of trouble, see ***Server Endpoint Configuration***
    below for more information about informing pyQuil where to find
    the relevant servers.

Before starting development in pyQuil or running a pyQuil program, be
sure to launch these servers as background processes as in the following
two console sessions:

::

    ### CONSOLE 1
    $ quilc -S
    port triggered: 6000.
    [2018-09-19 11:22:37] Starting server: 0.0.0.0 : 6000.

::

    ### CONSOLE 2
    $ qvm -S
    ******************************
    * Welcome to the Rigetti QVM *
    ******************************
    (Configured with 2048 MiB of workspace and 8 workers.)

    [2018-09-20 15:39:50] Starting server on port 5000.

With these two launched, we can immediately jump into a simple quantum
simulation in PyQuil:

::

    $ python3
    >>> from pyquil.api import get_qc
    >>> from pyquil import Program
    >>> qc = get_qc("9q-generic-qvm")
    >>> p = Program("H 0")
    >>> qc.run_and_measure(p, 10)
    array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]])

Here, we see 10 measurement trials, and in each row the measurement of
each of the nine qubits, the first one having a 50% probability of being
measured as ``0`` or ``1``, and the remaining having a 100% probability
of being measured as ``0``. You should also see activity in both of the
server terminals: this call to ``.run_and_measure()`` first uses the
local compiler to compile ``p`` to the "native gate set", then executes
the resulting program on the local QVM.

Server Endpoint Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected locations of the QVM and Compiler endpoints are
configurable in pyQuil. When running on a QMI, these configuration
values are automatically managed so as to point to the relevant
Rigetti-internal endpoints. When running locally, these default to
values reasonable for a user running local instances of the Rigetti
toolchain on their laptop. Ideally, little-to-no work will be required
for setting up this configuration environment locally or remotely, or
for transferring functioning code from one configured environment to
another.

In general, these values are read out of a pair of configuration files
(stored at the location described by the environment variables
``FOREST_CONFIG`` and ``QCS_CONFIG``, or else at the respective default
locations ``~/.forest_config`` and ``~/.qcs_config``), which by default
have the following respective contents:

::

    [Rigetti Forest]
    url = https://api.rigetti.com/
    key = None
    user_id = None

    [QPU]
    exec_on_engage = :

::

    [Rigetti Forest]
    qpu_endpoint_address = None
    qvm_address = http://localhost:5000
    compiler_server_address = http://localhost:6000

These values control the following behaviors:

-  ``Rigetti Forest``: This section contains network endpoint
   information about the entire Rigetti Forest infrastructure, e.g.,
   where to find information about which QPU devices are available.
-  ``url``: This is the endpoint where pyQuil looks for device
   information and for the 2.0 endpoints.
-  ``key``: This stores the pre-2.0 API key.
-  ``user_id``: This stores a 2.0 user ID.
-  ``qpu_endpoint_address``: This is the endpoint where pyQuil will try to
   communicate with the QPU orchestrating service during QPU-engagement.
-  ``qvm_address``: This is the endpoint where pyQuil will try to
   communicate with the Rigetti Quantum Virtual Machine. On a QMI, this
   points to the provided QVM instance. On a local installation, this
   should be set to the server endpoint for a locally running QVM
   instance.
-  ``compiler_server_address``: This is the endpoint where pyQuil will
   try to communicate with the compiler server. On a QMI, this points to
   a provided compiler server instance. On a local installation, this
   should be set to the server endpoint for a locally running quilc
   instance.
-  ``QPU``: This section contains configuration information pertaining
   to QPU access.
-  ``exec_on_engage``: This is the shell command that the QMI will
   launch when the QMI becomes QPU-engaged.

    **NOTE:** PyQuil itself reads these values out using the helper
    class ``pyquil._config.PyquilConfig``. PyQuil users should not ever
    need to touch this class directly.

Support
-------

This is a preview of the upcoming release of the Forest 2.0 SDK for
Rigetti Quantum Cloud Services. We welcome and encourage feedback.
Feedback, or in the event of difficulties, write to support@rigetti.com
with a detailed description of your problem.

To join our user community, connect to the Rigetti Slack workspace at
https://rigetti-forest.slack.com.

More extensive documentation of pyQuil 2.0, including a migration guide
from pyQuil 1.9, will be available soon.

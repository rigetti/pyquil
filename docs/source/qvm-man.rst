.. _qvm_man:

QVM Man Page
============

NAME
~~~~

``qvm`` - a quantum virtual machine for executing Quil

SYNOPSIS
~~~~~~~~

``qvm <options> -e`` # Execute Mode

``qvm <options> -S`` # Server Mode

``qvm <options> --benchmark [n]`` # Benchmark Mode

DESCRIPTION
~~~~~~~~~~~

The Rigetti QVM is a high-performance, classical implementation of a
quantum abstract machine. Specifically, it is capable of executing Quil
in a variety of ways. The QVM has three main modes of operation: Execute
Mode, Server Mode, and Benchmark Mode.

Execute Mode runs the QVM on a single Quil file, printing out
information about the execution, as well as a textual representation of
the wavefunction if ``--verbose`` is provided. (If one would like full
access to the wavefunction as an efficient representation, one should
use ``--shared``.)

Server Mode runs the QVM as an HTTP server, taking simple POST requests
with JSON payloads which are known to the companion library PyQuil. The
server is useful even to a single user wishing to run a variety of
computations.

Benchmark Mode is used for stress testing the QVM on a computer.

The QVM implements both a Quil interpreter and a just-in-time (JIT)
compiler. (Note that "compilation" here does not refer to the sense
related to translation of gate sets, but rather translation to machine
code for rapid execution.) Interpreted mode is enabled by default and
works for all modes of operation. JIT compilation is enabled by
supplying the ``--compile`` option.

The QVM does not have explicit options for running programs with noise
models. Instead, the Quil program itself specifies PRAGMAs for defining
Kraus operators and readout POVMs.

OPTIONS
~~~~~~~

``-e, --execute``
      (Execute Mode) Run the QVM in Execute Mode. Execute the Quil program supplied from stdin  and  print  some  information
      about the course of evaluation.

``-S, --server``
      (Server Mode) Run the QVM in Server Mode. This starts an HTTP server.

``--benchmark [<n>]``
      (Benchmark Mode) Run the QVM benchmark on <n> qubits. The default is 26.

``-p <port>, --port <port>``
      (Server Mode) Run the QVM server on port <port>. The default is 5000.

``--memory-limit <num-octets>``
      Limit the amount of classical memory to <num-octets> octets usable by an individual Quil program. The default is 65536.

``-w <n>, --num-workers <n>``
      Force the number of parallel workers to be <n>. By default, this is the number of logical cores of the host machine.

``--time-limit <limit-ms>``
      (Server  Mode) Limit the amount of time for a single request to approximately <limit-ms> milliseconds. By default, this
      value is 0, which indicates an unlimited amount of time is allowed.

``--qubit-limit <n>``
      (Server Mode) Limit requests to consuming <n> qubits.

``--benchmark-type <name>``
      (Benchmark Mode) Run the benchmark named <name>. Benchmarks include "bell", "qft", "hadamard".

``-h, --help``
      Show the help message.

``-v, --version``
      Show the version.

``--verbose``
      Print every instruction transition of the QVM with information about timing and allocation.

      (Execute Mode) Print each basis state and corresponding amplitude of the evolved wavefunction.

``-c, --compile``
      JIT compile the Quil programs to make them run faster.

``--safe-include-directory <dir>``
      Only allow <dir> to be included from with the Quil INCLUDE directive.

``--shared <name>``
      (Server Mode) Run the QVM in shared memory mode. This allocates the wavefunction in POSIX shared memory  named  <name>.
      If <name> is an empty string, then a name will be generated. The --qubits argument must be specified.


EXAMPLES
~~~~~~~~

``qvm -e < file.quil``
      Run a Quil file on the QVM.

``printf "H 0\nCNOT 0 1\nCNOT 1 2" | qvm --verbose -e``
      Create a 3-qubit Bell state, printing information about the execution along the way.

``qvm -S -p 1234``
      Start a QVM server for use with PyQuil on port 1234.

``qvm -c --benchmark 25 --benchmark-type qft``
      Benchmark a 25-qubit quantum Fourier transform in compiled mode.

BUGS
~~~~

Shared memory mode does not work with QVMs executing noisy programs (i.e., ones where Kraus operators or POVMs are specified).

The WAIT instruction does nothing.

SUPPORT
~~~~~~~

Contact <support@rigetti.com>.

COPYRIGHT
~~~~~~~~~

Copyright (c) 2018 Rigetti Computing

SEE ALSO
~~~~~~~~

:ref:`quilc(1) <quilc_man>`

version 0.16.0 (qvm: 0.20.0) [1b48c69]                    21 September 2018                                                    QVM(1)

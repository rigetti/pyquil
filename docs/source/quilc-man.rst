.. _quilc_man:

QUILC Man Page
==============

NAME
~~~~

``quilc`` - an optimizing, architecture-independent Quil compiler

SYNOPSIS
~~~~~~~~

``quilc <options>``

DESCRIPTION
~~~~~~~~~~~

The Rigetti Quil compiler, ``quilc``, is an optimizing compiler for Quil. It
takes a general Quil program along with a qubit architecture, called an
ISA, and produces another Quil program that is executable on that
architecture. The compiler will also attempt to optimize the program by
producing fewer gates and shorter gate depths. The compiler may either
be run as a server which takes requests (as is used with PyQuil), or it
may be run as a batch program reading from standard input.

Server Mode runs the compiler as an HTTP server, taking simple POST
requests with JSON payloads which are known to the companion library
pyQuil.

OPTIONS
~~~~~~~

``-S, --server``
      (Server Mode) Run the compiler in Server Mode. This starts an HTTP server.

``-?, -h, --help``
      Show the help message.

``-v, --version``
      Show the version.

``--verbose``
      Print what the compiler is thinking. (Warning: It thinks a lot.)

``--isa <string>``
      Compile  for  the  qubit  architecture  defined  by  <string>,  which can be either 8Q, 20Q, 16QMUX, or a path to a QPU
      description file.

``-p, --protoquil``
      Prescribe that the input and output must be ProtoQuil, which is Quil that is comprised of gates and measurements,  with
      no control flow.

``--port <port>``
      (Server Mode) Run quilc in server mode on port <port>.

``-d, --compute-gate-depth``
      Print a calculated gate depth for the provided circuit as an appended Quil comment. (Requires -p.)

``-2, --compute-2Q-gate-depth``
      Print  a  calculated  multiqubit gate depth for the provided circuit as an appended Quil comment. (Requires -p. Ignores
      the blacklist and whitelist.)

``--compute-gate-volume``
      Print a calculated gate volume for the provided circuit as an appended Quil comment. (Requires -p.)

``-r, --compute-runtime``
      Print a calculated estimated runtime for the provided circuit as an appended Quil comment. (Requires -p.)

``-f, --compute-fidelity``
      Print a calculated estimated compiled circuit fidelity for the provided circuit as an appended Quil comment.  (Requires
      -p.)

``-u, --compute-unused-qubits``
      Print a list of unused qubits as an appended Quil comment. (Requires -p.)

``-t, --show-topological-overhead``
      Print  the  number of SWAP gates incurred for topological reasons for the provided circuit as an appended Quil comment.
      (Requires -p.)

``--gate-blacklist <gate-list>``
      When calculating statistics, ignore the gates present in the comma-separated list of names of <gate-list>.

``--gate-whitelist <gate-list>``
      When calculating statistics, consider only the gates present in the comma-separated list of names of <gate-list>.

``--time-limit <limit-ms>``
      (Server Mode) Limit the amount of time for a single request to approximately <limit-ms> milliseconds. By default,  this
      value is 0, which indicates an unlimited amount of time is allowed.

``--without-pretty-printing``
      Disable pretty printing of numerical quantities (e.g., multiples of pi) in compiled output.

``--prefer-gate-ladders``
      Use gate ladders, instead of the SWAP gate, to implement long-ranged gates, when possible.

``-j, --json-serialize``
      Serialize the output of compilation as a JSON object.

``-s, --print-logical-schedule``
      Include the logically parallelized schedule in JSON output. (Requires -p.)

``-m, --compute-matrix-reps``
      Print  the  matrix  representation  of  a compiled ProtoQuil program. Additionally, verify that this matrix matches the
      matrix representation of the input program. (Requires -p. Note that this is a very expensive operation.)

``--enable-state-prep-reductions``
      Perform program optimizations by assuming that the quantum state starts in the zero state.

EXAMPLES
~~~~~~~~

``quilc --isa "8Q" < file.quil``
      Compile a Quil file (printing the result to stdout) for an eight qubit ring.

SUPPORT
~~~~~~~

Contact <support@rigetti.com>.

COPYRIGHT
~~~~~~~~~

Copyright (c) 2018 Rigetti Computing

SEE ALSO
~~~~~~~~

:ref:`qvm(1) <qvm_man>`

0.13.0 (cl-quil: 0.19.0) [e9b41e3]                        24 September 2018                                                  QUILC(1)




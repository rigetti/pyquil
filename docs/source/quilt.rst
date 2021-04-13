.. _quilt:

Quil-T
======

Quil-T is an extension to Quil which introduces pulse-level control to
quantum programs. With Quil-T one can describe a program at a level
lower than is typically permitted in circuit-type programs, with
explicit control over the RF waveforms played by the QPU's control
hardware. In particular this imbues programs with a notion of *time*,
hence the `T` suffix.

The Quil compiler `quilc` was developed to support most users in their
pursuit for producing an optimal program from a high-level
language. In contrast Quil-T was developed to enable the low-level and
precise control desired by power-users. For example, for many users
the implementation details of a Hadamard gate are not particularly
important, and indeed the behind-the-scenes realisation of a Hadamard
gate are likely to change over time as gate implementations are
recalibrated to provide the best results. If you instead you are
interested in those details, and in particular you want to control
those details, then pulse-level control with Quil-T is the way to
go. With Quil-T you can define precisely what you mean by `H 0`, you
can perform experiments to characterize the underlying hardware such
as determining `T1`. The hardware is almost at your fingertips.

For examples, see the adjacent notebooks. For more information, see
the `Quil project homepage <https://github.com/rigetti/quil>`_.

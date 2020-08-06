# Quilt Overview and Proposal

## Introduction

Quil enables users to specify high-level, gate-based, timing-agnostic quantum programs. Only a subset of experiments can be specified this way (albeit a large subset), others require lower level control.

In particular there is a desire to:

- be able to define custom waveforms
- have full control over pulse timing and length
- be able to introduce more sophisticated translation for ideas eg. dynamic decoupling, crosstalk correction, gate decompositions, etc.
- define new gates (e.g. Toffoli and Fredkin) by composing pulses on frames and/or that exploit levels beyond the two level qubit approximation
- remove channels where discrepancies between internal and external performance can be introduced

This RFC proposes adding analog control to Quil which will be accessible through Quil language bindings and used by compilers and simulators.

## Language Proposal

Quilt extends Quil with the following new instructions

- `DEFCAL`
- `DEFFRAME`, `DEFWAVEFORM`
- `DELAY`
- `FENCE`
- `PULSE`
- `CAPTURE`, `RAW-CAPTURE`
- `SET-SCALE`
- `SET-FREQUENCY`, `SHIFT-FREQUENCY`
- `SET-PHASE`, `SHIFT-PHASE`, `SWAP-PHASES`

### Frames and Waveforms

Each qubit (or ordered list of qubits) can have multiple frames, denoted by string names such as "xy", "cz",
or "ro". For example, `0 "rf"` denotes the "rf" frame on qubit 0, whereas `2 3 "cz"` denotes the "cz" frame on qubits 
2 and 3. 

A frame is an abstraction that captures the instantaneous frequency, amplitude scaling, and
phase that will be mixed into the control signal. The frame frequencies are with
respect to the absolute "lab frame". 

Each frame has an associated definition, specified with a `DEFFRAME` block. As an example, the `2 3 "cz"` frame might
have a definition as below.

```
DEFFRAME 2 3 "cz":
    DIRECTION: "tx"
    INITIAL-FREQUENCY: 299306382.26093024
    SAMPLE-RATE: 1000000000.0
    ...
```

Each definition consists of a sequence of field-value pairs. The specific interpretation of these fields is hardware specific.

#### Waveforms

Quilt has two notions of _waveforms_. Custom waveforms are defined using
`DEFWAVEFORM` as a list of complex numbers which represent the desired waveform
envelope, along with a sample rate. Each complex number represents one sample of
the waveform. The exact duration (in seconds) of a waveform can be determined by dividing
the length of the waveform by the _sample rate_ of the frame, which is in units of samples 
per second.

As an example, a custom waveform representing a "linear ramp" might be defined as

```
DEFWAVEFORM linear_ramp:
    0.001, 0.002, 0.003, 0.004, ...
```

There are also some built-in waveform generators which take as a parameter the
duration of the waveform in seconds, alleviating the need to know the sample
rate to calculate duration. These are valid for use regardless of the frame's
underlying sample rate.


#### Frame State

In order to materialize the precise waveforms to be played the waveform
envelopes must be modulated by the frame's frequency, in addition to applying
some scaling and phasing factors. Although in theory it would be mostly possible
to simply define new waveforms that did the modulation, scaling, and phasing
manually, this is both tedious and doesn't take advantage of hardware which has
specialized support for tracking these things.

Therefore, each frame has associated with it a triple (frequency, phase, scale)
that can be modified throughout the program using SET-* instructions (and
additional instructions for phase).

Here's a table explaining the differences between these three values that are
tracked through the program:

| Name      | Initial Value                     | Valid Values          | Can be parameterized? |
|-----------|-----------------------------------|-----------------------|-----------------------|
| Frequency | INITIAL-FREQUENCY in DEFFRAME     | Real numbers          | Yes                   |
| Phase     | 0.0                               | Real numbers          | Yes                   |
| Scale     | 1.0                               | Real numbers          | Yes                   |


### Pulses

Now that frequency, phase, and scale on a frame have been established we can
play pulses. Pulses can be played by using the `PULSE` instruction and
specifying both the qubit frame as well as the waveform.

Given a waveform `my_custom_waveform` and the following program:
```
SET-FREQUENCY 0 "xy" 5400e6
SET-PHASE 0 "xy" pi/2
SET-SCALE 0 "xy" 1/2
PULSE 0 "xy" my_custom_waveform
```
A compiler would have several options depending on the hardware backend. It
could create a new waveform (eg. `my_custom_waveform_2`) and apply the
(frequency, phase, scale) to it. Or it could take advantage of built-in hardware
instructions to apply those values internally. This would be the responsibility
of the compiler to make a trade-off between number of instructions and number of
waveforms, given some hardware constraints.

### Readout

To readout a qubit the capture instruction is used. It takes a qubit frame, a
waveform, and a classical memory reference. In this case the waveform is used
as an integration kernel. The inner product of the integration kernel and the
list of measured IQ values is evaluated to produce a single complex result.

This complex number needs to be stored in Quil classical memory. Quil does not
currently support complex typed memory regions, so a real array of length 2 is used instead:
```
# Simple capture of an IQ point
DECLARE iq REAL[2]
CAPTURE 0 "ro" boxcar_kernel(duration: 1e-6) iq
```

### Timing

Analog control instructions introduce the concept of time into Quil. In this new interpretation, each instruction in
Quil has an associated execution duration (which may be effectively zero for
certain operations). It is up to the translator/hardware provider to provide semantics for how
pulse and capture operations are scheduled, as long as certain consistency requirements are met.
In general:

1. "Events" on a frame happen at a well defined time since eg. updating a
   frame frequency means that it starts to accumulate phase at a new rate.
2. Events happen in the order listed in the program.
3. Pulses on a common frame may not overlap in time.
4. Pulses on distinct frames which involve a common qubit may not overlap in
   time unless one is marked as `NONBLOCKING`.
   
A more precise specification of the timing semantics is discussed TODO

#### Pulse Operations

The duration of a pulse operation, i.e. `PULSE`, `CAPTURE`, or `RAW-CAPTURE`, is
the duration of the associated waveform.

Each frame is defined relative to a set of qubits. The execution of a pulse
operation on a frame blocks pulse operations on intersecting frames, i.e. frames
with a qubit in common with the pulse frame.

##### NONBLOCKING

In certain instances it may be desirable to support multiple concurrent pulses
on the same qubit, for example in measurements where `CAPTURE` performs a
readout which may overlap with a transmission `PULSE`. 

A pulse operation (`PULSE`, `CAPTURE`, and `RAW-CAPTURE`) with the `NONBLOCKING`
modifier does not exclude pulse operations on other frames. For example,
in

```
NONBLOCKING PULSE 0 "xy" flat(duration: 1.0, iq: 1.0)
NONBLOCKING PULSE 0 1 "ff" flat(duration: 1.0, iq: 1.0)
```

the two pulses could be emitted simultaneously. Nonetheless, a `NONBLOCKING`
pulse does still exclude the usage of the pulse frame, so e.g. `NONBLOCKING
PULSE 0 "xy" ... ; NONBLOCKING PULSE 0 "xy" ...` would require serial execution.

#### Delay

A `DELAY` instruction is equivalent to a `NONBLOCKING` no-operation on all
specified frames. For example, `DELAY 0 "xy" 1.0` delays frame `0 "xy"` by one
second.

If the `DELAY` instruction presents a list of qubits with no frame names, _all
frames on exactly these qubits are delayed_. Thus `DELAY 0 1.0` delays all one
qubit frames on qubit 0, but does not affect `0 1 "cz"`.

#### Fence

The `FENCE` instruction provides a means of synchronization of all frames
involving a set of qubits. In particular, it guarantees that all instructions
involving any of the fenced qubits preceding the `FENCE` are completed before
any instructions involving the fenced qubits which follow the `FENCE`. If `FENCE`
has no arguments, then it applies to all qubits on the device.

#### Frame Mutations

Single frame mutations (`SET-FREQUENCY`, `SHIFT-FREQUENCY`, `SET-PHASE`, `SHIFT-PHASE`,
`SET-SCALE`) have a hardware dependent duration (which may be effectively zero).
These operations block pulses on the targeted frame.

The `SWAP-PHASE` instruction introduces an implicit synchronization on the two
involved frames. In other words, any operations involving either of the swapped
frames and preceding the `SWAP-PHASE` must complete prior to the `SWAP-PHASE`
event.

### Calibrations

Calibrations can be associated with gates in Quil to aid the compiler in
converting a list of gates into the corresponding series of pulses.

Calibrations can be parameterized and include concrete values, which are
resolved in "Haskell-style", with later definitions being prioritized over
earlier ones. For example, given the following list of calibration definitions
in this order:
1. `DEFCAL RX(%theta) %qubit:`
2. `DEFCAL RX(%theta) 0:`
3. `DEFCAL RX(pi/2) 0:`
The instruction `RX(pi/2) 0` would match (3), the instruction `RX(pi) 0` would
match (2), and the instruction `RX(pi/2) 1` would match (1).

The body of a DEFCAL is a list of analog control instructions that ideally
enacts the corresponding gate.


## Hypothetical Examples

Here are some example calibrations for various types of gates and measurements.

Setting up frequencies:
```
SET-FREQUENCY 0 "xy" 4678266018.71412
SET-FREQUENCY 1 "xy" 3821271416.79618

SET-FREQUENCY 0 1 "cz" 137293415.829024

SET-FREQUENCY 0 "ro" 5901586914.0625
SET-FREQUENCY 1 "ro" 5721752929.6875

SET-FREQUENCY 0 "out" 5901586914.0625
SET-FREQUENCY 1 "out" 5721752929.6875
```

Calibrations of RX:
```
DEFCAL RX(%theta) 0:
    SET-SCALE %theta/pi*0.936
    PULSE 0 "xy" draggaussian(duration: 80e-9, fwhm: 40e-9, t0: 40e-9, anh: -210e6, alpha: 0)

DEFCAL RX(pi/2) 0:
    SET-SCALE 0.468
    PULSE 0 "xy" draggaussian(duration: 80e-9, fwhm: 40e-9, t0: 40e-9, anh: -210e6, alpha: 0)

# With crosstalk mitigation - no pulses on neighbors
DEFCAL RX(pi/2) 0:
    FENCE 0 1 7
    PULSE 0 "xy" draggaussian(duration: 80e-9, fwhm: 40e-9, t0: 40e-9, anh: -210e6, alpha: 0)
    FENCE 0 1 7
```

RZ:
```
DEFCAL RZ(%theta) %qubit:
    # RZ of +theta corresponds to a frame change of -theta
    SHIFT-PHASE %qubit "xy" -%theta
```

Calibrations of CZ:
```
DEFCAL CZ 0 1:
    PULSE 0 1 "cz" erfsquare(duration: 340e-9, risetime: 20e-9, padleft: 8e-9, padright: 8e-9)
    SHIFT-PHASE 0 "xy" 0.00181362669
    SHIFT-PHASE 1 "xy" 3.44695296672

# With no parallel 2Q gates
DEFCAL CZ 0 1:
    FENCE
    PULSE 0 1 "cz" erfsquare(duration: 340e-9, risetime: 20e-9, padleft: 8e-9, padright: 8e-9)
    SHIFT-PHASE 0 "xy" 0.00181362669
    SHIFT-PHASE 1 "xy" 3.44695296672
    FENCE
```

Readout:
```
DEFCAL MEASURE 0 dest:
    DECLARE iq REAL[2]
    PULSE 0 "ro" flat(duration: 1.2e-6, iq: ???)
    CAPTURE 0 "out" flat(duration: 1.2e-6, iq: ???) iq
    LT %dest iq[0] ??? # thresholding
```

Toffoli gate:
```
SET-FREQUENCY 12 13 "cz" 283.5e6
SET-FREQUENCY 13 14 "iswap" 181e6

DEFCAL CCNOT 12 13 14:
    # iSWAP_02 on 13-14
    FENCE 12 13 14
    PULSE 13 14 "iswap" erfsquare(tmax: 131e-9, risetime: 20e-9, padleft: 12e-9, pad_right: 13e-9)

    # CZ_20 or 12-13
    FENCE 12 13 14
    PULSE 12 13 "cz" erfsquare(tmax: 332e-9, risetime: 20e-9, padleft: 12e-9, pad_right: 12e-9)

    # iSWAP_02 on 13-14
    FENCE 12 13 14
    SHIFT-PHASE 13 14 "iswap" 0.5 # iSWAP_phase from the original code snippet
    PULSE 13 14 "iswap" erfsquare(tmax: 131e-9, risetime: 20e-9, padleft: 12e-9, pad_right: 13e-9)

    FENCE 12 13 14
```

Single point of a parametric gate chevron:
(parameterized in amplitude, frequency, and time)
```
RX(pi) 0
RX(pi) 1
SET-FREQUENCY 0 "cz" 160e6
SET-SCALE 0 "cz" 0.45
PULSE 0 1 "cz" erfsquare(duration: 100e-9, risetime: 20e-9, padleft: 0, padright: 0)
```

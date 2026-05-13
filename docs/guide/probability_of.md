# Trajectory Probabilities

`clifft.probability_of()` returns the exact probability that
`clifft.sample()` would assign to each requested measurement record under a
compiled program. It is the deterministic-record counterpart to sampling:
the same trajectory model, evaluated analytically rather than by drawing
shots.

This is useful when:

- You want a deterministic check that a particular outcome has the
  probability you expect.
- You are cross-validating sampling against an analytical prediction
  without paying for enough shots to resolve a rare branch.
- You want a likelihood for a measurement record from a circuit that
  contains classical feedback.

## Warm-up: Bell State

The smallest example is a Bell measurement. Only the records `00` and `11`
are emitted, each with probability 0.5:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    M 0 1
""")

records = ["00", "01", "10", "11"]
ps = clifft.probability_of(program, records)

for record, probability in zip(records, ps):
    print(f"{record}: {probability:.1f}")
```

Output:

```text
00: 0.5
01: 0.0
10: 0.0
11: 0.5
```

Each record is interpreted in measurement order: position `i` is the
`i`-th entry `sample().measurements` would emit for that shot. A single
record string is also accepted and returns a length-one array.

## Joint Probabilities Under Feedback

`probability_of()` handles classical feedback (`CX rec[-1] ...` and
related conditional gates). For trajectories where later operations
depend on earlier outcomes, it returns the exact joint probability of
the full record:

```python
import clifft

program = clifft.compile("""
    H 0
    M 0
    CX rec[-1] 1
    M 1
""")

records = ["00", "01", "10", "11"]
ps = clifft.probability_of(program, records)

for record, probability in zip(records, ps):
    print(f"{record}: {probability:.2f}")
```

Output:

```text
00: 0.50
01: 0.00
10: 0.00
11: 0.50
```

Qubit 1 is flipped exactly when the first measurement returned 1, so
records `01` and `10` are unreachable.

## Cross-Check Against Sampling

For circuits ending in `M ...`, `sample()` produces an empirical
distribution and `probability_of()` produces the exact distribution they
should converge to. The two should agree up to the usual binomial
sampling error:

```python
import clifft
import numpy as np

program = clifft.compile("H 0\nT 0\nH 0\nM 0")

probs = clifft.probability_of(program, ["0", "1"])

shots = 200_000
samples = clifft.sample(program, shots=shots, seed=42).measurements
freq_1 = float(samples.sum()) / shots
freq_0 = 1.0 - freq_1

print(f"exact: {probs[0]:.4f} {probs[1]:.4f}")
print(f"freq:  {freq_0:.4f} {freq_1:.4f}")
```

Probabilities here are around 0.85 and 0.15 (the
$(2 \pm \sqrt{2})/4$ outcomes of an $HTH$ rotation), and 200,000 shots
resolve them to within a few thousandths.

## Programmatic Record Arrays

For generated queries, pass a 2D `bool` or `uint8` array with one row
per record and one column per measurement slot. `program.num_measurements`
tells you the required column count:

```python
import clifft
import numpy as np

program = clifft.compile("H 0\nCX 0 1\nM 0 1")

records = np.array(
    [
        [0, 0],
        [1, 1],
    ],
    dtype=np.uint8,
)

ps = clifft.probability_of(program, records)
print(ps)
```

Output:

```text
[0.5 0.5]
```

Column order matches the order in which measurements appear in the
compiled program. If you measure qubits out of declared order, the
columns follow the order you wrote them.

## Log Probabilities for Deep Circuits

For circuits with many measurements, joint probabilities can underflow
float64. Pass `return_log=True` to get natural-log values instead:

```python
import clifft
import numpy as np

program = clifft.compile("H 0\nM 0")

log_ps = clifft.probability_of(program, ["0", "1"], return_log=True)
print(log_ps)
```

Output:

```text
[-0.69314718 -0.69314718]
```

Unreachable records are reported as `0.0` in linear output and `-inf`
in log output.

## Limitations

`probability_of()` requires that the program contain at least one
measurement and evolve a pure state up to feedback. It rejects programs
that include noise channels, readout noise, detectors, observables, or
post-selection -- the trajectory model is single-valued and these
constructs make it multi-valued or signal-conditioned. Use
[`sample()`](simulation.md#sampling) for those circuits.

Programs with hidden measurement slots (from `R` / reset gates lowered
to measure-then-feedback) are also rejected. Recompile without resets,
or use `sample()` to marginalize over the hidden outcomes.

For unitary programs ending in `M` on every qubit, the linear-output
result of `probability_of()` equals `probabilities()` on the
corresponding bitstrings. `probabilities()` is the right entry point
for that case because it does not run the measurement; it queries the
final statevector directly. See
[Strong Simulation](strong-simulation.md) for that workflow.

## How It Works

Internally, `probability_of()` rewrites each sampling measurement
opcode to a forced-outcome sibling and runs the program once per
record. The forced kernels replace the PRNG draw with the
user-supplied outcome and accumulate the log-probability of that
choice into a running scalar, using the same dust-clamping convention
as the sampler. The original `CompiledModule` is not mutated; the
rewrite runs on a private shallow copy.

For the algorithmic underpinnings shared with `probabilities()`, see
[Basis-State Probabilities](../theory/probabilities.md).

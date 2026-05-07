# Strong Simulation with Exact Probabilities

Strong simulation asks for exact output probabilities, not sampled shots.
For unitary circuits, `clifft.probabilities()` computes exact probabilities
for selected full-register bitstrings without expanding the full statevector.

This is useful when:

- You care about a sparse set of output bitstrings.
- The circuit is wider than is convenient for dense statevector extraction.
- The active rank stays small, so exact sparse queries are cheap.
- You want deterministic checks for compiled circuits.

## Warm-up: Bell State

The smallest example is a Bell state. The circuit has four possible two-bit
outputs, but only `00` and `11` have nonzero probability:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
""")

bitstrings = ["00", "01", "10", "11"]
ps = clifft.probabilities(program, bitstrings)

for bitstring, probability in zip(bitstrings, ps):
    print(f"{bitstring}: {probability:.1f}")
```

Output:

```text
00: 0.5
01: 0.0
10: 0.0
11: 0.5
```

By default, `bit_order="big"` maps the first bitstring character to qubit 0.
Each queried bitstring must include every qubit in the program.

## Sparse Queries on a Wider Circuit

For small circuits, `get_statevector()` is often the simplest way to inspect a
state. It returns all $2^n$ amplitudes, so it is not the right interface when
you only need a few exact probabilities from a wider output space.

The following circuit creates a small rare branch on qubit 0, then fans that
branch out across the rest of the register:

```python
import clifft

n = 12

lines = [
    "H 0",
    "R_Z(0.02) 0",
    "H 0",
]
lines.extend(f"CX 0 {q}" for q in range(1, n))

program = clifft.compile("\n".join(lines))

bitstrings = [
    "0" * n,
    "1" * n,
    "1" + "0" * (n - 1),
    "0" * (n - 1) + "1",
]

ps = clifft.probabilities(program, bitstrings)

print(f"qubits: {program.num_qubits}")
print(f"peak active rank: {program.peak_rank}")
for bitstring, probability in zip(bitstrings, ps):
    print(f"{bitstring}: {probability:.12g}")
```

Output:

```text
qubits: 12
peak active rank: 1
000000000000: 0.999013364214
111111111111: 0.000986635785864
100000000000: 0
000000000001: 0
```

The full output space has 4096 bitstrings, but the query asks for only four.
The non-Clifford work is localized to one active qubit, so the exact query
scales with the active rank rather than by constructing a 4096-entry
statevector. The same pattern is useful at larger widths when the active rank
remains small and the set of target outputs stays sparse.

## Batch Related Queries

Pass all target bitstrings in one call:

<!--pytest-codeblocks:cont-->

```python
ps = clifft.probabilities(program, bitstrings)
```

Clifft shares circuit-dependent stabilizer work across the batch. Repeated
single-bitstring calls are correct, but they rebuild work that one batched call
can reuse.

## Programmatic Bitstring Arrays

String bitstrings are convenient in examples and tests. For generated queries,
pass a 2D NumPy array with one row per bitstring and one column per qubit:

<!--pytest-codeblocks:cont-->

```python
import numpy as np

query_bits = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

ps = clifft.probabilities(program, query_bits)
print(ps)
```

Output:

```text
[0.99901336 0.00098664 0.        ]
```

`bool` arrays are also accepted. As with strings, the default
`bit_order="big"` maps the first column to qubit 0; use
`bit_order="little"` if the last column should map to qubit 0.

## Compare with Sampling

Sampling estimates probabilities from repeated shots. Exact probability queries
return deterministic values:

<!--pytest-codeblocks:cont-->

```python
measurement = "M " + " ".join(str(q) for q in range(n))
sample_program = clifft.compile("\n".join([*lines, measurement]))

samples = clifft.sample(sample_program, shots=1000, seed=5)
rare_hits = samples.measurements.all(axis=1).sum()
print(rare_hits)
```

For a branch with probability around $10^{-3}$, a 1000-shot sample may see the
rare outcome zero, one, or a few times. `probabilities()` reports the exact
value directly.

## Limitations

`probabilities()` applies to unitary programs. It rejects programs containing
measurements, feedback, noise, readout noise, detectors, observables, or
post-selection. Use `sample()` or `sample_survivors()` for mixed-circuit
workflows.

If you intentionally want to query the unitary skeleton of a mixed circuit,
compile with [`DropNonUnitaryPass`](../reference/passes.md). That changes the
circuit semantics by dropping non-unitary operations; it is not equivalent to
sampling or marginalizing the original circuit.

## How It Works

Clifft combines the active state vector with the final Clifford frame. For each
queried physical bitstring, it sums over the active basis states and evaluates
the remaining Clifford matrix elements as stabilizer amplitudes. The result is
an exact full-register probability without expanding the entire physical
statevector.

See [Basis-State Probabilities](../theory/probabilities.md) for the algorithm.

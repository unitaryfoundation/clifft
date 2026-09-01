# Strong Simulation: Exact Probabilities

Clifft provides two exact-probability APIs covering complementary regimes:

- **`clifft.basis_probabilities(unitary_program, bitstrings)`** — exact
  computational-basis probabilities for a unitary program.
- **`clifft.record_probabilities(measured_program, records)`** — exact
  joint probabilities of measurement records for a unitary circuit that
  contains measurements (with or without classical feedback). Noise is
  not supported.

You can convert the former problem into the latter by adding explicit
terminal measurements to a unitary program, and the two distributions match
on the bitstrings the records encode. The runtime cost can differ
substantially — see
[Performance on overlapping circuits](#performance-on-overlapping-circuits).

## When to use which

| Your circuit… | Use |
|---|---|
| has post-selection | neither exact API - use `sample_survivors()` |
| has noise, detectors, or observables | neither exact API - use `sample()` |
| has no measurements or other non-unitary operations | `basis_probabilities()` |
| has measurements but no noise, detectors, observables, or post-selection | `record_probabilities()` |
| has classical feedback but no other unsupported construct | `record_probabilities()` |

## Inspect a small statevector

For debugging and validation of a small pure-unitary circuit,
`get_statevector()` returns the final dense statevector:

```python
import clifft

program = clifft.compile("H 0\nCNOT 0 1")
state = clifft.get_statevector(program)

print(state)
```

The result contains all $2^n$ amplitudes, is normalized, and is defined only up
to global phase. Relative amplitudes and phases are preserved. Dense expansion
is limited to 10 qubits; use `basis_probabilities()` when a larger circuit only
needs probabilities for selected output bitstrings.

## `basis_probabilities()`: probabilities of a unitary state

The smallest example is a Bell state. The circuit has four possible two-bit
outputs, but only `00` and `11` have nonzero probability:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
""")

bitstrings = ["00", "01", "10", "11"]
ps = clifft.basis_probabilities(program, bitstrings)

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

### Sparse queries on a wider circuit

For small circuits, `get_statevector()` is often the simplest way to inspect
a state. It returns all $2^n$ amplitudes, so it is not the right interface
when you only need a few exact probabilities from a wider output space.

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

ps = clifft.basis_probabilities(program, bitstrings)

print(f"qubits: {program.num_qubits}")
print(f"peak active width: {program.peak_active_width}")
for bitstring, probability in zip(bitstrings, ps):
    print(f"{bitstring}: {probability:.12g}")
```

Output:

```text
qubits: 12
peak active width: 1
000000000000: 0.999013364214
111111111111: 0.000986635785864
100000000000: 0
000000000001: 0
```

The full output space has 4096 bitstrings, but the query asks for only four.
The non-Clifford work is localized to one active qubit, so the exact query
scales with the active-state dimension rather than by constructing a
4096-entry statevector. The same pattern is useful on larger circuits when the
active width remains small and the set of target outputs stays sparse.

### Batch related queries

Pass all target bitstrings in one call:

<!--pytest-codeblocks:cont-->

```python
ps = clifft.basis_probabilities(program, bitstrings)
```

Clifft shares circuit-dependent stabilizer work across the batch. Repeated
single-bitstring calls are correct, but they rebuild work that one batched
call can reuse.

### Programmatic bitstring arrays

String bitstrings are convenient in examples and tests. For generated
queries, pass a 2D NumPy array with one row per bitstring and one column per
qubit:

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

ps = clifft.basis_probabilities(program, query_bits)
print(ps)
```

Output:

```text
[0.99901336 0.00098664 0.        ]
```

`bool` arrays are also accepted. As with strings, the default
`bit_order="big"` maps the first column to qubit 0; use
`bit_order="little"` if the last column should map to qubit 0.

## `record_probabilities()`: probabilities of measurement records

`record_probabilities()` is the analytical counterpart of `sample()`: same
trajectory model, but exact rather than estimated. The smallest example
again is a Bell measurement, this time with the measurement included:

```python
import clifft

program = clifft.compile("""
    H 0
    CNOT 0 1
    M 0 1
""")

records = ["00", "01", "10", "11"]
ps = clifft.record_probabilities(program, records)

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

Each record is interpreted in measurement order: position `i` is the `i`-th
entry `sample().measurements` would emit for that shot. A single record
string is also accepted and returns a length-one array.

### Joint probabilities under feedback

`record_probabilities()` handles classical feedback (`CX rec[-1] ...` and
related conditional gates). For trajectories where later operations depend
on earlier outcomes, it returns the exact joint probability of the full
record:

```python
import clifft

program = clifft.compile("""
    H 0
    M 0
    CX rec[-1] 1
    M 1
""")

records = ["00", "01", "10", "11"]
ps = clifft.record_probabilities(program, records)

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

### Cross-check against sampling

For circuits ending in `M ...`, `sample()` produces an empirical
distribution and `record_probabilities()` produces the exact distribution
they should converge to. The two should agree up to the usual binomial
sampling error:

```python
import clifft

program = clifft.compile("H 0\nT 0\nH 0\nM 0")

probs = clifft.record_probabilities(program, ["0", "1"])

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

### Log probabilities for deep circuits

For circuits with many measurements, joint probabilities can underflow
float64. Pass `return_log=True` to get natural-log values instead:

```python
import clifft

program = clifft.compile("H 0\nM 0")

log_ps = clifft.record_probabilities(program, ["0", "1"], return_log=True)
print(log_ps)
```

Output:

```text
[-0.69314718 -0.69314718]
```

Unreachable records are reported as `0.0` in linear output and `-inf` in
log output.

## Performance on overlapping circuits

When the circuit is a unitary prefix followed by terminal `M`-all, either
API mathematically applies. They use very different execution strategies,
though, and the difference can be 100×+ in either direction:

- `basis_probabilities()` evolves the program once, then walks
  active-state amplitudes per queried bitstring. The up-front cost is
  amortized across the batch.
- `record_probabilities()` forces the requested measurement outcomes and
  replays the executable plan once per record. No
  amortization, but the compiler's `StatevectorSqueezePass` can reorder
  measurements next to their non-Clifford gates to reduce active width early.

A practical way to choose is to compile both forms and compare their
`program.peak_active_width` values. When the *measured* form has a noticeably
lower peak active width than the unitary form, `record_probabilities()` is
typically faster per query. When the two widths match, `basis_probabilities()` wins
because it amortizes plan execution across queries.

```python
import clifft

# A circuit with a final H sandwich on every qubit: the squeeze pass can
# pair each terminal H-T-H with an M and free that qubit's active dim early.
circuit = """
H 0 1 2 3 4
CX 0 1
CX 2 3
CX 1 4
H 0 1 2 3 4
T 0 1 2 3 4
H 0 1 2 3 4
"""

unitary = clifft.compile(circuit)
measured = clifft.compile(circuit + "\nM 0 1 2 3 4")

print(f"basis_probabilities()  peak active width: {unitary.peak_active_width}")
print(f"record_probabilities() peak active width: {measured.peak_active_width}")
```

A gap in `peak_active_width` indicates roughly a
$2^{(k_{\text{unitary}} - k_{\text{measured}})}$ speedup ceiling for
`record_probabilities()` over `basis_probabilities()` on this circuit,
independent of batch size. At equal peak active width,
`basis_probabilities()` wins by roughly the plan-to-amplitude-walk
cost ratio for any moderately sized batch.

If neither performance regime dominates your workload, use the circuit
constraints in the table at the top: unitary programs use
`basis_probabilities()`, while eligible programs with measurements use
`record_probabilities()`.

## Limitations

Both APIs reject programs that include noise, readout noise, detectors,
observables, or post-selection. These constructs make the trajectory
multi-valued or signal-conditioned in a way neither API models. Use
[`sample()`](simulation.md#ordinary-sampling) or
[`sample_survivors()`](simulation.md) for those workflows.

`basis_probabilities()` additionally rejects programs that contain any
measurement, including terminal `M`-all. The rejection criterion is sharp:
any measurement breaks the single-pure-final-state assumption that the
amplitude-walk algorithm relies on. If you intentionally want the unitary
skeleton of a mixed circuit, compile with
[`DropNonUnitaryPass`](../reference/passes.md) — but note that this
changes the circuit's semantics; it is not equivalent to marginalizing the
original circuit.

`record_probabilities()` requires at least one measurement. Programs with
hidden measurement slots (from `R` / reset gates lowered to
measure-then-feedback) are also rejected. Recompile without resets, or use
`sample()` to marginalize over the hidden outcomes.

## How it works

`basis_probabilities()` combines the active state vector with the final
Clifford frame. For each queried physical bitstring, it sums over the
active basis states and evaluates the remaining Clifford matrix elements as
stabilizer amplitudes. The result is an exact full-register probability
without expanding the entire physical statevector. See
[Basis-State Probabilities](../theory/basis_probabilities.md) for the
algorithm.

`record_probabilities()` replays each requested measurement record through the
executable sampling plan. Replay replaces each measurement draw with the
requested outcome and accumulates its log-probability using the same
dust-clamping convention as ordinary sampling. The plan is immutable and can
be reused for subsequent samples or queries.

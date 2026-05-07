# Basis-State Probabilities

For unitary programs, `clifft.probabilities()` computes exact probabilities for
full-register computational-basis bitstrings without materializing the full
$2^n$ statevector. This page summarizes the algorithm behind that API. For
usage examples, see [Strong Simulation with Exact Probabilities](../guide/strong-simulation.md).

Clifft starts from the factored state

$$
|\psi\rangle = \gamma \, U_C \, P \, \Big( |\phi\rangle_A \otimes |0\rangle_D \Big),
$$

where $U_C$ is the Clifford frame, $P$ is the Pauli frame, and
$|\phi\rangle_A$ is the dense active state over $k$ active qubits. Writing

$$
|\phi\rangle_A = \sum_{i \in \{0,1\}^k} v_i |i\rangle_A,
\qquad
P = X^{p_x} Z^{p_z},
$$

the probability of a physical bitstring $x$ is

$$
\Pr[x] = \left| \gamma \sum_{i \in \{0,1\}^k} v_i \, (-1)^{\langle p_z[A], i\rangle} \, \langle x | U_C | y(i)\rangle \right|^2,
$$

where $y(i) = p_x \oplus (i, 0_D)$. The sum ranges over $2^k$ active basis
states, not over all $2^n$ physical basis states. The remaining task is to
evaluate the Clifford matrix element $\langle x | U_C | y\rangle$.

## $\langle x | U_C | y\rangle$ as a stabilizer-state amplitude

$U_C |y\rangle$ is a Clifford applied to a computational basis state,
which is a stabilizer state. The amplitude of a basis state $|x\rangle$
in a stabilizer state has a direct combinatorial form.

Clifft pushes $U_C$ onto the bra:

$$
\langle x | U_C | y\rangle = \langle U_C^\dagger \, x | y\rangle .
$$

The state $|U_C^\dagger \, x\rangle$ is stabilized by

$$
\big\{(-1)^{x_q} \, U_C^\dagger \, Z_q \, U_C\big\}_{q=0}^{n-1}.
$$

The Heisenberg image $U_C^\dagger Z_q U_C$ is the $q$-th Z-row of the
inverse tableau stored in `program.constant_pool.final_tableau`. The signs
$(-1)^{x_q}$ come from the queried bitstring.

## Stabilizer amplitude structure

Gaussian elimination on the X-block of the stabilizer generator matrix
separates the rows into two groups:

**X-pivoted rows** ($r_X$ of them). Each has a unique X-pivot column.
These are the free directions of the basis-state support. Applying subsets of
their X-masks $x_1, \ldots, x_{r_X}$ generates the affine subspace.

**Pure-Z rows** ($r_Z = n - r_X$ of them). These rows have no X-support and
act diagonally in the computational basis. Their pivot columns determine a
base state $b \in \{0,1\}^n$.

The stabilizer state is then a uniform superposition over an affine
subspace:

$$
|\Psi\rangle =
\frac{1}{\sqrt{2^{r_X}}}
\sum_{c \in \{0,1\}^{r_X}}
\omega(c) \,
\Big| b \oplus \bigoplus_i c_i \, x_i \Big\rangle .
$$

The phase $\omega(c)$ is accumulated from the generator signs, Y terms, and
Z-mask action while walking from $b$ to the target basis state.

## Evaluating one amplitude

For a target basis state $|y\rangle$, the amplitude $\langle y | \Psi\rangle$
is zero unless $y \oplus b$ lies in the span of the X-pivot masks. If it does,
there is a unique $c$ such that

$$
b \oplus \bigoplus_i c_i x_i = y,
$$

and the amplitude is

$$
\frac{1}{\sqrt{2^{r_X}}} \, \omega(c).
$$

Clifft finds $c$ with a forward sweep over the X-pivots. It starts with the
residual $r = y \oplus b$, clears pivot bits by XORing the corresponding
X-mask, and returns zero if any residual support remains.

## Why share the elimination across batched queries

The pivot structure, X-masks, Y-counts, and eliminated Z-masks depend only on
the inverse-tableau rows $U_C^\dagger Z_q U_C$. That is, they depend on the
circuit, not on the queried bitstring. The signs $(-1)^{x_q}$ are the only
piece that changes per query.

`make_stabilizer_amplitude_structure` decomposes each row sign as

$$
\text{sign}_i = \text{static\_sign}_i \;\oplus\; \langle \text{sign\_mask}_i, \, x\rangle ,
$$

so static and dynamic parts are tracked separately through the
elimination. Static parts XOR under row multiplication (handled by Stim's
`PauliString::operator*=`); dynamic sign-masks XOR linearly. After
elimination, `StabilizerAmplitudeStructure` captures the circuit-dependent
work once.

For each queried bitstring, `bind(x)` recomputes only the per-row signs and
the base state $b$ in $O(n)$ work. Then `amplitude(y)` evaluates an amplitude
in $O(r_X \cdot \lceil n/64 \rceil)$ word-bit operations.

## Complexity

Per `clifft.probabilities()` call, with $M$ queried bitstrings, $n$
qubits, and active rank $k$:

| Step | Cost | Frequency |
|------|------|-----------|
| `execute(program, state)` | full bytecode pass | once |
| `final_tableau.inverse()` | $O(n^2)$ | once |
| `make_stabilizer_amplitude_structure` | $O(n^3 / W)$, $W = 64$ | once |
| `bind(x)` | $O(n)$ | per bitstring |
| `amplitude(y)` (called $2^k$ times per bitstring) | $O(r_X \cdot n / W)$ | per amplitude |

Total: $O(\text{compile-once terms}) + M \cdot 2^k \cdot O(r_X \cdot n / W)$.

The exponential cost is in $k$, not $n$. The same scaling principle that makes
the SVM efficient on near-Clifford circuits applies to probability queries. For
pure-Clifford circuits ($k = 0$), the inner loop runs once per bitstring; each
query costs $O(r_X \cdot n / W)$.

## When to use this versus dense statevector

For very small circuits ($n \lesssim 10$),
[`clifft.get_statevector()`](../guide/simulation.md) returns the full
$2^n$-amplitude vector and squaring its absolute value is the fastest
path to a probability table. `probabilities()` shines when:

- $n$ is large enough that materializing the full $2^n$ statevector is
  impractical, but you only care about a sparse set of bitstrings.
- The circuit's active rank $k$ is small (so the $2^k$ inner loop is
  cheap).
- You want to query many bitstrings against the same circuit (the
  structure is shared across the batch).

For mixed circuits, including measurements, noise, and observables,
`probabilities()` is not applicable, since the state is no longer a single pure
vector. Use [sampling](../guide/simulation.md#sampling) for those workflows, or
[`DropNonUnitaryPass`](../reference/passes.md) if you intentionally want
to query the unitary skeleton of a mixed circuit.

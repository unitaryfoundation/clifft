# Basis-State Probabilities: Algorithm Derivation

The [theory overview](overview.md#exact-basis-state-probabilities) shows
that for a unitary program, the probability of observing the physical
bitstring $x$ reduces to

$$
\Pr[x] = \left| \gamma \sum_{i \in \{0,1\}^k} v_i \, (-1)^{\langle p_z[A], i\rangle} \, \langle x | U_C | y(i)\rangle \right|^2,
$$

where $y(i) = p_x \oplus (i, 0_D)$. Everything outside the sum is known
from the factored state. The whole problem reduces to evaluating the
Clifford matrix element $\langle x | U_C | y\rangle$ for $2^k$ different
$y$, plus one $x$.

This page derives the algorithm Clifft uses to evaluate that matrix
element and explains why each step is the right tool.

## $\langle x | U_C | y\rangle$ as a stabilizer-state amplitude

$U_C |y\rangle$ is a Clifford applied to a computational basis state,
which is a **stabilizer state**. The amplitude of a basis state $|x\rangle$
in a stabilizer state is not a simulation problem — it has a closed-form
combinatorial structure that can be evaluated directly.

It is more convenient to push $U_C$ onto the bra:

$$
\langle x | U_C | y\rangle = \langle U_C^\dagger \, x | y\rangle .
$$

The state $|U_C^\dagger \, x\rangle$ is stabilized by

$$
\big\{(-1)^{x_q} \, U_C^\dagger \, Z_q \, U_C\big\}_{q=0}^{n-1}.
$$

The compiler hands these to us for free: the Heisenberg image
$U_C^\dagger Z_q U_C$ is exactly the $q$-th Z-row of the inverse tableau
that the back end stores in `program.constant_pool.final_tableau`. The
signs $(-1)^{x_q}$ are read off the queried bitstring $x$.

So the question becomes: **what is the amplitude of $|y\rangle$ in the
stabilizer state defined by these $n$ Pauli generators (with signs
decorated by $x$)?**

## The structure of a stabilizer state's amplitudes

A stabilizer state on $n$ qubits is the unique simultaneous $+1$ eigenstate
of $n$ commuting Pauli generators. Put the generator matrix into reduced
row echelon form by Gaussian-eliminating the X-block. Two kinds of rows
result.

**X-pivoted rows** ($r_X$ of them). Each has a unique X-pivot column.
These are the "free directions": applying any subset of them to a base
state generates the basis-state superposition. Call their X-masks
$x_1, \ldots, x_{r_X}$.

**Pure-Z rows** ($r_Z = n - r_X$ of them). After elimination these have
no X-support and act diagonally in the computational basis. They pin down
a unique base state $b \in \{0,1\}^n$: at each Z-pivot column, $b$'s bit
is fixed by the row's sign. (A Z-only stabilizer with sign $+1$ at column
$c$ forces $b_c = 0$; with sign $-1$, $b_c = 1$. After reduced row
echelon, each Z-pivot's row depends only on its own pivot bit, so $b$ is
uniquely determined.)

The stabilizer state is then a uniform superposition over an affine
subspace:

$$
|\Psi\rangle =
\frac{1}{\sqrt{2^{r_X}}}
\sum_{c \in \{0,1\}^{r_X}}
\omega(c) \,
\Big| b \oplus \bigoplus_i c_i \, x_i \Big\rangle .
$$

The phase $\omega(c)$ is the product of phases picked up when applying
$S_1^{c_1} S_2^{c_2} \cdots S_{r_X}^{c_{r_X}}$ in order to $|b\rangle$.
Each $S_i$ contributes a factor of $\pm 1$ from its sign, $i$ for each
Y in its support, and $\pm 1$ from
$(-1)^{\langle Z\text{-mask}_i, \, \text{current}\rangle}$ where "current"
is the basis state just before $S_i$ is applied. Because the $S_i$
commute, the resulting $\omega(c)$ is independent of the order in which
we apply them — but we must update "current" between steps to keep the
Z-action factor correct.

## Evaluating one amplitude

For a target basis state $|y\rangle$, the amplitude $\langle y | \Psi\rangle$
is determined by whether $y$ lies in the support of $|\Psi\rangle$:

- If $y \oplus b$ is **not** in the linear span of
  $\{x_1, \ldots, x_{r_X}\}$ over $\mathbb{F}_2$: the amplitude is $0$.
  No combination of free generators reaches $y$ from $b$.
- If it is: there is a unique $c \in \{0,1\}^{r_X}$ such that
  $b \oplus \bigoplus_i c_i x_i = y$, and the amplitude is
  $\frac{1}{\sqrt{2^{r_X}}} \, \omega(c)$.

Finding $c$ is a single forward sweep over the X-pivots in pivot-column
order: walk the residual $r = y \oplus b$, and at each pivot $i$, if
$r_{\text{pivot}_i}$ is set, set $c_i = 1$ and XOR $x_i$ into $r$. Because
the elimination is reduced row echelon, $x_i$'s only set pivot column is
$\text{pivot}_i$ itself, so this clears the bit without disturbing later
pivots. If $r$ is zero at the end, $c$ is found; otherwise the amplitude
is zero.

This is the inner loop in
[`BoundStabilizerAmplitudeQuery::amplitude`](https://github.com/unitaryfoundation/clifft/blob/main/src/clifft/api/probabilities.cc).

## Why share the elimination across batched queries

The pivot structure — which columns are X-pivots, which are Z-pivots, the
values of the $x_i$ masks, the Y-counts of each generator, and the
eliminated-row Z-masks — depends only on the inverse-tableau rows
$U_C^\dagger Z_q U_C$. That is, only on the **circuit**. The signs
$(-1)^{x_q}$ are the only piece that depends on the queried bitstring
$x$, and they enter only through the row signs (and downstream, through
$b$ and $\omega$).

`make_stabilizer_amplitude_structure` decomposes each row sign as

$$
\text{sign}_i = \text{static\_sign}_i \;\oplus\; \langle \text{sign\_mask}_i, \, x\rangle ,
$$

so static and dynamic parts are tracked separately through the
elimination. Static parts XOR under row multiplication (handled by Stim's
`PauliString::operator*=`); dynamic sign-masks XOR linearly. After
elimination we have a `StabilizerAmplitudeStructure` that captures
everything circuit-dependent in one pass.

For each queried bitstring, `bind(x)` recomputes only the per-row signs
and the base state $b$ — $O(n)$ work — then `amplitude(y)` evaluates an
amplitude in $O(r_X \cdot \lceil n/64 \rceil)$ word-bit operations.
Identity-row contradictions ($I = -I$, theoretically unreachable for
valid Cliffords) are checked at bind time.

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

The exponential cost is in $k$, not $n$ — the same scaling principle
that makes the SVM efficient on near-Clifford circuits applies to
probability queries. For pure-Clifford circuits ($k = 0$) the inner loop
runs once per bitstring; each query costs $O(r_X \cdot n / W)$.

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

For mixed circuits — measurements, noise, observables — `probabilities()`
is not applicable, since the state is no longer a single pure vector. Use
[sampling](../guide/simulation.md#sampling) for those workflows, or
[`DropNonUnitaryPass`](../reference/passes.md) if you intentionally want
to query the unitary skeleton of a mixed circuit.

# Tableau Conventions

Clifft uses Clifford tableaus during compilation to rewind Pauli observables.
This page documents the representation and composition conventions used
throughout the compiler.

## Pauli representation

An `n`-qubit Hermitian Pauli is represented by two `n`-bit masks `x` and `z`
and a sign bit `s`:

\[
P = (-1)^s i^{|x \mathbin{\&} z|} X^x Z^z.
\]

For one qubit, `(x, z)` therefore maps `00 -> I`, `10 -> X`, `01 -> Z`, and
`11 -> Y`. The explicit sign is the real `+1` or `-1` multiplying the
Hermitian Pauli. Implementations may use a phase modulo four while multiplying
Paulis, but values stored in HIR masks and tableau generator rows are
Hermitian.

Bit `q` always refers to physical qubit `q`. In a dense statevector index,
qubit 0 is the least-significant bit. Public `basis_probabilities()` bit-string
and array inputs instead use the requested `bit_order`; by default, the first
character or column maps to qubit 0. Masks are stored in 64-bit words, with
qubit `q` at bit `q % 64` of word `q / 64`. Unused high bits in the final word
must be zero.

## Tableau rows

A forward tableau for a Clifford unitary `U` stores the images of the Pauli
generators:

\[
X_q \mapsto U X_q U^\dagger, \qquad
Z_q \mapsto U Z_q U^\dagger.
\]

The `xs[q]` row is the image of `X_q`; `zs[q]` is the image of `Z_q`. The
image of `Y_q` is obtained by multiplying those two rows with the phase needed
to preserve `Y_q = i X_q Z_q`.

Tableau application is a homomorphism on Paulis. For a Pauli product, apply
the tableau to each selected generator and multiply the resulting rows in
qubit order, accounting for anti-commutation phases.

## Composition and inversion

If tableau `a` represents `A` and tableau `b` represents `B`, `a.then(b)`
means that `A` is applied first and `B` second. The result represents `B A`:

\[
P \mapsto B(A P A^\dagger)B^\dagger.
\]

`inverse()` represents the inverse conjugation map. The following identities
must hold exactly for every generator row:

- `identity.then(a) == a` and `a.then(identity) == a`;
- `a.then(a.inverse()) == identity`;
- `a(P * Q) == a(P) * a(Q)` including phase;
- composition is associative.

## Frontend rewinding

The frontend processes gates in circuit order but holds the inverse tableau of
the Clifford prefix. If the processed prefix implements `U`, rewinding an
observable `P` returns

\[
U^\dagger P U.
\]

After appending a gate `G` to the circuit prefix, the new circuit unitary is
`G U`, so the inverse map must become

\[
P \mapsto U^\dagger(G^\dagger P G)U.
\]

This is why frontend gate updates prepend the inverse gate action to the
inverse tableau. The direction is significant: tracking `U P U^dagger`
instead often produces plausible masks with incorrect signs.

The same rewinding rule applies to T-like rotations, measurements, resets,
noise channels, classical feedback, expectation probes, and arbitrary Pauli
products. Ordinary sampling dispatch never performs tableau evolution.

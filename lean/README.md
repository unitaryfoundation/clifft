# Lean mathematical companion

This project proves mathematical rules used by Clifft. It supports development
and review of the handwritten C++ implementation: Lean checks the mathematical
argument, and reviewers check that the C++ follows the model. Existing C++ tests
and independent simulator comparisons remain necessary.

The current scope is single-qubit Pauli multiplication, including its exact
complex phase, and the interpretation of signed I, X, Y, and Z. It is the first
reviewable increment of [the formalization work](https://github.com/unitaryfoundation/clifft/issues/497).
There is no connection to Clifft's runtime, CMake build, or Python installation.

## Build and check

Install [Elan](https://github.com/leanprover/elan#installation) and make its
`lake` command available on `PATH`. From the repository root:

```sh
cd lean
export MATHLIB_NO_CACHE_ON_UPDATE=1
lake exe cache get \
  Mathlib.Basic.Complex.Basic \
  Mathlib.LinearAlgebra.Matrix.Notation \
  Mathlib.Tactic.FinCases \
  Mathlib.Tactic.NormNum
lake build
lake exe mk_all --lib ClifftProofs --check
```

`lean-toolchain` pins Lean 4.34.1. `lakefile.toml` pins Mathlib to the commit for
v4.34.1, and `lake-manifest.json` records every transitive dependency revision.
The first build downloads dependencies and the imported Mathlib artifacts;
`MATHLIB_NO_CACHE_ON_UPDATE` prevents an automatic download of all of Mathlib.
Routine proof checking does not require `lake update`; dependency upgrades should be
explicit, reviewed changes to these pins and the manifest.

The default build checks the proof library and `ClifftProofsAudit`. Warnings are
errors. The audit examines the transitive axiom dependencies of every declaration
in imported `ClifftProofs` modules, regardless of the declaration's namespace. It
allows only `propext`, `Classical.choice`, and `Quot.sound`, the standard logical
axioms used by Mathlib. It rejects `sorry`/`admit`, added axioms, and the extra
trust used by `native_decide`. These checks do not replace reviewing what the
definitions and theorem statements mean.

The import check ensures every proof file is included by `ClifftProofs.lean`, so
it participates in the build and audit. A separate
[GitHub Actions workflow](../.github/workflows/lean.yml) runs both checks.
No Lean installation is needed for ordinary Clifft development.

## First result and conventions

Read [ClifftProofs/Pauli.lean](ClifftProofs/Pauli.lean) in this order:

1. `xMatrix`, `yMatrix`, and `zMatrix` are the usual explicit two-by-two complex
   matrices in the ordered basis `|0>, |1>`.
2. `Pauli` stores Boolean `x` and `z` and a phase in `Fin 4`. Its semantics,
   `Pauli.denote`, is `i^phase X^x Z^z`. `HSMul.hSMul` is Lean's scalar
   multiplication of a complex number and a matrix; it is written out to keep
   the source ASCII-only.
3. `Pauli.mul` XORs the bodies and computes
   `phase_p + phase_q + 2 * (z_p AND x_q)` modulo four.
4. `Pauli.denote_mul` proves, for **every** pair of these Paulis, that
   `denote (mul p q) = denote p * denote q` as complex matrices. Equality
   includes global phase. In this product, `q` acts first on a column vector.
5. `Pauli.signed x z negative` uses phase
   `(x AND z) + 2 * negative`. The four `denote_signed_I/X/Y/Z` theorems identify
   the positive matrices, and `denote_signed_negative` proves that the sign bit
   negates the operator.

The representation has 16 elements because it includes phases `1, i, -1, -i`.
The signed constructor gives the eight Hermitian Paulis. Products need the
larger representation: `X * Y = i Z`. In particular, body `x = z = true` with
phase zero means `XZ = -i Y`; positive Y has phase one, and negative Y has phase
three.

The multiplication proof has two small pieces. `phaseFactor_add` checks that
addition modulo four agrees with multiplication of powers of `i`. `body_mul`
checks the Z/X crossing sign against ordinary matrix multiplication. Both use
exhaustive finite cases and kernel-checked arithmetic. `denote_mul` combines
them using matrix scalar-multiplication identities. No multiplication law or
Pauli identity is assumed as an axiom.

## Correspondence with C++

The conventions follow
[Tableau Conventions](../docs/development/tableau-conventions.md).
The implementation links below are **review obligations**, not verified
translations between Lean and C++.

| Formal definition or result | C++ counterpart and inspection required |
| --- | --- |
| `Pauli.x`, `.z`, `.phase` | [`PauliString` and `PauliStringView`](../src/clifft/tableau/pauli_string.h): for one qubit, the low mask bits are `x` and `z`, and `phase()` is the exponent modulo four. |
| `Pauli.signed` and `denote_signed_*` | [`y_phase`, `set_sign`, `sign`, and `is_hermitian`](../src/clifft/tableau/pauli_string.cc): with one qubit, `popcount(x & z)` is the Boolean conjunction. Adding two negates a Hermitian Pauli. `set_pauli` changes only the body; `from_text` establishes the phase by calling `set_sign` afterward. |
| `Pauli.mul` and `denote_mul` | [`PauliString::right_multiply`](../src/clifft/tableau/pauli_string.cc): the crossing uses the left Z and right X, the phase sum is masked with `3U`, and both bodies are XORed. This computes the operator product `P Q`. |
| Positive and negative Y, and multiplication phases | [`Native Pauli phase convention preserves Hermitian signs`](../tests/test_tableau.cc) remains an implementation regression test. |

This increment has no preconditions beyond the types: Boolean masks, a valid
phase modulo four, and exact complex arithmetic. It does not prove packed-word
operations, padding, memory safety, the correspondence of C++ executions to
Lean values, floating-point behavior, or any compiler pass. Arbitrary-qubit
tensor products, Clifford conjugation, symbolic records, noise, measurement,
and approximation policies remain outside the current formal model.

## Review and subsequent increments

For this increment, review the matrix definitions, the phase convention, the
statement of `denote_mul`, and the C++ correspondence table before the proof
tactics. The principal question is whether the statement captures the actual
mathematical contract used by Clifft.

Keep subsequent pull requests independently reviewable:

1. Generalize to arbitrary-qubit signed Pauli multiplication and commutation,
   with tensor-product matrix semantics and explicit qubit ordering.
2. Prove affine symbolic-frame signs, including classical-bit substitution and
   composition. Together with the Pauli algebra, this completes the initial
   foundation described in the issue.
3. Prove a representative noise-aware rotation reorder rule, stating its
   allowed noise and quantum/classical dependency conditions. Extend to
   selected active-state transitions and other rewrites separately.

Each increment should state its exact claim, assumptions, exclusions, proof
outline, and C++ mapping. Changes to a modeled semantic contract should update
the formalization in the same pull request. New proof modules must be imported
by `ClifftProofs.lean` and pass the default build and import check.

Certificates, translation validation, and generated implementation code are
deferred. A possible later bridge is a checker for compiler-exported Pauli and
symbolic-sign calculations, after their mathematical contracts are established.

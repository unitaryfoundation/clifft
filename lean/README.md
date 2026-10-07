# Lean mathematical companion

This project proves mathematical rules used by Clifft. It supports development
and review of the handwritten C++ implementation: Lean checks the mathematical
argument, and reviewers check that the C++ follows the model. Existing C++ tests
and independent simulator comparisons remain necessary.

The current scope is single-qubit Pauli multiplication, including its exact
complex phase, Hermitian classification, sign recovery, and faithful matrix
semantics. See [the formalization issue](https://github.com/unitaryfoundation/clifft/issues/497)
for the broader scope and subsequent work.
There is no connection to Clifft's runtime, CMake build, or Python installation.

## Build and check

Install [Elan](https://github.com/leanprover/elan#installation) and make its
`lake` command available on `PATH`. From the repository root:

```sh
cd lean
export MATHLIB_NO_CACHE_ON_UPDATE=1
bash cache.sh
lake build
lake exe mk_all --lib ClifftProofs --check
LEAN_NUM_THREADS=2 lake env leanchecker -v ClifftProofs ClifftProofsAudit
```

`lean-toolchain` pins Lean 4.34.1. `lakefile.toml` pins Mathlib to the commit for
v4.34.1, and `lake-manifest.json` records every transitive dependency revision.
The first build downloads dependencies and the imported Mathlib artifacts;
`MATHLIB_NO_CACHE_ON_UPDATE` prevents an automatic download of all of Mathlib.
`cache.sh` derives the module list from the proof sources' import lines. Keep
one module per import line; both `import` and `public import` are supported.
Routine proof checking does not require `lake update`; dependency upgrades should be
explicit, reviewed changes to these pins and the manifest.

The default build checks the proof library and `ClifftProofsAudit`. Warnings are
errors. The audit examines the transitive axiom dependencies of every declaration
in imported `ClifftProofs` modules, regardless of the declaration's namespace. It
allows only `propext`, `Classical.choice`, and `Quot.sound`, the standard logical
axioms used by Mathlib. It rejects `sorry`/`admit`, added axioms, and the extra
trust used by `native_decide`.

`leanchecker`, bundled with the pinned Lean toolchain, replays the compiled
project declarations through the kernel. This also catches ill-typed declarations
introduced by bypassing normal kernel checking, which an axiom audit alone does
not catch. It uses the same Lean kernel and trusts imported Mathlib artifacts;
it is not an independent verifier. These checks do not replace reviewing what
the definitions and theorem statements mean.

The import check ensures every proof file is included by `ClifftProofs.lean`, so
it participates in the build and audit. The reusable
[Lean workflow](../.github/workflows/lean.yml) runs all three checks on every CI
run, and `ci-gate` requires its success. It has no path filter that could leave
the required check missing on other pull requests.
No Lean installation is needed for ordinary Clifft development.

## Pauli model and results

The definitions and theorems are in [ClifftProofs/Pauli.lean](ClifftProofs/Pauli.lean):

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
6. `isHermitian_iff_conjTranspose` proves that the relative phase test used by
   C++ is equivalent to matrix Hermiticity. `hermitian_iff_signed` characterizes
   exactly those Paulis as outputs of `signed`, and `sign_signed` proves sign
   recovery. As in C++, `sign` is meaningful only for Hermitian Paulis.
7. `denote_injective` proves that two representations denote the same matrix
   only if their masks and phase agree.

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
| `Pauli.signed` and `denote_signed_*` | [`y_phase` and `set_sign`](../src/clifft/tableau/pauli_string.cc): with one qubit, `popcount(x & z)` is the Boolean conjunction. Adding two negates a Hermitian Pauli. `set_pauli` changes only the body; `from_text` establishes the phase by calling `set_sign` afterward. |
| `Pauli.sign`, `sign_signed`, and `isHermitian_iff_conjTranspose` | [`sign` and `is_hermitian`](../src/clifft/tableau/pauli_string.cc): `delta = (phase - y_phase) & 3U` gives a real factor `i^delta` exactly when it is 0 or 2; `delta == 2` recovers the negative sign for Hermitian inputs. |
| `Pauli.mul` and `denote_mul` | [`PauliString::right_multiply`](../src/clifft/tableau/pauli_string.cc): the crossing uses the left Z and right X, the phase sum is masked with `3U`, and both bodies are XORed. This computes the operator product `P Q`. |
| `Pauli.mul`, `denote_mul`, and `phaseFactor_add` | [`right_multiply_masks`](../src/clifft/tableau/tableau.cc), called by `right_multiply_row` and `right_multiply_row_by_pauli`: the same crossing and XOR rule, with `phase_delta` added to the right operand's phase. The extra term multiplies the resulting operator by `i^phase_delta`. This maps the one-qubit case; correctness of the caller's choice of `phase_delta` is not proved. |
| `Pauli.mul` and `denote_mul` specialized to X or Z | [`right_multiply_generator`](../src/clifft/tableau/tableau.cc): the right operand has phase zero and body `(not z_generator, z_generator)`. Only X can cross an existing Z, contributing phase two. This maps the one-qubit specialization, not an arbitrary-width tableau update. |
| `denote_injective` | [`PauliString::operator==`](../src/clifft/tableau/pauli_string.cc): equality of the masks and phase is equivalent to operator equality in the single-qubit model. Packed storage and padding remain implementation obligations. |
| Positive and negative Y, and multiplication phases | [`Native Pauli phase convention preserves Hermitian signs`](../tests/test_tableau.cc) remains an implementation regression test. |

This increment has no preconditions beyond the types: Boolean masks, a valid
phase modulo four, and exact complex arithmetic. It does not prove packed-word
operations, padding, memory safety, the correspondence of C++ executions to
Lean values, floating-point behavior, or any compiler pass. Arbitrary-qubit
tensor products, Clifford conjugation, symbolic records, noise, measurement,
and approximation policies remain outside the current formal model.

## Contributing proofs

Each increment should state its exact claim, assumptions, exclusions, proof
outline, and C++ mapping. Changes to a modeled semantic contract should update
the formalization in the same pull request. New proof modules must be imported
by `ClifftProofs.lean` and pass the default build, import check, and kernel replay.

# Native prefix handoff and terminal sampling

The native prototype now samples complete attempts of the pinned physical
SOFT circuit about **10 times faster** than ordinary scalar Clifft on the
same CPU. It executes the original noisy prefix using Clifft, obtains the
boundary state through compiler-prepared expressions, samples fresh physical
faults in the terminal gadget, and produces all visible and hidden records,
raw detectors, and the observable.

Unlike the preceding [contraction microbenchmark](gadget_contraction.md),
these timings include the prefix, handoff, fault draws, parameter binding,
measurement RNG, and output parities. There is no per-history Stim/Cirq
analysis, input-state preparation in Python, or selection of favorable fault
histories in the timed path.

This is an opt-in native profiling prototype, not a new public `clifft.compile`
mode. Production compilation and sampling are unchanged. The input's
[cultivation interpretation remains unverified](soft_cultivation.md): its
noiseless detectors do not satisfy the expected cultivation sanity checks.
The result establishes a speedup for this exact physical circuit distribution,
not a validated distance-seven logical-error-rate calculation.

The subsequent [author-corpus audit](terminal_coverage.md) finds that the
prototype declines all four validated d3/d5 inputs. Their logical readout and
flagged extraction require additional support; d5 also exposes an avoidable
prefix-parser restriction. This coverage work precedes public packaging.

## How the handoff works

The bundle compiler recognizes the final T / Clifford-measurement-reset /
inverse-T gadget and its terminal CSS measurements. It appends internal
expectation probes to the original prefix for the 36 CSS checks, all-data
X/Y/Z, and both X/Z axes of each spectator. Clifft's existing HIR passes and
sampling planner lower these probes into active-coordinate actions and
affine Boolean signs.

Before any shot, the native constructor inspects those planned actions and
requires:

- Active width one at the handoff.
- Every CSS check has an identity active projection. Its sign can depend on
  earlier measurements and faults, but its eigenvalue is definite for every
  history admitted by the plan.
- The three logical observables map to three distinct nonidentity Paulis on
  that active qubit.
- Every spectator has a compiler-certified product X or Z state. The two
  gadget branches act proportionally on it, and each measured spectator has
  a definite X result.

These are symbolic compiler certificates, replacing the earlier per-history
CH-state inspection. A changed input that fails them is explicitly declined.
The ordinary Clifft path continues to handle such circuits; the experimental
CLI does not silently substitute an approximate state.

During a shot, the existing executor supplies the signed syndromes and three
logical expectations. The Bloch vector gives two complex amplitudes up to an
irrelevant common phase. Fixed binary duals map the Z syndrome to a code-basis
offset. This avoids synthesizing a physical encoder or recovering stabilizer
amplitudes during execution.

## Fault and measurement execution

The compiler propagates every supported suffix Pauli channel through the
fixed CNOT networks once. Each channel becomes a small set of XOR masks:
data X/Z effects before and after the inverse T layer, spectator-record flips,
and the two parities governing the central measurement and hidden reset.
Symmetric readout flips retain their own physical noise sites.

The native sampler draws those sites afresh on every attempt. It binds both
central outcomes to the precompiled local phase tables and uses the fixed
partial-record contractions to sample the central bit, logical Y bit, and
18 X-check bits. Remaining Z-check and spectator results are reconstructed
from the compiled maps. Prefix hidden records are moved to the full circuit's
hidden-record region, and all detector/observable parities use the original
record conventions.

Common Pauli global phases cancel in probabilities. Relative phases between
the two coherent branches are retained by the local operator tables and
spectator eigenvalues; the branches are never treated as a classical mixture.

All native model storage, contraction scratch, input tables, records, and
noise-choice buffers are allocated before sampling. The hot functions are
`noexcept` and perform fixed indexed arithmetic, parity evaluations, and the
repository's deterministic Xoshiro draws. They perform no tableau evolution,
commutation analysis, dependency discovery, or memory allocation. Stim is
used by the Python bundle compiler only; it is neither modified nor invoked
per shot.

## Complete-attempt measurements

[Raw results](native_terminal_data.json) include source hashes, native-binary
hashes, preparation times, all three timing repetitions, and the complete
validation traces. AMD EPYC 9554P, one pinned CPU, native Release build, one
worker. Each row uses three batches of 256 complete attempts after warmup.
Both paths run in the same native process and produce full records and raw
outputs, without postselection or early rejection.

| Noise setting | Native compiled terminal | Ordinary Clifft | Speedup |
| --- | ---: | ---: | ---: |
| Source, p = 0.0005 | 0.706 ms/attempt | 7.265 ms/attempt | 10.3x |
| Zero noise | 0.690 ms/attempt | 7.142 ms/attempt | 10.4x |
| Stress, p = 0.005 | 0.729 ms/attempt | 7.167 ms/attempt | 9.8x |

Compilation is excluded from per-attempt timings. Python bundle preparation
takes roughly half a second, plus native prefix planning and model loading.
The JSON reports those costs separately from ordinary native preparation.
Interpreter startup and process launch are outside both native timers.

The ordinary prefix peaks at ten active coordinates and reaches one at the
handoff. The full ordinary circuit peaks at nineteen. The new path therefore
avoids the large coefficient state for the last gadget while preserving
Clifft's existing small-state prefix execution. No matched total-RSS or
multithreaded/batched throughput comparison is claimed.

## Validation

For each of the three noise settings, 16 freshly generated complete attempts
are captured with their actual prefix/suffix Pauli choices and readout flips.
The validation driver materializes those choices into the original physical
circuit and uses ordinary Clifft forced replay on both the full circuit and
the prefix. Their log-probability difference agrees with the native terminal
probability to floating-point precision (below `1e-12` in all 48 traces).
This checks both the handoff and the generated records; it does not merely
score previously prepared terminal states. Stress traces contain dozens of
simultaneous physical faults, including histories with 77 selected sites.

Stim independently checks the generated visible-record to detector/observable
mapping with `skip_reference_sample=True`, preserving the raw convention used
by this source. No deterministic reference syndrome is subtracted.

Additional tests cover:

- A seven-data-qubit code with a nontrivial logical input, noisy prefix,
  mixed T signs, two-qubit Pauli faults, noisy central/terminal measurements,
  and faults between measurement and reset. Twenty-four fresh native
  histories agree with physical replay, and repeated seeded runs reproduce
  the complete records and selected fault sites.
- A zero-noise version whose full measurement distribution is computed using
  dense physical Qiskit gate matrices and explicit projectors. The 2,048-shot
  native distribution passes a conservative simultaneous binomial check,
  with no samples of impossible outcomes.
- Rejection when an added boundary Hadamard breaks the CSS certificate and
  when a nonterminal gate follows the output checks.

Together with the preceding independent Aer, Stim, contraction, and circuit
tests, **43 focused tests pass**. Source lint, formatting, mypy, and repository
hygiene checks also pass for the new files.

## Scope of the implementation

The bundle compiler takes a circuit, not a hard-coded source line or a list
of preselected histories. The current recognizer nevertheless has a narrow
contract: balanced one-logical-qubit CSS data, all-data X gadget parity,
product spectators, inverse CNOT networks, mixed physical T signs, and a
complete terminal CSS/logical readout. Its compact masks support at most
63 data qubits and 63 spectators. The research parser reserves S spellings
for its physical-T proxy and declines genuine S instructions; that is a
tooling limitation, not an interference-algorithm requirement.

General feedback, arbitrary continuations after the terminal measurements,
fixed Pauli gates inside the gadget, and random spectator measurements are
outside this implementation. The native certificate rejects unsupported
boundaries before execution. No speedup on the validated small author corpus
or on other large circuit families is claimed.

The important remaining engineering work is production packaging: integrate
recognition and lowering with the ordinary compiler, make unsupported cases
fall back automatically, and support the public sampling policies and output
interfaces. The core feasibility questions are now answered for this input:
the handoff can be compiled without runtime stabilizer analysis, and a native
sampler including that handoff is substantially faster on complete attempts.

## Reproduction

```bash
cmake --build build-study --target sample_terminal_gadget replay_cultivation -j 4
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --group dev --with cirq-core==1.6.1 \
  python tools/profile/benchmark_native_terminal.py \
    --output tools/profile/research/native_terminal_data.json
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --group dev --with cirq-core==1.6.1 \
  python -m pytest -q tools/profile/test_native_terminal.py \
    tools/profile/test_gadget_contraction.py tools/profile/test_clifford_gadget.py \
    tools/profile/test_soft_cultivation_study.py tools/profile/test_cultivation_study.py \
    tools/profile/test_cultivation_aer.py
```

To inspect a compiled bundle separately, run `compile_terminal_gadget.py`
with a source path and output directory, then invoke `sample_terminal_gadget`
with that directory, timing-shot count, validation-trace count, and seed.
The generated bundle is an internal research format, not a stable API.

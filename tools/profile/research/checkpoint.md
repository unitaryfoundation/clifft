# Representation study checkpoint

The current candidate is named **compiled logical-block simulation**. Production
recognition and integration are paused while bounded independent alternatives
are evaluated. The user prioritizes sustained sampling of large residual states,
simple invariants, and cheap eligibility/fallback. Few-shot startup and one-round
memory are outside this study.

## Resume compiled logical blocks

The reproducible checkpoint is commit `ce702482` on
`codex/factored-state-research`. Start with [the full f7 report](fold_f7.md),
[the f5 design](fold_blocks.md), and their adjacent raw JSON data.

- The noisy f7 reconstruction is `../fixtures/fold_cultivation_f7.stim`.
  It has 99 physical qubits, 221 non-Clifford gates, and 2,205 noise sites.
- The native sampler measured 45.41 us per attempted f7 shot over 100,000
  attempts; process maximum RSS was 4,980 KiB. Ordinary scalar Clifft plans
  width 44, but sampling was skipped. There is no measured f7 speedup ratio.
- f5 measured 17.73 us per attempt against 364.66 ms in the specific scalar
  diagnostic comparison. Neither result establishes best-in-class performance.
- f7 validation: 115 histories against the independent coherent reference,
  48 research tests, and native ASAN/UBSAN checks. Both uniform verification
  records are accepted, and nonzero growth sectors are preserved.
- The invariant is two logical amplitudes plus a physical Pauli frame at code
  boundaries, with up to four coherent monomial terms between boundaries.
  Contraction schedules and fault maps are compiled once; no hot topology work.
- This is a structured-generator prototype, not an arbitrary-circuit recognizer.
  Its terminal contract is ideal syndrome postselection and logical XYZ probes.
  It lacks final-error-correction decoding and a general residual-state handoff.
  The circuit is paper-guided, not verified equivalent to an authors' artifact.

The next logical-block work would be an output-matched competing benchmark,
then a minimal sound eligibility certificate and explicit fallback. Do not
start a general representation-switching framework or optimize SIMD first.

## Bounded independent studies

1. **Pauli-proxy / Clifford-stabilizer reduction.** Read the actual applicability
   conditions and output contract. Test the smallest exact reduction on our
   fixed-fault f3/f5/f7 inputs, including hooks and noisy growth. Distinguish
   protocol-outcome equivalence from physical-state or arbitrary-probe
   equivalence. Measure all per-history preprocessing when comparing timings.
   Stop at a correctness-backed proceed/defer/reject result; a new production
   trajectory interpreter requires a separate architectural decision.
2. **Clifford-frame improvement plus exact tensors.** First measure whether
   better coordinate orders or Clifford frames reduce exact residual ranks on
   noisy trajectories, including five-round coherent memory as an independent
   workload. Include frame-selection cost. This is a feasibility screen, not
   permission to build a full MPS backend or truncate bonds.

Previously measured product components and ordinary per-shot recompilation
did not justify proceeding for the user's target. These do not rule out a
different frame or a restricted proxy. Short coherent sums are partly explored
by the current method; direct noise-averaged probability contraction remains a
separate, unstarted aggregate-output direction.

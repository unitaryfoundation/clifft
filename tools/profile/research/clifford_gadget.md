# Coherent Clifford reduction of the physical cultivation gadget

## Result

The proposed reduction works algebraically and passes the physical-history
checks implemented here. It has **not demonstrated a speedup**. This is an
offline research reference, with no changes to Clifft's planner or executor.

The three T-conjugated parity gadgets in the pinned SOFT input reduce to two
coherent Clifford terms apiece. An offline signed-stabilizer intersection
check finds a one-logical-qubit entry space for all eight evaluated histories.
Compressing into its two basis states keeps every tested gadget at four
coherent stabilizer components, including the 80-wire, 37-data final gadget.
This avoids constructing the ordinary width-19 dense coefficient array.

The Python/Cirq implementation samples full attempts in roughly 1-2 seconds;
ordinary native Clifft takes roughly 7 milliseconds. These measurements
establish a compact representation and a usable correctness reference, not
an accelerated backend. The remaining engineering question is whether its
fault-dependent interference calculations can be compiled into a sufficiently
cheap fixed schedule. Porting its per-shot tableau work directly would violate
the current execution architecture.

The source remains an [unverified cultivation workload](soft_cultivation.md).
This experiment preserves its physical behavior, including its nontrivial
noiseless detector statistics. It neither repairs that circuit nor validates
its claimed code/fault distance.

## Exact reduction and physical noise

The recognizer checks inverse mixed T/T_DAG layers, a single X measurement
followed by an X reset of the same wire, and inverse CNOT networks around them.
The final gadget contains 73 CNOTs on each side. This inverse condition is
verified by tableau equality, not by gate-count equality. The smaller source
gadgets operate on 7 and 19 data wires; the final one operates on 37.

Let U be the first T layer, A the first CNOT network, and q the measured wire.
For a fixed physical fault history, propagate the Pauli errors through the
Clifford networks to form three Pauli products:

- F_pre, at the end of A and before the measurement;
- F_mid, between the measurement and reset;
- F_post, at the end of the inverse CNOT network.

Every original one- and two-qubit Pauli channel is sampled at its original
location. A two-qubit fault remains a correlated draw, including its identity
factor when applicable. The reduction does not replace these channels by
independent single-qubit errors.

Let m be the true measurement result. Let delta_pre and delta_mid indicate
whether the corresponding Pauli product anticommutes with X_q. The hidden
reset result is necessarily

```text
r = m xor delta_mid.
```

A requested history with a different reset result has probability zero.
Define

```text
P = A^dagger X_q A
Q = F_post A^dagger Z_q^r F_mid F_pre A.
```

Both are Paulis, retaining their complex Pauli phases. The complete conditional
operator is

```text
K_m = (C_0 + (-1)^(m xor delta_pre) C_1) / 2
C_0 = U^dagger Q U
C_1 = U^dagger (Q P) U.
```

Conjugation of a Pauli by a tensor product of T/T_DAG gates gives a Clifford
operation. For each affected X or Y factor, the implementation applies that
Pauli followed by S_DAG (for a forward T) or S (for a forward T_DAG), retaining
the corresponding exp(+/- i*pi/4) phase. Spectator factors are unchanged.
Thus the high-width interval becomes a sum of two Clifford actions, including
the reset correction and the actual fault history.

A readout flip b changes the reported result to m xor b. It does not change m,
the physical projection, or the reset correction. Keeping these quantities
separate is necessary: treating the reported bit as the true reset branch
would give an incorrect coherent operator.

The two Clifford terms are **not a probabilistic mixture**. The reference
keeps their relative phases and evaluates cross terms when computing norms
and branch probabilities. Subsequent Clifford gates and Pauli projections
act on each component without doubling the component count.

## Boundary compression

Naively retaining every expansion through three gadgets can leave 16 terms,
even though the compiled input to the final gadget has active width one.
Merging identical stabilizer rays alone did not always recover two terms.

The reference instead intersects the signed Pauli-stabilizer groups of its
components. If they share at least n-1 independent positive stabilizers, it
constructs an orthogonal basis for that at-most-two-dimensional space and
computes the state's coherent coordinates in that basis. Global phases are
retained through CH amplitudes and overlaps. If this certificate fails, it
retains the components rather than assuming that the state fits one qubit.

All eight tested histories pass this entry check, including histories with
61 and 68 physical faults in the elevated-noise diagnostic. All final-gadget
samples and all full samples in the report peak at four components. This is
evidence on those histories, not an exhaustive proof of the boundary property
for every possible fault assignment.

Finding these common generators, synthesizing Clifford inverses for overlaps,
and evolving the CH/tableau components are dynamic operations in this offline
reference. None of them runs inside ordinary Clifft dispatch.

## Validation

The [raw report](clifford_gadget_data.json) pins the source and materialized
history hashes, fault selections, complete records, timings and versions.

- Eight complete physical histories are sampled by ordinary Clifft and
  evaluated by the coherent reference: two zero-noise histories, four at the
  source p=0.0005, and two at p=0.005 as stress cases. Each includes all 355
  visible measurements and 361 hidden reset measurements. The largest
  absolute log-probability difference is below 1e-12.
- Eight histories are generated independently by the terminal reference,
  conditioned on an ordinary full-history prefix. Restoring those prefix
  records and replaying the original physical circuit agrees with the sum of
  prefix and conditional-terminal log probabilities.
- Three complete noisy attempts are generated by the coherent sampler,
  including their physical fault draws. Each is reachable under ordinary
  physical-gate replay and agrees in conditional log probability.
- Fifty-four three-qubit operator comparisons check all X/Y/Z combinations
  at the pre-measurement, between-measurement-and-reset, and post-reset
  fault positions, for both true outcomes. Qiskit matrices verify the full
  conditional operators, so these tests check phase and arbitrary input states.
- Sixty-four basis-state comparisons against Aer unitary matrices check
  conjugated-Pauli phases with mixed T/T_DAG signs.
- Small native bidirectional replays cover deterministic readout flips and
  faults that change the hidden reset outcome. Tests also cover phase-preserving
  boundary compression and rejection of noninverse networks, duplicate T
  targets and ambiguous use of the S-proxy spelling.

The algebraic reduction is exact. Its CH implementation uses floating-point
coefficients and relative numerical-zero tolerances, not rational arithmetic.
The comparisons above test that numerical implementation; they are not a
proof of every possible full-circuit outcome distribution. No logical-error
rate is inferred from these small samples.

Ordinary Clifft replay explicitly declines readout-channel actions. The native
comparison therefore writes each already-selected readout flip as an inverted
measurement target, and removes zero-probability readout channels. This changes
the bit convention while preserving the physical projection and subsequent
record references. It does not replace readout noise by a data-qubit error.
The new native diagnostic rejects remaining readout-channel actions rather
than reporting their unsupported replay as a physical impossibility.

## Timing and scope

The benchmark pins one worker to one logical CPU on the AMD EPYC 9554P VM.
Ordinary Clifft is the same production sampling implementation as the earlier
study. The final run records exact package versions in the JSON report.

| Operation | Observed scale | Contract |
| --- | ---: | --- |
| Ordinary Clifft full sampling | About 7 ms/attempt | Three batches of 256 full-record attempts, no postselection |
| Coherent reference full sampling | About 1-2 s/attempt | Three complete attempts, fresh fault histories |
| Coherent terminal sampling | About 0.4-0.6 s/attempt | Eight conditional samples at two prepared entry states |
| Coherent full-history evaluation | About 0.2-0.5 s/history | Forced-record probability, not sampling |

Fault materialization and fault-conditioned plan preparation are timed
separately from reference sampling. Ordinary compilation is likewise separate.
The terminal timings also exclude prefix-state preparation, which is recorded
separately. Consequently the terminal row is not directly comparable to an
unconditional attempted-shot rate. The tiny reference sample counts provide
an order-of-magnitude feasibility check, not a precise performance study.

Avoiding dense coefficients does not by itself establish a total-memory win:
the reference also owns Python, Cirq and Stim objects, synthesized Clifford
circuits and scratch storage. No matched RSS comparison is claimed.

## Implication for Clifft

This gives a more concrete compiler target than merely increasing the existing
CSS prototype's size limit: recognize and certify the physical measurement
gadget, preserve its fault/reset/readout maps, and compile its coherent branch
effects. The terminal gadget plus Clifford suffix remains the smallest useful
integration target because it need not return an arbitrary physical state to
a later non-Clifford circuit.

The missing production capability is efficient **precompiled interference
evaluation with fault-dependent Clifford effects**. Existing affine Pauli
frames do not automatically represent those Clifford changes. The reference
currently discovers stabilizer intersections and overlap circuits per history;
moving that code into a C++ executor would still contradict the prohibition
on runtime topology planning. A native port's speed is therefore not the only
question.

The next bounded experiment should first express the branch-overlap calculations
as fixed compiler-prepared phase/parity contractions and measure their cost.
It must preserve arbitrary incoming logical amplitudes, all reported and hidden
records, and the physical faults. If that fixed representation grows too large
or needs fault-specific elimination, it does not fit the present architecture
without an explicit architectural decision. This study does not make that
decision or introduce a workaround.

The follow-up [fixed-contraction experiment](gadget_contraction.md) now answers
that bounded question positively for this terminal gadget, including the
partial-record probabilities needed for sampling. Its remaining compiler
handoff and timing limitations are documented separately.

## Reproduction

After preparing the ordinary development environment and configuring
`build-study` with `CLIFFT_BUILD_PROFILER=ON`:

```bash
cmake --build build-study --target replay_cultivation -j 4
MPLCONFIGDIR=/tmp/clifft-gadget-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --group dev --with cirq-core==1.6.1 \
  python tools/profile/benchmark_clifford_gadget.py \
    --native build-study/replay_cultivation \
    --output tools/profile/research/clifford_gadget_data.json
MPLCONFIGDIR=/tmp/clifft-gadget-mpl \
  uv run --frozen --group dev --with cirq-core==1.6.1 \
  python -m pytest -q tools/profile/test_clifford_gadget.py
```

Cirq is an optional research dependency supplied by `uv --with`; it is not
added to Clifft's production dependencies. Run benchmark and tests sequentially.

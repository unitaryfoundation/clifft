# Complete f5 fold-transversal MSC reconstruction

Date: 2026-09-20. The physical f5 protocol now runs through final fictitious
error detection (FED). It supplies a real large-cost baseline and a checked
extension of the coherent-Clifford reduction to folded checks with CCZ gates.
The reduction remains an offline reference, not a production sampler.

Baseline update (2026-09-20): the measurements below use revision
`41a18d455e855650f3c5dffa75db301f67ff2ded` and the default compile pipeline
(peephole fusion and statevector squeeze). They do not include the opt-in
`ActiveWidthSchedulePass` from [PR 471](https://github.com/unitaryfoundation/clifft/pull/471),
whose inspected head is `8f69b3c305303483e5dd33dcf4d13dbbebae1fa9`.
[PR 472](https://github.com/unitaryfoundation/clifft/pull/472) documents that
work and adds no runtime optimization. The reported width, memory and time
therefore establish a default-pipeline baseline. The completed
[scheduled comparison](scheduled_baseline.md) reruns identical f3/f5 inputs
and the synthetic sweep with both bounded and unbounded scheduling. For f5
at p=0.001 with early rejection, unbounded scheduling reduces 445 ms to
208 ms per attempt in the matched run, while peak width stays 22 and the
coefficient array stays 64 MiB. Use that stronger baseline for subsequent
gadget comparisons; the historical timings below are retained for provenance.

Implementation follow-up: the [folded-check contraction kernel](folded_contraction.md)
now samples the two checks and returns coherent logical amplitudes for
continuation. Full-history and logical-phase audits pass. Fault binding
remains offline; its 3.43 ms kernel timing is not a full-protocol speedup.

## Physical construction and provenance

`../folded_msc_family.py` retains the [f3 reference](folded_msc.md) unchanged
when asked for distance three. For f5 it relabels that prefix into a larger
patch, stopping after the second small logical check, then adds growth,
stabilizer extraction, two flagged logical checks, and terminal evaluation.

The regular-d3 to rotated-d5 growth follows the gate sequence in
[Sahay's grow_3u5r](https://github.com/kaavyas99/MSC_foldedH/blob/9378fd228ba83c592875d155136fcbcd16a45680/src/full_clifford_sim/s3_fxns.py).
It inserts twelve data qubits between the old thirteen. A half-cycle then
adds sixteen data qubits to complete the regular-d5 patch. Code, logical-map,
and reset-support checks independently validate this transcription.

The protocol sequence, fold factors, cat assignment and flagged preparation
follow [Takada et al., Appendix A](https://arxiv.org/html/2609.16929v1#A1).
The five cat wires use offsets (0,3,0,1,2), with one verification wire.
The generated input has the reported 47 physical qubits and 97 non-Clifford
gates in the noisy protocol. The noiseless evaluation adds 34 non-Clifford
gates. Final evaluation is FED, not decoder-based correction (FEC).

The generated artifact is `../fixtures/folded_msc/f5_fed_p1e-3.stim`, with
operation-level stage and source references in its JSON sidecar. There are
151 visible measurements, 150 acceptance detectors, and 192 hidden reset
outcomes. Noise remains attached to each physical operation: 261 reset/
readout X-error sites, 178 one-qubit, 454 two-qubit and 40 three-qubit
depolarizing sites, totaling 933 locations. No idle noise is added.

As for f3, sequential stabilizer CNOT ordering is an explicit reconstruction
choice. This is not an identical author export or a reproduction of the
published logical-error coefficients. The f5 label identifies the target
construction; a complete fault-distance-five proof has not been performed.

## Validation

Fourteen focused tests pass across `test_folded_msc.py` and
`test_folded_msc_family.py`. New checks establish that:

- General geometry exactly reproduces the previously verified d3 checks.
  The generated f3 physical input remains byte-for-byte unchanged.
- Stim verifies that growth preserves the encoded X, Y and Z eigenstates
  and establishes every regular-d5 CSS stabilizer. Only the 28 new data
  qubits are reset.
- The independent folded Clifford expression preserves the code space
  and has the expected logical action on X, Y and Z.
- Exhausting the 100 single-location Pauli faults in cat preparation
  confirms that accepted cat states have at most one X-type error modulo
  the all-cat X stabilizer. This checks the cat component, not the whole
  protocol's fault distance.
- Complete noiseless f5 sampling has zero detectors and logical failures.
- A dense Qiskit Aer comparison covers all 256 combinations of conjugation
  direction, computational control bit, and Pauli faults after the first T
  and the intervening CX. An entangled reference qubit checks the complete
  local map, including phase.

`folded_msc_f5_validation.json` records two further audits:

1. The new folded-check oracle agrees with all 24 previously saved dense-Aer
   f3 histories, including the full visible and hidden measurement records.
2. It agrees with native physical Clifft replay on 24 complete f5 histories.
   Fault patterns are sampled at p=0, 0.001, 0.01 and 0.03, including cases
   with up to 35 physical faults. Maximum absolute log-probability difference
   is 1.78e-15. Both accepted and rejected full trajectories are represented.

These are conditional-on-fault-pattern probability checks, not logical-error
estimates. A full f5 dense state vector is impractical. Aer MPS attempts in
two qubit orders did not finish within separate 120-second limits; a prefix
through growth did complete. Therefore there is no claimed full f5 MPS
cross-check. The full f5 comparison uses the coherent-Clifford oracle,
supported by the independent small dense tests and code-algebra checks.

## What the offline reduction demonstrates

`../folded_msc_oracle.py` reuses the existing `CoherentState`/`Term` reference.
For each check, it resolves the cat's coherent computational branches.
Within each branch, a T-conjugated CX with its sampled intervening Pauli
faults is a single-qubit Clifford, including its phase. A CCZ is either an
identity or CZ according to its cat control. Physical ancilla errors,
flag measurements, unencoding and all data syndromes remain present.

The largest representation in the 24 sampled f5 histories contains **16
coherent Clifford terms**; the clean history peaks at eight. These observed
counts are not a general constant-width bound. Terms carry full stabilizer
states, so counting them is not a byte-for-byte comparison with amplitudes.

The reference uses dynamic Stim/Cirq tableau operations and computes the
probability of a supplied complete history. Its timings are not comparable
to full shot generation, and it is not allowed in Clifft's hot execution.
It establishes correctness of the folded reduction and gives concrete
compiler work to pursue without replacing Clifft's architecture.

## Ordinary-Clifft cost

`folded_msc_f5_benchmark.json` records three batches of eight attempted shots
per case, after warmup, on one pinned AMD EPYC 9554P core. Compilation is
excluded and returned records are included. This deliberately small run
measures large per-shot cost; its acceptance counts are not precise rate
estimates.

| Physical p | Full sampling seconds/attempt | Early rejection seconds/attempt |
| --- | ---: | ---: |
| 0 | 0.686 | 0.674 |
| 0.001 | 0.758 | 0.416 |

At p=0.001, seven of 24 attempts pass in each mode. No logical-error rate
can be inferred from this sample. Peak active width is **22**, or 4,194,304
complex amplitudes / **64 MiB per lane**, excluding other runtime storage.
The corresponding f3 coefficient payload was 4 KiB.

`folded_msc_f5_stages.json` maps static unfused plan diagnostics to source
stages. Both f5 checks and the final logical measurement reach width 22.
They account for approximately 115 million, 558 million and 176 million
estimated coefficient visits, respectively. These are compiler estimates,
not measured per-stage timings. They locate the large expansion in the
Clifford-measurement gadgets rather than in the Clifford growth circuit.

## Reproduction and next step

```bash
PYTHONPATH=tools/profile uv run --frozen --group dev python tools/profile/folded_msc_family.py \
  --distance 5 --probability 0.001 --output /tmp/f5_fed.stim

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools/profile \
  uv run --frozen --group dev --with cirq-core==1.6.1 pytest -q \
  tools/profile/test_folded_msc.py tools/profile/test_folded_msc_family.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools/profile \
  uv run --frozen --group dev --with cirq-core==1.6.1 python \
  tools/profile/audit_folded_msc_family.py --output /tmp/f5_validation.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools/profile \
  uv run --frozen --group dev python tools/profile/benchmark_folded_msc.py \
  --distance 5 --probabilities 0 0.001 --output /tmp/f5_benchmark.json
```

The next optimization experiment should compile the folded logical-check
reduction, including sampled fault effects and required continuation, into
preplanned arithmetic. Use this f5 circuit as the complete-protocol baseline.
The existing terminal-only recognizer cannot simply accept it unchanged.
Success requires independent probability checks and an actual end-to-end
speed/memory comparison. f7 reconstruction and PyMatching FEC remain on the
retained plan; no production backend or runtime tableau planning was added.

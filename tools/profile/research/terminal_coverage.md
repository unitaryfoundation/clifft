# Terminal prototype coverage and distance scaling

The current native terminal prototype accepts **zero of the four validated
author circuits**. The earlier roughly 10x full-attempt speedup remains a
result for the single SOFT workload, whose cultivation interpretation is
unverified. This audit does not establish a speedup on a validated protocol.

## Completed checks

The reproducible [audit](../audit_terminal_coverage.py) checks the unchanged
pinned author circuits after the documented physical S-to-T substitution.
[Results](terminal_coverage_data.json) include hashes and recognition failures.

| Author circuit | Complete bundle | Isolated last physical gadget | Additional obstruction |
| --- | --- | --- | --- |
| d3a6 | Declined: unsupported middle | Recognized | Final T / MPP-X / inverse-T evaluation |
| d3a6f2 | Declined: unsupported middle | Declined | Flag resets and measurements; final evaluation |
| d5a19 | Declined: prefix feedback | Recognized | Final T / MPP-X / inverse-T evaluation |
| d5a19f13 | Declined: prefix feedback | Declined | Flag resets and measurements; final evaluation |

Recognition of an isolated gadget means its inverse networks and
measurement/reset structure pass the existing algebraic parser. It does not
mean a native boundary certificate or full sampler was validated for it.

All four also passed fresh ordinary-Clifft checks:

- 256 noiseless physical-T attempts each, with all detector and observable
  bits zero: 1,024 attempts total.
- 8,192 noisy S-proxy attempts each against independent Stim sampling:
  32,768 attempts total. Detector/observable marginals and acceptance pass
  the existing simultaneous seven-standard-error smoke-test tolerance.
- Exact detector/observable parity agreement between Clifft's records and
  Stim's record converter for all those proxy attempts.

These are baseline correctness checks, not noisy physical-T joint-distribution
or logical-error-rate validation. The earlier independent d3 Aer checks are
documented in the [corpus study](cultivation_corpus.md); they were not rerun.

The feedback rejection is an avoidable research-parser restriction: the
prototype parses the entire prefix even though ordinary Clifft executes it.
Both d5 sources contain two feedback instructions. The terminal measurement
and flagged-gadget failures require actual support beyond that parser fix.
All four end with a T-conjugated product-X logical measurement followed by
CSS checks. Replacing this with the prototype's product-Y output would change
the experiment. The flag records must also remain in the physical circuit.

## A useful distance-scaling experiment

Yes, construct a parameterized **gadget benchmark** using the published color
code geometry, starting with d=3 and d=5 and extending to d=7 and d=9. Check
the small instances against ordinary Clifft and an independent physical
oracle before using larger instances. Measure compilation, peak contraction
scope, memory, coherent-term count, and complete gadget sampling time under
ordinary noise and conditioned faults. Include the cost of preparing or
handing off the state when reporting complete-attempt timing. State an ideal
encoded-input assumption explicitly when using it.

This can answer whether the compiled contraction approach scales without
waiting for more author artifacts. It does not supply a validated larger
injection-and-cultivation protocol. In particular, growth, feedback, flags,
and the resulting circuit fault distance need separate validation.

The original pinned [cultivation generator](https://github.com/Strilanc/magic-state-cultivation/blob/871e68ff6df2f75190b1bfd6351459d1b5a037e3/src/cultiv/_construction/_cultivation_stage.py)
has explicit d3 checks, d3-to-d5 growth, and d5 checks and assembly functions.
It is not a general distance parameter for the full cultivation stage.
The pinned [Chan circuit factory](https://github.com/timchan0/cliffordea/blob/a62a353e77468a2ff54c24e83133ad65a63e2294/cliffordea/enum/circuits/main.py)
loads prewritten files for listed d3/d5 combinations; passing distance 7 does
not synthesize a circuit. Its documentation distinguishes nominal code
distance from the lower fault distance of some unflagged constructions.

The next implementation priority is therefore coverage on these real small
circuits: leave prefix execution with Clifft, support their exact final
logical measurement, and handle the flagged physical checks. Validate joint
fault-conditioned probabilities and records before measuring speed. The
distance-scaling benchmark then tests whether that useful coverage remains
efficient as the code grows. A separate d7 cultivation design would need
fault-distance evidence, not just noiseless success or a few sampled faults.

## Importance sampling is a separate requirement

Ordinary noisy attempts suffice to measure runtime and memory versus distance.
Estimating a tiny accepted logical-error rate is a different statistical
task. If acceptance is A and the error probability among accepted attempts is
L, approximately 100/(A L) independent attempts are needed for 100 expected
failures, giving roughly 10% relative counting error in the rare-event limit.
For illustration, A=0.01 and L=1e-12 would require about 1e16 attempts. These
are hypothetical values, not measurements or predictions for our fixtures.

[Tuloup and Ayral, Section V.2 and Figure 13](https://arxiv.org/html/2603.14670v1#S5.SS2)
already use fault-count importance sampling for d5. Their figure reports
1e9 to 4e9 samples for the sampled fault-count strata and only 1 to 4 failure
events per stratum. Thus fixed-count sampling helps but can still leave rare
malignant fault configurations within each stratum.

Clifft already provides `sample_k` and `sample_k_survivors`; the
[existing corpus study](cultivation_corpus.md) exercised weights 3 and 5.
The new native terminal experiment currently samples ordinary noise only.
To combine them, retain one global physical noise-site catalog and draw a
single conditioned fault set across prefix and suffix. Independently imposing
k faults in each part would implement the wrong distribution. Reuse Clifft's
existing conditional sampler and retain the original channel probabilities.

For stratum weight w_k=P(K=k), use attempted-shot denominators for both rates:

    L = sum_k w_k P(accept AND error | K=k)
        / sum_k w_k P(accept | K=k).

Use binomial weights for equal site probabilities and Poisson-binomial weights
otherwise. Track numerator/denominator uncertainty and bound the omitted
tail relative to the small acceptance/error target. Do not assume all k<d
failure probabilities vanish from the nominal distance or from zero observed
failures. Establish the circuit's fault distance first. If failures remain
too rare within important strata, evaluate targeted fault proposals or
splitting with correct weights; fixed-k sampling alone has no efficiency
guarantee for large d.

## Reproduction

```bash
UV_CACHE_DIR=/tmp/clifft-study-uv-cache \
MPLCONFIGDIR=/tmp/clifft-gadget-mpl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools/profile \
uv run --offline --frozen --group dev --with cirq-core==1.6.1 \
  python tools/profile/audit_terminal_coverage.py \
  --output tools/profile/research/terminal_coverage_data.json
```

`--offline` assumes the dependencies are already cached. This audit changes
no production compiler or executor behavior.

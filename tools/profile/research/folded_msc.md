# Reconstructed fold-transversal MSC baseline

Date: 2026-09-20. The first complete f3 cultivation circuit is now generated
and checked, through final fictitious error detection (FED) and a noiseless
logical measurement. This is the first step of the
[retained f3/f5/f7 reconstruction plan](protocol_sources.md), not a new
gadget performance result. No production simulator changes were made.

Performance update: the [PR 471 comparison](scheduled_baseline.md) explicitly
enables active-width scheduling and supersedes the default-only timings below.
At p=0.001 with early rejection, f3 improves from 5.71 us to 3.58 us per
attempt in the matched run; peak width remains eight.

## Circuit and scope

`../folded_msc.py` independently transcribes the physical construction in
[Takada, Bartlett, and Williamson](https://arxiv.org/html/2609.16929v1#A1).
Injection and growth follow Figures 18 and 7; controlled Hermitian factors,
cat wiring and assignments follow Appendix A.1, Table 4 and Figure 19a.
The circuit uses 13 data qubits plus three reusable ancillas. There are 29
non-Clifford gates in the noisy preparation protocol. The final noiseless
evaluation adds 14 more, and uses a single ancilla for the logical check.

Stages are injection, rotated-to-regular growth, stabilizer extraction,
two logical checks, stabilizer extraction, and final noiseless evaluation.
The latter rejects nonzero syndromes and measures the logical H_XY operator.
It does not perform fictitious error correction (FEC) or an escape stage.

There are 253 physical noise locations: 73 reset/readout X-error sites,
56 one-qubit, 116 two-qubit and eight three-qubit depolarizing channels.
Physical H, T, T_DAG, CX and CCZ gates each retain their own subsequent
noise location. There is no idle noise. Final evaluation is noiseless.
All 30 noisy measurement outcomes and the 12 final syndromes must be zero;
the last logical measurement is the observable, not an acceptance detector.
Thus the complete FED circuit has 43 measurements and 42 detectors.

**Schedule limitation:** the publication specifies sequential stabilizer
extraction with one reusable ancilla but does not enumerate every CNOT
order in that extraction. This reconstruction uses X checks then Z checks,
in the generator order displayed in Sec. III.A, with ascending data index
within each check. These choices affect hook errors. The circuit is a
documented reconstruction, not a byte-identical author artifact or a
reproduction of the paper's numerical coefficients. The claimed protocol
label f3 is inherited from the construction; tests below are not a complete
fault-distance proof. Author files can resolve this remaining convention.

Every generated operation records a stage and paper reference in the
sidecar JSON, together with measurement roles and the physical-text hash.
The checked-in generated input is
`../fixtures/folded_msc/f3_fed_p1e-3.stim`; the corresponding JSON is beside it.

## Independent validation

Five focused tests pass in `../test_folded_msc.py`:

- Qiskit Aer evolves the 13-qubit injection/growth state. All twelve
  published CSS stabilizers have expectation +1. Applying the published
  logical Clifford expression directly leaves the state unchanged; this
  comparison does not use the generated controlled-check circuit.
- Stim verifies the complete CSS stabilizer algebra. Enumerating both
  logical Pauli cosets establishes code distance three.
- Ordinary Clifft produces all-zero detectors and logical observable on
  256 noiseless complete attempts. Physical noise placement and noiseless
  terminal evaluation are checked separately.
- Aer and native Clifft replay agree on fault-conditioned complete histories,
  including the otherwise hidden outcomes discarded by reset.
- An explicitly inserted logical Z error leaves the final syndromes zero
  and flips the final logical result, verified independently with Aer and
  native replay. Thus the success test is not merely insensitive to error.

The larger saved audit in `folded_msc_validation.json` contains 24 complete
trajectories with physical faults sampled at stress probability 0.02,
including a clean reference case. The maximum absolute log-probability
difference is 4.89e-15. These are exact probabilities conditioned on each
chosen Pauli fault pattern; they do not estimate a logical error rate.

The audit also exhausts all **2,485 single-location nonidentity Pauli
faults**, including all 63 possibilities at each three-qubit channel.
Native replay assigns zero probability to accepted logical failure in every
case. This exhausts single faults only, not all pairs. The generator and
reference retain all physical operations; no Clifford substitution is used.

## Ordinary-Clifft measurements

`folded_msc_benchmark.json` contains public-API timings, seeds, source hashes,
native-profiler hash, CPU information and raw counts. Execution used one
pinned AMD EPYC 9554P core, three batches of 262,144 attempted shots per case,
with compilation outside the timer and returned records included.

| Physical p | Full sampling us/attempt | Early rejection us/attempt | Acceptance with early rejection |
| --- | ---: | ---: | ---: |
| 0 | 7.84 | 8.07 | 1.0000 |
| 0.001 | 6.84 | 5.82 | 0.7894 |
| 0.003 | 6.65 | 4.62 | 0.4922 |
| 0.01 | 7.85 | 2.12 | 0.0949 |

Peak active width is **8**: 256 complex coefficients, or 4 KiB per lane,
excluding other runtime storage. Several expensive source locations are
CCZ operations. The current terminal-gadget recognizer does not support
this fold-transversal circuit, and no optimized-path comparison is claimed.

Separate fixed-fault runs use one million attempted shots at each k:

| Fault count k | Accepted | Accepted logical failures |
| --- | ---: | ---: |
| 1 | 68,700 | 0 |
| 2 | 9,228 | 0 |
| 3 | 1,351 | 31 |

These are conditional-on-k experiments, not unconditional logical-error
probabilities. The two-fault sample is useful evidence but not exhaustive.
The nonzero three-fault result confirms that the physical noisy protocol
and logical evaluator can produce accepted failures. Direct sampling found
too few failures to support a precise low-p estimate. Importance sampling
remains relevant to statistics independently of per-shot simulation cost.

## Reproduction

Use the existing native `profile_cultivation` and `replay_cultivation`
research targets and the development Python environment:

```bash
PYTHONPATH=tools/profile uv run --frozen --group dev python tools/profile/folded_msc.py \
  --probability 0.001 --output /tmp/f3_fed.stim

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools/profile \
  uv run --frozen --group dev pytest -q tools/profile/test_folded_msc.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools/profile \
  uv run --frozen --group dev python tools/profile/validate_folded_msc.py \
  --single-faults --output /tmp/folded_msc_validation.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools/profile \
  uv run --frozen --group dev python tools/profile/benchmark_folded_msc.py \
  --output /tmp/folded_msc_benchmark.json
```

## Next milestone

The [f5 extension](folded_msc_f5.md) now completes growth, flagged checks and
FED, with a checked coherent-Clifford oracle and ordinary-Clifft profile.
The next optimization experiment is a compiler-planned folded check with
the required continuation. The f7 growth and eight-qubit verified cat remain
subsequent work, as does matching PyMatching FEC. The small f3 result is a
correctness baseline, not evidence of a complete-protocol gadget speedup.

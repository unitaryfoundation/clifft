# Other non-Clifford families for the gadget experiment

Audited 2026-09-20. The intended outcome remains an extension of Clifft that
improves real workloads or makes previously inaccessible ones tractable.
A large physical qubit count or a distillation label is insufficient evidence.
The complete cultivation reconstruction plan is retained in
[protocol_sources.md](protocol_sources.md): reconstruct Takada f3, validate,
then f5/f7 with the published noise, cat acceptance, and logical evaluation.
That work has not been implemented. This candidate search supplements it.

## Structural selection criterion

For a Hermitian Clifford operator C with C^2 = I, an ideal ancilla-mediated
measurement has outcome operator (I + (-1)^m C)/2. This is two coherent
Clifford terms even when implementing controlled-C requires many non-Clifford
gates. This algebraic observation motivates testing other Clifford checks.
It is not a claim that the existing recognizer accepts them.

The current implementation recognizes a much narrower physical T-conjugated
CNOT parity gadget, one logical CSS input, and a terminal Y readout.
Multi-block inputs, flags, repeated checks, different readouts, and arbitrary
continuations require further compiler work. Two terms per check do not
guarantee bounded cost across many checks. Every physical fault location must
survive the reduction; an ideal controlled-C replacement alone is not a noisy
simulation. Profile ordinary Clifft before extending this implementation.

## Candidate ranking

| Family | Public artifact actually inspected | Relevance and next action |
| --- | --- | --- |
| Encoded 5-to-1 distillation | QuEra Tsim notebook and an 85-qubit extended-Stim benchmark | Runnable control; ordinary Clifft already needs only five active coordinates on the benchmark |
| Bottom-up Toffoli-state preparation | Parameterized distance/round generator in Ruiz's repository, but with Toffoli replaced by CNOTs | Best distinct structural candidate: recover physical gates from the paper, then test one noisy Clifford check before a full sweep |
| MEK / Hadamard-check distillation and flagged H-state preparation | Published circuit constructions; the inspected Chamberland repository contains result analysis, not a generator | Close algebraic match; reconstruct a small check as an independent validation case, without promising a speedup |
| Unfolded distillation | Circuit schedules, code data, noise and decoding scripts; inspected simulation uses Clifford rotations | Useful real distillation target after restoring physical rotations; not a direct fit for the present gadget recognizer |
| Code switching / transversal-T code conversion | Published 15-qubit Reed-Muller to Steane construction | Potential test of a more general compiled logical operation; no verified runnable author generator found in this search |

## Immediately usable distillation control

[QuEra's Tsim notebook](https://github.com/QuEraComputing/tsim/blob/44f64ba79c91e6ada77d6b609595ad0a3d767c11/docs/demos/magic_state_distillation.ipynb)
contains both a five-qubit logical circuit and a five-block [[17,1,5]]
color-code implementation. The latter assumes ideal encoding, inserts noise
in the transversal distillation circuit, and uses X/Y/Z output tomography.
Acceptance combines code syndromes with the logical distillation syndrome
1011. Preserving both conditions matters for fidelity reproduction.

The already exported
[Clifft-paper Z-basis benchmark](https://github.com/unitaryfoundation/clifft-paper/blob/db7dc9f13a2c2854690e92390c779048a1ac1400/qec_bench/circuits/distillation.stim)
has 85 qubits, 85 measurements, 40 detectors, and five observables. SHA-256:
`188bd53c48dbc21f840fb297df6f41c61f5bad6a856bba621f00ff42078921c1`.
Its only non-Clifford instruction lines are:

```text
R_X(-0.3040867239846964) 7 24 41 58 75
T_DAG 7 24 41 58 75
```

Fresh check with this worktree's native `profile_cultivation` harness:
225 compiled actions; maximum of the `before` and `after` active widths is
**5**, so the double-complex coefficient state is 32 amplitudes / 512 bytes
per lane, excluding other executor storage. This was a compilation/profile
check, not a throughput or fidelity experiment. Reproduce after downloading
the pinned file as `/tmp/distillation85.stim`:

```bash
build-study/profile_cultivation /tmp/distillation85.stim > /tmp/distillation85-plan.tsv
python - <<'PY'
import csv
with open('/tmp/distillation85-plan.tsv') as stream:
    rows = list(csv.DictReader(stream, delimiter='\t'))
print(max(int(row[key]) for row in rows for key in ('before', 'after')))
PY
```

This is a useful negative control for dispatch overhead and a candidate for
full notebook reproduction. It is not evidence of a new memory bottleneck,
and detector-only selection on its Z output is not the notebook's complete
distillation fidelity calculation.

## Best distinct gadget candidate: a Toffoli-state Clifford check

[Ruiz et al.'s paper, Appendix A](https://arxiv.org/html/2507.12511v1#A1)
describes the bottom-up comparison: measure logical X1*CX23 on three
repetition-code blocks with a GHZ ancilla. This is a Hermitian Clifford
check whose controlled implementation includes Toffoli gates. The family
varies code distance and check repetition count.

The inspected author repository is
[Unfolded_distillation](https://github.com/DiegoRuiz-Git/Unfolded_distillation/tree/c20a7268889b717e3c0fa3eccbeb39928f9810bf),
MIT licensed. Its `Toffoli Bottom-up scheme/pz = 1e-3/Logical error.py`
contains `CreateCircuit(d,r)` and a noise model. Crucially, its own comment
says `transversal CNOT (in place of Toffoli)`: paired CNOTs sharing a target
are a surrogate. `AddNoise` recognizes the shared target and attaches
three-qubit phase-error channels. Loading this file unchanged would only
benchmark Clifford simulation.

Proposed experiment, not yet executed:

1. Match the gate mapping and preparation/output basis against the cited
   physical protocol. Export actual CCX operations (or H-CCZ-H) with all
   physical noise and measurement records. Do not blindly replace every
   repeated-target CNOT sequence in arbitrary circuits.
2. Validate the smallest complete check with a dense physical reference,
   including fault-conditioned branches and GHZ outcomes.
3. Measure ordinary-Clifft peak active width and time over distance and
   rounds before considering a compiler extension. Compare biased noise
   to the author surrogate only within its stated approximation; additional
   X/Y noise experiments must be labeled a changed noise model.
4. If transient width is expensive, propose support for this three-block
   Clifford measurement, with precomputed fault effects and contraction
   schedules. The present one-logical-qubit terminal interface is insufficient.

This tests whether the reduction generalizes beyond a transversal
single-qubit Clifford, and whether repeated checks destroy the advantage.
Neither a speedup nor bounded coherent-branch growth has been demonstrated.

## Hadamard-based distillation / verification

[Meier, Eastin, and Knill](https://arxiv.org/abs/1204.4221) give the 10-to-2
Hadamard-state distillation construction using a four-qubit error-detecting
code. Its Clifford-check viewpoint makes it a natural small independent
algebra test, though its small unencoded support is unlikely to challenge
ordinary Clifft. The published circuit must be reconstructed; this search
did not verify an author circuit export.

[Chamberland and Cross](https://arxiv.org/abs/1811.00566) and
[Chamberland and Noh](https://arxiv.org/abs/2003.03049) provide flagged
H-state preparation constructions. Controlled-H checks are closely related
to the existing gadget after a Clifford change of axes. Actual flags and
interleaved error correction make them a more demanding implementation test.

Do not mistake the
[MagicStatePrep repository](https://github.com/einsteinchris/MagicStatePrep/tree/5bcc7f767b3edd44ba6949120ce2271d281b23f2)
for a public simulator. Its sole file, `DataMagicStatePrepSim.nb`, contains
data import, fitting and overhead analysis. The
[2020 publication's code-availability statement](https://www.nature.com/articles/s41534-020-00319-5)
explicitly says the simulation code cannot be shared for proprietary reasons.
These are paper-reconstruction candidates, not ready circuit downloads.

## Other useful targets and misleading availability claims

- **Unfolded distillation:** the same Ruiz repository includes code geometry,
  CNOT schedules, noise and BP+OSD analysis. In the inspected repetition-code
  script, `rotations` identifies 15 sites but the emitted gate is `SQRT_X`.
  [Appendix E](https://arxiv.org/html/2507.12511v1#A5) explains the physical
  quarter-turn to Clifford half-turn substitution. Restore the physical
  rotations and validate decoding/output semantics before making a
  non-Clifford benchmark. Exact physical simulation is potentially useful,
  but the current paired-layer gadget does not directly apply.
- **Code switching:** [Daguerre and Kim](https://arxiv.org/abs/2410.07327)
  provide a different physical non-Clifford setting, using a transversal T
  layer and conversion to a 2D color code. It motivates a broader logical
  operation interface, not an assertion that the present recognizer works.
- **Logical factories:**
  [Gidney's factory scripts](https://github.com/Strilanc/magic-state-cultivation)
  include Reed-Muller and CCZ examples. Logical scripts and lattice-surgery
  descriptions are useful inputs, but are not automatically complete noisy
  physical circuits with validated acceptance and final logical evaluation.
- **Scheduling is not circuit simulation:** the
  [Lee preparation-cycle repository](https://github.com/seokhyung-lee/msd-magic-state-prep-cycle-simulation/tree/c5613fff9915904d8575b44ccd3438a440b7f0cb)
  accepts external success probabilities and simulates preparation timing.
  It is not a substitute for a physical non-Clifford circuit generator.

Recommendation: retain the complete-cultivation track; use 5-to-1 as an
available control, and investigate one physical bottom-up Toffoli check as
the next distinct gadget family. Gate structure and measured ordinary-Clifft
cost should determine further investment. No production architecture change,
new circuit vendoring, or non-Clifford reconstruction was performed here.

# Complete cultivation circuit acquisition

Search date: 2026-09-20. There is enough published detail to pursue a faithful
reconstruction instead of waiting for author replies. The strongest larger
target is Takada's fold-transversal surface-code family. This is a different
protocol from the generated unflagged color-code gadget benchmark.

Implementation update: the [f3 reconstruction](folded_msc.md) now runs
through final fictitious error detection, with independent Aer validation,
single-fault enumeration, and ordinary-Clifft timings. The report records
the unresolved stabilizer gate-order convention. The [f5 extension](folded_msc_f5.md)
also runs through FED, with validated growth, flagged cats, a large-cost
ordinary-Clifft baseline and coherent-Clifford history checks. The f7 stage
and decoder-based final error correction below remain planned work.

The [optimized-baseline prerequisite](scheduled_baseline.md) is complete:
the existing inputs were rerun with PR 471's explicitly enabled active-width
scheduler (PR 472 is its documentation), including unbounded search. It
roughly halves f5 sampling cost but leaves width 22 and its 64 MiB coefficient
array unchanged. The synthetic d7 gadget still wins by about 4.1x. The
complete native folded-check experiment below uses these stronger baselines;
the offline oracle remains a correctness reference, not the timed sampler.

The [native folded-check kernel](folded_contraction.md) now samples the two
checks and their X syndrome and returns coherent logical amplitudes. Both
complete-history probabilities and continuation phases pass f3/f5 audits.
Native fresh-fault binding, a compiler-certified prefix handoff, and the
complete-attempt comparison are now implemented in the
[complete native sampler](folded_protocol.md). The fresh-noise f5 sampler is
about 90-94 times faster than the explicitly enabled unbounded PR 471
baseline. The user selected f7 testing before fallback or consolidation.
The [complete f7 reconstruction and sampler](folded_protocol_f7.md) now
implement that priority, with verified growth/cats and fresh complete-history
checks. Automatic selection, a native f5-to-f7 handoff and plan-memory
improvements remain open. The earlier acquisition plan below is preserved
as the source and scope record.

## Primary reconstruction target

[Takada, Bartlett, and Williamson, Appendix A](https://arxiv.org/html/2609.16929v1#A1)
specifies fault-distance 3/5/7 injection, growth, and verification sequences
(Table 3), cat assignments (Table 4), injection and verified-cat circuits
(Figures 18/19), and the noise model. The f7 cat has eight data wires and six
verification wires; both uniform verification outcomes are accepted.
Three-qubit depolarization follows CCZ; idle noise is omitted. Sequential
syndrome extraction and fictitious error correction also need matching.

Dependencies are [Sahay et al.](https://arxiv.org/abs/2509.05212),
[Tsai and Puri's unitary encoder](https://arxiv.org/abs/2506.04084),
[Higgott et al.'s local growth construction](https://quantum-journal.org/papers/q-2021-08-05-517/),
and [Peham et al.'s verified cats](https://arxiv.org/abs/2601.03343).
The intended reconstruction includes injection through logical evaluation;
it does not add a later escape/storage stage.

Neither inspected arXiv source archive contains executable circuits:
2609.16929v1 has 24 files; 2609.19116v1 has 16. Both contain paper sources,
figures, and 00README.json but no .stim, .qasm, .py, .jl, or .ipynb files.
No exact public Takada or Hartweg circuit release was located in this search.
This is not proof that none exists: GitHub's exact Takada-ID code search
timed out twice, while repository/web searches returned no relevant release.

## Public repositories inspected

| Source | Pinned revision | What is actually available |
| --- | --- | --- |
| [Sahay: MSC_foldedH](https://github.com/kaavyas99/MSC_foldedH/tree/9378fd228ba83c592875d155136fcbcd16a45680) | `9378fd228ba83c592875d155136fcbcd16a45680` | f3 Y-state generator and f3/f5 Clifford sample circuits; useful scaffold, not the newer T-state artifacts |
| [Vaknin et al.: surface-code-magic-state-cultivation](https://github.com/tomirendo/surface-code-magic-state-cultivation/tree/ec28f6b95d7e7dad30f01e2ac8f37e9ed6c69dbc) | `ec28f6b95d7e7dad30f01e2ac8f37e9ed6c69dbc` | Gate-level non-Clifford circuit objects, state-vector interpreter, generators and Clifford simulation exports, including expansion stages |
| [Chen et al.: Cultiv_T_RP2](https://github.com/Zihan-Chen-PhMA/Cultiv_T_RP2/tree/77ad9b3dcc858ff9a5b92e80eca33d191aa16b3d) | `77ad9b3dcc858ff9a5b92e80eca33d191aa16b3d` | MSC3/MSC5 and end-to-end surface-code outputs; inspected T-named file is a Clifford proxy |
| [MQT QECC cat circuits](https://github.com/munich-quantum-toolkit/qecc/tree/ac7e1835e87204571afb3612e3d3a08e7d84cece/scripts/cat_states) | `ac7e1835e87204571afb3612e3d3a08e7d84cece` | Concrete verified-cat .stim files, construction data, and synthesis implementation |

The original Gidney color-code generator and corrected Chan d3/d5 circuits
remain the established small-protocol sources in the
[existing corpus study](cultivation_corpus.md). The SOFT d7 artifact retains
the previously documented failed cultivation sanity check.

### Export semantics matter

In the Vaknin repository, `2025/HCultivationSurfaceCode/simulation.py` defines
the actual gate objects and has a `to_stim_circuit` conversion. The generation
script frequently sets `apply_non_cliffords=False`. Even the true setting
uses a private encoding: `2025/StateVecSimulator/latte/vec_sim.py` maps T to
the spelling SQRT_Y, CCZ to SQRT_X_DAG, and CH to XCX, then interprets those
spellings as non-Clifford gates. Loading that private format directly into
ordinary Stim or Clifft would implement different gates. A proper adapter
must export the original operations and preserve noise placement and outputs.
Several preparation routines explicitly specialize to d2/d3/d5, so this is
not a verified arbitrary-distance full-protocol generator.

The RP2 sampling script calls `Y_to_rp3_sign_correction` and
`magic_Y_measurement_post_selection`; its `d3_color_T_check_compact` emits S
gates in the inspected implementation. Its circuit names alone do not imply
physical T simulation. A larger final surface-code distance also does not
imply cultivation was performed at that distance.

Fresh file inspections using unmodified Stim:

- Sahay `sample_circuits/fd5_base_new.stim`: qubit index span 415, 779
  measurements, 704 detectors, one observable; entirely Clifford/noise.
  SHA-256 `90801d17ba9d1aeaa1162fa2aa8c30aa700732eb90f350341a7cc806630702a1`.
- RP2 `circuit_garage/rp_3_rp_5_T_cult.stim`: qubit index span 263, 141
  measurements, 132 detectors, one observable; entirely Clifford/noise.
  SHA-256 `78260e20d9a8a28bc8719c70e1f7704ad954ad3530c5386a5d78d595c63c4802`.

These checks classify released files, not their physical T counterparts or
their logical-error-rate accuracy. No new third-party circuit was vendored.

### Usable cat preparation building block

MQT's [ft_ghz_8_4.stim](https://github.com/munich-quantum-toolkit/qecc/blob/ac7e1835e87204571afb3612e3d3a08e7d84cece/scripts/cat_states/circuits/ft_ghz_8_4.stim)
uses 14 qubits, of which six are measured for verification. SHA-256:
`0b34b9ec72b8d4850a4bed65f4401f2793adaa7ceec1cf45cb5e1fb1d8b6399f`.
For each uniform verification branch, exact Stim postselection confirms
all-data X and each Z0*Zj have expectation +1 on the remaining eight wires.
This is an ideal-state check, not a fault-distance proof or a claim that its
wire labeling exactly matches the paper's figure. Match the published
schedule and wiring before using it in the target reconstruction.

## Hartweg reconstruction is also possible

[Hartweg and Pineiro Orioli, Appendix A](https://arxiv.org/html/2609.19116v1#A1)
describes Reg3/Reg5 via Sahay primitives, growth, and noiseless unitary
decoding. Rot3/Rot5 have separately specified checks. Their SD6 scheduling,
idle noise, terminal decoding, and logical-error definition differ from
Takada's choices. Implementing one is not an exact reproduction of the other.
The author artifact remains especially valuable for matching those details
and the published performance comparison.

## Concrete next work

1. Reconstruct the f3 physical circuit from the specified sources. Keep an
   instruction-to-source map for injection, growth, syndrome extraction,
   verified cats, controlled logical checks, and final evaluation.
2. Check noiseless outputs and exact small physical branch probabilities;
   verify cat acceptance and the logical measurement as separate components.
   Compare acceptance and logical-error statistics only under matching noise
   and evaluation conventions.
3. Add the published f5 stage, then f7, retaining all physical noise sites.
   Treat downloaded author circuits as an independent comparison when they
   arrive. Do not claim a reconstructed protocol is byte-identical to them.
4. Profile ordinary Clifft before deciding whether and how to extend the
   current contraction path. These checks contain controlled-Clifford/CCZ
   structure beyond its present CNOT-parity recognizer.

A regular surface-code patch has d^2+(d-1)^2 data qubits: 13, 41, and 85 at
d3/d5/d7. Thus the final target also exceeds the prototype's 63-data-qubit
mask limit. Clifft's existing parser supports CCZ and DEPOLARIZE3, so circuit
generation can use the production dialect without inventing a new simulator.
No production architecture change or complete reconstructed protocol was
implemented during this acquisition search.

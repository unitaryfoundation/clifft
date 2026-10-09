# Review: clifft `codex/bt27-fault-specialization` / `research/conditional_clifford`

Date: 2026-10-09. Reviewed at branch tip `300b41e` (base PR 552 `ed2a125a`, clifft 0.11.1.dev11).
Measurements below were run on this VM with the Codex worktree's build
(branch `.venv`, `build-profile/` research targets), one thread, pinned
nowhere, 8–1024 shots. Scripts, circuits and raw rows are in `scripts/`, `circuits/`, `results/` next to this file; run the scripts from `circuits/` with the branch `.venv` and the opt-in `build-profile` targets.

## Verdict in three lines

1. The mathematics is right and the measured capability is real: it is the first clifft path that
   samples noisy BT81 / factory circuits at width ≤ 9 where ordinary clifft needs 33–87.
2. The packaging is wrong for a product mode: it rewrites a Stim circuit and recompiles per shot
   (4–53 ms/shot, 20 s setup on BT81). The per-shot work is avoidable, because the fault dependence is
   affine over GF(2) and the executor already evaluates per-lane sign expressions.
3. It applies to exactly the class clifft most needs: CNOT + diagonal-phase regions on code states
   (transversal CCZ/T factories, code switching, fold-transversal CS cultivation checks). It does not yet
   apply to Gidney-style cultivation, for a reason that is now identified (see §4).

## 1. What the prototype actually is

- Clifford prefix → affine computational-basis support `q = A x ⊕ b`. A CNOT + {T, S, Z, CZ, (CCZ)} region
  becomes a phase polynomial mod 8 on the free coordinates `x`.
- Pauli faults and measurement/reset outcomes only shift `b` and stabilizer signs. Complementing a variable
  of a Clifford+T phase polynomial changes it by a diagonal *Clifford* (Clifford-hierarchy level 3), so the
  non-Clifford core is history-independent and only S/CZ/Z/X corrections vary. Per shot: draw the full
  categorical fault history, evaluate packed GF(2) controls, render a reduced circuit with the original
  physical labels, hand it (plus the untouched continuation) to ordinary clifft.
- A one-rotation carrier transports a single surviving T through a Clifford/measured bridge. Regions must
  start from a stabilizer state; any non-diagonal gate (H, SQRT_X, CH) or non-deferrable measurement ends one.
- Size: ~3.8k lines of host Python across ten modules plus two opt-in native targets (`tools/profile/`);
  production `src/clifft` untouched. 33-input survey, 230 fixed-history audits, Aer/Stim checks on small
  controls; the large-circuit checks are 128-shot moment screens (gross errors only), as the docs say.

## 2. What is genuinely new versus production (measured)

Production at this base already has a state-aware phase pass (PR 545 TOHPE, PR 552 constraint recovery).
On the planar fold-transversal controlled-S check (§3) with the code state prepared by a **deterministic
unitary encoder**, production alone reaches width 2 at d=3 and d=5, even with two checks, a reused ancilla
and ancilla-mediated syndrome rounds between them. It fails on the same circuits when:

- the code state is prepared by **random-sign stabilizer projection** (MPP or CX/M ancilla rounds), i.e.
  signs are records rather than proven constants → width 7–11 (ideal);
- there is **any Pauli noise** → width 8 (d=3) / 17–18 (d=5), because the pass discards signed knowledge at
  every nonzero-probability fault.

So the prototype's real contribution is record-dependent signs and fault-dependent support offsets treated
as classical controls. The noise-free structural reduction is production, up to caps (noise-free BT81 stays
56 in production, 0/6 in the prototype).

## 3. New circuits from the monitoring list: fold-transversal CS cultivation checks

Unrotated planar surface code `[[d²+(d−1)², 1, d]]` with Moussa's fold: CS on the 2d−1 diagonal qubits,
CCZ (7-T decomposition) on the (n−2d+1)/2 mirror pairs, CZ sign fix from a GF(2) solve, verified with Stim
(all stabilizers preserved, X_L → Y_L). This is the structure of Sahay 2509.05212 / Caliper 2610.06828 and
of QPerfect's Reg3/Reg5 rows (clifft 0.7.0: k = 8 / 22). Variants: one or two checks, optional exact logical
T (CX-parity gadget on Z_L), MPP or ancilla (CX+M+R) syndrome rounds between checks, fresh or reused check
ancilla, p = 0 or 0.001 depolarizing.

| circuit (noisy p=0.001 unless noted) | T gates | ordinary width (dev build) | prototype width | final T | prototype ms/shot |
|---|---:|---:|---:|---:|---:|
| d=3 one check | 43 | 8 (7 ideal) | 1 | 3 | 0.6 |
| d=3 T + two checks, MPP bridge | 87 | 8 | 1 | 5 | 2.3 |
| d=3 T + two checks, reused ancilla | 87 | 8 | 1 | 5 | 2.7 |
| d=3 T + two checks, CX/M/R syndrome bridge | 87 | 8 | 2 | 5 | — |
| d=5 one check | 139 | 17 (10 ideal) | 1 | 3 | 1.1 |
| d=5 T + two checks, MPP bridge | 279 | 18 (11 ideal) | 1 | 7 | 4.0 |
| d=5 T + two checks, reused ancilla | 279 | 18 | 1 | 7 | 4.0 |
| d=5 T + two checks, CX/M/R syndrome bridge | 279 | 22 | 2 | 7 | — |
| Steane [[7,1,3]] controlled-S^⊗7 (form A) | 28 | 4 | 1–2 | 3 | 0.3 |

Cross-check: on the d=3 two-check circuits, 1024 prototype shots vs 1024 ordinary shots agree on all 103
record features (max |z| = 2.5, none above 3). Full rows: `results/planar_results.jsonl`, `results/xcheck.jsonl`.

Reading: the d² wall for fold-transversal controlled-S is a representation artefact, as suspected on
2026-09-17, and this prototype removes it; chaining across stabilizer rounds and a reused check ancilla
works because the second check's rotations still commute with the first region's core and production's pass
merges them once the reduced prefix is deterministic.

## 4. Where it stops: Gidney cultivation (clifft-bench msc d3/d5)

Both tools leave Gidney d=5 at width 10 (prototype: region 1 = the injection only, boundary at the next
`RX`; the plan dump shows width-4 `ROTATE` blocks per check at d=3). Gidney's check is written as
`T†^⊗7 · (CX/CZ to six ancillas, MX 7, RX 7) · T^⊗7`: two T layers separated by a non-diagonal Clifford and a
mid-check measurement. Syntactically that is not CNOT + diagonal and the two layers do not commute, so
neither the prototype's region model nor production's commuting-rotation collector can reduce it. Logically
it is a controlled Clifford (stabilizer nullity 2). Two ways forward:

- rewrite the sandwich: `T_q · CX_{a,q} · T†_q = CX_{a,q} · CS†_{a,q} · T_a`, so `D · C · D†` with `C` an
  ancilla-controlled X/Z layer becomes `C · D'` (CNOT + diagonal) and the existing reduction applies; the
  mid-check `MX 7 / RX 7` needs separate handling;
- or the controlled-Clifford / Pauli-dressed frame primitive from Brad's proposal (`C_b(|φ⟩_A ⊗ |0⟩_D)`),
  which handles non-diagonal controlled Cliffords directly.

Also outside scope for any phase-polynomial method: Vaknin's H_XY check (CH/CSWAP) and `T^⊗n` on a surface
code (no transversal T, the restricted polynomial keeps ~d²/2 independent non-Clifford terms).

## 5. The simpler extraction

Measured on the Steane toy and on noisy BT27 direct-X (10 histories incl. 6 faults): the reduced circuits
differ only in X/Z/CZ/S lines and record values; T targets and CX wiring are identical. In the T-parity
(Reed–Muller) form a complemented variable is a sign flip of the rotation, so every fault/record dependence
is a GF(2)-affine sign on a fixed rotation (degree-2 monomial corrections such as `Z(l3)` for a doubly
flipped CCZ term are linear in parity form). clifft's executor already carries per-lane sign expressions
(`ExecutePromotion.sign`, `ROTATE ... sign=eN`, `assign_symbol`). Therefore:

1. Move the algebra (affine support, pull-back, control responses: `shared_phase_specialization.py`) into
   the HIR phase pass as a *symbolic* constraint tracker: record outcomes and fault symbols become affine
   offsets/signs instead of "discard knowledge". Emit the core once with sign expressions; emit support
   offsets as Pauli-frame feedback. No per-shot rewrite, no fresh planning, no carrier: region k's entry is
   the frame state, support constraints survive diagonal gates, active coordinates are free variables.
2. Cost floor measured: the prototype's reduced BT27 circuit (width 6) batch-samples in ordinary clifft at
   1.6 µs/shot versus 6.9 ms/shot in the survey; Steane 0.1 µs vs 0.3 ms. Expect ≥1000× on the cases that
   today lose to merlin on setup.
3. Keep the prototype as the oracle: its per-shot renderer is an independent check of the static emission
   (compare record laws on the 33 retained inputs plus the planar set).

## 6. Does it warrant an opt-in mode?

As a host workflow: no. It loses to ordinary clifft on all 24 inputs ordinary can run, loses to merlin on
setup, and samples at ms/shot. As a static optimizer pass with symbolic signs: yes, and probably not even
opt-in, because the pass only fires on commuting diagonal regions over a stabilizer-constrained support and
otherwise leaves the HIR alone. The circuit class is the one clifft's users bring (STAR, Aachen, QPerfect
15-to-1, Navarra code switching, LPX/BT factories, Sahay/Caliper cultivation), and clifft's differentiators
survive: noise anywhere, readout flips, direct-X readouts merlin rejects, full record sampling, and later
loss/leakage through the same frame.

## 7. Other circuits to try next

- QPerfect 15-to-1 with the conditional logical-S gadget (QPerfect release qperfect-io/paper-mps-for-qec, Stim export in the arxiv-stim-clifft-monitor repo):
  the Stim export applies the gadget unconditionally because `S rec[...]` is inexpressible; the symbolic
  tracker handles record-controlled S natively (phase term `2·r·x`). Needs an input-language extension first.
- Navarra 2608.11160 App. E code switching and Bharti/Haug 2609.29890: small widths already (7, 5); only as
  correctness controls.
- Sahay/Caliper real proxy circuits with escape and postselection (Reg5 full protocol), to confirm the
  one-check result survives their growth schedule.
- Gidney cultivation after the sandwich rewrite of §4, as the test of the controlled-Clifford gap.
- Negative controls: Vaknin H_XY, `T^⊗n` on a d=5 surface code.

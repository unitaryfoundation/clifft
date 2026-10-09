# External review of the conditional Clifford prototype (2026-10-09)

[REVIEW.md](REVIEW.md) assesses the branch, measures the prototype on new circuits from the
near-Clifford monitoring list (planar fold-transversal controlled-S cultivation checks, Steane
controlled-S), tests whether the reduced core is history-invariant, and proposes a static
extraction into the production phase pass.

- `scripts/planar_fold.py` builds and Stim-verifies the fold-transversal logical S on the unrotated
  planar code and writes the circuit variants plus `results/ordinary_widths.json` (run inside `circuits/`).
- `scripts/unitary_prep.py` prepares the same checks with a deterministic encoder (production-only check).
- `scripts/run_planar.sh` runs the `ordinary` and `combined` workers on every variant.
- `scripts/diff_histories2.py <circuit.stim>` compares the prototype's reduced circuits across fault histories.
- `circuits/` are the exact inputs measured; `results/` are the raw worker rows and the summary table.

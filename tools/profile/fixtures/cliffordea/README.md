# Physical colour-code cultivation corpus

These four files are byte-for-byte copies from
[timchan0/cliffordea](https://github.com/timchan0/cliffordea/tree/a62a353e77468a2ff54c24e83133ad65a63e2294),
the authors' repository for [Chan et al., arXiv:2609.17706](https://arxiv.org/abs/2609.17706).
The accompanying MIT license is retained. `manifest.json` records the exact
upstream paths, revision and SHA-256 checksums. Do not edit the circuit files.

The files encode **S-state proxies**, not T-state cultivation. The study script
reproduces the authors' `make_variant_text` transformation: replace the physical
noise strength `0.001`, then replace instruction names `S` and `S_DAG` with `T`
and `T_DAG`. Gate order, measurements, feedback, idle noise, detector references,
and observable declarations remain unchanged. Derived T text has its own hash
in the measurements, and can be generated with `variant_text`.

`d3a6f2` and `d5a19f13` are the flagged variants. Both d5 files already include
the authors' Bell-growth feedforward and detector healing. Do not apply the
correction twice. See the pinned
[simulation guide](https://github.com/timchan0/cliffordea/blob/a62a353e77468a2ff54c24e83133ad65a63e2294/cliffordea/sim/README.md)
and [correction specification](https://github.com/timchan0/cliffordea/blob/a62a353e77468a2ff54c24e83133ad65a63e2294/cliffordea/sim/d5-correction.md).

The physical noise model is exactly the supplied instruction-level model,
including one- and two-qubit depolarization, Pauli errors and measurement
readout flips. All declared detectors are postselected. Observable 0 indicates
logical failure among accepted attempts. Terminal noiseless logical and code
measurements are included in both the circuit and the timing; this is
injection-and-cultivation evaluation, not an escape/decoding benchmark. Detector
normalization is not added by the study.

These are author-provided physical workloads, not the synthetic CSS controls
and not the surface-code Reg5 circuit from the CAMPS paper. The unflagged d5
file also has a different checksum from the fixture identified by Wan and
Zapirain; no byte-level equivalence or matching rare logical-error reference is
assumed here.

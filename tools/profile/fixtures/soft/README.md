# Unverified SOFT physical T workload

These files are byte-for-byte copies from
[SOFT revision 686051afe06e28c433fffec1c61686458728ca2e](https://github.com/haoliri0/SOFT/tree/686051afe06e28c433fffec1c61686458728ca2e).
The manifest pins paths and SHA-256 hashes. `LICENSE` is SOFT's distributed
Apache-2.0 license. `upstream_config.json` preserves the mapping of `msc_d7` to
this file with **postselection disabled**.

The file first appears in commit `77f1f388f4c2c5d6412c52259a8c9388677b8f9f`,
"sync symft with SOFT". The repository does not provide its generator or a
derivation validating this T circuit. Its benchmark README documents the
canonical d3/d5 files and a separate `msc_proxy_d7_unverified_p1e-3.stim`;
neither description certifies this file. The root license is retained as
distributed, without attributing the undocumented construction to other authors.

This is an extended Stim-like **physical T/T_DAG** input, not a Stim Clifford
circuit. It is distinct from both the named unverified proxy and the
fold-transversal surface-code circuits requested from Takada et al.

**Do not treat its distance label, detectors, or observable as validated
cultivation performance.** The [study](../../research/soft_cultivation.md)
finds a noiseless detector firing with exact probability 3/8. The S proxy has
a deterministic raw reference bit at detector 270, which Stim's default
detector sampler subtracts. The T variant has genuinely random noiseless
outputs; a fixed reference does not resolve this.

The study preserves the source, noise channels, records and terminal
measurement. Zero-noise and S-proxy copies are created only in memory or
temporary files. No detector is removed or repaired.

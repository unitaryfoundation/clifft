# Acknowledgments and References

Clifft has benefited from open research and software that shaped its algorithms,
interfaces, and physical models. We acknowledge the following work and describe
how it contributed to Clifft.

## Stim

Stim's circuit format and compile-once, sample-many interfaces shaped Clifft's
frontend and sampling APIs. Clifft originally used Stim's C++ tableau
implementation; current production builds use Clifft's native Clifford
implementation. Stim continues to serve as an independent reference for
correctness tests and as a source of QEC circuit fixtures.

- Craig Gidney, ["Stim: a fast stabilizer circuit simulator"](https://doi.org/10.22331/q-2021-07-06-497),
  *Quantum* 5, 497 (2021).
- [Stim source](https://github.com/quantumlib/Stim).

## SOFT and SymFT

SOFT's generalized-stabilizer simulation and released magic-state cultivation
circuits informed Clifft's development and provided workloads for its examples
and benchmarks.

SymFT, the second-generation successor to SOFT, combines SOFT's
generalized-stabilizer simulation with Clifft's original dense active-state
representation. Clifft's current sampler adopts SymFT's symbolic
Clifford-Pauli-frame factorization, adaptive stabilizer-coordinate planning, and
direct multi-coordinate kernels. SymFT's packed cross-shot execution also
inspired Clifft's packed sampling path.

The [method provenance](theory/overview.md#method-provenance) describes this
lineage and the implementation boundaries. The [symbolic sampling](updates/symbolic-sampling.md)
and [packed sampling](updates/packed-sampling.md) updates explain the adaptations.

- Riling Li et al., ["SOFT: a high-performance simulator for universal fault-tolerant quantum circuits"](https://arxiv.org/abs/2512.23037)
  (2025).
- Wang Fang, Huazhe Lou, and Riling Li, ["SymFT: Universal Fault-Tolerant Quantum Circuit Simulation via Symbolic Clifford-Pauli Frames and Stabilizer Coordinates"](https://arxiv.org/abs/2607.28600)
  (2026).
- [SOFT and SymFT source](https://github.com/haoliri0/SOFT).

## Tsim

Tsim's Stim-compatible approach to universal noisy-circuit simulation informed
Clifft's circuit-format and interface design. Its open code and data enabled
direct comparisons in Clifft's original performance study.

- Rafael Haenel, Xiuzhe Luo, and Chen Zhao, ["Tsim: Fast Universal Simulator for Quantum Error Correction"](https://arxiv.org/abs/2604.01059)
  (2026).
- [Tsim source](https://github.com/QuEraComputing/tsim).

## SqaleSim

Infleqtion's SqaleSim motivated Clifft's leakage/loss instrument design and its
five-level neutral-atom modeling. The public schedules and noise model from the
Sqale study underpin Clifft's [logical Shor noise-sweep tutorial](guide/neutral-atom-leakage.md).
That tutorial compares matched model assumptions with Clifft's exact conditional
back-action.

- Rines et al., ["Demonstration of a Logical Architecture Uniting Motion and In-Place Entanglement"](https://arxiv.org/abs/2509.13247).
- [Public supplementary artifact](https://zenodo.org/records/17137995), including
  the circuits and noise model used for Figure 9.

## deltakit-stim

Riverlane's deltakit-stim inspired Clifft's configurable leakage partner effects,
including partner noise and leakage spreading. The
[partner-interactions guide](guide/partner-interactions.md) describes Clifft's
model and API.

- [deltakit-stim source and documentation](https://github.com/Deltakit/deltakit-stim).

## TOHPE

Clifft's phase-polynomial optimizer implements Vivien Vandaele's TOHPE algorithm
for reducing T count. The author's Rust implementation served as a validation
reference. Clifft supplies the surrounding state constraints, non-Clifford core
extraction, and Clifford correction.

- Vivien Vandaele, ["Lower T-count with faster algorithms"](https://doi.org/10.22331/q-2025-09-16-1860),
  *Quantum* 9, 1860 (2025), Algorithm 2.
- [Author's implementation](https://github.com/VivienVandaele/quantum-circuit-optimization),
  with the [revision used for validation](https://github.com/VivienVandaele/quantum-circuit-optimization/blob/231e6fe9f92d5bb1ebf7459c2a9233f5e74d148e/src/t_opt.rs#L79).

## MQT Bench

Clifft's fixed-input 32-qubit Draper QFT adder fixture was generated with
MQT Bench. It provides an arithmetic workload for validating and benchmarking
arbitrary-angle rotation simplification, as described in
[Smaller Active States in Clifft](updates/smaller-active-states.md#simplify-rotations-using-the-prepared-state).

- Nils Quetschlich, Lukas Burgholzer, and Robert Wille, ["MQT Bench: Benchmarking Software and Design Automation Tools for Quantum Computing"](https://doi.org/10.22331/q-2023-07-20-1062),
  *Quantum* 7, 1062 (2023).
- [MQT Bench source](https://github.com/munich-quantum-toolkit/bench).

## Pauli Frame Sparse Representation

Thomas Tuloup and Thomas Ayral's work informed Clifft's stratified
importance-sampling approach to rare-event estimation. Clifft's
[importance-sampling tutorial](guide/importance-sampling-tutorial.md) follows
their cultivation analysis by conditioning on fault count and weighting each
stratum by its probability.

- Thomas Tuloup and Thomas Ayral, ["Computing logical error thresholds with the Pauli Frame Sparse Representation"](https://arxiv.org/abs/2603.14670)
  (2026).

## Magic-state cultivation

The magic-state cultivation protocol and its released circuits, data, and
analysis tools underpin Clifft's cultivation examples and benchmarks. Clifft
uses artifacts from the original work as well as T-gate circuits distributed
with SOFT; the fixture headers and tutorials record their individual sources.

- Craig Gidney, Noah Shutty, and Cody Jones, ["Magic state cultivation: growing T states as cheap as CNOT gates"](https://arxiv.org/abs/2409.17595)
  (2024).
- [Code and circuits](https://github.com/Strilanc/magic-state-cultivation).
- [Published data](https://doi.org/10.5281/zenodo.13777072).

## xoshiro256++

Clifft uses David Blackman and Sebastiano Vigna's xoshiro256++ random-number
generator, seeded with SplitMix64, for sampling. The implementation follows the
public reference code, providing a small per-shot state and deterministic
integer output across platforms.

- David Blackman and Sebastiano Vigna, ["Scrambled Linear Pseudorandom Number Generators"](https://doi.org/10.1145/3460772),
  *ACM Transactions on Mathematical Software* 47(4), Article 36 (2021).
- [xoshiro256++ reference implementation](https://prng.di.unimi.it/xoshiro256plusplus.c)
  and [SplitMix64 seeding implementation](https://prng.di.unimi.it/splitmix64.c).

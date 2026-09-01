# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.9.0] - 2026-08-24

Clifft 0.9.0 adds parallel sampling across ordinary, post-selected, forced-fault, and noncomputational workloads. A single `threads` budget can spread work across independent shots, or, for some large `k`/active-width workloads, use OpenMP within each shot. Expert callers can select an explicit hybrid layout.

The release also defines exact statevectors projectively, up to global phase. That simpler contract lets the compiler absorb Clifford-valued rotations earlier and remove global-phase bookkeeping, while new SIMD measurement kernels accelerate important active-state paths. See [CPU Execution and Tuning](https://unitaryfoundation.github.io/clifft/stable/guide/cpu-execution/) for the threading model, resource tradeoffs, and advanced controls.

### Added

- Added parallel sampling to `sample()`, `sample_survivors()`, `sample_k()`, `sample_k_survivors()`, and `noncomp.sample()`. `threads` accepts a positive worker budget or `"auto"`; fixed-plan sampling automatically chooses cross-shot or intra-shot execution, while `thread_layout` and `intra_shot_min_active_width` provide expert control. This stack preserves seeded results across worker layouts and keeps noncomputational result sidecars in deterministic row order, by @bachase in [#352](https://github.com/unitaryfoundation/clifft/pull/352), [#354](https://github.com/unitaryfoundation/clifft/pull/354), [#355](https://github.com/unitaryfoundation/clifft/pull/355), and [#379](https://github.com/unitaryfoundation/clifft/pull/379).

### Changed

- **Breaking:** `threads` now represents a total sampling worker budget and may select intra-shot OpenMP execution for wide, undersubscribed fixed-plan workloads. Use `thread_layout=(shot_workers, 1)` to preserve cross-shot-only scheduling. Seeded ordinary and fixed-fault sampling now derives each shot from the call seed and global shot index, so rows differ from v0.8 for the same seed even though v0.9 results are reproducible across worker layouts, by @bachase in [#352](https://github.com/unitaryfoundation/clifft/pull/352) and [#379](https://github.com/unitaryfoundation/clifft/pull/379).
- **Breaking:** `get_statevector()` now returns a normalized representative of the final state ray and no longer guarantees the source matrix's global phase. Compare statevectors up to global phase, or compare probabilities or fidelity. Building on this contract, the compiler now removes global-phase bookkeeping and absorbs Clifford-valued axis and Pauli-product rotations during tracing and peephole fusion, avoiding unnecessary actions and active-width growth, by @bachase in [#369](https://github.com/unitaryfoundation/clifft/pull/369), [#372](https://github.com/unitaryfoundation/clifft/pull/372), [#378](https://github.com/unitaryfoundation/clifft/pull/378), and [#383](https://github.com/unitaryfoundation/clifft/pull/383).

### Performance

- Vectorized high-pivot and diagonal active-measurement probability and collapse paths for AVX2 and AVX-512 while retaining scalar fallbacks, by @bachase in [#374](https://github.com/unitaryfoundation/clifft/pull/374).

### Fixed

- Fixed `basis_probabilities()` for outcomes where multiple active coordinates interfere with complex relative weights. Both the Gray-code and fallback paths now conjugate the inverse-frame expansion correctly, with analytic and randomized independent-reference regressions, by @bachase in [#381](https://github.com/unitaryfoundation/clifft/pull/381).

### CI

- Consolidated the post-0.8 CI work into a stricter release-confidence stack: deterministic AVX-512 execution under Intel SDE, Linux arm64 coverage, the full Python suite against optimized builds, nightly ASan and UBSan, abi3 verification on the newest stable CPython, bounded tool setup and jobs, an aggregate merge gate, packaging round trips on every pull request, and a shorter dedicated benchmark-history workflow, by @bachase in [#353](https://github.com/unitaryfoundation/clifft/pull/353), [#356](https://github.com/unitaryfoundation/clifft/pull/356), [#357](https://github.com/unitaryfoundation/clifft/pull/357), [#358](https://github.com/unitaryfoundation/clifft/pull/358), [#359](https://github.com/unitaryfoundation/clifft/pull/359), [#363](https://github.com/unitaryfoundation/clifft/pull/363), [#364](https://github.com/unitaryfoundation/clifft/pull/364), [#365](https://github.com/unitaryfoundation/clifft/pull/365), [#368](https://github.com/unitaryfoundation/clifft/pull/368), [#370](https://github.com/unitaryfoundation/clifft/pull/370), and [#373](https://github.com/unitaryfoundation/clifft/pull/373).

### Documentation

- Corrected the arXiv badge identifier and made documentation links stable across versioned deployments, by @bachase in [#351](https://github.com/unitaryfoundation/clifft/pull/351) and [#362](https://github.com/unitaryfoundation/clifft/pull/362).

## [0.8.0] - 2026-08-18

Clifft 0.8.0 replaces the original localized-Pauli Schrodinger virtual machine (SVM) with a symbolic-coordinate compiler and sampler. Drawing on the recent [SymFT paper](https://arxiv.org/abs/2607.28600) and [reference implementation](https://github.com/haoliri0/SOFT), the new backend combines symbolic Clifford-Pauli-frame factorization and adaptive stabilizer-coordinate planning with Clifft's [original factored active-state representation](https://arxiv.org/abs/2604.27058). Its planner resolves Clifford coordinates, affine Pauli-frame effects, active Pauli shapes, and symbolic dependencies ahead of time, while prepared scalar, AVX2, and AVX-512 kernels apply multi-coordinate operations directly. The usual `compile()` and sampling APIs retain their output contracts, and the new path also powers exact state queries and leakage/loss continuations.

Across the release benchmark corpus, the new sampler is faster than the SVM on five of seven real workloads and remains close on the other two. See [Symbolic Sampling in Clifft](https://unitaryfoundation.github.io/clifft/stable/updates/symbolic-sampling/) for the design, API migration guide, matched benchmark results, and method provenance.

### Added

- Added `Program.inspect()` and `Program.inspect_action()` for diagnostic views of prepared sampling plans. The playground now presents the same symbolic plan with circuit-source provenance.
- Added generalized Pauli-product phase gates: Clifford `SPP` / `SPP_DAG` and non-Clifford `TPP` / `TPP_DAG` by @danielgaskins in [#333](https://github.com/unitaryfoundation/clifft/pull/333).
- Added the `LEAKAGE(p)` circuit annotation for source-preserving transitions from `g` to `leak_g` and `e` to `leak_e` by @bachase in [#244](https://github.com/unitaryfoundation/clifft/pull/244).
- Added `Program.peak_active_width` as the canonical name for the largest dense active-state width. `Program.peak_rank` remains as a deprecated alias for this release, by @bachase in [#338](https://github.com/unitaryfoundation/clifft/pull/338).

### Changed

- `clifft.compile()` now returns the symbolic sampling `Program`. It is an opaque, reusable compiled program rather than an iterable bytecode module; `num_actions` replaces `num_instructions` for diagnostic action counts.
- `clifft.noncomp.sample(..., max_rank=...)` is now `clifft.noncomp.sample(..., max_active_width=...)`.
- The default optimizer removes rotations whose phase is unobservable at a later terminal measurement, including safe cases across resets, disjoint conditional corrections, noise, and classical bookkeeping, by @bachase in [#239](https://github.com/unitaryfoundation/clifft/pull/239) and [#243](https://github.com/unitaryfoundation/clifft/pull/243).
- A fixed seed remains reproducible for a fixed version and call, but sampled rows need not match v0.7 because the new executor has a different random number schedule.

### Removed

- Removed the SVM, public `Opcode`, `Instruction`, bytecode `Program` iteration and inspection helpers (`source_map`, `active_k_history`, and `as_dict()`), bytecode pass APIs, and the `bytecode_passes` argument to `compile()`. Use `Program.inspect()` for diagnostic plan text; HIR passes remain the supported compiler-customization boundary.
- Removed the mutable `State` / `execute()` inspection workflow. Use `get_statevector(program)` for eligible final states.
- Removed the SVM backend and OpenMP controls: `svm_backend()`, `get_num_threads()`, and `set_num_threads()`. Parallel sampling and intra-shot parallel kernels are tracked as future work.

### Fixed

- Reject invalid inverted targets instead of accepting inversion on operations where it has no defined meaning, by @bachase in [#241](https://github.com/unitaryfoundation/clifft/pull/241).
- Harden noncomputational continuation planning and reuse its storage without changing trajectory semantics.

## [0.7.0] - 2026-07-31

Clifft 0.7.0 adds experimental simulation of leakage and loss through the new `clifft.noncomp` Python API. It samples trajectories across two computational levels, two leaked levels, and loss, with state-dependent transitions, state-selective measurement, and back-action on the computational state. Transitions can be attached to gates or placed explicitly in the circuit, and results include measurement records, detector and observable values, heralds, and final per-qubit status.

The [Leakage and Loss guide](https://unitaryfoundation.github.io/clifft/stable/guide/leakage-and-loss/) introduces the model and API. The [Delayed Loss tutorial](https://unitaryfoundation.github.io/clifft/stable/guide/delayed-loss/) applies it to a surface-code memory experiment, where losses of the same data qubit at different times produce the same final herald but different detector histories.

### CI

- pin NumPy for mypy hook (#221) by @bachase in [#221](https://github.com/unitaryfoundation/clifft/pull/221)

### Documentation

- add delayed-loss surface-code tutorial (#233) by @bachase in [#233](https://github.com/unitaryfoundation/clifft/pull/233)

### Features

- add leakage and loss simulation (#231) by @bachase in [#231](https://github.com/unitaryfoundation/clifft/pull/231)

### Testing

- align and simplify unit coverage (#230) by @bachase in [#230](https://github.com/unitaryfoundation/clifft/pull/230)

## [0.6.0] - 2026-07-14

Clifft 0.6.0 adds Stim-compatible correlated Pauli noise chains through `CORRELATED_ERROR` / `E` and `ELSE_CORRELATED_ERROR`. It also adds a dedicated [front-end integrations guide](https://unitaryfoundation.github.io/clifft/stable/getting-started/integrations/) for using Clifft from Qiskit and Cirq through their separately maintained companion packages.

### Documentation

- add front-end integration guide (#182) by @bachase in [#182](https://github.com/unitaryfoundation/clifft/pull/182)

### Features

- add correlated Pauli noise channels (#158) by @bachase in [#158](https://github.com/unitaryfoundation/clifft/pull/158)

## [0.5.0] - 2026-06-15

Clifft 0.5.0 broadens the circuits users can write directly and makes the project easier to try from both the playground and Python docs. The parser now accepts common controlled gates (`CCZ`, `CCX`, `CH`) through exact Clifford+T rewrites, the simulator supports three-qubit Pauli noise channels (`DEPOLARIZE3`), and phase-sensitive statevector behavior is better tested and documented.

This release also includes several [unitaryHACK 2026](https://unitaryhack.dev/) contributions: @Samfresh-ai added starter circuits to the playground, @manasa-manoj-nbr added an auto-generated Python API reference, and @ashmitjsg completed the Qiskit backend bounty as the companion [unitaryfoundation/clifft-qiskit](https://github.com/unitaryfoundation/clifft-qiskit) package, so Qiskit users can run supported circuits on Clifft without hand-writing Stim.

### Bug Fixes

- preserve global phase across tableau rewrites (#140) by @bachase in [#140](https://github.com/unitaryfoundation/clifft/pull/140)

### Build

- hide vendored Stim symbols (#131) by @bachase in [#131](https://github.com/unitaryfoundation/clifft/pull/131)

### CI

- enforce uv lockfile in workflows (#128) by @bachase in [#128](https://github.com/unitaryfoundation/clifft/pull/128)
- skip PR preview cleanup for forks (#122) by @bachase in [#122](https://github.com/unitaryfoundation/clifft/pull/122)
- replace stripped-binary audit with QEMU wheel smoke (#100) by @bachase in [#100](https://github.com/unitaryfoundation/clifft/pull/100)

### Documentation

- clarify statevector phase checks (#153) by @bachase in [#153](https://github.com/unitaryfoundation/clifft/pull/153)
- auto-generate Python API reference via mkdocstrings (#121) by @manasa-manoj-nbr in [#121](https://github.com/unitaryfoundation/clifft/pull/121)
- clarify Stim gate descriptions (#110) by @bachase in [#110](https://github.com/unitaryfoundation/clifft/pull/110)

### Features

- add parser rewrites for controlled gates (#151) by @bachase in [#151](https://github.com/unitaryfoundation/clifft/pull/151)
- add DEPOLARIZE3 and PAULI_CHANNEL_3 support (#149) by @bachase in [#149](https://github.com/unitaryfoundation/clifft/pull/149)
- add starter circuit catalog (#118) by @Samfresh-ai in [#118](https://github.com/unitaryfoundation/clifft/pull/118)

### Testing

- widen componentwise global-phase coverage (#145) by @bachase in [#145](https://github.com/unitaryfoundation/clifft/pull/145)
- add optimization pass docs drift test (#101) by @puneetdixit200 in [#101](https://github.com/unitaryfoundation/clifft/pull/101)

## [0.4.1] - 2026-05-19

This patch makes Linux wheel CPU targeting consistent and portable across Clifft and its dependencies. It fixes a build configuration issue where host-specific CPU settings could leak into binaries, potentially producing unsupported instructions on some Linux x86_64 systems.

### Bug Fixes

- set stim's SIMD_WIDTH so libstim respects the wheel baseline (#95) by @bachase in [#95](https://github.com/unitaryfoundation/clifft/pull/95)
- tighten AVX-2 dispatch and trap CLIFFT_FORCE_ISA misconfig (#94) by @bachase in [#94](https://github.com/unitaryfoundation/clifft/pull/94)

### CI

- fix Windows configure command (#98) by @bachase in [#98](https://github.com/unitaryfoundation/clifft/pull/98)
- align x86 wheel baseline with runtime dispatch (#97) by @bachase in [#97](https://github.com/unitaryfoundation/clifft/pull/97)

## [0.4.0] - 2026-05-15

This release expands strong simulation with a new `clifft.record_probabilities()` API that returns the joint probability `sample()` would assign to a given measurement record (or batch of records). Combined with the existing basis-probability path, Clifft now answers two complementary "what's the exact probability of …" questions: bitstring outcomes for unitary circuits, and measurement-record outcomes for circuits that contain measurements with or without classical feedback. See the [Exact Probabilities guide](https://unitaryfoundation.github.io/clifft/stable/guide/strong-simulation/) for both APIs side-by-side and more detail on when to choose one versus another.

The probability surface is also faster and clearer. `basis_probabilities()` is roughly 17× faster on representative inputs via a Gray-code walk over X-generators when the dormant block reduces cleanly during RREF. As part of unifying the docs, the two queries were renamed to make the queried object explicit:

  - `clifft.probabilities()` → `clifft.basis_probabilities()`
  - the newly-added `clifft.record_probabilities()`

There are no backward-compatibility aliases; callers must rename `basis_probabilities()`.

Beyond the probability work, 0.4.0 lands a scheduled benchmark-history workflow that records Catch2 and pytest-benchmark results daily ([charts](https://unitaryfoundation.github.io/clifft/bench/cpp/)), versioned docs deployments via `mike`, and a compile-time speedup on QEC circuits.

### CI

- add scheduled benchmark history workflow (#38) (#70) by @bachase in [#70](https://github.com/unitaryfoundation/clifft/pull/70)
- serialize docs preview deployments (#86) by @bachase in [#86](https://github.com/unitaryfoundation/clifft/pull/86)
- make MSVC debug builds ccache-friendly (#68) by @bachase in [#68](https://github.com/unitaryfoundation/clifft/pull/68)

### Documentation

- add versioned docs deployment (#85) by @bachase in [#85](https://github.com/unitaryfoundation/clifft/pull/85)

### Features

- rename probability APIs and unify strong-simulation docs (#84) by @bachase in [#84](https://github.com/unitaryfoundation/clifft/pull/84)
- add probability_of Python API (#82) by @bachase in [#82](https://github.com/unitaryfoundation/clifft/pull/82)
- add probability_of C++ entry point (#80) by @bachase in [#80](https://github.com/unitaryfoundation/clifft/pull/80)
- wire forced measurement opcodes into the SVM dispatcher (#79) by @bachase in [#79](https://github.com/unitaryfoundation/clifft/pull/79)
- add forced-outcome measurement kernels (#78) by @bachase in [#78](https://github.com/unitaryfoundation/clifft/pull/78)
- add forced-execution state fields to SchrodingerState (#77) by @bachase in [#77](https://github.com/unitaryfoundation/clifft/pull/77)
- add forced-measurement opcodes to the Opcode enum (#76) by @bachase in [#76](https://github.com/unitaryfoundation/clifft/pull/76)

### Performance

- ~17x faster probabilities() via gray-code walk over X-generators (#73) by @bachase in [#73](https://github.com/unitaryfoundation/clifft/pull/73)
- add profile_probability harness for strong-simulation profiling (#72) by @bachase in [#72](https://github.com/unitaryfoundation/clifft/pull/72)
- cut compile time on QEC circuits via heap-alloc fix and eager V_cum flush (#67) by @bachase in [#67](https://github.com/unitaryfoundation/clifft/pull/67)

### Testing

- cover SchrodingerState::reset() reuse semantics (#75) by @bachase in [#75](https://github.com/unitaryfoundation/clifft/pull/75)

## [0.3.0] - 2026-05-08

This release adds strong simulation for unitary circuits through exact computational-basis probability queries of the factored state. The new `clifft.probabilities()` API evaluates selected bitstrings without materializing the full $2^n$ statevector, so sparse output queries can scale with active width rather than output-space size. See the [strong-simulation tutorial](https://unitaryfoundation.github.io/clifft/stable/guide/strong-simulation/) for examples.

Clifft 0.3.0 also removes the old compile-time qubit ceiling by moving Pauli mask storage to runtime-width arenas. That fixed-width limit kept the early implementation simple and fast; the new arena path keeps the overhead localized while supporting circuits above the former inline-width bound.

The release also improves performance in the playground for larger circuits. The prior playground had some pauses when unnecessarily re-rendering the current program counter line in the active-width timeline.

### Bug Fixes

- version playground wasm assets (#57) to ensure users load the latest playground code by @bachase in [#57](https://github.com/unitaryfoundation/clifft/pull/57)


### Features

- add exact full-bitstring probabilities (#60) by @bachase in [#60](https://github.com/unitaryfoundation/clifft/pull/60)
- runtime-width SVM Pauli frame, drop kMaxInlineQubits ceiling (#53) by @bachase in [#53](https://github.com/unitaryfoundation/clifft/pull/53)
- migrate AOT-side Pauli mask storage to runtime-width arenas (#52) by @bachase in [#52](https://github.com/unitaryfoundation/clifft/pull/52)
- add runtime-width Pauli mask views and arena (#49) by @bachase in [#49](https://github.com/unitaryfoundation/clifft/pull/49)

### Performance

- decouple K-history highlight via recharts hooks (#59) by @bachase in [#59](https://github.com/unitaryfoundation/clifft/pull/59)
- O(1) cursor highlight via reverse source maps (#56) by @bachase in [#56](https://github.com/unitaryfoundation/clifft/pull/56)

### Testing

- cover introspection formatters (#61) by @bachase in [#61](https://github.com/unitaryfoundation/clifft/pull/61)
- add baseline benchmarks for runtime-qubit migration (#47) by @bachase in [#47](https://github.com/unitaryfoundation/clifft/pull/47)

## [0.2.0] - 2026-05-01

Version 0.2.0 of clifft is primarily a cleanup release to coincide with the release of the clifft [preprint](https://arxiv.org/abs/2604.27058) on the arXiv. There are no major functionality changes or fixes.

### Bug Fixes

- refresh uv lockfile (#32) by @bachase in [#32](https://github.com/unitaryfoundation/clifft/pull/32)
- silence libomp false positives via ignore_noninstrumented_modules (#30) by @bachase in [#30](https://github.com/unitaryfoundation/clifft/pull/30)
- tolerance-based EXP_VAL check for OpenMP determinism test (#20) by @bachase in [#20](https://github.com/unitaryfoundation/clifft/pull/20)

### Documentation

- add links to arXiv paper (#42) by @bachase in [#42](https://github.com/unitaryfoundation/clifft/pull/42)
- use docs group for mkdocs recipes (#35) by @bachase in [#35](https://github.com/unitaryfoundation/clifft/pull/35)
- add performance summaries to README and doc page (#31) by @bachase in [#31](https://github.com/unitaryfoundation/clifft/pull/31)
- make README URLs absolute so PyPI renders correctly (#29) by @bachase in [#29](https://github.com/unitaryfoundation/clifft/pull/29)
- brand logos, color scheme, and Unitary Foundation attribution (#16) by @bachase in [#16](https://github.com/unitaryfoundation/clifft/pull/16)
- align terminology and exposition with the paper draft (#12) by @bachase in [#12](https://github.com/unitaryfoundation/clifft/pull/12)
- drop "RISC" terminology in favor of "VM bytecode" (#14) by @bachase in [#14](https://github.com/unitaryfoundation/clifft/pull/14)
- correct OP_FRAME to mutate the virtual Pauli frame, not U_C (#11) by @bachase in [#11](https://github.com/unitaryfoundation/clifft/pull/11)

### Features

- highlight target panes during guided tour (#17) by @bachase in [#17](https://github.com/unitaryfoundation/clifft/pull/17)
- run default optimization passes by default in compile() (#15) by @bachase in [#15](https://github.com/unitaryfoundation/clifft/pull/15)
- load-from-URL button and origin-aware Share (#10) by @bachase in [#10](https://github.com/unitaryfoundation/clifft/pull/10)

### Refactoring

- move reference_syndrome from backend/ to api/ (#26) by @bachase in [#26](https://github.com/unitaryfoundation/clifft/pull/26)
- rename OP_PHASE_* bytecode opcodes to OP_ARRAY_* (#13) by @bachase in [#13](https://github.com/unitaryfoundation/clifft/pull/13)

## [0.1.0] - 2026-04-16

### Bug Fixes

- add pretend version for TestPyPI manual dispatch (#8) by @bachase in [#8](https://github.com/unitaryfoundation/clifft/pull/8)
- remove unused setup-uv from wheel jobs (#7) by @bachase in [#7](https://github.com/unitaryfoundation/clifft/pull/7)
- use manylinux_2_28 for Linux wheel builds (#6) by @bachase in [#6](https://github.com/unitaryfoundation/clifft/pull/6)
- set MACOSX_DEPLOYMENT_TARGET for macOS arm64 wheel (#5) by @bachase in [#5](https://github.com/unitaryfoundation/clifft/pull/5)
- update playground links after docs page removal (#4) by @bachase in [#4](https://github.com/unitaryfoundation/clifft/pull/4)
- playground link serves docs page instead of SPA (#3) by @bachase in [#3](https://github.com/unitaryfoundation/clifft/pull/3)

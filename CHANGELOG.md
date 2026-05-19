# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.1] - 2026-05-19

This patch makes Linux wheel CPU targeting consistent and portable across Clifft and its dependencies. It fixes a build configuration issue where host-specific CPU settings could leak into binaries, potentially producing unsupported instructions on some Linux x86_64 systems.

### Bug Fixes

- pin stim's SIMD_WIDTH so libstim respects the wheel baseline (#95) by @bachase in [#95](https://github.com/unitaryfoundation/clifft/pull/95)
- tighten AVX-2 dispatch and trap CLIFFT_FORCE_ISA misconfig (#94) by @bachase in [#94](https://github.com/unitaryfoundation/clifft/pull/94)

### CI

- fix Windows configure command (#98) by @bachase in [#98](https://github.com/unitaryfoundation/clifft/pull/98)
- align x86 wheel baseline with runtime dispatch (#97) by @bachase in [#97](https://github.com/unitaryfoundation/clifft/pull/97)

## [0.4.0] - 2026-05-15

This release expands strong simulation with a new `clifft.record_probabilities()` API that returns the joint probability `sample()` would assign to a given measurement record (or batch of records). Combined with the existing basis-probability path, Clifft now answers two complementary "what's the exact probability of …" questions: bitstring outcomes for unitary circuits, and measurement-record outcomes for circuits that contain measurements with or without classical feedback. See the [strong-simulation tutorial](https://unitaryfoundation.github.io/clifft/guide/strong-simulation/) for both APIs side-by-side and more detail on when to choose one versus anothr.

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

This release adds strong simulation for unitary circuits through exact computational-basis probability queries of the factored state. The new `clifft.probabilities()` API evaluates selected bitstrings without materializing the full $2^n$ statevector, so sparse output queries can scale with active rank rather than output-space size. See the [strong-simulation tutorial](https://unitaryfoundation.github.io/clifft/guide/strong-simulation/) for examples.

Clifft 0.3.0 also removes the old compile-time qubit ceiling by moving Pauli mask storage to runtime-width arenas. That fixed-width limit kept the early implementation simple and fast; the new arena path keeps the overhead localized while supporting circuits above the former inline-width bound.

The release also improves performance in the playground for larger circuits. The prior playground had some pauses when unnecessarily re-rendering the current program counter line in the active dimensions timeline.

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

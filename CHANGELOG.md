# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

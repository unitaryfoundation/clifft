# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2025-04-16

Initial preview release of Clifft.

### Features

- Multi-level AOT compiler: parser, HIR frontend, optimizer passes, bytecode backend
- Schrodinger Virtual Machine (SVM) with AVX2/AVX-512/scalar backends
- Stim circuit format support (parse and simulate)
- Importance sampling for near-Clifford circuits
- Python bindings via nanobind (`import clifft`)
- HIR and bytecode optimization passes (peephole fusion, single-axis fusion, statevector squeeze, noise block, expand-T, and more)
- Expectation value computation
- WebAssembly build for browser-based Playground
- MkDocs documentation site

# Python API Reference

This page documents the public Python API surface of `clifft`. The reference is auto-generated
from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

For usage guides and tutorials, see the [User Guide](../guide/compilation.md).

## Compilation

Functions for parsing Stim-format circuits and compiling them to executable bytecode.

::: clifft.compile
    options:
      show_source: true
      members: []

::: clifft.parse
    options:
      show_source: true
      members: []

::: clifft.parse_file
    options:
      show_source: true
      members: []

::: clifft.trace
    options:
      show_source: true
      members: []

::: clifft.lower
    options:
      show_source: true
      members: []

## Sampling

Monte Carlo sampling from compiled programs.

::: clifft.sample
    options:
      show_source: true
      members: []

::: clifft.sample_survivors
    options:
      show_source: true
      members: []

::: clifft.sample_k
    options:
      show_source: true
      members: []

::: clifft.sample_k_survivors
    options:
      show_source: true
      members: []

## Strong Simulation

Exact probability calculations without sampling.

::: clifft.basis_probabilities
    options:
      show_source: true
      members: []

::: clifft.record_probabilities
    options:
      show_source: true
      members: []

## State Inspection

Execute programs and inspect the quantum state without sampling.

::: clifft.execute
    options:
      show_source: true
      members: []

::: clifft.get_statevector
    options:
      show_source: true
      members: []

## Result Types

::: clifft.SampleResult
    options:
      show_source: true
      members: true

## Compiled Programs and Execution State

::: clifft.Program
    options:
      show_source: true
      members: false

::: clifft.State
    options:
      show_source: true
      members: false

## Circuit and IR Inspection

::: clifft.Circuit
    options:
      show_source: true
      members: false

::: clifft.AstNode
    options:
      show_source: true
      members: false

::: clifft.Target
    options:
      show_source: true
      members: false

::: clifft.HirModule
    options:
      show_source: true
      members: false

::: clifft.HeisenbergOp
    options:
      show_source: true
      members: false

::: clifft.Instruction
    options:
      show_source: true
      members: false

## Pass Managers

::: clifft.HirPassManager
    options:
      show_source: true
      members: false

::: clifft.BytecodePassManager
    options:
      show_source: true
      members: false

::: clifft.default_hir_pass_manager
    options:
      show_source: true
      members: []

::: clifft.default_bytecode_pass_manager
    options:
      show_source: true
      members: []

## HIR Passes

Optimization passes that operate on the high-level intermediate representation.

::: clifft.PeepholeFusionPass
    options:
      show_source: true
      members: false

::: clifft.StatevectorSqueezePass
    options:
      show_source: true
      members: false

::: clifft.RemoveNoisePass
    options:
      show_source: true
      members: false

::: clifft.DropNonUnitaryPass
    options:
      show_source: true
      members: false

## Bytecode Passes

Optimization passes that operate on the low-level bytecode representation.

::: clifft.NoiseBlockPass
    options:
      show_source: true
      members: false

::: clifft.ExpandTPass
    options:
      show_source: true
      members: false

::: clifft.ExpandRotPass
    options:
      show_source: true
      members: false

::: clifft.SwapMeasPass
    options:
      show_source: true
      members: false

::: clifft.MultiGatePass
    options:
      show_source: true
      members: false

::: clifft.SingleAxisFusionPass
    options:
      show_source: true
      members: false

## Enums and Types

::: clifft.Opcode
    options:
      show_source: true
      members: false

::: clifft.OpType
    options:
      show_source: true
      members: false

::: clifft.GateType
    options:
      show_source: true
      members: false

::: clifft.ParseError
    options:
      show_source: true
      members: false

## Utilities

::: clifft.get_num_threads
    options:
      show_source: true
      members: []

::: clifft.set_num_threads
    options:
      show_source: true
      members: []

::: clifft.svm_backend
    options:
      show_source: true
      members: []

::: clifft.compute_reference_syndrome
    options:
      show_source: true
      members: []

::: clifft.version
    options:
      show_source: true
      members: []

## Type Aliases

::: clifft.BasisBitstrings
    options:
      show_source: true
      members: []

::: clifft.MeasurementRecords
    options:
      show_source: true
      members: []

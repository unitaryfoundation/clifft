# Python API Reference

## Compilation

::: clifft.compile

::: clifft.parse

::: clifft.parse_file

::: clifft.trace

::: clifft.lower

::: clifft.ParseError

## Sampling

::: clifft.sample

::: clifft.sample_survivors

::: clifft.sample_k

::: clifft.sample_k_survivors

## Symbolic Sampling Backend (experimental)

This explicit opt-in surface exercises the new scalar symbolic-coordinate
backend without changing the default `clifft.compile` and `clifft.sample`
pipeline. It currently accepts noiseless rotations and measurements only.

::: clifft.experimental.compile

::: clifft.experimental.sample

::: clifft.experimental.record_probabilities

::: clifft.experimental.Program

## Leakage and Loss (experimental)

Sampling under a five-level leakage/loss model. See the
[Leakage and Loss guide](../guide/leakage-and-loss.md).

::: clifft.noncomp.sample

::: clifft.noncomp.Model

::: clifft.noncomp.Classifier

::: clifft.noncomp.NonComputationalSample

::: clifft.noncomp.Level

::: clifft.noncomp.QubitStatus

## Strong Simulation

::: clifft.basis_probabilities

::: clifft.record_probabilities

## State Inspection

::: clifft.execute

::: clifft.get_statevector

## Result Types

::: clifft.SampleResult

## Compiled Programs and Execution State

::: clifft.Program

::: clifft.State

## Circuit and IR Inspection

::: clifft.Circuit

::: clifft.AstNode

::: clifft.Target

::: clifft.HirModule

::: clifft.HeisenbergOp

::: clifft.Instruction

::: clifft.Opcode

::: clifft.OpType

::: clifft.GateType

## Pass Managers

::: clifft.HirPassManager

::: clifft.BytecodePassManager

::: clifft.default_hir_pass_manager

::: clifft.default_bytecode_pass_manager

## HIR Passes

::: clifft.PeepholeFusionPass

::: clifft.StatevectorSqueezePass

::: clifft.RemoveNoisePass

::: clifft.DropNonUnitaryPass

## Bytecode Passes

::: clifft.NoiseBlockPass

::: clifft.ExpandTPass

::: clifft.ExpandRotPass

::: clifft.SwapMeasPass

::: clifft.MultiGatePass

::: clifft.SingleAxisFusionPass

## Utilities

::: clifft.get_num_threads

::: clifft.set_num_threads

::: clifft.svm_backend

::: clifft.version

::: clifft.compute_reference_syndrome

## Type Aliases

::: clifft.BasisBitstrings

::: clifft.MeasurementRecords

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

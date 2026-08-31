# Python API Reference

The stable CPU workflow starts with `clifft.compile()`, which returns a reusable
`Program` consumed by the sampling and exact-query functions below. These
functions do not take a backend switch. Experimental hardware backends use
separate, explicitly selected APIs and program types.

See [Choose a Workflow](../getting-started/choosing-a-workflow.md) to select an
API by result and circuit constraints before using this reference.

## Compilation

::: clifft.compile

::: clifft.parse

::: clifft.parse_file

::: clifft.parse_qasm2

::: clifft.parse_qasm2_file

::: clifft.trace

::: clifft.lower

::: clifft.ParseError

## Sampling

::: clifft.sample

::: clifft.sample_survivors

::: clifft.sample_k

::: clifft.sample_k_survivors

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

::: clifft.get_statevector

## Result Types

::: clifft.SampleResult

## Compiled Programs

::: clifft.Program

## Circuit and IR Inspection

::: clifft.Circuit

::: clifft.Qasm2Import

::: clifft.AstNode

::: clifft.Target

::: clifft.HirModule

::: clifft.HeisenbergOp

::: clifft.OpType

::: clifft.GateType

## Pass Managers

::: clifft.HirPassManager

::: clifft.default_hir_pass_manager

## HIR Passes

::: clifft.PeepholeFusionPass

::: clifft.StatevectorSqueezePass

::: clifft.RemoveNoisePass

::: clifft.DropNonUnitaryPass

## Utilities

::: clifft.version

::: clifft.compute_reference_syndrome

## Type Aliases

::: clifft.BasisBitstrings

::: clifft.MeasurementRecords

# Python API Reference

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

The four fixed-plan functions share CPU batching and threading arguments. See
[CPU Execution and Tuning](../guide/cpu-execution.md) for their interaction and
compatibility limits.

::: clifft.sample

::: clifft.sample_survivors

::: clifft.sample_k

::: clifft.sample_k_survivors

## Experimental Hardware Backends

!!! warning "Experimental"
    These APIs require backend-specific source builds and may change without
    compatibility guarantees. They are not selected by the regular CPU API.

See [HIP Backend](../development/hip-backend.md) for the current hardware and
workflow limits.

::: clifft.experimental.hip.is_built

::: clifft.experimental.hip.is_available

::: clifft.experimental.hip.backend_info

::: clifft.experimental.hip.compile

::: clifft.experimental.hip.Program

::: clifft.experimental.hip.Sampler

::: clifft.experimental.hip.ReplayResult

## Leakage and Loss

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

::: clifft.HirPass

::: clifft.HirPassManager

::: clifft.default_hir_pass_manager

## HIR Passes

::: clifft.PeepholeFusionPass

::: clifft.StatevectorSqueezePass

::: clifft.ActiveWidthSchedulePass

::: clifft.RemoveNoisePass

::: clifft.DropNonUnitaryPass

## Utilities

::: clifft.version

::: clifft.runtime_isa

::: clifft.compute_reference_syndrome

## Type Aliases

::: clifft.BasisBitstrings

::: clifft.MeasurementRecords

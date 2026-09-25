# Using with Sinter

[Sinter](https://github.com/quantumlib/Stim/tree/main/glue/sample) is part of the
Stim project and runs Monte Carlo experiments on quantum error-correction
circuits. It coordinates parallel sampling and decoding, saves results, and
helps compare logical error rates across codes and noise levels.

Clifft's first integration is `PerfectionistSampler`, for Clifford experiments
that accept a shot only when **every detector is quiet**. Any logical observable
flip on an accepted shot counts as an error. This performs the same task as
Sinter's built-in `"perfectionist"` sampler, using Clifft to produce the counts.

Clifft can be faster than Stim on some Clifford error-detection workloads.
Broader integration, including non-Clifford experiments, remains an area for
exploration; specific research workflows will guide further models and APIs.

## Collect statistics

Install the optional integration:

<!--pytest-codeblocks:skip-->
```bash
pip install 'clifft[sinter]'
```

The regular `clifft` installation requires neither Stim nor Sinter.

Save this example as a Python script and run it. The main guard is required
for Sinter's multiprocessing workers.

```python
import sinter
import stim

from clifft.sinter import PerfectionistSampler


def main():
    circuit = stim.Circuit.generated(
        "color_code:memory_xyz",
        distance=5,
        rounds=5,
        after_clifford_depolarization=0.001,
        before_round_data_depolarization=0.001,
        before_measure_flip_probability=0.001,
        after_reset_flip_probability=0.001,
    )
    task = sinter.Task(
        circuit=circuit,
        # No graphlike decomposition is needed for zero-prediction error detection.
        detector_error_model=circuit.detector_error_model(),
    )
    results = sinter.collect(
        num_workers=2,
        tasks=[task],
        decoders=["clifft-perfectionist"],
        custom_decoders={
            "clifft-perfectionist": PerfectionistSampler(),
        },
        max_shots=100_000,
    )
    for result in results:
        print(result.shots, result.discards, result.errors)


if __name__ == "__main__":
    main()
```

`shots` counts **all attempts**, including discarded shots. `errors` counts
accepted shots with at least one flipped observable, so several flipped
observables in the same shot still count as one error. Error probability
conditioned on acceptance is `errors / (shots - discards)` when the denominator
is nonzero.

This example estimates color-code error rates conditioned on detecting no
errors. It does not apply a decoder to correct detected errors.

## Options and limits

The adapter supports all-detector postselection on Clifford circuits using
[instructions supported by Clifft](../reference/gates.md). Partial detector
postselection, observable postselection, and arbitrary decoders are not
supported. For non-Clifford circuits, use
[Clifft's native sampling workflows](simulation.md).

Start with the defaults: each Sinter worker uses one Clifft thread and up to
1024 packed shots. For longer runs, try `PerfectionistSampler(batch_size=2048)`
with `sinter.collect(max_batch_size=16384, ...)` and compare the complete run
against Sinter's built-in `"perfectionist"`. Larger batches can improve
throughput but use more memory. See [CPU Execution and Tuning](cpu-execution.md)
for general tuning guidance.

See the [sampler API reference](../reference/python-api.md#optional-sinter-integration)
for parameter limits, mask formats, and unsupported Stim features.

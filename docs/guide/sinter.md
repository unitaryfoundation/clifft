# Error detection with Sinter

Use `clifft.sinter.PerfectionistSampler` when a shot is accepted only if **every
detector is quiet**, and any logical observable flip on an accepted shot is an
error. Clifft samples and rejects shots natively, returning only counts to
Sinter. Sinter schedules workers and collects the results.

Install the optional integration:

<!--pytest-codeblocks:skip-->
```bash
pip install 'clifft[sinter]'
```

The regular `clifft` installation requires neither Stim nor Sinter. This
integration supports Stim/Sinter 1.16 and uses their public sampler interface.

Sinter's built-in `"perfectionist"` sampler computes the same counts using Stim.
Register the Clifft sampler as shown below to compare them on your workload.

## Collect statistics

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

The example is color-code **error detection**, not error correction with a
color-code decoder. The task's detector error model participates in Sinter's
task identity, but this sampler does not use it to predict corrections.

## Supported task contract

- Supply a `stim.Circuit` using [instructions supported by Clifft](../reference/gates.md).
  Stim instruction tags are ignored when compiling. Some Stim features, such as
  heralded noise, sweep-bit controls, and Pauli targets in `OBSERVABLE_INCLUDE`,
  are not supported.
- An absent `postselection_mask` selects every detector, matching Sinter's
  built-in `"perfectionist"`. An explicit mask must be a little-endian `uint8`
  array selecting every detector. Padding bits in the final byte are ignored.
  With no detectors, an absent or empty mask is accepted.
- Detector and observable bits are normalized against the noiseless reference.
  The sampler predicts zero observable flips for every accepted shot.
- Observable postselection, partial detector postselection, and arbitrary
  decoders are unsupported and are not silently substituted.
- Leave Sinter's `count_detection_events` and `count_observable_error_combos`
  options false. Sinter 1.16 rejects these options for custom samplers.
- Non-Clifford circuit/model pairing is outside this adapter's scope. Use
  [Clifft's native survivor API](simulation.md) for those circuits.

Import and construct `PerfectionistSampler` in the parent process. Its
configuration is pickle-safe; each Sinter worker compiles its own program once
per task. The compiled sampler is worker-local. Sampling uses fresh native
randomness on every call; this adapter exposes no seed or cross-call stream
reproducibility guarantee. Direct `sample(0)` calls return zero counts.

## Capacity and performance

Each Sinter process uses one Clifft native thread. Sinter's `max_batch_size`
controls attempts requested per call; the adapter's `batch_size` controls native
lane capacity within that call. Both default to 1024 with Sinter 1.16, though
Sinter starts with smaller calls and increases them as collection proceeds.
Native capacity is capped at 2048 lanes and at the shots requested in each call.
To use `batch_size=2048`, also increase Sinter's `max_batch_size` to at least 2048;
larger native capacity requests are capped, not rejected.

The adapter defaults to packed execution for Clifford error detection. Set
`batch_size=1` for scalar execution or `batch_size="auto"` to use Clifft's core
policy, which currently chooses scalar execution for postselection and can be
substantially slower on these workloads. Packed execution uses more memory;
the best capacity and any speedup over Stim depend on the circuit and hardware.
Compare complete collection runs, including compilation and worker startup,
when choosing a sampler. See [CPU Execution and Tuning](cpu-execution.md).

The adapter calls `sample_survivors(..., keep_records=False)` through the public
API. It retains the compiled program but constructs native workers for each
sampling call. It produces no survivor-row matrices.

The [regular benchmark suite](https://github.com/unitaryfoundation/clifft/blob/main/tools/bench/README.md#sinter-postselection)
includes the fully Clifford S-gate cultivation fixture as a counts-only
postselection regression case. It excludes compilation and Sinter process
startup; measure those costs separately when assessing a complete workload.

## API

::: clifft.sinter.PerfectionistSampler

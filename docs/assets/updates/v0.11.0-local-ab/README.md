<!--pytest-codeblocks:skipfile-->

# Local A/B evidence for the v0.11.0 development post

These local measurements isolate two changes. They are not a release-over-release
campaign or measurements of published wheels.

- [Scheduler raw timings](scheduler.json): the same runtime from commit
  `229b1abb698c9fde7c9ff85be11e07acac666fe2`, using default compilation versus
  default compilation followed by `ActiveWidthSchedulePass()`.
- [Deferred-noise raw timings](deferred-noise.json): parent
  `51c23ca63d05660012492dd3b59f09ccd50e28a8` versus change
  `2731fa02f665dc3262e1efa9e797802e06570682`. Only the deferred-noise change
  separates these two revisions.
- [Measurement script](measure.py): persistent worker processes pinned to one
  logical CPU, using identical fixtures, APIs, shot counts, and seed per pair.

## Method

All builds used GCC 13.3.0, Release mode, native CPU tuning, and no interprocedural
optimization. Runtime dispatch selected AVX-512. Python was 3.14.2. Each variant
pair used the same NumPy version, recorded in its JSON. The noise builds used
`SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0` because they were built from source archives;
the exact source commits, rather than that placeholder package version, identify
the measured code.

The script first compiles each program six times and discards the first timing.
It calibrates a common shot count for both variants to target roughly 250 ms on
the slower variant, then warms each variant at that shot count. It records 15
pairs, alternating A/B and B/A order, with seed `100 + pair_index` shared within
each pair. The reported throughput ratio is the median of `time_A / time_B`.
Raw timings and survivor counts remain available for every pair. Sampling times
include public-API setup and result allocation, and exclude compilation, process
startup, interprocess communication, and destruction of the returned result.

All runs force one thread and scalar sampling (`batch_size=1`). Coherent and
Quantum Volume scheduler cases use ordinary sampling. Cultivation uses
all-detector postselection with aggregate counts. Noise cases use the output
mode recorded in the JSON. The synthetic early-rejection case is the existing
benchmark circuit: a 99% rejection check followed by 4096 noisy reset/measurement
steps, with survivor records retained.

The four scheduler fixtures and five noise cases were selected before the
reported runs. The tables include the scheduler slowdown and the noise cases
with little change. Timings on a VM vary; these results illustrate workload
dependence and do not predict other hardware or execution modes.

## Reproduce

Create source checkouts or archives for the three exact revisions above. Build
each with the same compiler and settings into its own Python environment. The
candidate pair must use the same environment; the noise pair must use separate
environments for the two installed revisions. To match the measured build,
use Python 3.14.2, GCC 13.3.0, the default Release configuration,
`CLIFFT_CPU_BASELINE=native`, and `CLIFFT_INTERPROCEDURAL_OPTIMIZATION=OFF`.
The recorded NumPy versions are 2.4.2 for the candidate and 2.5.3 for both noise
builds. Source archives also require `SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0`.

Run the script from this directory, substituting your checkout and interpreter
paths. `--root` supplies fixtures from the candidate checkout; it does not select
the installed Clifft runtime. Choose a logical CPU available to your process.

```bash
python measure.py --mode scheduler \
  --root /path/to/candidate-checkout \
  --python-a /path/to/candidate-env/bin/python \
  --python-b /path/to/candidate-env/bin/python \
  --sha-a 229b1abb698c9fde7c9ff85be11e07acac666fe2 \
  --sha-b 229b1abb698c9fde7c9ff85be11e07acac666fe2 \
  --compiler 'GCC 13.3.0' --cpu 2 --seconds 0.25 --repeats 15 \
  --output scheduler.json

python measure.py --mode noise \
  --root /path/to/candidate-checkout \
  --python-a /path/to/before-env/bin/python \
  --python-b /path/to/after-env/bin/python \
  --sha-a 51c23ca63d05660012492dd3b59f09ccd50e28a8 \
  --sha-b 2731fa02f665dc3262e1efa9e797802e06570682 \
  --compiler 'GCC 13.3.0' --cpu 2 --seconds 0.25 --repeats 15 \
  --output deferred-noise.json
```

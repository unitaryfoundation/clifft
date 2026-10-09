# Reproduction and selected correctness evidence

## Environment and build

Use a checkout of this research branch with its Python package installed from
source, following [the build guide](../../docs/development/building.md).
The scripts need NumPy, Stim, Qiskit and Qiskit Aer from the development
dependencies. The survey harness currently uses Linux affinity and process
resource interfaces. The comparison with Merlin is optional and additionally
requires its Python package.

Build the two current native research tools:

```bash
cmake -S . -B build-profile \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCLIFFT_BUILD_TESTS=OFF -DCLIFFT_BUILD_PROFILER=ON
cmake --build build-profile \
  --target profile_prefix_trace_reuse export_optimized_prefix -j2
```

The work is based on PR 552's commit `ed2a125a`. Building against a different
Clifft version changes the baseline and may change the research worker's
internal interfaces. No production installation exposes this worker as a
supported public sampling mode.

## Run the current survey from retained inputs

All 33 complete circuit texts are included in the gzip JSON artifacts listed
by [capability-survey.json](capability-survey.json). The new replay option
verifies both archive and individual circuit SHA-256 hashes. It needs neither
a Merlin generator checkout nor the original `/tmp` benchmark directory.

A small review run exercises a large reduced circuit, surviving non-Clifford
work, measured chaining and fallback:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  tools/profile/survey_conditional_capability.py \
  --retained-survey research/conditional_clifford/capability-survey.json \
  --backends ordinary combined \
  --native-worker build-profile/profile_prefix_trace_reuse \
  --exporter build-profile/export_optimized_prefix \
  --cases quadcycle-noisy.stim bt81_direct_x_ideal chain_three_noisy \
    branch_dependent unsupported_rotation \
  --shots 8 --output /tmp/conditional-review.json
```

For the complete 33-case assessment, omit `--cases` and use `--shots 128`.
Add `merlin` to `--backends` to include that installed simulator. Original
measurements pinned the process to CPU 2; choose an available fixed CPU with
`taskset` for timing comparisons. Setup is reported separately. An eight-shot
review run exercises the implementation; it is not a replacement throughput
measurement or statistical equivalence test.

Each backend runs in its own process with a 180-second default timeout. Width
above 12 is reported without allocating the corresponding dense state. Read
the `status` fields: `error`, `timeout`, `width_budget` and
`partial_width_budget` are retained outcomes, not silently skipped cases.
The main driver writes a result even when a backend rejects a source.

The historical generator route still accepts `--benchmark-dir` and
`--merlin-checkout`, as recorded in the survey. Merlin helpers are pinned to
`097380fac1a3968ca47925146e211fe990f4c396`; the supplied factory exporter and
independent references came from a separate checkout at `1bcbf960`. That
separate checkout is not included in this branch. The replay option avoids
that dependency for the current benchmark inputs, but regenerating the
independent D/E rational references still requires the original reference
package.

## Independent small checks

These validators use generated small circuits and need no external benchmark
checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_conditional_phase_frontend.py \
  --output /tmp/conditional-instruments.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_continuation_trace_reuse.py \
  --mode coordinate-identity \
  --exporter build-profile/export_optimized_prefix \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/conditional-continuation.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  tools/profile/validate_boundary_composition.py \
  --mode coordinate-identity \
  --worker build-profile/profile_prefix_trace_reuse \
  --output /tmp/conditional-boundary.json
```

`validate_coordinate_reuse.py` additionally exercises stable, dense, changing
and identity-query frames across all research coordinate policies.
`validate_compiled_prefix_reuse.py` independently checks exported preparations
and conditional instruments. Both accept the worker/exporter flags described
by their `--help`.

## Evidence retained for review

| Evidence | What it establishes | Artifact |
| --- | --- | --- |
| Complete-circuit survey | Observed capability/cost across 33 inputs; 230 fixed-history audits; 83 full recompilation comparisons; 40 independent small instruments | [Survey index](capability-survey.json) and its listed gzip groups |
| Generic frontend instruments | 70 original/renamed controls, fixed histories and enumerated measurement/reset branches; complete small physical-state and record laws; correlated-noise controls | [Frontend validation](conditional-frontend-validation.json) |
| Fixed preparation export | Independent small prepared states and full conditional instruments after exporting the final frame | [Preparation validation](compiled-prefix-validation.json) |
| Current continuation policy | 256 Aer state checks, 288 tomography record laws, 144 exact Stim laws, 72 conditional-trace checks and seven rejection controls | [Continuation validation](coordinate-identity-validation.json) |
| Boundary algebra | All 512 three-qubit diagonal Cliffords; exact traces across physical mask-word boundaries | [Boundary validation](coordinate-boundary-validation.json) |
| Planner policies | Independent small controls for identity, dense, long-lived and changing frames | [Coordinate controls](coordinate-reuse-controls.json) |
| Selected large references | 50 exported D/E rational queries, 3,422 coarse-law probabilities, 550 raw-record probabilities, 368 state probes, 160 factory conditional-record comparisons | [Factory/preparation validation](compiled-prefix-factory-validation.json) |
| Measurement deferral | Small instruments and noise mixtures; dependency barriers; BT scored laws, distillation/cultivation and noisy-scoring cases | [Deferral regression](factory-deferred-regression.json) |
| Why identity is the selected planner policy | Paired timing, query audits and exact comparisons for alternative coordinate policies | [Coordinate study](coordinate-reuse-study.json) |
| Original supplied benchmark contract | Hashes, dimensions, noise descriptions and selected exact queries for the supplied factory package | [Benchmark manifest](factory-benchmark-manifest.json) |

Independent Aer checks compare unnormalized physical-state blocks for each
visible record, including hidden reset mixtures. Matching optimized HIR and
planner inspection against fresh compilation checks the reuse mechanism;
it does not independently re-prove the original large-circuit reduction.
The factory conditional reference still uses existing HIR optimization.
Selected D/E raw probabilities and state probes do not exhaust every possible
large quantum instrument. Complete coarse laws are marginal laws.

The large survey's 128-shot moment screens detect gross errors only. They do
not certify rare acceptance/logical-error rates or every possible fault
history. No validation convenience changes the actual sampling law.

## Provenance and historical drivers

The broad survey ran immediately before commit `d126c29e`; its metadata records
the parent Git revision and the actual dirty-tree source/binary hashes.
Those source files are captured by `d126c29e`. Later review changes add the
retained-input loader and documentation; they do not retroactively change
recorded benchmark hashes or times.

Superseded reports and output files were removed from the current tree, with
lessons consolidated in [LESSONS.md](LESSONS.md). Their original versions are
still available in Git at `d126c29e`. Several earlier scripts remain because
the current generators and validators import their helpers. Reproducing an
old standalone driver's historical experiment may require its old data at
that revision; it is not part of the current reproduction path above.

To inspect a retained group without running a benchmark:

```python
import gzip
import json
from pathlib import Path

root = Path("research/conditional_clifford")
index = json.loads((root / "capability-survey.json").read_text())
case = "bt81_direct_x"
archive = root / index["cases"][case]["artifact"]
row = json.loads(gzip.decompress(archive.read_bytes()))[case]
print(row["source"])
print(row["backends"]["combined"]["widths"])
```

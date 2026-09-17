# Parsed-region logical-block planning

The research adapter now derives supported syndrome and final-growth plans from
Clifft's parsed circuit and runs them in one reusable native worker. A changed
schedule no longer requires generating and compiling another C++ executable.
The logical-block representation and hot executor are unchanged. This closes
the schedule-catalog gap; it does not establish a second physical circuit family.

## What changed

`fold_adapter.py` reads the actual Clifft AST through `fold_plan_worker.cpp`.
It identifies the known injection, morphing, cat preparation, fold, and decode
anchors. Between those anchors it extracts the supplied syndrome operations
and the final d5-to-d7 growth operations. Stage comments are not inputs to this
process. Existing offline certificates check the complete syndrome instrument,
fresh-wire initialization, code transport, logical axes, and syndrome transport.
Noise maps are derived from these supplied gates, including their order.

A structural check retains detector dependencies and physical noise locations.
The existing native recognizer then checks the actual circuit against the newly
certified body, binds optional/per-site noise probabilities, and certifies the
Clifford suffix and physical Pauli probes. This final body match is a guard on
the derived plan, not a lookup in a list of precompiled syndrome schedules.
Changing a CCZ to its explicit parser-equivalent T/CX decomposition is accepted;
adding noise inside that decomposition is declined because it changes the
certified physical fault locations.

`fold_plan_data.py` writes a private, versioned data file containing the sparse
fault maps, code geometry, contraction schedules, coherent actions, detector
metadata, and optional fixed-history fixtures. The worker loads owned storage,
checks capacities, ranges, dimensions, and contraction dependencies, and creates
the same spans consumed by `fold_blocks_native.h`. It then uses the existing
survivor-result adapter. Loading, certification, topology work, and allocations
remain outside hot execution. This is research tooling, not a public plan-file
API or a production dispatcher.

The f7 sampling plan occupies about 2.07 MB on disk. It retains 6,199 fault bits,
2,205 physical noise sites, at most four coherent terms, and at most 2,851
complex scratch entries. The data format repeats some geometry to keep ownership
simple; deduplication is not needed to establish this result. Output storage is
additional and depends on the number of shots requested.

## Held-out validation

The held-out f7 circuit combines the dual-rotated growth encoder with a seeded
shuffle of every syndrome check and its CNOT ordering. It has no stage comments.
Neither the adapter nor worker contains this schedule. The old combined
growth/reverse-checks catalog declines it. The new adapter reconstructs all
supplied stages exactly and certifies them.

The reference study uses 200 fixed histories: an ideal history, 64 natural
histories at p=0.001, 32 stress histories at p=0.01, paired hooks, a logical-error
tail, nonzero syndrome sectors, f7 flag diagnostics, and 84 growth-specific
faults. The independent coherent CH/tableau reference follows elementary gates
and retains relative phases. Native loaded-plan execution is also compared
against each certified Python fixture. These histories are correctness probes,
not a logical-error-rate estimate.

Validation results and timings are recorded in
[fold_adapter_data.json](fold_adapter_data.json).

Maximum acceptance discrepancy is 1.12e-16 and maximum conditional logical-XYZ
discrepancy is 2.23e-16. Native execution matches all 200 fixtures to 2.23e-16.
Both nonzero growth-sector diagnostics retain acceptance 1/4. The full suite
passes 91 tests; two subsequently added decomposition/layout checks also pass,
for 93 distinct passing research tests. Seven adapter tests pass with the
ASAN/UBSAN worker; the old-catalog comparison is skipped in that sanitizer run.
All 200 fixtures and 2,500 sampled attempts additionally pass ASAN/UBSAN, with
leak detection disabled. The research worker and kernel are instrumented; the
linked Clifft core library is the existing release build.

The regression checks also compare the reusable worker with the existing
compiled catalog on f3/f5/f7 using identical RNG seeds. Survivor counts,
measurement records, detectors, probes, and physical noise probabilities agree
exactly. Other checks cover missing/nonuniform noise, aliases, reset-erased
prefixes, Clifford/probe suffixes, and malformed region rejection. Partial or
nonzero detector postselection and fixed-fault sampling decline before sampling.
Truncated, oversized, and trailing plan data are rejected during loading.

## Sustained sampling and decline cost

| f7 execution path | Median us/attempt | Range us/attempt | Survivors per 100,000 |
| --- | ---: | ---: | ---: |
| Original circuit, compiled catalog | 45.10 | 44.96-45.59 | 12,667 |
| Original circuit, loaded plan | 45.17 | 44.95-45.72 | 12,667 |
| Held-out circuit, loaded plan | 45.99 | 45.44-46.41 | 12,553 |

Each row uses three sequential runs of 100,000 attempts at p=0.001, seed 19331,
with full survivor records and logical XYZ probes. Reusing the seed stabilizes
the comparison; these repetitions are not independent statistical datasets.
There is no material throughput change from loading the same plan as data.
The held-out circuit changes physical fault propagation, so its acceptance
need not equal the original circuit's acceptance. Its measured acceptance is
not evidence of a better or worse logical error rate.

Offline preparation takes 1.96 seconds for the held-out circuit, including the
zero-shot final contract check. Native process wall time is 4.58-4.66 seconds
for its 100,000 attempts, excluding Python planning. Maximum process RSS is
about 77 MiB with preallocated full survivor outputs; this is not the small
coefficient/scratch storage alone. Native C++ is compiled once for the worker,
while circuit-specific maps are still planned once per input, not per shot.

All eight available corpus circuits other than one-round inputs decline, taking
5.5-25.9 ms through this Python/subprocess AST bridge. They include five-round
d5 coherent memory, existing MSC d3/d5, distillation, quantum volume, and pure
surface-code circuits. A separate native fallback request successfully runs
ordinary Clifft tracing, optimization, and planning on each. Its planning costs
1.3-38.1 ms. Thus this research frontend's rejection overhead is noticeable for
small ordinary compilations; it is not yet an optimized production eligibility
check. The adapter does not automatically dispatch fallback or sample those
ordinary plans. No one-round circuit was evaluated.

These measurements establish reusable planning and preserve the f7 throughput
finding. They do not establish broad corpus applicability, a measured generic
f7 speedup, or best-in-class performance.

## Applicability and next research step

The adapter requires canonical physical wire labels and the current protocol
stage skeleton, injection, cats, and fold geometry. Its variable regions are
syndrome extraction and the final regular-d5-to-regular-d7 growth. The previous
catalog recognizer's general qubit relabeling is not yet available during this
region discovery. Unsupported inputs remain available to ordinary Clifft;
there is no live residual-state handoff.

The accepted output contract still ends in ideal syndrome postselection and
physical Pauli expectations, with optional terminal Clifford gates. There is
no final-error-correction decoder or certified fault-distance claim. The input
is a paper-guided reconstruction, not an authors' circuit artifact. The changed
growth encoder still preserves the original syndrome-transfer relation.

This is enough schedule-composition evidence to stop extending the catalog
harness for now. The next independent research target is the deferred exact
Clifford-frame/tensor screen on large fold residuals. The earlier screen used
five-round coherent memory; it did not answer that large-fold question. Keep
compiled logical blocks as the leading bounded candidate while measuring
whether another representation covers states outside these certified block
boundaries. More schedule permutations or serializer optimization would not
answer that physics question. Production integration and any architecture
change remain separate decisions.

## Reproduction

Build one worker, then use the existing research Python environment with NumPy,
Stim, Cirq, and Aer:

```sh
c++ -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic -I src -I build-research/generated tools/profile/fold_plan_worker.cpp build-research/src/clifft/libclifft_core.a -o /tmp/clifft-plan-worker
OPENBLAS_NUM_THREADS=1 python tools/profile/study_fold_adapter.py --worker /tmp/clifft-plan-worker --output /tmp/fold-adapter-study --shots 100000 --trials 3 --catalog /tmp/clifft-growth-combined-recognizer
OPENBLAS_NUM_THREADS=1 python tools/profile/fold_adapter.py --worker /tmp/clifft-plan-worker --circuit /tmp/fold-adapter-study/held.stim --plan /tmp/fold-adapter-study/new.bin --shots 100000
CLIFFT_PLAN_WORKER=/tmp/clifft-plan-worker CLIFFT_RECOGNIZER_NATIVE=/tmp/clifft-recognizer CLIFFT_SCHEDULE_RECOGNIZER_NATIVE=/tmp/clifft-schedule-recognizer CLIFFT_GROWTH_RECOGNIZER_NATIVE=/tmp/clifft-growth-recognizer CLIFFT_PROXY_NATIVE=/tmp/clifft-proxy-build/profile_proxy OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tools/profile -p 'test_*.py'
```

The study's initial planning time includes parsing, certification, serialization,
and a zero-shot output-contract check. Sampling timers include native setup,
output allocation, all physical noise draws, acceptance, survivor records, and
probes. They exclude Python planning, native plan loading, parsing, and JSON
output; process-wall measurements include those native overheads. Run timings
sequentially after tests and builds finish. The optional old catalog is built
as described in [the growth report](fold_growth.md).

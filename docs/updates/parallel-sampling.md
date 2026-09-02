# Parallel Sampling in Clifft (v0.9.0, August 2026)

Clifft's original localized-Pauli SVM already used OpenMP to divide wide
active-state kernels across CPU cores. The
[original Clifft paper benchmarks](https://github.com/unitaryfoundation/clifft-paper/tree/main/qv_bench)
used that path for their 16-threaded Quantum Volume runs.

The symbolic-coordinate rewrite in v0.8.0 deliberately cut over to a clean
single-shot baseline without carrying the SVM-specific threading machinery
forward. Version 0.9.0 restores intra-shot multicore execution on the new
symbolic sampler and adds integrated cross-shot workers for sampling APIs.

The goal is that users should automatically get the best performance for
their workload while retaining the flexibility to override Clifft's decisions
when they know better. A sampling call supplies a total worker budget; Clifft
decides whether to spend it across independent shots or within a single
expensive shot.

Just as importantly, that execution choice does not change seeded results.
Parallelism affects how Clifft computes a sample, not which experiment the
sample represents.

## One worker budget, two forms of parallelism

Near-Clifford workloads have different shapes.

Most Monte Carlo and importance-sampling jobs contain many independent shots.
For these, Clifft uses cross-shot parallelism: several workers execute
different shots concurrently. The same path supports ordinary sampling,
postselected survivor sampling, forced-fault sampling, and noncomputational
leakage and loss trajectories.

Other jobs request only a few shots but reach a large active width. Their dense
active state contains $2^k$ coefficients, so individual rotations and
measurements can dominate execution time. When there are not enough shots to
occupy the worker budget, an OpenMP-enabled build can instead divide those
coefficient traversals among several workers within each shot.

The normal interface exposes this as one setting:

<!--pytest.mark.skip-->

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")
result = clifft.sample(
    program,
    shots=100_000,
    seed=42,
    threads=8,
)
```

`threads` is the total CPU budget. Clifft uses cross-shot workers when there
are enough shots and can select intra-shot execution for undersubscribed
workloads whose active state is wide enough to benefit. One thread remains the
default, and `threads="auto"` uses the implementation-reported hardware
concurrency.

The default policy deliberately chooses a simple layout. It does not
automatically combine cross-shot and intra-shot workers.

## Automatic by default, controllable when needed

No static rule can be optimal for every circuit and machine. Performance
depends on the number of shots, active width over time, postselection
lifetime, output requirements, CPU topology, and available memory.

Advanced callers can therefore replace the automatic decision with an
explicit layout:

<!--pytest.mark.skip-->

```python
import clifft

program = clifft.compile("H 0\nT 0\nM 0")
result = clifft.sample(
    program,
    shots=8,
    thread_layout=(2, 4),
    intra_shot_min_active_width=17,
)
```

This layout runs two shots concurrently and gives each shot up to four OpenMP
workers after its active width reaches 17.

The automatic policy is intended to provide a useful default without making
users understand the executor. The override exists for measured production
tuning, unusual hardware, and applications that already know more about their
resource constraints than Clifft can infer. The
[CPU execution guide](../guide/cpu-execution.md) documents the current policy
and controls.

## Did v0.9.0 recover multicore scaling?

The `clifft-bench`
[Quantum Volume campaign](https://github.com/unitaryfoundation/clifft-bench/pull/28)
measured the restored intra-shot path from 1 to 16 physical cores. It used the
exact v0.9.0 release commit with OpenMP enabled on an AWS `c8i.8xlarge`, with
one logical CPU selected from each physical core. Each cell below is the
median end-to-end compilation plus one-sample latency across three
deterministic circuit seeds. Speedup is the median of the three paired
per-seed speedups.

| Workload | 1 core | 16 cores | Speedup |
|---|---:|---:|---:|
| QV18 | 0.0514 s | 0.0419 s | 1.21x |
| QV20 | 0.200 s | 0.0560 s | 3.62x |
| QV22 | 0.927 s | 0.118 s | 7.51x |
| QV24 | 7.13 s | 0.701 s | 10.17x |
| QV26 | 38.4 s | 5.00 s | 7.69x |
| QV28 | 209 s | 26.5 s | 7.91x |

Quantum Volume circuits are dense in non-Clifford gates and are not Clifft's
primary near-Clifford workload. Here they provide a useful stress test for
wide active states. The small gain at QV18 and larger gains at wider sizes also
show why Clifft uses an active-width threshold instead of entering the OpenMP
runtime for every kernel.

These results recover a capability present in the original SVM rather than
establishing multicore Clifft for the first time. What changes in v0.9.0 is the
execution engine beneath it, the unified worker-budget interface, and the
ability to choose between intra-shot and cross-shot work automatically.

## Reproducibility across worker layouts

Parallel scheduling creates a subtle reproducibility problem. If every worker
draws from one shared random stream, changing the worker count or execution
order can change which random values belong to which shot.

Clifft 0.9.0 instead derives a separate random stream for every shot from the
call seed, global shot index, and a domain identifying the source of
randomness. A shot therefore receives the same stream regardless of which
worker executes it.

Workers may dynamically claim work to balance uneven shot costs, while results
are restored to global shot order. With a fixed seed, changing `threads`
preserves measurement, detector, and observable rows, survivor order, and
noncomputational result sidecars.

Seeded rows do differ from v0.8 because v0.9 introduced this per-shot
derivation. Within v0.9, however, the worker layout is not part of the seeded
result.

## Parallelism and memory

Cross-shot workers each own an executor and its dense active-state storage. At
peak active width $k$, coefficient and measurement scratch storage uses
approximately

$$
24 \times 2^k \text{ bytes}
$$

per worker, before records, outputs, symbolic state, and other metadata.

Intra-shot workers cooperate on one executor instead of replicating it. This
makes intra-shot execution useful not only when there are too few shots to
occupy the CPU, but also when duplicating a wide active state would consume too
much memory.

This tradeoff is another reason for exposing a worker budget rather than
treating "number of threads" as synonymous with "number of simultaneous
shots."

## Other changes in v0.9.0

The release vectorized additional active-measurement probability and collapse
kernels for AVX2 and AVX-512. It also fixed `basis_probabilities()` for cases
where several active coordinates interfere with complex relative weights.

Finally, `get_statevector()` is now defined only up to global phase.
Statevector extraction was primarily a debugging and validation interface,
while preserving an otherwise irrelevant source-matrix phase required extra
compiler bookkeeping. Removing that requirement allowed Clifft to absorb more
Clifford-valued rotations and avoid unnecessary actions or active-width
growth. Relative amplitudes, probabilities, and fidelity remain unchanged.

<!--pytest-codeblocks:skipfile-->
# Benchmark History

Clifft uses two complementary benchmark layers: long reproducible campaigns
for user-facing performance claims and short scheduled benchmarks for developer
regression tracking.

## Campaigns and scheduled benchmarks

| | Reproducible campaigns | Scheduled regression benchmarks |
|---|---|---|
| Main question | How do releases and tools compare on application circuits? | Did a focused operation drift over time? |
| Workloads | Versioned QEC corpus and Quantum Volume matrices | Targeted C++ and Python cases |
| Runtime | The current QEC campaign uses five samples totaling about 150 seconds per workload, run, and placement | Short enough to run as a daily CI job |
| Hardware | Named, pinned AWS hardware epochs; fresh boot IDs are recorded | Shared GitHub-hosted runners |
| Cadence | Infrequent: releases, tool updates, and new hardware epochs | Daily and manual |
| Output | Raw JSON, derived CSV tables, and reviewed figures | Trend dashboards on `gh-pages` |
| Best use | Canonical absolute rates and cross-tool or cross-release conclusions | Detecting trends and choosing what to investigate |

The long campaigns live in
[`clifft-bench`](https://github.com/unitaryfoundation/clifft-bench). They record
the circuit digest, software lock, host, boot, timed samples, and comparison
policy. The current results are summarized in the
[Performance guide](../guide/performance.md).

The scheduled charts are developer telemetry, not a second source of absolute
performance claims. Shared-runner noise, shorter cases, different units, and a
different measurement contract mean their values should not be compared
directly with `clifft-bench` rates.

## Scheduled benchmark dashboards

Each scheduled run appends to a Chart.js viewer hosted on `gh-pages`:

- [C++ Catch2 benchmarks (scalar)](https://unitaryfoundation.github.io/clifft/bench/cpp-scalar/)
- [C++ Catch2 benchmarks (AVX2)](https://unitaryfoundation.github.io/clifft/bench/cpp-avx2/)
- [C++ Catch2 benchmarks (AVX-512)](https://unitaryfoundation.github.io/clifft/bench/cpp-avx512/)
- [Python pytest-benchmark suite](https://unitaryfoundation.github.io/clifft/bench/python/)

ISA-specific dashboards are updated only when the runner exposes the required
CPU feature. Each viewer's `data.js` contains its complete history.

## What runs and when

The [`bench.yml`](https://github.com/unitaryfoundation/clifft/blob/main/.github/workflows/bench.yml)
workflow records:

- C++ sampling cases from
  [`tests/test_benchmarks.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_benchmarks.cc),
  tagged `[bench]` and run in scalar, AVX2, and AVX-512 modes where supported.
- Python compile and sampling cases under
  [`tools/bench/`](https://github.com/unitaryfoundation/clifft/tree/main/tools/bench).

It runs daily at 06:17 UTC and by manual dispatch from **Actions > Benchmark
history > Run workflow**. It does not run on pull requests or pushes to `main`,
and it records data without gating, comments, or alerts.

## Reading the scheduled results

- **Runner noise:** GitHub-hosted `ubuntu-24.04` runners share hardware. Trends
  across many days are more meaningful than isolated spikes.
- **Different units:** Catch2 plots elapsed time in a reporter-selected unit;
  lower is better. The Python chart plots iterations per second; higher is
  better. The charts are not directly comparable.
- **Investigation workflow:** If a trend looks suspicious, reproduce the
  relevant suite locally against `main` and the suspect commit with `just
  bench` for Python or `ctest -R Bench` for C++.

## Adding scheduled benchmarks

New `[bench]` cases in
[`tests/test_benchmarks.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_benchmarks.cc)
or new cases under
[`tools/bench/`](https://github.com/unitaryfoundation/clifft/tree/main/tools/bench)
are picked up by the next scheduled run without a workflow change. Add
application-scale or cross-tool workloads to `clifft-bench` instead.

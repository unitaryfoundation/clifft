<!--pytest-codeblocks:skipfile-->
# Archived Benchmark History

Clifft recorded daily C++ Catch2 and Python pytest-benchmark results from May
through August 2026. The scheduled publisher was retired because absolute
timings from heterogeneous GitHub-hosted runners did not provide a reliable,
actionable regression signal.

The existing `gh-pages` results remain available as a historical archive:

- [C++ scalar ISA](https://unitaryfoundation.github.io/clifft/bench/cpp-scalar/)
- [C++ AVX2](https://unitaryfoundation.github.io/clifft/bench/cpp-avx2/)
- [C++ AVX-512](https://unitaryfoundation.github.io/clifft/bench/cpp-avx512/)
- [Python](https://unitaryfoundation.github.io/clifft/bench/python/)

These series may mix CPU models and are not suitable for comparing absolute
performance between commits. Release-quality, hardware-scoped measurements
live in the [clifft-bench](https://github.com/unitaryfoundation/clifft-bench)
project.

The benchmark suites remain useful for local, paired comparisons:

- C++ sampling benchmarks are tagged `[bench]` in
  [`tests/test_benchmarks.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_benchmarks.cc).
- Python benchmarks live under
  [`tools/bench/`](https://github.com/unitaryfoundation/clifft/tree/main/tools/bench).

Run `ctest --test-dir build -R Bench` for C++ or `just bench` for Python.
Paired pull-request canaries are tracked in
[issue #434](https://github.com/unitaryfoundation/clifft/issues/434).

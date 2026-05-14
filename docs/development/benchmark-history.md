<!--pytest-codeblocks:skipfile-->
# Benchmark History

Clifft records benchmark results from its existing C++ Catch2 and Python pytest-benchmark suites on a daily schedule, so maintainers can spot performance drift between releases. The setup is intentionally small: it only records and stores history. It does not gate pull requests, post comments, or alert on regressions.

## Where to view the charts

Each scheduled run appends to a Chart.js viewer hosted on `gh-pages`:

- C++ Catch2 benchmarks: <https://unitaryfoundation.github.io/clifft/bench/cpp/>
- Python pytest-benchmark suite: <https://unitaryfoundation.github.io/clifft/bench/python/>

The two charts live on the same `gh-pages` branch as the docs site but under their own `/bench/` subpath, so they are not part of the versioned documentation tree.

## What gets recorded

Each chart's `data.js` contains the full history as a JS array, easy to skim if you need raw numbers:

- C++ Catch2 results come from the sampling benchmarks in [`tests/test_benchmarks.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_benchmarks.cc) (tagged `[bench]`).
- Python pytest-benchmark results come from the suites under [`tools/bench/`](https://github.com/unitaryfoundation/clifft/tree/main/tools/bench).

## When it runs

The [`bench.yml`](https://github.com/unitaryfoundation/clifft/blob/main/.github/workflows/bench.yml) workflow runs:

- Daily at 06:17 UTC.
- On manual dispatch from the **Actions** tab → **Benchmark history** → **Run workflow**.

It does not run on pull requests or pushes to `main`.

## Reading the results

A few caveats worth keeping in mind when interpreting the chart:

- **Runner noise.** The workflow uses GitHub-hosted `ubuntu-24.04` runners, which share hardware with other tenants. Expect roughly 5–10% variance run-to-run, more on the smaller fixtures. Trends across many days are meaningful; isolated spikes generally are not.
- **Two charts, two units.** The C++ chart plots elapsed time per benchmark in whatever unit Catch2's console reporter chose (ns/us/ms/s, picked per case to keep the printed mean readable). The Python chart plots throughput in iterations per second, the default `pytest-benchmark` metric the action records. Lower is better on the C++ chart; higher is better on the Python chart. The two are not directly comparable.
- **No alerts.** The workflow records data and stops. If you suspect a regression, run the relevant suite locally against `main` and the suspect commit (`just bench` for Python, `ctest -R Bench` for C++).

## Adding new benchmarks

New cases added to [`tests/test_benchmarks.cc`](https://github.com/unitaryfoundation/clifft/blob/main/tests/test_benchmarks.cc) (with the `[bench]` Catch2 tag) or under [`tools/bench/`](https://github.com/unitaryfoundation/clifft/tree/main/tools/bench) are picked up automatically by the next scheduled run; no workflow change is required.

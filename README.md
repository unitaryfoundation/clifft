# clifft benchmark history

This branch stores benchmark history for the clifft project. It is
auto-managed by `.github/workflows/bench.yml`, which runs the project
benchmarks on a daily schedule and on `workflow_dispatch`, then appends
results here via `benchmark-action/github-action-benchmark`.

Do not edit this branch by hand — pushes from the workflow are the only
expected source of new commits. The contents are not user-facing source
code; if you need to inspect history, see the rendered chart linked from
`docs/development/benchmark-history.md` on `main`.

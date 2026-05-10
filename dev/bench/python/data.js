window.BENCHMARK_DATA = {
  "lastUpdate": 1778418699502,
  "repoUrl": "https://github.com/unitaryfoundation/clifft",
  "entries": {
    "Python pytest-benchmark suite": [
      {
        "commit": {
          "author": {
            "email": "brad@unitary.foundation",
            "name": "Brad Chase",
            "username": "bachase"
          },
          "committer": {
            "email": "brad@unitary.foundation",
            "name": "Brad Chase",
            "username": "bachase"
          },
          "distinct": true,
          "id": "b5e69ffee6eaeed0812f5a4d239706ff6f709ad7",
          "message": "TEMP: trigger bench workflow on push to this branch (revert before merge)\n\nAdds a one-line `push: branches: [chore/bench-history-workflow]`\ntrigger so we can verify the workflow runs end-to-end before merging\nand writing to bench-data is no longer recoverable.\n\nThis commit will be reverted in the PR once both jobs have completed\nsuccessfully.\n\nAssisted-by: Claude (Opus 4.7) <noreply@anthropic.com>",
          "timestamp": "2026-05-10T13:09:03Z",
          "tree_id": "57442fd937679d082fb76ff80e8e451d826e3348",
          "url": "https://github.com/unitaryfoundation/clifft/commit/b5e69ffee6eaeed0812f5a4d239706ff6f709ad7"
        },
        "date": 1778418699235,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1464.5669976618044,
            "unit": "iter/sec",
            "range": "stddev: 0.000013455288063989599",
            "extra": "mean: 682.7956669763212 usec\nrounds: 1078"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 622.9249461610463,
            "unit": "iter/sec",
            "range": "stddev: 0.00002749569798876611",
            "extra": "mean: 1.6053298333335133 msec\nrounds: 534"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 54.25676756670908,
            "unit": "iter/sec",
            "range": "stddev: 0.0007331691243065412",
            "extra": "mean: 18.430880512195884 msec\nrounds: 41"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.1359793495239647,
            "unit": "iter/sec",
            "range": "stddev: 0.0014970980206172927",
            "extra": "mean: 880.2976924000006 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 18422.17328538478,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026284739879735798",
            "extra": "mean: 54.282411988456836 usec\nrounds: 6840"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1576.0173510156603,
            "unit": "iter/sec",
            "range": "stddev: 0.000014348378268302881",
            "extra": "mean: 634.5107808334423 usec\nrounds: 1200"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 102.84163921433209,
            "unit": "iter/sec",
            "range": "stddev: 0.00009722123443050594",
            "extra": "mean: 9.723687872340323 msec\nrounds: 94"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 11.5060768112571,
            "unit": "iter/sec",
            "range": "stddev: 0.0003598234717079268",
            "extra": "mean: 86.91059658333226 msec\nrounds: 12"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 75.26169992997809,
            "unit": "iter/sec",
            "range": "stddev: 0.00008725258777355701",
            "extra": "mean: 13.28697067605939 msec\nrounds: 71"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.387318091592981,
            "unit": "iter/sec",
            "range": "stddev: 0.0010670782468938857",
            "extra": "mean: 185.62111666666206 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.5414211819353127,
            "unit": "iter/sec",
            "range": "stddev: 0.002103359841498055",
            "extra": "mean: 1.8469909072000008 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}
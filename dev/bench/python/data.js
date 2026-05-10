window.BENCHMARK_DATA = {
  "lastUpdate": 1778419533796,
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
      },
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
          "id": "69d28cc80c60f81f1f2ba648f2eba3cfe10e414d",
          "message": "test(bench): shorten exp-val BENCHMARK name to fit Catch2 column\n\nThe first end-to-end run of the bench-history workflow failed at the\nCatch2 publish step with \"No benchmark found for bench suite\". Root\ncause: the BENCHMARK label \"exp-val 20q 200 probes x100000 shots\" (36\nchars) exceeded Catch2 v3's console-reporter name column (~35 chars)\nand wrapped onto two lines. The workflow's parser\n(benchmark-action/github-action-benchmark) reads each suite's first\nline for the benchmark stats; on a wrapped name it sees a row that\nmatches no number+unit pattern, returns null, and aborts the whole\npublish. The other 5 benches parsed fine but were discarded because\nthe action throws on any suite that yields zero results.\n\nShortens \"x100000 shots\" to \"x100k\" (28 chars) and adds a comment at\nthe call site warning future contributors about the column constraint\nso this doesn't recur.\n\nAssisted-by: Claude (Opus 4.7) <noreply@anthropic.com>",
          "timestamp": "2026-05-10T13:22:43Z",
          "tree_id": "50848cec7db0ae634af8baedba7eb3bd0bb6de07",
          "url": "https://github.com/unitaryfoundation/clifft/commit/69d28cc80c60f81f1f2ba648f2eba3cfe10e414d"
        },
        "date": 1778419532962,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1455.4790301113428,
            "unit": "iter/sec",
            "range": "stddev: 0.000013271430073626315",
            "extra": "mean: 687.0590227078029 usec\nrounds: 1145"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 595.9773120709764,
            "unit": "iter/sec",
            "range": "stddev: 0.000018456265308184534",
            "extra": "mean: 1.6779162222217405 msec\nrounds: 540"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 57.902463849726566,
            "unit": "iter/sec",
            "range": "stddev: 0.0005466028429183629",
            "extra": "mean: 17.27042224999761 msec\nrounds: 44"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0397895166725297,
            "unit": "iter/sec",
            "range": "stddev: 0.0022198685723466296",
            "extra": "mean: 961.7331045999947 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17760.872489434838,
            "unit": "iter/sec",
            "range": "stddev: 0.000003722753989332419",
            "extra": "mean: 56.30354030157336 usec\nrounds: 7692"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1527.9086786385628,
            "unit": "iter/sec",
            "range": "stddev: 0.000011931988984020524",
            "extra": "mean: 654.4893775268338 usec\nrounds: 1237"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 109.27054089954792,
            "unit": "iter/sec",
            "range": "stddev: 0.00009959584224899582",
            "extra": "mean: 9.151597418368203 msec\nrounds: 98"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 9.336374775677399,
            "unit": "iter/sec",
            "range": "stddev: 0.000284692864675567",
            "extra": "mean: 107.10795399999837 msec\nrounds: 9"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 70.83145740324576,
            "unit": "iter/sec",
            "range": "stddev: 0.00005515500318425169",
            "extra": "mean: 14.11802095652173 msec\nrounds: 69"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.770368258622493,
            "unit": "iter/sec",
            "range": "stddev: 0.0002767401768458125",
            "extra": "mean: 173.29916483332397 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.578866974205754,
            "unit": "iter/sec",
            "range": "stddev: 0.0027564112586944394",
            "extra": "mean: 1.7275126144000013 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}
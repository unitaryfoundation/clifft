window.BENCHMARK_DATA = {
  "lastUpdate": 1778419876262,
  "repoUrl": "https://github.com/unitaryfoundation/clifft",
  "entries": {
    "C++ Catch2 benchmarks": [
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
          "id": "7bb90715922069c3eb94dfc5c067f5ddc8226f95",
          "message": "ci(bench): pin CPU baseline to x86-64-v3 for cross-runner consistency\n\nThe bench-cpp job's second run failed at link-time when\ncatch_discover_tests executed the freshly-built binary and got\nSIGILL (\"Illegal instruction\"). Cause: GitHub-hosted runners draw\nfrom a heterogeneous pool, and the default CLIFFT_CPU_BASELINE=native\nemits whatever ISA the build host supports — instructions the next\nrunner's CPU may not. Pinning to x86-64-v3 (AVX2/FMA, ~2015 CPUs)\ngives a fixed instruction floor that any current GitHub runner can\nexecute, and as a bonus makes day-to-day timings more comparable\nthan letting -march=native vary across heterogeneous hardware.\n\nAssisted-by: Claude (Opus 4.7) <noreply@anthropic.com>",
          "timestamp": "2026-05-10T13:27:30Z",
          "tree_id": "b8400cb047ff481d8cef347749f17df931fd3597",
          "url": "https://github.com/unitaryfoundation/clifft/commit/7bb90715922069c3eb94dfc5c067f5ddc8226f95"
        },
        "date": 1778419875431,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 23.3255,
            "range": "± 103.591",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 40.0452,
            "range": "± 98.108",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 77.1566,
            "range": "± 1.16562",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 86.4966,
            "range": "± 554",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 31.697,
            "range": "± 108.956",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 109.814,
            "range": "± 2.05144",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          }
        ]
      }
    ]
  }
}
window.BENCHMARK_DATA = {
  "lastUpdate": 1779130217408,
  "repoUrl": "https://github.com/unitaryfoundation/clifft",
  "entries": {
    "C++ Catch2 benchmarks": [
      {
        "commit": {
          "author": {
            "name": "Brad Chase",
            "username": "bachase",
            "email": "14430+bachase@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0bdf0ebab61aa3eddfc7056636726a45a91ad330",
          "message": "ci: add scheduled benchmark history workflow (#38) (#70)\n\nRecords C++ Catch2 ([bench] tag) and Python pytest-benchmark results on\na daily schedule, published via benchmark-action/github-action-benchmark\nto gh-pages under /bench/{cpp,python}/. Two independent jobs so a Python\nfailure does not gate the C++ history. Concurrency group is shared with\ndocs.yml/release.yml/PR-preview so scheduled runs cannot race a mike\npush. CPU baseline pinned to x86-64-v3 for cross-runner consistency.\nRecord-only: no alerts or PR comments.\n\nAssisted-by: Claude (Opus 4.7) <noreply@anthropic.com>",
          "timestamp": "2026-05-14T13:07:18Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/0bdf0ebab61aa3eddfc7056636726a45a91ad330"
        },
        "date": 1778768147729,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 27.044,
            "range": "± 328.966",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 56.4419,
            "range": "± 544.85",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 95.6666,
            "range": "± 275.845",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 103.241,
            "range": "± 106.35",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 39.4697,
            "range": "± 229.65",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 164.599,
            "range": "± 711.334",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Brad Chase",
            "username": "bachase",
            "email": "14430+bachase@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "609e10d203ca797befcc7645148529801f330c17",
          "message": "fix(svm): tighten AVX-2 dispatch and trap CLIFFT_FORCE_ISA misconfig (#94)\n\nfix(svm): tighten AVX-2 dispatch and trap CLIFFT_FORCE_ISA misconfig\n\n* AVX-2 dispatch now requires fma in addition to avx2 + bmi2, matching\n  what svm_avx2.cc is compiled with.\n* CLIFFT_FORCE_ISA verifies the host can execute the requested ISA and\n  installs a trap function that throws std::runtime_error on first\n  execute() if not. Parser is exact, case-insensitive, and rejects\n  unknown values via the same trap mechanism.\n* svm_backend() reports trap:avx2 / trap:avx512 / trap:unknown when a\n  trap is installed; C++ header and Python docstring updated.\n\nCloses #93.\n\nAssisted-by: Claude (Opus 4.7) <noreply@anthropic.com>",
          "timestamp": "2026-05-18T16:45:49Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/609e10d203ca797befcc7645148529801f330c17"
        },
        "date": 1779130214766,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 26.6278,
            "range": "± 1.18317",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 54.0604,
            "range": "± 1.66355",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 96.5821,
            "range": "± 474.882",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 102.962,
            "range": "± 246.748",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 40.2854,
            "range": "± 777.153",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 176.178,
            "range": "± 1.93603",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          }
        ]
      }
    ]
  }
}
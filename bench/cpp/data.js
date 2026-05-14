window.BENCHMARK_DATA = {
  "lastUpdate": 1778768150025,
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
      }
    ]
  }
}
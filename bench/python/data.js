window.BENCHMARK_DATA = {
  "lastUpdate": 1779002462005,
  "repoUrl": "https://github.com/unitaryfoundation/clifft",
  "entries": {
    "Python pytest-benchmark suite": [
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
        "date": 1778768063285,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1440.6321560734198,
            "unit": "iter/sec",
            "range": "stddev: 0.000014836441703994988",
            "extra": "mean: 694.1397190005777 usec\nrounds: 1121"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 615.7244424220887,
            "unit": "iter/sec",
            "range": "stddev: 0.000025742098042784203",
            "extra": "mean: 1.624103139492527 msec\nrounds: 552"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 57.17234129051469,
            "unit": "iter/sec",
            "range": "stddev: 0.0005011522319604521",
            "extra": "mean: 17.49097513636209 msec\nrounds: 44"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0605997953216715,
            "unit": "iter/sec",
            "range": "stddev: 0.03207498360898713",
            "extra": "mean: 942.8627126000038 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17603.509414140644,
            "unit": "iter/sec",
            "range": "stddev: 0.000003956630158358559",
            "extra": "mean: 56.80685461483689 usec\nrounds: 7628"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1497.8817628406196,
            "unit": "iter/sec",
            "range": "stddev: 0.00008142791258926484",
            "extra": "mean: 667.6094367445769 usec\nrounds: 1241"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 109.37860084506005,
            "unit": "iter/sec",
            "range": "stddev: 0.0001243561355367914",
            "extra": "mean: 9.142556151514018 msec\nrounds: 99"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 9.70131850289942,
            "unit": "iter/sec",
            "range": "stddev: 0.00034549636022571415",
            "extra": "mean: 103.07877219999853 msec\nrounds: 10"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 68.6494611287336,
            "unit": "iter/sec",
            "range": "stddev: 0.00007406535353061622",
            "extra": "mean: 14.566756731342275 msec\nrounds: 67"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.727882693145733,
            "unit": "iter/sec",
            "range": "stddev: 0.0011595973232076792",
            "extra": "mean: 174.5845810000001 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.5696013293572721,
            "unit": "iter/sec",
            "range": "stddev: 0.03851816732699114",
            "extra": "mean: 1.755613880200002 sec\nrounds: 5"
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
          "id": "d3b34225e3611526edcfdc8b810c002706f7e9c2",
          "message": "chore(bench): serialize bench-python after bench-cpp to avoid gh-pages race (#87)\n\n`needs: bench-cpp` with `always() && !cancelled()` so the two\nauto-push jobs don't race each other on gh-pages, while keeping\nPython independent of a C++ failure and still honoring cancellation.\n\nAssisted-by: Claude (Opus 4.7) <noreply@anthropic.com>",
          "timestamp": "2026-05-14T14:23:14Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/d3b34225e3611526edcfdc8b810c002706f7e9c2"
        },
        "date": 1778830331868,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1438.8982195502492,
            "unit": "iter/sec",
            "range": "stddev: 0.000011837760587754486",
            "extra": "mean: 694.9761883175908 usec\nrounds: 1147"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 615.9735430586952,
            "unit": "iter/sec",
            "range": "stddev: 0.000024117825497127502",
            "extra": "mean: 1.6234463497155613 msec\nrounds: 529"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 56.41814656037566,
            "unit": "iter/sec",
            "range": "stddev: 0.0006182705781185618",
            "extra": "mean: 17.724793545457118 msec\nrounds: 44"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 0.9523820477110069,
            "unit": "iter/sec",
            "range": "stddev: 0.10531581273906734",
            "extra": "mean: 1.0499987924000038 sec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17513.448893962817,
            "unit": "iter/sec",
            "range": "stddev: 0.00000440124184646052",
            "extra": "mean: 57.09897610999493 usec\nrounds: 7409"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1536.8965825086896,
            "unit": "iter/sec",
            "range": "stddev: 0.000014140242002971694",
            "extra": "mean: 650.6618671554929 usec\nrounds: 1227"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 111.55082714465576,
            "unit": "iter/sec",
            "range": "stddev: 0.00009569925620873186",
            "extra": "mean: 8.96452339795948 msec\nrounds: 98"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 9.107935115215648,
            "unit": "iter/sec",
            "range": "stddev: 0.001594603898160893",
            "extra": "mean: 109.79437022222606 msec\nrounds: 9"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 67.76242256958136,
            "unit": "iter/sec",
            "range": "stddev: 0.00011693484194361132",
            "extra": "mean: 14.757441692306044 msec\nrounds: 65"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.732917855654426,
            "unit": "iter/sec",
            "range": "stddev: 0.0006495405411941205",
            "extra": "mean: 174.4312451666635 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.5685892808260812,
            "unit": "iter/sec",
            "range": "stddev: 0.032041252462023155",
            "extra": "mean: 1.7587387481999996 sec\nrounds: 5"
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
          "id": "d0f8f5e5065bd89dc7ba5459c061e2ff306f777b",
          "message": "chore: guard benchmark CPU baseline (#89)\n\nThe failed nightly run hit an illegal instruction while Catch2 discovery\nexecuted the freshly linked test binary. Since GitHub-hosted x64 runners\ndo not explicitly guarantee the x86-64-v3 feature set, this avoids\ntreating rare runner placement as a source regression without adding a\ncompile-and-run CPU probe.",
          "timestamp": "2026-05-15T14:19:15Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/d0f8f5e5065bd89dc7ba5459c061e2ff306f777b"
        },
        "date": 1778915323477,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1460.1649054666536,
            "unit": "iter/sec",
            "range": "stddev: 0.00001181112297471878",
            "extra": "mean: 684.8541532919601 usec\nrounds: 1109"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 629.9631203753061,
            "unit": "iter/sec",
            "range": "stddev: 0.000022504009520042378",
            "extra": "mean: 1.5873945119267319 msec\nrounds: 545"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 54.616036338743406,
            "unit": "iter/sec",
            "range": "stddev: 0.0005094006486087923",
            "extra": "mean: 18.309640666666656 msec\nrounds: 42"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.1624280633717872,
            "unit": "iter/sec",
            "range": "stddev: 0.003701891068960202",
            "extra": "mean: 860.2682879999975 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 18407.03486822157,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029046639404597373",
            "extra": "mean: 54.32705523508452 usec\nrounds: 6934"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1584.8993135161118,
            "unit": "iter/sec",
            "range": "stddev: 0.000014329530350979587",
            "extra": "mean: 630.9549076537185 usec\nrounds: 1202"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 103.24572936456923,
            "unit": "iter/sec",
            "range": "stddev: 0.00008281229618033992",
            "extra": "mean: 9.685630642105469 msec\nrounds: 95"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 12.300834352347554,
            "unit": "iter/sec",
            "range": "stddev: 0.00017700803268555455",
            "extra": "mean: 81.29529846153524 msec\nrounds: 13"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 75.00918776603031,
            "unit": "iter/sec",
            "range": "stddev: 0.000038335281128067045",
            "extra": "mean: 13.331700152776135 msec\nrounds: 72"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.4186992516126296,
            "unit": "iter/sec",
            "range": "stddev: 0.00015036290223061682",
            "extra": "mean: 184.54613433332648 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.5424045136517058,
            "unit": "iter/sec",
            "range": "stddev: 0.0033885782184684353",
            "extra": "mean: 1.8436424749999958 sec\nrounds: 5"
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
          "id": "d0f8f5e5065bd89dc7ba5459c061e2ff306f777b",
          "message": "chore: guard benchmark CPU baseline (#89)\n\nThe failed nightly run hit an illegal instruction while Catch2 discovery\nexecuted the freshly linked test binary. Since GitHub-hosted x64 runners\ndo not explicitly guarantee the x86-64-v3 feature set, this avoids\ntreating rare runner placement as a source regression without adding a\ncompile-and-run CPU probe.",
          "timestamp": "2026-05-15T14:19:15Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/d0f8f5e5065bd89dc7ba5459c061e2ff306f777b"
        },
        "date": 1779002460553,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1466.6320414920772,
            "unit": "iter/sec",
            "range": "stddev: 0.000014780879497362964",
            "extra": "mean: 681.8342786120031 usec\nrounds: 1066"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 620.8772169216699,
            "unit": "iter/sec",
            "range": "stddev: 0.0000316507593740737",
            "extra": "mean: 1.6106244080883394 msec\nrounds: 544"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 54.52123076262132,
            "unit": "iter/sec",
            "range": "stddev: 0.0004885503354690836",
            "extra": "mean: 18.341478833335145 msec\nrounds: 42"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.1599916238582466,
            "unit": "iter/sec",
            "range": "stddev: 0.002566871272943033",
            "extra": "mean: 862.0751903999974 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 18512.84587639194,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027092602041591576",
            "extra": "mean: 54.0165464930071 usec\nrounds: 6829"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1571.9027833912762,
            "unit": "iter/sec",
            "range": "stddev: 0.00001451089245395403",
            "extra": "mean: 636.1716580478127 usec\nrounds: 1199"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 102.82668689751925,
            "unit": "iter/sec",
            "range": "stddev: 0.00008568654008718406",
            "extra": "mean: 9.725101821053864 msec\nrounds: 95"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 11.251046881785053,
            "unit": "iter/sec",
            "range": "stddev: 0.00008534605080083559",
            "extra": "mean: 88.88061799999747 msec\nrounds: 12"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 73.50517271844491,
            "unit": "iter/sec",
            "range": "stddev: 0.00005608013022824658",
            "extra": "mean: 13.604484732393079 msec\nrounds: 71"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.392754072368341,
            "unit": "iter/sec",
            "range": "stddev: 0.0006093782129893003",
            "extra": "mean: 185.43400766666687 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.5374807507154543,
            "unit": "iter/sec",
            "range": "stddev: 0.031812886712965394",
            "extra": "mean: 1.8605317468000009 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}
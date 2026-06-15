window.BENCHMARK_DATA = {
  "lastUpdate": 1781512577648,
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
        "date": 1779090563590,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1459.4723493503084,
            "unit": "iter/sec",
            "range": "stddev: 0.000013182229218381714",
            "extra": "mean: 685.1791337089429 usec\nrounds: 1062"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 613.9275536996967,
            "unit": "iter/sec",
            "range": "stddev: 0.000020701167573430332",
            "extra": "mean: 1.628856685082343 msec\nrounds: 543"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 54.54339585101052,
            "unit": "iter/sec",
            "range": "stddev: 0.0005380636953634874",
            "extra": "mean: 18.334025309527426 msec\nrounds: 42"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.1649441451499876,
            "unit": "iter/sec",
            "range": "stddev: 0.0011352777752921593",
            "extra": "mean: 858.4102544000075 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 18488.309708149296,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023588984945924087",
            "extra": "mean: 54.08823282310221 usec\nrounds: 6928"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1528.810802968669,
            "unit": "iter/sec",
            "range": "stddev: 0.00003758842200708288",
            "extra": "mean: 654.1031748717267 usec\nrounds: 1178"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 102.81746266376351,
            "unit": "iter/sec",
            "range": "stddev: 0.00032612849921846465",
            "extra": "mean: 9.72597430526201 msec\nrounds: 95"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 12.325542003004747,
            "unit": "iter/sec",
            "range": "stddev: 0.0003376184705147459",
            "extra": "mean: 81.13233476923106 msec\nrounds: 13"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 73.5085348753627,
            "unit": "iter/sec",
            "range": "stddev: 0.0005433224819160259",
            "extra": "mean: 13.60386248611197 msec\nrounds: 72"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.41933297886865,
            "unit": "iter/sec",
            "range": "stddev: 0.001167566337390347",
            "extra": "mean: 184.52455383333208 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.5438934003361636,
            "unit": "iter/sec",
            "range": "stddev: 0.004923283367901401",
            "extra": "mean: 1.8385955765999937 sec\nrounds: 5"
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
        "date": 1779130386222,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1452.0761561334525,
            "unit": "iter/sec",
            "range": "stddev: 0.000013880536270735205",
            "extra": "mean: 688.6691140654577 usec\nrounds: 1166"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 623.0865482520069,
            "unit": "iter/sec",
            "range": "stddev: 0.000020786702383316186",
            "extra": "mean: 1.6049134792034552 msec\nrounds: 553"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 57.85601849155118,
            "unit": "iter/sec",
            "range": "stddev: 0.00045397944855190067",
            "extra": "mean: 17.28428651110916 msec\nrounds: 45"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.058017636009204,
            "unit": "iter/sec",
            "range": "stddev: 0.003876445062213851",
            "extra": "mean: 945.1638290000119 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17375.363546504163,
            "unit": "iter/sec",
            "range": "stddev: 0.000005878530348976368",
            "extra": "mean: 57.55275262722173 usec\nrounds: 7992"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1517.4443549094663,
            "unit": "iter/sec",
            "range": "stddev: 0.0000155438589345393",
            "extra": "mean: 659.002748117022 usec\nrounds: 1195"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 111.32890806470904,
            "unit": "iter/sec",
            "range": "stddev: 0.00013827565536939258",
            "extra": "mean: 8.982392959596424 msec\nrounds: 99"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 9.508082953375855,
            "unit": "iter/sec",
            "range": "stddev: 0.0019811481792487125",
            "extra": "mean: 105.1736722222169 msec\nrounds: 9"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 68.66859966703144,
            "unit": "iter/sec",
            "range": "stddev: 0.00012934214767823822",
            "extra": "mean: 14.562696849053573 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.753086496626411,
            "unit": "iter/sec",
            "range": "stddev: 0.0002725132881169704",
            "extra": "mean: 173.81974016667337 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.5745240683035939,
            "unit": "iter/sec",
            "range": "stddev: 0.010379663107994088",
            "extra": "mean: 1.7405711182000005 sec\nrounds: 5"
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
        "date": 1779176183919,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1435.0712310450606,
            "unit": "iter/sec",
            "range": "stddev: 0.000055021701867502045",
            "extra": "mean: 696.8295220243324 usec\nrounds: 1067"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 628.317370766796,
            "unit": "iter/sec",
            "range": "stddev: 0.00003311531973663367",
            "extra": "mean: 1.591552369114997 msec\nrounds: 531"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 54.11993596095474,
            "unit": "iter/sec",
            "range": "stddev: 0.0005471804497649833",
            "extra": "mean: 18.477479365856198 msec\nrounds: 41"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.2035252858718843,
            "unit": "iter/sec",
            "range": "stddev: 0.00108681162491539",
            "extra": "mean: 830.8923890000017 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 18423.179623781645,
            "unit": "iter/sec",
            "range": "stddev: 0.000003785267343003754",
            "extra": "mean: 54.2794468935832 usec\nrounds: 6261"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1546.5522215875915,
            "unit": "iter/sec",
            "range": "stddev: 0.00006439323045432338",
            "extra": "mean: 646.5995690552654 usec\nrounds: 1144"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 103.39226228851011,
            "unit": "iter/sec",
            "range": "stddev: 0.000045477522686533695",
            "extra": "mean: 9.6719036595752 msec\nrounds: 94"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 11.48416033166034,
            "unit": "iter/sec",
            "range": "stddev: 0.005712526275290765",
            "extra": "mean: 87.07645758333153 msec\nrounds: 12"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 73.22602638785479,
            "unit": "iter/sec",
            "range": "stddev: 0.00037159641882229843",
            "extra": "mean: 13.656346647888835 msec\nrounds: 71"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.348142387142381,
            "unit": "iter/sec",
            "range": "stddev: 0.00268921580989938",
            "extra": "mean: 186.98081083333307 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.5389856532269532,
            "unit": "iter/sec",
            "range": "stddev: 0.006263676325121646",
            "extra": "mean: 1.8553369538000026 sec\nrounds: 5"
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
          "id": "687944d50e517e5c1bdaee3748f4d93cf1af2919",
          "message": "ci: replace stripped-binary audit with QEMU wheel smoke (#100)\n\nThe audit's AVX-512 scan never ran on the published wheel and produced\nfalse positives on stripped binaries. Replace it with a qemu-x86_64\nsmoke against Haswell + Nehalem that asserts both pass paths and clean\nCLIFFT_FORCE_ISA traps. Runs on every PR (~70s) and against the\ncibuildwheel artifact in release.yml. The compile-flag check is\nretained as audit_build_flags.py.\n\nAssisted-by: Claude (Opus 4.7) <noreply@anthropic.com>",
          "timestamp": "2026-05-19T16:45:30Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/687944d50e517e5c1bdaee3748f4d93cf1af2919"
        },
        "date": 1779262841938,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1412.024846642714,
            "unit": "iter/sec",
            "range": "stddev: 0.0001086540718120113",
            "extra": "mean: 708.2028353662752 usec\nrounds: 1148"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 629.7836176382696,
            "unit": "iter/sec",
            "range": "stddev: 0.000020128795156946534",
            "extra": "mean: 1.5878469556735477 msec\nrounds: 564"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 56.19592940064878,
            "unit": "iter/sec",
            "range": "stddev: 0.0012684582290606306",
            "extra": "mean: 17.79488320000016 msec\nrounds: 45"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0202712489020564,
            "unit": "iter/sec",
            "range": "stddev: 0.0018528598278371568",
            "extra": "mean: 980.1315101999876 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17518.566613516603,
            "unit": "iter/sec",
            "range": "stddev: 0.000004032220409818009",
            "extra": "mean: 57.08229571867148 usec\nrounds: 7825"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1465.4484174369,
            "unit": "iter/sec",
            "range": "stddev: 0.000019538098840829735",
            "extra": "mean: 682.3849874900551 usec\nrounds: 1199"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 110.8655904621751,
            "unit": "iter/sec",
            "range": "stddev: 0.00011170935913482612",
            "extra": "mean: 9.019931214285807 msec\nrounds: 98"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 7.391249021515524,
            "unit": "iter/sec",
            "range": "stddev: 0.001414576164916692",
            "extra": "mean: 135.2951303749954 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.79817293446854,
            "unit": "iter/sec",
            "range": "stddev: 0.0002284458996671748",
            "extra": "mean: 18.587991849055882 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.396266679696725,
            "unit": "iter/sec",
            "range": "stddev: 0.00028142158241372",
            "extra": "mean: 156.341198714281 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6346902127948162,
            "unit": "iter/sec",
            "range": "stddev: 0.03286887028399888",
            "extra": "mean: 1.5755717983999886 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1779349339191,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1441.987942711674,
            "unit": "iter/sec",
            "range": "stddev: 0.0000183614941266057",
            "extra": "mean: 693.4870746002836 usec\nrounds: 1126"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 623.562634058467,
            "unit": "iter/sec",
            "range": "stddev: 0.00010852347365819827",
            "extra": "mean: 1.6036881387383408 msec\nrounds: 555"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 56.7846137814685,
            "unit": "iter/sec",
            "range": "stddev: 0.000583221999921746",
            "extra": "mean: 17.610404181816364 msec\nrounds: 44"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.028683320584116,
            "unit": "iter/sec",
            "range": "stddev: 0.0020254886471471104",
            "extra": "mean: 972.1164715999976 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17566.81954065923,
            "unit": "iter/sec",
            "range": "stddev: 0.000004041594791493083",
            "extra": "mean: 56.92550081051683 usec\nrounds: 8019"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1465.98731314864,
            "unit": "iter/sec",
            "range": "stddev: 0.000015292196542967694",
            "extra": "mean: 682.1341433386659 usec\nrounds: 1186"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 111.09160599355546,
            "unit": "iter/sec",
            "range": "stddev: 0.00013278448183815482",
            "extra": "mean: 9.00158019191847 msec\nrounds: 99"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 8.008217295832917,
            "unit": "iter/sec",
            "range": "stddev: 0.0032166141184284",
            "extra": "mean: 124.87173650000116 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.40073585167181,
            "unit": "iter/sec",
            "range": "stddev: 0.000162977037292256",
            "extra": "mean: 18.726333711536167 msec\nrounds: 52"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.371009683730893,
            "unit": "iter/sec",
            "range": "stddev: 0.0013880104075688375",
            "extra": "mean: 156.96099199999887 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6331743641452453,
            "unit": "iter/sec",
            "range": "stddev: 0.0516475595281424",
            "extra": "mean: 1.579343790000013 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1779435485445,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1453.7609926321768,
            "unit": "iter/sec",
            "range": "stddev: 0.000013945291701569107",
            "extra": "mean: 687.8709809027149 usec\nrounds: 1152"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 621.1989614858219,
            "unit": "iter/sec",
            "range": "stddev: 0.00004827229718636808",
            "extra": "mean: 1.6097901992755081 msec\nrounds: 552"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 57.34219631139779,
            "unit": "iter/sec",
            "range": "stddev: 0.0004822420307063031",
            "extra": "mean: 17.439164600000367 msec\nrounds: 45"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.022514492411305,
            "unit": "iter/sec",
            "range": "stddev: 0.007647243606009758",
            "extra": "mean: 977.9812485999969 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17524.352176073804,
            "unit": "iter/sec",
            "range": "stddev: 0.000004168674320814668",
            "extra": "mean: 57.06345033200779 usec\nrounds: 6171"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1458.3501936547195,
            "unit": "iter/sec",
            "range": "stddev: 0.000019429651024628293",
            "extra": "mean: 685.7063580140074 usec\nrounds: 1148"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 111.15831596452678,
            "unit": "iter/sec",
            "range": "stddev: 0.00011330714732124211",
            "extra": "mean: 8.996178030612873 msec\nrounds: 98"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 7.549256670045269,
            "unit": "iter/sec",
            "range": "stddev: 0.0016789003365683332",
            "extra": "mean: 132.46337271428388 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.95313104942483,
            "unit": "iter/sec",
            "range": "stddev: 0.00006151438436858378",
            "extra": "mean: 18.534605509436147 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.3166603141216315,
            "unit": "iter/sec",
            "range": "stddev: 0.0023921309712571785",
            "extra": "mean: 158.3115048571447 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6269778376400614,
            "unit": "iter/sec",
            "range": "stddev: 0.03630105173564982",
            "extra": "mean: 1.5949527079999997 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1779520784987,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1455.9040217060913,
            "unit": "iter/sec",
            "range": "stddev: 0.00001261787027328161",
            "extra": "mean: 686.8584639447295 usec\nrounds: 1151"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 632.06745668484,
            "unit": "iter/sec",
            "range": "stddev: 0.00001769498413300002",
            "extra": "mean: 1.5821096141303437 msec\nrounds: 552"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 57.32942077755858,
            "unit": "iter/sec",
            "range": "stddev: 0.0006439998321296372",
            "extra": "mean: 17.44305081818386 msec\nrounds: 44"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0287304633880034,
            "unit": "iter/sec",
            "range": "stddev: 0.003409121813942917",
            "extra": "mean: 972.0719231999965 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17618.027478857708,
            "unit": "iter/sec",
            "range": "stddev: 0.000003850983981918779",
            "extra": "mean: 56.76004315466288 usec\nrounds: 8064"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1478.0473712745422,
            "unit": "iter/sec",
            "range": "stddev: 0.000015679726089095668",
            "extra": "mean: 676.5683018249172 usec\nrounds: 1206"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 112.6254285080234,
            "unit": "iter/sec",
            "range": "stddev: 0.00005594376043394612",
            "extra": "mean: 8.878989525254152 msec\nrounds: 99"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 8.072000997739057,
            "unit": "iter/sec",
            "range": "stddev: 0.0007864995936737486",
            "extra": "mean: 123.88501937500962 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.91030361776264,
            "unit": "iter/sec",
            "range": "stddev: 0.00015772947581279618",
            "extra": "mean: 18.549329773585526 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.408173261010032,
            "unit": "iter/sec",
            "range": "stddev: 0.0002071813813415187",
            "extra": "mean: 156.05071200000356 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6423449374811698,
            "unit": "iter/sec",
            "range": "stddev: 0.004962179293214251",
            "extra": "mean: 1.5567959543999905 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1779607966081,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1439.8342574013009,
            "unit": "iter/sec",
            "range": "stddev: 0.000013577259570428412",
            "extra": "mean: 694.5243835251288 usec\nrounds: 1129"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 626.6386391351019,
            "unit": "iter/sec",
            "range": "stddev: 0.000039812314670840014",
            "extra": "mean: 1.595816053380012 msec\nrounds: 562"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 55.911384714656336,
            "unit": "iter/sec",
            "range": "stddev: 0.0006050111485011157",
            "extra": "mean: 17.885445068182417 msec\nrounds: 44"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0191371621605454,
            "unit": "iter/sec",
            "range": "stddev: 0.0006064454731454159",
            "extra": "mean: 981.2221917999977 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17694.188776019117,
            "unit": "iter/sec",
            "range": "stddev: 0.000004357169579299842",
            "extra": "mean: 56.51573025802105 usec\nrounds: 7522"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1466.4915624190858,
            "unit": "iter/sec",
            "range": "stddev: 0.000014294290431132015",
            "extra": "mean: 681.899593305826 usec\nrounds: 1195"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 104.29161355236735,
            "unit": "iter/sec",
            "range": "stddev: 0.00016066069720872982",
            "extra": "mean: 9.588498690720474 msec\nrounds: 97"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 7.873313124579094,
            "unit": "iter/sec",
            "range": "stddev: 0.0034531086399543333",
            "extra": "mean: 127.01133362499917 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.919393050757186,
            "unit": "iter/sec",
            "range": "stddev: 0.00007442532310250502",
            "extra": "mean: 18.546202830188516 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.409610430699809,
            "unit": "iter/sec",
            "range": "stddev: 0.0004066709970071042",
            "extra": "mean: 156.01572214285397 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6397655130213317,
            "unit": "iter/sec",
            "range": "stddev: 0.01700279185118948",
            "extra": "mean: 1.5630726878000019 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1779696076357,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1409.7868558797677,
            "unit": "iter/sec",
            "range": "stddev: 0.00000833934271556246",
            "extra": "mean: 709.3270843243585 usec\nrounds: 1091"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 615.0704341891253,
            "unit": "iter/sec",
            "range": "stddev: 0.00002596082658197926",
            "extra": "mean: 1.6258300585010959 msec\nrounds: 547"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 59.37382902099805,
            "unit": "iter/sec",
            "range": "stddev: 0.00039016820650269036",
            "extra": "mean: 16.84243742552534 msec\nrounds: 47"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.1294602028477716,
            "unit": "iter/sec",
            "range": "stddev: 0.0077349272501351886",
            "extra": "mean: 885.37869459999 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 20683.97102705491,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018539321178287814",
            "extra": "mean: 48.34661577759834 usec\nrounds: 7631"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1492.2450390898757,
            "unit": "iter/sec",
            "range": "stddev: 0.000009543770963987753",
            "extra": "mean: 670.1312276500531 usec\nrounds: 1085"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 120.72978839069906,
            "unit": "iter/sec",
            "range": "stddev: 0.00006649972365988802",
            "extra": "mean: 8.282959933333565 msec\nrounds: 105"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 13.260286619748507,
            "unit": "iter/sec",
            "range": "stddev: 0.00013635295388234384",
            "extra": "mean: 75.413151214296 msec\nrounds: 14"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 57.55947414379182,
            "unit": "iter/sec",
            "range": "stddev: 0.00005326633903192434",
            "extra": "mean: 17.373334535716165 msec\nrounds: 56"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 7.136372206741491,
            "unit": "iter/sec",
            "range": "stddev: 0.0012574180944280072",
            "extra": "mean: 140.1272202499939 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.7136435043552116,
            "unit": "iter/sec",
            "range": "stddev: 0.008227383468906148",
            "extra": "mean: 1.4012598642000058 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1779781096529,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1438.3977490591244,
            "unit": "iter/sec",
            "range": "stddev: 0.000012228411374656686",
            "extra": "mean: 695.2179956163819 usec\nrounds: 1141"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 624.1587595333126,
            "unit": "iter/sec",
            "range": "stddev: 0.00003079070986189201",
            "extra": "mean: 1.6021564781814586 msec\nrounds: 550"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 55.0174818048447,
            "unit": "iter/sec",
            "range": "stddev: 0.0014322563355308655",
            "extra": "mean: 18.17604090909051 msec\nrounds: 44"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.019080002121837,
            "unit": "iter/sec",
            "range": "stddev: 0.00448288598787126",
            "extra": "mean: 981.2772284000175 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17609.72279839059,
            "unit": "iter/sec",
            "range": "stddev: 0.000004099510851798239",
            "extra": "mean: 56.786810982135 usec\nrounds: 7248"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1474.097473658117,
            "unit": "iter/sec",
            "range": "stddev: 0.000017771361535098737",
            "extra": "mean: 678.3811911151317 usec\nrounds: 1193"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 106.02182517080996,
            "unit": "iter/sec",
            "range": "stddev: 0.00006249859390190466",
            "extra": "mean: 9.432020231578894 msec\nrounds: 95"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 8.197375161330706,
            "unit": "iter/sec",
            "range": "stddev: 0.0010202619608974386",
            "extra": "mean: 121.99026887500253 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.43454290121038,
            "unit": "iter/sec",
            "range": "stddev: 0.00035362994919543514",
            "extra": "mean: 18.71448590565839 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.387570479893718,
            "unit": "iter/sec",
            "range": "stddev: 0.00022144063023430641",
            "extra": "mean: 156.55404557142967 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6346790847651431,
            "unit": "iter/sec",
            "range": "stddev: 0.03684627717067216",
            "extra": "mean: 1.5755994234000013 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1779868313213,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1437.075471386646,
            "unit": "iter/sec",
            "range": "stddev: 0.00002002666080170144",
            "extra": "mean: 695.8576775617022 usec\nrounds: 1132"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 622.2000766159013,
            "unit": "iter/sec",
            "range": "stddev: 0.000024210927830191496",
            "extra": "mean: 1.6072000592460927 msec\nrounds: 557"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 56.516149300397956,
            "unit": "iter/sec",
            "range": "stddev: 0.0004586023497597249",
            "extra": "mean: 17.69405758139574 msec\nrounds: 43"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0121526412556034,
            "unit": "iter/sec",
            "range": "stddev: 0.005619220266410214",
            "extra": "mean: 987.9932722000035 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17607.280275997382,
            "unit": "iter/sec",
            "range": "stddev: 0.000004210011920714815",
            "extra": "mean: 56.79468857908857 usec\nrounds: 7530"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1471.0169971664175,
            "unit": "iter/sec",
            "range": "stddev: 0.000015327451011913272",
            "extra": "mean: 679.801798297555 usec\nrounds: 1175"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 110.13716541020854,
            "unit": "iter/sec",
            "range": "stddev: 0.0001084578737032528",
            "extra": "mean: 9.079587224489353 msec\nrounds: 98"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 7.989791658921737,
            "unit": "iter/sec",
            "range": "stddev: 0.004016518285348979",
            "extra": "mean: 125.15970912500052 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.505711101173205,
            "unit": "iter/sec",
            "range": "stddev: 0.0001109903206913498",
            "extra": "mean: 18.68959367924508 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.388553505294197,
            "unit": "iter/sec",
            "range": "stddev: 0.0003762899073915421",
            "extra": "mean: 156.52995614285763 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6323362388125846,
            "unit": "iter/sec",
            "range": "stddev: 0.03607309054645187",
            "extra": "mean: 1.5814371193999932 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1779954436707,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1449.7084058406062,
            "unit": "iter/sec",
            "range": "stddev: 0.000012151666407115827",
            "extra": "mean: 689.7938895650916 usec\nrounds: 1150"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 624.2798708930487,
            "unit": "iter/sec",
            "range": "stddev: 0.00012202851384011577",
            "extra": "mean: 1.6018456570920245 msec\nrounds: 557"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 56.72786547950158,
            "unit": "iter/sec",
            "range": "stddev: 0.0005584728337017649",
            "extra": "mean: 17.628020930231305 msec\nrounds: 43"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0238038655721933,
            "unit": "iter/sec",
            "range": "stddev: 0.0014712812658945473",
            "extra": "mean: 976.7495842000073 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17586.788582023226,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037770694289216628",
            "extra": "mean: 56.86086435485868 usec\nrounds: 7844"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1474.9629477078788,
            "unit": "iter/sec",
            "range": "stddev: 0.000015515901653983996",
            "extra": "mean: 677.9831327654838 usec\nrounds: 1175"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 112.07054427347069,
            "unit": "iter/sec",
            "range": "stddev: 0.00009477833195018256",
            "extra": "mean: 8.922951222221554 msec\nrounds: 99"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 7.971227384453082,
            "unit": "iter/sec",
            "range": "stddev: 0.0004281646800247572",
            "extra": "mean: 125.451194875005 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.288933771266116,
            "unit": "iter/sec",
            "range": "stddev: 0.00034397047378126945",
            "extra": "mean: 18.765622226414468 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.378508924143772,
            "unit": "iter/sec",
            "range": "stddev: 0.00022279193371353977",
            "extra": "mean: 156.77645228570978 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6252689258202389,
            "unit": "iter/sec",
            "range": "stddev: 0.04413269831071633",
            "extra": "mean: 1.5993118459999949 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1780040732035,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1438.5615398089324,
            "unit": "iter/sec",
            "range": "stddev: 0.000013681208379339644",
            "extra": "mean: 695.1388399642733 usec\nrounds: 1131"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 617.9111916152079,
            "unit": "iter/sec",
            "range": "stddev: 0.000042018562367646994",
            "extra": "mean: 1.6183555397111666 msec\nrounds: 554"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 56.79466839020323,
            "unit": "iter/sec",
            "range": "stddev: 0.0005175560907504404",
            "extra": "mean: 17.607286534883524 msec\nrounds: 43"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0203542635719234,
            "unit": "iter/sec",
            "range": "stddev: 0.0005719143076690967",
            "extra": "mean: 980.0517679999984 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17406.878136836665,
            "unit": "iter/sec",
            "range": "stddev: 0.00000660900509603072",
            "extra": "mean: 57.448555228509754 usec\nrounds: 7478"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1474.5646156281448,
            "unit": "iter/sec",
            "range": "stddev: 0.00002455753879522374",
            "extra": "mean: 678.1662799998855 usec\nrounds: 1200"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 111.007196940005,
            "unit": "iter/sec",
            "range": "stddev: 0.0001039083543804626",
            "extra": "mean: 9.008424927083425 msec\nrounds: 96"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 7.648678965582733,
            "unit": "iter/sec",
            "range": "stddev: 0.0006196486878960252",
            "extra": "mean: 130.7415312500062 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.842040051449686,
            "unit": "iter/sec",
            "range": "stddev: 0.00012477997176418398",
            "extra": "mean: 18.57284751923279 msec\nrounds: 52"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.365418501427758,
            "unit": "iter/sec",
            "range": "stddev: 0.0006685077681813031",
            "extra": "mean: 157.09886157142705 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6284954696289312,
            "unit": "iter/sec",
            "range": "stddev: 0.03587784570523032",
            "extra": "mean: 1.5911013655999908 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1780125950402,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1414.4489350208269,
            "unit": "iter/sec",
            "range": "stddev: 0.00005300625318457424",
            "extra": "mean: 706.9891144463802 usec\nrounds: 1066"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 613.2791240133531,
            "unit": "iter/sec",
            "range": "stddev: 0.00011016089358296579",
            "extra": "mean: 1.630578900934881 msec\nrounds: 535"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 54.47055102962696,
            "unit": "iter/sec",
            "range": "stddev: 0.0005388394195520976",
            "extra": "mean: 18.358543857140205 msec\nrounds: 42"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.092477890510577,
            "unit": "iter/sec",
            "range": "stddev: 0.002710235127073872",
            "extra": "mean: 915.3503321999892 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17974.673876903067,
            "unit": "iter/sec",
            "range": "stddev: 0.000009895570825487463",
            "extra": "mean: 55.633832738683004 usec\nrounds: 4765"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1476.419029358162,
            "unit": "iter/sec",
            "range": "stddev: 0.00004669873753242767",
            "extra": "mean: 677.3144887157993 usec\nrounds: 1152"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 102.92573891403023,
            "unit": "iter/sec",
            "range": "stddev: 0.00010441772556294607",
            "extra": "mean: 9.715742734043038 msec\nrounds: 94"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 11.713333063888802,
            "unit": "iter/sec",
            "range": "stddev: 0.001519039032076",
            "extra": "mean: 85.37279649999145 msec\nrounds: 12"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 63.59096399171807,
            "unit": "iter/sec",
            "range": "stddev: 0.0001633941606333255",
            "extra": "mean: 15.725504650790285 msec\nrounds: 63"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.987064393147272,
            "unit": "iter/sec",
            "range": "stddev: 0.001001009568917164",
            "extra": "mean: 167.02676542857782 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6023371331289498,
            "unit": "iter/sec",
            "range": "stddev: 0.0036815003749798114",
            "extra": "mean: 1.6601998200000025 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1780213215704,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1444.5087905373427,
            "unit": "iter/sec",
            "range": "stddev: 0.000018423017055030497",
            "extra": "mean: 692.2768532464313 usec\nrounds: 1063"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 620.472799649766,
            "unit": "iter/sec",
            "range": "stddev: 0.00006008698629834488",
            "extra": "mean: 1.611674195169334 msec\nrounds: 538"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 54.5671016413619,
            "unit": "iter/sec",
            "range": "stddev: 0.0005515051356527874",
            "extra": "mean: 18.326060390240688 msec\nrounds: 41"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0850132716968053,
            "unit": "iter/sec",
            "range": "stddev: 0.005289353143862924",
            "extra": "mean: 921.6477125999972 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 18335.376607781727,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026487189032909463",
            "extra": "mean: 54.53937605926182 usec\nrounds: 6608"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1512.7828717080208,
            "unit": "iter/sec",
            "range": "stddev: 0.000015982478751376515",
            "extra": "mean: 661.0333965977161 usec\nrounds: 1117"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 103.45093314644515,
            "unit": "iter/sec",
            "range": "stddev: 0.00024252103239659446",
            "extra": "mean: 9.66641836458256 msec\nrounds: 96"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 12.020298329200683,
            "unit": "iter/sec",
            "range": "stddev: 0.0004604828123286823",
            "extra": "mean: 83.19261074999436 msec\nrounds: 12"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 63.88236281324166,
            "unit": "iter/sec",
            "range": "stddev: 0.0000970831949669209",
            "extra": "mean: 15.6537729032264 msec\nrounds: 62"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.010412966622474,
            "unit": "iter/sec",
            "range": "stddev: 0.0004572814171237253",
            "extra": "mean: 166.37791871428524 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.591369656683398,
            "unit": "iter/sec",
            "range": "stddev: 0.053646227668175867",
            "extra": "mean: 1.6909897028000045 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Puneet Dixit",
            "username": "puneetdixit200",
            "email": "puneetdixit4321@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8",
          "message": "test: add optimization pass docs drift test (#101)\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>\nSigned-off-by: Puneet Dixit <236133619+puneetdixit200@users.noreply.github.com>",
          "timestamp": "2026-05-20T14:00:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2655e48c6133f0c29b72cdd8be4bc0ecf176c6e8"
        },
        "date": 1780301916617,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1404.3198651345867,
            "unit": "iter/sec",
            "range": "stddev: 0.000012783218536255923",
            "extra": "mean: 712.0884812835446 usec\nrounds: 1122"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 612.795668756536,
            "unit": "iter/sec",
            "range": "stddev: 0.00005044796869345838",
            "extra": "mean: 1.6318653198531345 msec\nrounds: 544"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 59.50758406054903,
            "unit": "iter/sec",
            "range": "stddev: 0.00047043016535342223",
            "extra": "mean: 16.804580723399877 msec\nrounds: 47"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.1568195178874208,
            "unit": "iter/sec",
            "range": "stddev: 0.005281460315683064",
            "extra": "mean: 864.4390801999918 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 20648.521834485426,
            "unit": "iter/sec",
            "range": "stddev: 0.000001682350145703464",
            "extra": "mean: 48.42961680336285 usec\nrounds: 6832"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1492.208242981173,
            "unit": "iter/sec",
            "range": "stddev: 0.000012348840457771632",
            "extra": "mean: 670.1477523017656 usec\nrounds: 1086"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 120.07821209241463,
            "unit": "iter/sec",
            "range": "stddev: 0.00006924067765367202",
            "extra": "mean: 8.327905475727599 msec\nrounds: 103"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 13.197123620921907,
            "unit": "iter/sec",
            "range": "stddev: 0.00038214063210956033",
            "extra": "mean: 75.77408749999596 msec\nrounds: 14"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 57.46370456456467,
            "unit": "iter/sec",
            "range": "stddev: 0.00010165292372276795",
            "extra": "mean: 17.402289107142874 msec\nrounds: 56"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 7.327701380516817,
            "unit": "iter/sec",
            "range": "stddev: 0.001184889862132317",
            "extra": "mean: 136.46844324999918 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.7343489403656891,
            "unit": "iter/sec",
            "range": "stddev: 0.019098403574938527",
            "extra": "mean: 1.3617504499999995 sec\nrounds: 5"
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
          "id": "769d5294ba704e047d94c0aefdc76f1c93ef4ac2",
          "message": "docs: clarify Stim gate descriptions (#110)\n\nCorrect SPP/SPP_DAG terminology, clarify H_NXZ and MPAD behavior,\nand document Clifft's current OBSERVABLE_INCLUDE target support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-01T14:40:33Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/769d5294ba704e047d94c0aefdc76f1c93ef4ac2"
        },
        "date": 1780387502529,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1409.9138104830456,
            "unit": "iter/sec",
            "range": "stddev: 0.000009699010774607042",
            "extra": "mean: 709.2632135133093 usec\nrounds: 1110"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 614.9917827818053,
            "unit": "iter/sec",
            "range": "stddev: 0.000021173428819624553",
            "extra": "mean: 1.6260379861933747 msec\nrounds: 507"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 55.81152771283379,
            "unit": "iter/sec",
            "range": "stddev: 0.0005412816063730437",
            "extra": "mean: 17.91744539130491 msec\nrounds: 46"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.154969146694733,
            "unit": "iter/sec",
            "range": "stddev: 0.006063355719880414",
            "extra": "mean: 865.8239944000059 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 20488.882246181638,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028746151917260196",
            "extra": "mean: 48.806957255384816 usec\nrounds: 7229"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1495.4408849333424,
            "unit": "iter/sec",
            "range": "stddev: 0.000011223200667874077",
            "extra": "mean: 668.6991174810456 usec\nrounds: 1064"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 115.99655864140998,
            "unit": "iter/sec",
            "range": "stddev: 0.00024474624277417875",
            "extra": "mean: 8.620945411763335 msec\nrounds: 102"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 13.18209140719794,
            "unit": "iter/sec",
            "range": "stddev: 0.0006873804101366469",
            "extra": "mean: 75.86049657142878 msec\nrounds: 14"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 56.62172677393066,
            "unit": "iter/sec",
            "range": "stddev: 0.00014403981768188213",
            "extra": "mean: 17.66106505357255 msec\nrounds: 56"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.88978008850531,
            "unit": "iter/sec",
            "range": "stddev: 0.005186700465693531",
            "extra": "mean: 145.14251357142274 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.7023888319611504,
            "unit": "iter/sec",
            "range": "stddev: 0.041459871358694035",
            "extra": "mean: 1.4237128418000111 sec\nrounds: 5"
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
          "id": "769d5294ba704e047d94c0aefdc76f1c93ef4ac2",
          "message": "docs: clarify Stim gate descriptions (#110)\n\nCorrect SPP/SPP_DAG terminology, clarify H_NXZ and MPAD behavior,\nand document Clifft's current OBSERVABLE_INCLUDE target support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-01T14:40:33Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/769d5294ba704e047d94c0aefdc76f1c93ef4ac2"
        },
        "date": 1780474220791,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1445.2764895662576,
            "unit": "iter/sec",
            "range": "stddev: 0.000013607010442496459",
            "extra": "mean: 691.9091310342357 usec\nrounds: 1160"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 628.0304892585805,
            "unit": "iter/sec",
            "range": "stddev: 0.00001981363387513131",
            "extra": "mean: 1.5922793830926059 msec\nrounds: 556"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 56.705743064655,
            "unit": "iter/sec",
            "range": "stddev: 0.0003909295785217454",
            "extra": "mean: 17.634898088890495 msec\nrounds: 45"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0198224798395195,
            "unit": "iter/sec",
            "range": "stddev: 0.000649975313628233",
            "extra": "mean: 980.562813399996 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17612.755731137717,
            "unit": "iter/sec",
            "range": "stddev: 0.000004372407735918454",
            "extra": "mean: 56.77703224101909 usec\nrounds: 7568"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1464.0847858731513,
            "unit": "iter/sec",
            "range": "stddev: 0.00002192307793973714",
            "extra": "mean: 683.0205529412832 usec\nrounds: 1190"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 109.0588520919098,
            "unit": "iter/sec",
            "range": "stddev: 0.00012387516504343938",
            "extra": "mean: 9.169361136840555 msec\nrounds: 95"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 7.988231306275413,
            "unit": "iter/sec",
            "range": "stddev: 0.0034204653884893723",
            "extra": "mean: 125.18415674999517 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.650729065539764,
            "unit": "iter/sec",
            "range": "stddev: 0.00010541908442872645",
            "extra": "mean: 18.63907569230605 msec\nrounds: 52"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.384917258030417,
            "unit": "iter/sec",
            "range": "stddev: 0.0004994697887986691",
            "extra": "mean: 156.61910085714632 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6229738298702785,
            "unit": "iter/sec",
            "range": "stddev: 0.05529031604258975",
            "extra": "mean: 1.6052038657999959 sec\nrounds: 5"
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
          "id": "769d5294ba704e047d94c0aefdc76f1c93ef4ac2",
          "message": "docs: clarify Stim gate descriptions (#110)\n\nCorrect SPP/SPP_DAG terminology, clarify H_NXZ and MPAD behavior,\nand document Clifft's current OBSERVABLE_INCLUDE target support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-01T14:40:33Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/769d5294ba704e047d94c0aefdc76f1c93ef4ac2"
        },
        "date": 1780560180713,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1451.9829532695728,
            "unit": "iter/sec",
            "range": "stddev: 0.000015109977064487684",
            "extra": "mean: 688.7133197729364 usec\nrounds: 1057"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 623.6543126649702,
            "unit": "iter/sec",
            "range": "stddev: 0.000018129186507325916",
            "extra": "mean: 1.6034523929239695 msec\nrounds: 537"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 54.50732438780892,
            "unit": "iter/sec",
            "range": "stddev: 0.0005671396566387559",
            "extra": "mean: 18.346158268294296 msec\nrounds: 41"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0939590125952277,
            "unit": "iter/sec",
            "range": "stddev: 0.0011652523061121427",
            "extra": "mean: 914.1110302000016 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 18351.736540043898,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027016679993760335",
            "extra": "mean: 54.49075611008134 usec\nrounds: 5933"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1500.0858154563032,
            "unit": "iter/sec",
            "range": "stddev: 0.000014423346461500127",
            "extra": "mean: 666.6285286457531 usec\nrounds: 1152"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 103.8202449198076,
            "unit": "iter/sec",
            "range": "stddev: 0.00020777924199175994",
            "extra": "mean: 9.63203275789241 msec\nrounds: 95"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 11.673094975219877,
            "unit": "iter/sec",
            "range": "stddev: 0.0002583370222011162",
            "extra": "mean: 85.66708333332684 msec\nrounds: 12"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 63.46026255215694,
            "unit": "iter/sec",
            "range": "stddev: 0.00012775936963174044",
            "extra": "mean: 15.757892573767977 msec\nrounds: 61"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.998301395012742,
            "unit": "iter/sec",
            "range": "stddev: 0.0006204962964381942",
            "extra": "mean: 166.7138634999977 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.5998635399466351,
            "unit": "iter/sec",
            "range": "stddev: 0.013350364054103477",
            "extra": "mean: 1.6670458086000053 sec\nrounds: 5"
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
          "id": "769d5294ba704e047d94c0aefdc76f1c93ef4ac2",
          "message": "docs: clarify Stim gate descriptions (#110)\n\nCorrect SPP/SPP_DAG terminology, clarify H_NXZ and MPAD behavior,\nand document Clifft's current OBSERVABLE_INCLUDE target support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-01T14:40:33Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/769d5294ba704e047d94c0aefdc76f1c93ef4ac2"
        },
        "date": 1780646030446,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1880.9010021738711,
            "unit": "iter/sec",
            "range": "stddev: 0.00001083605659328527",
            "extra": "mean: 531.6600920751488 usec\nrounds: 1325"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 801.661137413922,
            "unit": "iter/sec",
            "range": "stddev: 0.000015727351997827127",
            "extra": "mean: 1.2474098510324438 msec\nrounds: 678"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 70.31478075276223,
            "unit": "iter/sec",
            "range": "stddev: 0.0005052951362511936",
            "extra": "mean: 14.221760905664436 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.4078235129017926,
            "unit": "iter/sec",
            "range": "stddev: 0.002974837622521742",
            "extra": "mean: 710.3163080000058 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 23574.455273641935,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026176324466620624",
            "extra": "mean: 42.418795615527 usec\nrounds: 8484"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1918.1272142055448,
            "unit": "iter/sec",
            "range": "stddev: 0.00002181449314199809",
            "extra": "mean: 521.3418550104784 usec\nrounds: 1407"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 130.95654482287452,
            "unit": "iter/sec",
            "range": "stddev: 0.00007409881554048484",
            "extra": "mean: 7.636120831933613 msec\nrounds: 119"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 16.130606583841473,
            "unit": "iter/sec",
            "range": "stddev: 0.0008417675328633939",
            "extra": "mean: 61.993948882352065 msec\nrounds: 17"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 80.7028308557221,
            "unit": "iter/sec",
            "range": "stddev: 0.00006007823457215993",
            "extra": "mean: 12.391139064102564 msec\nrounds: 78"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 9.237757736616128,
            "unit": "iter/sec",
            "range": "stddev: 0.0010849026188035685",
            "extra": "mean: 108.25137750000238 msec\nrounds: 10"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.9280503724359888,
            "unit": "iter/sec",
            "range": "stddev: 0.016527659997453918",
            "extra": "mean: 1.077527717999999 sec\nrounds: 5"
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
          "id": "3ea2486f0bf810a34e1587a6240ca23904229d52",
          "message": "ci: skip PR preview cleanup for forks (#122)\n\nCo-authored-by: Shelley <shelley@exe.dev>",
          "timestamp": "2026-06-05T15:28:02Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/3ea2486f0bf810a34e1587a6240ca23904229d52"
        },
        "date": 1780731004588,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1461.8864360190837,
            "unit": "iter/sec",
            "range": "stddev: 0.000014075048839194113",
            "extra": "mean: 684.0476629109006 usec\nrounds: 1065"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 625.254963802267,
            "unit": "iter/sec",
            "range": "stddev: 0.000018688385427065187",
            "extra": "mean: 1.599347558824409 msec\nrounds: 510"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 54.51624964656939,
            "unit": "iter/sec",
            "range": "stddev: 0.000562085525177377",
            "extra": "mean: 18.343154682925412 msec\nrounds: 41"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0898439868563068,
            "unit": "iter/sec",
            "range": "stddev: 0.0028515180124080355",
            "extra": "mean: 917.5625245999981 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 18337.165307031435,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028214361498277227",
            "extra": "mean: 54.53405601445645 usec\nrounds: 6052"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1490.1917812237798,
            "unit": "iter/sec",
            "range": "stddev: 0.000013238764026923912",
            "extra": "mean: 671.0545666671018 usec\nrounds: 1110"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 104.24623981042008,
            "unit": "iter/sec",
            "range": "stddev: 0.00007621645579204884",
            "extra": "mean: 9.592672136842328 msec\nrounds: 95"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 12.043208563467813,
            "unit": "iter/sec",
            "range": "stddev: 0.00015205278252511156",
            "extra": "mean: 83.03435041666773 msec\nrounds: 12"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 63.802115704703795,
            "unit": "iter/sec",
            "range": "stddev: 0.00012714411144593268",
            "extra": "mean: 15.673461435484578 msec\nrounds: 62"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.9145008510434565,
            "unit": "iter/sec",
            "range": "stddev: 0.0072343898638435305",
            "extra": "mean: 169.07597533333293 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6007951270932793,
            "unit": "iter/sec",
            "range": "stddev: 0.003811008715917018",
            "extra": "mean: 1.6644609034000042 sec\nrounds: 5"
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
          "id": "83d8d0d9f9e08dadc7e9fad3d123de74bc33ca2e",
          "message": "build: hide vendored Stim symbols (#131)\n\nRestrict the Python extension's dynamic export table to the module init\nsymbol on Linux and macOS, so statically linked Stim internals do not\nbecome part of Clifft's accidental ABI.\n\nAdd an export audit for installed Python extensions and run it in CI and\ncibuildwheel smoke tests to prevent symbol-export regressions.\n\nCloses #109.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-08T17:34:12Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/83d8d0d9f9e08dadc7e9fad3d123de74bc33ca2e"
        },
        "date": 1780990748690,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1447.614166033922,
            "unit": "iter/sec",
            "range": "stddev: 0.000012324010035829655",
            "extra": "mean: 690.7918031361452 usec\nrounds: 1148"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 627.8792130501583,
            "unit": "iter/sec",
            "range": "stddev: 0.00001991350149081937",
            "extra": "mean: 1.5926630141840905 msec\nrounds: 564"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 57.693295726987465,
            "unit": "iter/sec",
            "range": "stddev: 0.0004470138187841095",
            "extra": "mean: 17.33303648888662 msec\nrounds: 45"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0202008553284383,
            "unit": "iter/sec",
            "range": "stddev: 0.0032777893149406523",
            "extra": "mean: 980.1991390000012 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17734.67674157687,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035445194793756512",
            "extra": "mean: 56.38670580646205 usec\nrounds: 6200"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1496.5840123554567,
            "unit": "iter/sec",
            "range": "stddev: 0.000016320066257942136",
            "extra": "mean: 668.1883487624001 usec\nrounds: 1253"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 112.56790375970128,
            "unit": "iter/sec",
            "range": "stddev: 0.00005812369969935987",
            "extra": "mean: 8.883526889997881 msec\nrounds: 100"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 8.012350750236967,
            "unit": "iter/sec",
            "range": "stddev: 0.0009930987477821797",
            "extra": "mean: 124.80731699999836 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 54.22084433894226,
            "unit": "iter/sec",
            "range": "stddev: 0.00012819475813963867",
            "extra": "mean: 18.443091622639752 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.368260728268363,
            "unit": "iter/sec",
            "range": "stddev: 0.001612113806816766",
            "extra": "mean: 157.02874657142323 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6237906042892091,
            "unit": "iter/sec",
            "range": "stddev: 0.0967006210780813",
            "extra": "mean: 1.6031020556000044 sec\nrounds: 5"
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
          "id": "83d8d0d9f9e08dadc7e9fad3d123de74bc33ca2e",
          "message": "build: hide vendored Stim symbols (#131)\n\nRestrict the Python extension's dynamic export table to the module init\nsymbol on Linux and macOS, so statically linked Stim internals do not\nbecome part of Clifft's accidental ABI.\n\nAdd an export audit for installed Python extensions and run it in CI and\ncibuildwheel smoke tests to prevent symbol-export regressions.\n\nCloses #109.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-08T17:34:12Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/83d8d0d9f9e08dadc7e9fad3d123de74bc33ca2e"
        },
        "date": 1781078059792,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1440.695320442526,
            "unit": "iter/sec",
            "range": "stddev: 0.000051785325237306514",
            "extra": "mean: 694.1092858501397 usec\nrounds: 1053"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 620.9105852288079,
            "unit": "iter/sec",
            "range": "stddev: 0.00012308308914267253",
            "extra": "mean: 1.61053785164815 msec\nrounds: 546"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 54.367741947889016,
            "unit": "iter/sec",
            "range": "stddev: 0.0005483299008966698",
            "extra": "mean: 18.393259756097482 msec\nrounds: 41"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0871252565771885,
            "unit": "iter/sec",
            "range": "stddev: 0.0073960245309693876",
            "extra": "mean: 919.8572049999996 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 18230.678845801376,
            "unit": "iter/sec",
            "range": "stddev: 0.000014271348353065057",
            "extra": "mean: 54.85259262467373 usec\nrounds: 5749"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1532.8349381156277,
            "unit": "iter/sec",
            "range": "stddev: 0.00001913035470040799",
            "extra": "mean: 652.3859648119308 usec\nrounds: 1222"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 103.42141095869648,
            "unit": "iter/sec",
            "range": "stddev: 0.00025129824982878413",
            "extra": "mean: 9.66917769473645 msec\nrounds: 95"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 11.98857181000221,
            "unit": "iter/sec",
            "range": "stddev: 0.00009293795984519441",
            "extra": "mean: 83.41277141666599 msec\nrounds: 12"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 63.92698221938104,
            "unit": "iter/sec",
            "range": "stddev: 0.00013864405072061135",
            "extra": "mean: 15.642846968252872 msec\nrounds: 63"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 5.968584130622392,
            "unit": "iter/sec",
            "range": "stddev: 0.0013060107736621812",
            "extra": "mean: 167.54392299999665 msec\nrounds: 6"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6001092962738336,
            "unit": "iter/sec",
            "range": "stddev: 0.005609316647916446",
            "extra": "mean: 1.6663631211999985 sec\nrounds: 5"
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
          "id": "83d8d0d9f9e08dadc7e9fad3d123de74bc33ca2e",
          "message": "build: hide vendored Stim symbols (#131)\n\nRestrict the Python extension's dynamic export table to the module init\nsymbol on Linux and macOS, so statically linked Stim internals do not\nbecome part of Clifft's accidental ABI.\n\nAdd an export audit for installed Python extensions and run it in CI and\ncibuildwheel smoke tests to prevent symbol-export regressions.\n\nCloses #109.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-08T17:34:12Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/83d8d0d9f9e08dadc7e9fad3d123de74bc33ca2e"
        },
        "date": 1781165302160,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1434.8305211135307,
            "unit": "iter/sec",
            "range": "stddev: 0.00002060506012016206",
            "extra": "mean: 696.9464234869556 usec\nrounds: 1124"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 594.7264985445688,
            "unit": "iter/sec",
            "range": "stddev: 0.00006891226266764529",
            "extra": "mean: 1.6814451726082962 msec\nrounds: 533"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 55.68899050190916,
            "unit": "iter/sec",
            "range": "stddev: 0.0008992130775069831",
            "extra": "mean: 17.956870666666465 msec\nrounds: 39"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0227799214394961,
            "unit": "iter/sec",
            "range": "stddev: 0.0024889485035619807",
            "extra": "mean: 977.727445599993 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17561.011287362875,
            "unit": "iter/sec",
            "range": "stddev: 0.000004381795171761657",
            "extra": "mean: 56.94432875398312 usec\nrounds: 7352"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1443.7966638514438,
            "unit": "iter/sec",
            "range": "stddev: 0.000019778106834636904",
            "extra": "mean: 692.6183063288285 usec\nrounds: 1185"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 105.34457533136302,
            "unit": "iter/sec",
            "range": "stddev: 0.00029812796213152395",
            "extra": "mean: 9.492657755318527 msec\nrounds: 94"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 8.270280910769323,
            "unit": "iter/sec",
            "range": "stddev: 0.0013573905688322172",
            "extra": "mean: 120.91487711110618 msec\nrounds: 9"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 52.4260016841299,
            "unit": "iter/sec",
            "range": "stddev: 0.0002908648520175302",
            "extra": "mean: 19.074504403846504 msec\nrounds: 52"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.303207166985005,
            "unit": "iter/sec",
            "range": "stddev: 0.0010306541178109869",
            "extra": "mean: 158.64939442857104 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6369265397409847,
            "unit": "iter/sec",
            "range": "stddev: 0.007933702717279752",
            "extra": "mean: 1.5700397732000055 sec\nrounds: 5"
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
          "id": "aaa2d3bbd351e599a95d2eb27da7046edbbfb0e2",
          "message": "test: widen componentwise global-phase coverage (#145)\n\nRaise the random componentwise sweep from 20 to 100 seeds and add a\n20-seed 8-qubit depth-60 set whose longer virtual-frame gate logs\nstress the chained composition phase across many links. The end-to-end\ncomponentwise comparison is the only model-free check on the canonical\nphase tracking, so breadth here is what future changes to frame routing\nor the phase machinery get caught by. Adds about four seconds to the\nPython suite.\n\nAssisted-by: Claude (Fable 5) <noreply@anthropic.com>",
          "timestamp": "2026-06-11T21:08:47Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/aaa2d3bbd351e599a95d2eb27da7046edbbfb0e2"
        },
        "date": 1781251423364,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1873.630926233443,
            "unit": "iter/sec",
            "range": "stddev: 0.000012398317267944708",
            "extra": "mean: 533.7230433158456 usec\nrounds: 1339"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 802.8912267566635,
            "unit": "iter/sec",
            "range": "stddev: 0.000016644370121567213",
            "extra": "mean: 1.2454987259476873 msec\nrounds: 686"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 70.5273711591059,
            "unit": "iter/sec",
            "range": "stddev: 0.0003588711924524869",
            "extra": "mean: 14.178892301884535 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.4566268681917414,
            "unit": "iter/sec",
            "range": "stddev: 0.0032044071869616253",
            "extra": "mean: 686.5176125999938 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 23660.448575436294,
            "unit": "iter/sec",
            "range": "stddev: 0.000002182341053157127",
            "extra": "mean: 42.264625575957 usec\nrounds: 8250"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1973.5452962829183,
            "unit": "iter/sec",
            "range": "stddev: 0.000011404999176954657",
            "extra": "mean: 506.70233000653894 usec\nrounds: 1503"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 133.548497769714,
            "unit": "iter/sec",
            "range": "stddev: 0.0000710155739343508",
            "extra": "mean: 7.4879164999995895 msec\nrounds: 122"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 17.745951548539725,
            "unit": "iter/sec",
            "range": "stddev: 0.0002553731520747257",
            "extra": "mean: 56.350880777778734 msec\nrounds: 18"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 80.2554730741477,
            "unit": "iter/sec",
            "range": "stddev: 0.0000778249829178041",
            "extra": "mean: 12.460209400000721 msec\nrounds: 80"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 9.30901302914293,
            "unit": "iter/sec",
            "range": "stddev: 0.001429784919243567",
            "extra": "mean: 107.42277370000295 msec\nrounds: 10"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.9283802504094772,
            "unit": "iter/sec",
            "range": "stddev: 0.01811114393740523",
            "extra": "mean: 1.0771448440000029 sec\nrounds: 5"
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
          "id": "7b5f05a4df902ea2329f7b44e709b8ca79ea7812",
          "message": "feat: add DEPOLARIZE3 and PAULI_CHANNEL_3 support (#149)\n\nAdd DEPOLARIZE3 and PAULI_CHANNEL_3 parsing, frontend lowering, Python enum exposure, docs, and focused parser/frontend tests.\n\nCloses #148\n\nAssisted-by: Claude (Opus 4.6) <noreply@anthropic.com>",
          "timestamp": "2026-06-12T20:09:46Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/7b5f05a4df902ea2329f7b44e709b8ca79ea7812"
        },
        "date": 1781336564491,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1413.3342181032447,
            "unit": "iter/sec",
            "range": "stddev: 0.000008055979259623338",
            "extra": "mean: 707.5467268754329 usec\nrounds: 1146"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 619.947348916089,
            "unit": "iter/sec",
            "range": "stddev: 0.000016404265762076592",
            "extra": "mean: 1.6130402069601426 msec\nrounds: 546"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 59.44169160978645,
            "unit": "iter/sec",
            "range": "stddev: 0.0005048049200700308",
            "extra": "mean: 16.82320897871891 msec\nrounds: 47"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.1818898448923458,
            "unit": "iter/sec",
            "range": "stddev: 0.007792656002496831",
            "extra": "mean: 846.1025402000018 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 20590.749600787978,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018376586918192206",
            "extra": "mean: 48.56549758449452 usec\nrounds: 7659"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1524.8391824673859,
            "unit": "iter/sec",
            "range": "stddev: 0.000009447264591123155",
            "extra": "mean: 655.8068624534369 usec\nrounds: 1076"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 118.58917184054084,
            "unit": "iter/sec",
            "range": "stddev: 0.0000633828975694631",
            "extra": "mean: 8.43247308737964 msec\nrounds: 103"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 12.807202764528862,
            "unit": "iter/sec",
            "range": "stddev: 0.0002080769039832278",
            "extra": "mean: 78.08106253846657 msec\nrounds: 13"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 57.538743425192344,
            "unit": "iter/sec",
            "range": "stddev: 0.00008109388409577625",
            "extra": "mean: 17.379593999999788 msec\nrounds: 57"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 7.285803441960828,
            "unit": "iter/sec",
            "range": "stddev: 0.0014970742849098005",
            "extra": "mean: 137.25322237500137 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.7294608739205257,
            "unit": "iter/sec",
            "range": "stddev: 0.012815984836486946",
            "extra": "mean: 1.3708754448000036 sec\nrounds: 5"
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
          "id": "255aac95efeef93b34c5cadfe4c0d3aa8ce70b72",
          "message": "feat: add parser rewrites for controlled gates (#151)\n\nAdds parser-only rewrite gates for controlled gates using existing\nClifft gate support:\n\n- `CH c t` rewrites to `R_Y(0.25) t; CX c t; R_Y(-0.25) t`.\n- `CCZ a b c` rewrites to the textbook 7-T / 6-CX sequence.\n- `CCX a b t` rewrites to `H t; CCZ a b t; H t`.\n\nCloses #150.\n\nAssisted-by: Claude (Opus 4.6) <noreply@anthropic.com>",
          "timestamp": "2026-06-13T12:22:28Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/255aac95efeef93b34c5cadfe4c0d3aa8ce70b72"
        },
        "date": 1781424017044,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1452.2533716249532,
            "unit": "iter/sec",
            "range": "stddev: 0.00001163612295511002",
            "extra": "mean: 688.5850771901336 usec\nrounds: 1153"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 588.5254912487007,
            "unit": "iter/sec",
            "range": "stddev: 0.000016519652083049987",
            "extra": "mean: 1.699161743832464 msec\nrounds: 527"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 57.6636738340108,
            "unit": "iter/sec",
            "range": "stddev: 0.0006025321189916276",
            "extra": "mean: 17.341940488886898 msec\nrounds: 45"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0240193755344942,
            "unit": "iter/sec",
            "range": "stddev: 0.0021713832190980308",
            "extra": "mean: 976.5440224000088 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17586.044324138285,
            "unit": "iter/sec",
            "range": "stddev: 0.000003872081446846848",
            "extra": "mean: 56.86327075995243 usec\nrounds: 7394"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1481.251218424435,
            "unit": "iter/sec",
            "range": "stddev: 0.000011071546526536185",
            "extra": "mean: 675.1049299143677 usec\nrounds: 1170"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 109.8961268493329,
            "unit": "iter/sec",
            "range": "stddev: 0.00012531102056615665",
            "extra": "mean: 9.099501762887382 msec\nrounds: 97"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 7.710899887886951,
            "unit": "iter/sec",
            "range": "stddev: 0.0005150038922283288",
            "extra": "mean: 129.68654950000058 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.64620599626352,
            "unit": "iter/sec",
            "range": "stddev: 0.00007701317376547978",
            "extra": "mean: 18.640647207551837 msec\nrounds: 53"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.413308105394458,
            "unit": "iter/sec",
            "range": "stddev: 0.00038440303815650027",
            "extra": "mean: 155.92576928572402 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.642479117249402,
            "unit": "iter/sec",
            "range": "stddev: 0.003043136869738321",
            "extra": "mean: 1.5564708223999957 sec\nrounds: 5"
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
          "id": "75a855f48f1a4d36f3355597ab00135a3c346cf9",
          "message": "docs: clarify statevector phase checks (#153)\n\nFixes #152.\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-14T12:08:20Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/75a855f48f1a4d36f3355597ab00135a3c346cf9"
        },
        "date": 1781512575917,
        "tool": "pytest",
        "benches": [
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_stim_deep",
            "value": 1452.1927269474868,
            "unit": "iter/sec",
            "range": "stddev: 0.000011008701899939482",
            "extra": "mean: 688.6138330289002 usec\nrounds: 1096"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_compile_clifft_deep",
            "value": 589.6039442307044,
            "unit": "iter/sec",
            "range": "stddev: 0.000018504998458187105",
            "extra": "mean: 1.69605378285718 msec\nrounds: 525"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_stim_deep",
            "value": 57.091369698890105,
            "unit": "iter/sec",
            "range": "stddev: 0.0005353702441341315",
            "extra": "mean: 17.515782249999877 msec\nrounds: 44"
          },
          {
            "name": "tools/bench/test_bench_deep_clifford.py::test_sample_clifft_deep",
            "value": 1.0247842689948035,
            "unit": "iter/sec",
            "range": "stddev: 0.0006424586069155366",
            "extra": "mean: 975.8151352000027 msec\nrounds: 5"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_stim",
            "value": 17614.986113251296,
            "unit": "iter/sec",
            "range": "stddev: 0.000004288680146401302",
            "extra": "mean: 56.76984322160357 usec\nrounds: 7450"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_compile_clifft",
            "value": 1469.0775504841329,
            "unit": "iter/sec",
            "range": "stddev: 0.000018514174293528363",
            "extra": "mean: 680.6992589808831 usec\nrounds: 1197"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_stim",
            "value": 112.44471374537768,
            "unit": "iter/sec",
            "range": "stddev: 0.000038057868440078834",
            "extra": "mean: 8.893259333332665 msec\nrounds: 99"
          },
          {
            "name": "tools/bench/test_bench_qec.py::test_sample_clifft",
            "value": 7.640663069424579,
            "unit": "iter/sec",
            "range": "stddev: 0.0001003200210602998",
            "extra": "mean: 130.87869350000148 msec\nrounds: 8"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_compile_qv20",
            "value": 53.138076083002964,
            "unit": "iter/sec",
            "range": "stddev: 0.00014879975583066455",
            "extra": "mean: 18.818897365384018 msec\nrounds: 52"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_1shot",
            "value": 6.421561416235556,
            "unit": "iter/sec",
            "range": "stddev: 0.0005329390846633174",
            "extra": "mean: 155.72536571428128 msec\nrounds: 7"
          },
          {
            "name": "tools/bench/test_bench_qv.py::test_sample_qv20_10shots",
            "value": 0.6435556411693936,
            "unit": "iter/sec",
            "range": "stddev: 0.005189790487261456",
            "extra": "mean: 1.5538671965999982 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}
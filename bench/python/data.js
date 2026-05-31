window.BENCHMARK_DATA = {
  "lastUpdate": 1780213217190,
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
      }
    ]
  }
}
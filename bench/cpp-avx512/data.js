window.BENCHMARK_DATA = {
  "lastUpdate": 1784877308431,
  "repoUrl": "https://github.com/unitaryfoundation/clifft",
  "entries": {
    "C++ Catch2 benchmarks (AVX-512)": [
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
        "date": 1780474052477,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 29.0681,
            "range": "± 248.265",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.4292,
            "range": "± 569.509",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 96.0664,
            "range": "± 484.98",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 103.546,
            "range": "± 236.045",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 40.7953,
            "range": "± 278.236",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 168.439,
            "range": "± 2.24256",
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
          "id": "3ea2486f0bf810a34e1587a6240ca23904229d52",
          "message": "ci: skip PR preview cleanup for forks (#122)\n\nCo-authored-by: Shelley <shelley@exe.dev>",
          "timestamp": "2026-06-05T15:28:02Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/3ea2486f0bf810a34e1587a6240ca23904229d52"
        },
        "date": 1780730846007,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 29.1246,
            "range": "± 1.20694",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.2768,
            "range": "± 913.092",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 97.616,
            "range": "± 1.14735",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 103.187,
            "range": "± 175.675",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 40.1096,
            "range": "± 340.786",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 166.159,
            "range": "± 841.974",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Manasa Manoj",
            "username": "manasa-manoj-nbr",
            "email": "manasa23bcy41@iiitkottayam.ac.in"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "10bc24018dc615be99ef6c3174a12b2bf30dda77",
          "message": "docs: auto-generate Python API reference via mkdocstrings (#121)\n\nAdd a mkdocstrings-powered Python API reference page, including docs\ndependency updates, MkDocs configuration, and CI changes needed to build\nthe docs against the installed package.\n\nAssisted-by: Claude <noreply@anthropic.com>",
          "timestamp": "2026-06-06T11:42:27Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/10bc24018dc615be99ef6c3174a12b2bf30dda77"
        },
        "date": 1780818308219,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 13.8311,
            "range": "± 141.206",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 29.8888,
            "range": "± 564.701",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 61.3634,
            "range": "± 2.8421",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 72.765,
            "range": "± 1.85467",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 24.1761,
            "range": "± 2.28498",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 81.2691,
            "range": "± 1.78375",
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
          "id": "83d8d0d9f9e08dadc7e9fad3d123de74bc33ca2e",
          "message": "build: hide vendored Stim symbols (#131)\n\nRestrict the Python extension's dynamic export table to the module init\nsymbol on Linux and macOS, so statically linked Stim internals do not\nbecome part of Clifft's accidental ABI.\n\nAdd an export audit for installed Python extensions and run it in CI and\ncibuildwheel smoke tests to prevent symbol-export regressions.\n\nCloses #109.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-08T17:34:12Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/83d8d0d9f9e08dadc7e9fad3d123de74bc33ca2e"
        },
        "date": 1781165134644,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 29.2358,
            "range": "± 291.674",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.6902,
            "range": "± 538.177",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 95.5819,
            "range": "± 260.485",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 103.112,
            "range": "± 182.343",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 38.8023,
            "range": "± 302.947",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 162.557,
            "range": "± 574.283",
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
          "id": "7b5f05a4df902ea2329f7b44e709b8ca79ea7812",
          "message": "feat: add DEPOLARIZE3 and PAULI_CHANNEL_3 support (#149)\n\nAdd DEPOLARIZE3 and PAULI_CHANNEL_3 parsing, frontend lowering, Python enum exposure, docs, and focused parser/frontend tests.\n\nCloses #148\n\nAssisted-by: Claude (Opus 4.6) <noreply@anthropic.com>",
          "timestamp": "2026-06-12T20:09:46Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/7b5f05a4df902ea2329f7b44e709b8ca79ea7812"
        },
        "date": 1781336403361,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 26.4859,
            "range": "± 1.01218",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 53.1337,
            "range": "± 909.92",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 98.3594,
            "range": "± 313.399",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 103.223,
            "range": "± 217.427",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 40.0982,
            "range": "± 278.741",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 163.952,
            "range": "± 888.584",
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
          "id": "a4e3031b510374aba387f525f8391ee44a2112ff",
          "message": "feat: add correlated Pauli noise channels (#158)\n\nAdd Stim-compatible CORRELATED_ERROR/E and ELSE_CORRELATED_ERROR noise support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-15T22:19:40Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/a4e3031b510374aba387f525f8391ee44a2112ff"
        },
        "date": 1781769965031,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 27.2927,
            "range": "± 415.949",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 56.8762,
            "range": "± 775.309",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 100.796,
            "range": "± 1.49412",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 103.821,
            "range": "± 186.931",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 41.6145,
            "range": "± 430.078",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 165.669,
            "range": "± 852.521",
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
          "id": "a4e3031b510374aba387f525f8391ee44a2112ff",
          "message": "feat: add correlated Pauli noise channels (#158)\n\nAdd Stim-compatible CORRELATED_ERROR/E and ELSE_CORRELATED_ERROR noise support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-15T22:19:40Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/a4e3031b510374aba387f525f8391ee44a2112ff"
        },
        "date": 1782372848661,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 24.3156,
            "range": "± 83.5523",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 40.5167,
            "range": "± 111.643",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 89.4025,
            "range": "± 306.955",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 87.6481,
            "range": "± 882.187",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 36.0251,
            "range": "± 202.033",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 111.917,
            "range": "± 2.09669",
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
          "id": "a4e3031b510374aba387f525f8391ee44a2112ff",
          "message": "feat: add correlated Pauli noise channels (#158)\n\nAdd Stim-compatible CORRELATED_ERROR/E and ELSE_CORRELATED_ERROR noise support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-15T22:19:40Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/a4e3031b510374aba387f525f8391ee44a2112ff"
        },
        "date": 1783324593227,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 29.4448,
            "range": "± 45.1294",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.4925,
            "range": "± 159.941",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 100.702,
            "range": "± 217.599",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 103.93,
            "range": "± 295.814",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 41.2822,
            "range": "± 121.35",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 164.058,
            "range": "± 2.24011",
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
          "id": "46e51382dbe975fb0c8c6cd825212a7a181e8492",
          "message": "docs: add front-end integration guide (#182)",
          "timestamp": "2026-07-07T13:53:29Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/46e51382dbe975fb0c8c6cd825212a7a181e8492"
        },
        "date": 1783582405956,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 23.5337,
            "range": "± 182.281",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 40.6038,
            "range": "± 131.557",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 86.6898,
            "range": "± 2.54145",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 89.8454,
            "range": "± 151.813",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 36.8918,
            "range": "± 236.371",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 111.666,
            "range": "± 144.867",
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
          "id": "46e51382dbe975fb0c8c6cd825212a7a181e8492",
          "message": "docs: add front-end integration guide (#182)",
          "timestamp": "2026-07-07T13:53:29Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/46e51382dbe975fb0c8c6cd825212a7a181e8492"
        },
        "date": 1783840422952,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 25.5369,
            "range": "± 431.126",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 40.5538,
            "range": "± 88.7986",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 90.2223,
            "range": "± 151.511",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 89.8804,
            "range": "± 159.092",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 36.6537,
            "range": "± 378.32",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 112.096,
            "range": "± 2.57547",
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
          "id": "a65a7e7d1078280e453c81feaad4babe3e2d32c2",
          "message": "docs: prepare v0.6.0 release (#217)",
          "timestamp": "2026-07-14T22:10:01Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/a65a7e7d1078280e453c81feaad4babe3e2d32c2"
        },
        "date": 1784099170939,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 30.5386,
            "range": "± 85.5768",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.6271,
            "range": "± 174.016",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 100.644,
            "range": "± 932.203",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 104.287,
            "range": "± 867.63",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 41.559,
            "range": "± 141.328",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 169.067,
            "range": "± 839.547",
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
          "id": "a65a7e7d1078280e453c81feaad4babe3e2d32c2",
          "message": "docs: prepare v0.6.0 release (#217)",
          "timestamp": "2026-07-14T22:10:01Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/a65a7e7d1078280e453c81feaad4babe3e2d32c2"
        },
        "date": 1784272081944,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 16.642,
            "range": "± 183.486",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 51.25,
            "range": "± 786.899",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 102.8,
            "range": "± 1.56859",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 115.287,
            "range": "± 753.517",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.3118,
            "range": "± 353.095",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 157.838,
            "range": "± 3.61482",
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
          "id": "a65a7e7d1078280e453c81feaad4babe3e2d32c2",
          "message": "docs: prepare v0.6.0 release (#217)",
          "timestamp": "2026-07-14T22:10:01Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/a65a7e7d1078280e453c81feaad4babe3e2d32c2"
        },
        "date": 1784877307471,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 13.8409,
            "range": "± 46.7285",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 43.5096,
            "range": "± 2.03116",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 86.4284,
            "range": "± 997.695",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 97.974,
            "range": "± 315.322",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 34.8111,
            "range": "± 226.157",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 129.481,
            "range": "± 2.81957",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          }
        ]
      }
    ]
  }
}
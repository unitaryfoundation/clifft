window.BENCHMARK_DATA = {
  "lastUpdate": 1781336346794,
  "repoUrl": "https://github.com/unitaryfoundation/clifft",
  "entries": {
    "C++ Catch2 benchmarks (AVX2)": [
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
        "date": 1779262673048,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 37.7184,
            "range": "± 407.085",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.7874,
            "range": "± 179.458",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 115.199,
            "range": "± 2.74635",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 115.526,
            "range": "± 895.489",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 45.217,
            "range": "± 1.265",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 150.153,
            "range": "± 1.08635",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1779349175268,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 37.7194,
            "range": "± 369.768",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.9077,
            "range": "± 264.162",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 113.234,
            "range": "± 3.111",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 117.655,
            "range": "± 1.11447",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 48.3165,
            "range": "± 411.07",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 150.226,
            "range": "± 1.26194",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1779435325540,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 42.2304,
            "range": "± 298.374",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 67.3499,
            "range": "± 439.754",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 113.783,
            "range": "± 5.03952",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.968,
            "range": "± 1.33019",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.2503,
            "range": "± 664.044",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 170.455,
            "range": "± 2.08378",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1779520625167,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 44.768,
            "range": "± 267.794",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.0971,
            "range": "± 387.225",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.05,
            "range": "± 1.63607",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.233,
            "range": "± 1.66909",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.9185,
            "range": "± 305.306",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 169.658,
            "range": "± 646.297",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1779607801214,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.572,
            "range": "± 286.624",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.3335,
            "range": "± 157.419",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.913,
            "range": "± 692.43",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 111.063,
            "range": "± 1.12992",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.902,
            "range": "± 977.634",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 169.505,
            "range": "± 582.991",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1779695926344,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 43.2278,
            "range": "± 212.616",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.0535,
            "range": "± 854.432",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 115.074,
            "range": "± 462.387",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 114.781,
            "range": "± 973.031",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 47.6782,
            "range": "± 1.25059",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 150.107,
            "range": "± 167.957",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1779780935239,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.5731,
            "range": "± 252.073",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.969,
            "range": "± 883.828",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.257,
            "range": "± 1.18191",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.533,
            "range": "± 1.4246",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.6153,
            "range": "± 109.731",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 169.401,
            "range": "± 336.942",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1779868148431,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 47.6778,
            "range": "± 511.995",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 71.2073,
            "range": "± 933.926",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 119.167,
            "range": "± 2.25876",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 113.703,
            "range": "± 1.3141",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 47.7047,
            "range": "± 1.52636",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 198.805,
            "range": "± 2.47433",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1779954270674,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.7848,
            "range": "± 193.565",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 66.5,
            "range": "± 432.701",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 114.023,
            "range": "± 494.795",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 110.764,
            "range": "± 634.828",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.4439,
            "range": "± 153.578",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 169.09,
            "range": "± 246.482",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1780040569198,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 37.6431,
            "range": "± 181.16",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.7507,
            "range": "± 154.734",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 113.039,
            "range": "± 2.20765",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 116.374,
            "range": "± 384.553",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 48.9722,
            "range": "± 1.45339",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 150.871,
            "range": "± 2.18906",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1780125798128,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.462,
            "range": "± 168.078",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.8034,
            "range": "± 247.156",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 117.585,
            "range": "± 898.718",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 115.457,
            "range": "± 1.14477",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 47.4241,
            "range": "± 894.807",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 150.357,
            "range": "± 716.147",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1780213064402,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.6569,
            "range": "± 506.36",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.7853,
            "range": "± 154.42",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 118.178,
            "range": "± 1.84557",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 117.705,
            "range": "± 1.6204",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 47.5714,
            "range": "± 321.997",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 151.289,
            "range": "± 295.423",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
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
        "date": 1780301766210,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 44.2476,
            "range": "± 557.641",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.2545,
            "range": "± 1.51314",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 112.132,
            "range": "± 2.22968",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 117.735,
            "range": "± 1.64307",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 48.1705,
            "range": "± 413.313",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 151.18,
            "range": "± 1.97587",
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
          "id": "769d5294ba704e047d94c0aefdc76f1c93ef4ac2",
          "message": "docs: clarify Stim gate descriptions (#110)\n\nCorrect SPP/SPP_DAG terminology, clarify H_NXZ and MPAD behavior,\nand document Clifft's current OBSERVABLE_INCLUDE target support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-01T14:40:33Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/769d5294ba704e047d94c0aefdc76f1c93ef4ac2"
        },
        "date": 1780387337362,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.0142,
            "range": "± 340.682",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.0277,
            "range": "± 509.695",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 117.396,
            "range": "± 679.595",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 117.096,
            "range": "± 1.74939",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 47.7228,
            "range": "± 388.458",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 150.224,
            "range": "± 427.579",
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
          "id": "769d5294ba704e047d94c0aefdc76f1c93ef4ac2",
          "message": "docs: clarify Stim gate descriptions (#110)\n\nCorrect SPP/SPP_DAG terminology, clarify H_NXZ and MPAD behavior,\nand document Clifft's current OBSERVABLE_INCLUDE target support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-01T14:40:33Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/769d5294ba704e047d94c0aefdc76f1c93ef4ac2"
        },
        "date": 1780473992509,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 35.581,
            "range": "± 65.6002",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.3315,
            "range": "± 112.226",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 107.239,
            "range": "± 248.186",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 107.456,
            "range": "± 268.725",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.0456,
            "range": "± 267.436",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 168.264,
            "range": "± 923.393",
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
          "id": "769d5294ba704e047d94c0aefdc76f1c93ef4ac2",
          "message": "docs: clarify Stim gate descriptions (#110)\n\nCorrect SPP/SPP_DAG terminology, clarify H_NXZ and MPAD behavior,\nand document Clifft's current OBSERVABLE_INCLUDE target support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-01T14:40:33Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/769d5294ba704e047d94c0aefdc76f1c93ef4ac2"
        },
        "date": 1780560021002,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.5472,
            "range": "± 826.724",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.6598,
            "range": "± 459.875",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 112.256,
            "range": "± 830.534",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.895,
            "range": "± 251.846",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.2776,
            "range": "± 619.062",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 169.74,
            "range": "± 324.373",
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
          "id": "769d5294ba704e047d94c0aefdc76f1c93ef4ac2",
          "message": "docs: clarify Stim gate descriptions (#110)\n\nCorrect SPP/SPP_DAG terminology, clarify H_NXZ and MPAD behavior,\nand document Clifft's current OBSERVABLE_INCLUDE target support.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-01T14:40:33Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/769d5294ba704e047d94c0aefdc76f1c93ef4ac2"
        },
        "date": 1780645892726,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.4514,
            "range": "± 150.42",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 64.9366,
            "range": "± 393.901",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 109.82,
            "range": "± 1.43824",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 111.122,
            "range": "± 1.5473",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.2629,
            "range": "± 582.807",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 170.051,
            "range": "± 651.89",
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
        "date": 1780730786563,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 35.5257,
            "range": "± 52.59",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.1638,
            "range": "± 153.358",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 108.441,
            "range": "± 483.54",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.036,
            "range": "± 237.171",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.1708,
            "range": "± 233.57",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 168.409,
            "range": "± 1.11561",
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
        "date": 1780818272082,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 21.3401,
            "range": "± 1.68537",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 40.371,
            "range": "± 1.09826",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 67.8686,
            "range": "± 3.13879",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 74.5494,
            "range": "± 1.61587",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 25.33,
            "range": "± 493.828",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 83.2119,
            "range": "± 4.82794",
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
        "date": 1780906210859,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.507,
            "range": "± 175.707",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.6563,
            "range": "± 916.63",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 117.163,
            "range": "± 1.55801",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 110.469,
            "range": "± 1.20169",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 45.2413,
            "range": "± 1.43508",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 171.594,
            "range": "± 1.98368",
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
        "date": 1780990584662,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 42.0048,
            "range": "± 258.641",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.6795,
            "range": "± 1.07846",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.482,
            "range": "± 966.133",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.164,
            "range": "± 931.288",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.7514,
            "range": "± 1.04655",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 176.677,
            "range": "± 2.91168",
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
        "date": 1781077906803,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 45.1836,
            "range": "± 159.069",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.2183,
            "range": "± 392.482",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.754,
            "range": "± 1.23822",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.537,
            "range": "± 543.32",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.8802,
            "range": "± 338.265",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 170.618,
            "range": "± 928.811",
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
        "date": 1781165075753,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 35.5267,
            "range": "± 52.4766",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.3509,
            "range": "± 206.61",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 106.973,
            "range": "± 710.498",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.05,
            "range": "± 138.159",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.025,
            "range": "± 814.85",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 163.73,
            "range": "± 882.394",
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
          "id": "aaa2d3bbd351e599a95d2eb27da7046edbbfb0e2",
          "message": "test: widen componentwise global-phase coverage (#145)\n\nRaise the random componentwise sweep from 20 to 100 seeds and add a\n20-seed 8-qubit depth-60 set whose longer virtual-frame gate logs\nstress the chained composition phase across many links. The end-to-end\ncomponentwise comparison is the only model-free check on the canonical\nphase tracking, so breadth here is what future changes to frame routing\nor the phase machinery get caught by. Adds about four seconds to the\nPython suite.\n\nAssisted-by: Claude (Fable 5) <noreply@anthropic.com>",
          "timestamp": "2026-06-11T21:08:47Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/aaa2d3bbd351e599a95d2eb27da7046edbbfb0e2"
        },
        "date": 1781251290234,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.1037,
            "range": "± 1.71686",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.2393,
            "range": "± 2.26139",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 102.942,
            "range": "± 680.321",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 113.227,
            "range": "± 404.047",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 41.1284,
            "range": "± 415.439",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 144.868,
            "range": "± 463.256",
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
        "date": 1781336346076,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 35.6136,
            "range": "± 103.276",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.534,
            "range": "± 156.148",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 107.368,
            "range": "± 285.902",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.181,
            "range": "± 818.437",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.9855,
            "range": "± 279.507",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 164.624,
            "range": "± 1.10949",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          }
        ]
      }
    ]
  }
}
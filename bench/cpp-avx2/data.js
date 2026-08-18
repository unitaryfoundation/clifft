window.BENCHMARK_DATA = {
  "lastUpdate": 1787034777118,
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
        "date": 1781423857859,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.1676,
            "range": "± 387.674",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.0099,
            "range": "± 184.64",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 102.547,
            "range": "± 1.09554",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 111.094,
            "range": "± 1.99758",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 40.2486,
            "range": "± 346.389",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 144.106,
            "range": "± 352.318",
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
          "id": "75a855f48f1a4d36f3355597ab00135a3c346cf9",
          "message": "docs: clarify statevector phase checks (#153)\n\nFixes #152.\n\nAssisted-by: OpenAI Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-06-14T12:08:20Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/75a855f48f1a4d36f3355597ab00135a3c346cf9"
        },
        "date": 1781512408880,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.3307,
            "range": "± 540.146",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 68.9467,
            "range": "± 998.008",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 114.604,
            "range": "± 1.45103",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.899,
            "range": "± 691.767",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 45.8926,
            "range": "± 960.989",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 173.267,
            "range": "± 3.21905",
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
        "date": 1781598312525,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 43.0102,
            "range": "± 180.754",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.198,
            "range": "± 352.243",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.298,
            "range": "± 1.62912",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 111.349,
            "range": "± 2.2179",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.3601,
            "range": "± 1.21407",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 158.708,
            "range": "± 2.73247",
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
        "date": 1781684176150,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.2098,
            "range": "± 203.218",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.4118,
            "range": "± 895.329",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 109.522,
            "range": "± 1.92852",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 114.936,
            "range": "± 188.126",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 45.0615,
            "range": "± 1.28954",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 145.552,
            "range": "± 826.975",
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
        "date": 1781769905717,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 35.6675,
            "range": "± 63.1658",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 64.3197,
            "range": "± 575.759",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 108.751,
            "range": "± 205.547",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.55,
            "range": "± 153.667",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.5034,
            "range": "± 173.327",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 167.891,
            "range": "± 2.17879",
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
        "date": 1781857054720,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.1111,
            "range": "± 168.947",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 64.2724,
            "range": "± 6.60211",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 113.006,
            "range": "± 7.94186",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.448,
            "range": "± 354.195",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.1926,
            "range": "± 1.51447",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 157.814,
            "range": "± 2.17815",
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
        "date": 1781941231929,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.2728,
            "range": "± 360.082",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.7879,
            "range": "± 2.49654",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 105.769,
            "range": "± 2.00193",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 112.494,
            "range": "± 1.06382",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.2368,
            "range": "± 321.538",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 146.409,
            "range": "± 4.8458",
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
        "date": 1782028922404,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.217,
            "range": "± 256.393",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.5907,
            "range": "± 1.52398",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 105.52,
            "range": "± 893.266",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 113.446,
            "range": "± 1.23222",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 41.8698,
            "range": "± 860.683",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 144.444,
            "range": "± 789.967",
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
        "date": 1782117189312,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.221,
            "range": "± 504.058",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.1755,
            "range": "± 2.41447",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 106.682,
            "range": "± 429.154",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 116.49,
            "range": "± 978.868",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.6454,
            "range": "± 1.22493",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 144.605,
            "range": "± 2.4809",
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
        "date": 1782200205892,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.3922,
            "range": "± 355.64",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.4747,
            "range": "± 303.891",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 109.915,
            "range": "± 1.57785",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 113.625,
            "range": "± 1.16643",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.123,
            "range": "± 531.911",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 145.17,
            "range": "± 2.05517",
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
        "date": 1782286439296,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 44.3381,
            "range": "± 521.166",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 64.4529,
            "range": "± 405.825",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 109.45,
            "range": "± 2.28128",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 114.032,
            "range": "± 798.847",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 46.297,
            "range": "± 477.905",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 147.483,
            "range": "± 620.491",
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
        "date": 1782372801752,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 29.5711,
            "range": "± 144.343",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 47.3957,
            "range": "± 281.302",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 78.1275,
            "range": "± 530.543",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 88.6233,
            "range": "± 112.85",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 33.9859,
            "range": "± 244.252",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 113.426,
            "range": "± 3.43813",
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
        "date": 1782459597905,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 45.1201,
            "range": "± 769.009",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.2258,
            "range": "± 1.9209",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 115.771,
            "range": "± 4.54167",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.309,
            "range": "± 874.678",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 45.4219,
            "range": "± 1.71942",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 173.027,
            "range": "± 16.5757",
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
        "date": 1782545253180,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.3411,
            "range": "± 252.023",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.7698,
            "range": "± 308.257",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 112.037,
            "range": "± 2.18947",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.865,
            "range": "± 801.002",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.0463,
            "range": "± 253.057",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 160.197,
            "range": "± 3.89107",
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
        "date": 1782632513145,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.7147,
            "range": "± 513.356",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.3371,
            "range": "± 380.686",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 105.259,
            "range": "± 2.33568",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 115.173,
            "range": "± 488.706",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.0955,
            "range": "± 1.65064",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 144.473,
            "range": "± 1.08279",
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
        "date": 1782720707430,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.1449,
            "range": "± 122.998",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.5777,
            "range": "± 368.555",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 113.018,
            "range": "± 2.45734",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.81,
            "range": "± 794.15",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.6888,
            "range": "± 656.372",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 158.112,
            "range": "± 1.42414",
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
        "date": 1782805206992,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.2545,
            "range": "± 116.875",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.3021,
            "range": "± 321.875",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 112.177,
            "range": "± 1.85078",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.113,
            "range": "± 1.60883",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.2256,
            "range": "± 483.919",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 159.566,
            "range": "± 3.85741",
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
        "date": 1782892365437,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.1506,
            "range": "± 172.869",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.9059,
            "range": "± 289.156",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 109.759,
            "range": "± 359.627",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.445,
            "range": "± 775.706",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 45.0203,
            "range": "± 1.79109",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 159.142,
            "range": "± 3.63647",
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
        "date": 1782977523743,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.2137,
            "range": "± 412.119",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.0089,
            "range": "± 1.30853",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 112.317,
            "range": "± 2.61684",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 110.609,
            "range": "± 900.342",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.0625,
            "range": "± 367.567",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 159.695,
            "range": "± 3.10373",
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
        "date": 1783063853015,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.7344,
            "range": "± 387.226",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.8769,
            "range": "± 675.783",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 113.755,
            "range": "± 816.65",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.557,
            "range": "± 652.337",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 46.686,
            "range": "± 372.829",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 159.381,
            "range": "± 3.30738",
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
        "date": 1783149662122,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 45.3895,
            "range": "± 225.549",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.4176,
            "range": "± 319.928",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.712,
            "range": "± 791.415",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 111.24,
            "range": "± 619.55",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.9785,
            "range": "± 1.31057",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 158.495,
            "range": "± 2.3437",
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
        "date": 1783236671153,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.3435,
            "range": "± 1.13021",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.8389,
            "range": "± 580.403",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 107.37,
            "range": "± 1.26087",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 112.739,
            "range": "± 752.59",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.6619,
            "range": "± 379.898",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 147.046,
            "range": "± 5.58574",
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
        "date": 1783324533792,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 35.6021,
            "range": "± 145.788",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.9007,
            "range": "± 354.826",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 112.136,
            "range": "± 2.20771",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.824,
            "range": "± 1.30817",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.6614,
            "range": "± 375.953",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 170.296,
            "range": "± 2.39319",
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
        "date": 1783409592607,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.1116,
            "range": "± 488.501",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.6201,
            "range": "± 396.963",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 114.765,
            "range": "± 1.22526",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.943,
            "range": "± 737.796",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.3637,
            "range": "± 420.406",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 158.039,
            "range": "± 551.688",
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
        "date": 1783494935305,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.0979,
            "range": "± 141.712",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.3158,
            "range": "± 184.589",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.453,
            "range": "± 3.23031",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 110.991,
            "range": "± 628.548",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.8891,
            "range": "± 350.533",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 160.354,
            "range": "± 2.53566",
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
        "date": 1783582359422,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 32.5862,
            "range": "± 165.527",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 48.1903,
            "range": "± 206.913",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 86.9676,
            "range": "± 1.9651",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 89.3076,
            "range": "± 291.556",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 34.3697,
            "range": "± 260.922",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 115.915,
            "range": "± 2.03168",
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
        "date": 1783668757933,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.6131,
            "range": "± 496.382",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.0475,
            "range": "± 1.21528",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 109.786,
            "range": "± 2.13584",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 114.424,
            "range": "± 483.904",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.482,
            "range": "± 652.007",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 145.18,
            "range": "± 751.452",
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
        "date": 1783753456904,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.0989,
            "range": "± 138.262",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.3036,
            "range": "± 124.623",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.93,
            "range": "± 740.724",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 107.516,
            "range": "± 1.21539",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.3473,
            "range": "± 784.35",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 164.957,
            "range": "± 4.63093",
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
        "date": 1783840375621,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 30.0123,
            "range": "± 530.856",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 47.4971,
            "range": "± 235.648",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 84.464,
            "range": "± 1.53343",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 88.4923,
            "range": "± 1.15575",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 34.6293,
            "range": "± 780.41",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 114.488,
            "range": "± 4.05046",
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
        "date": 1783927817110,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 43.4325,
            "range": "± 281.42",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.166,
            "range": "± 244.571",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 107.543,
            "range": "± 1.59135",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 116.197,
            "range": "± 1.15094",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.6071,
            "range": "± 620.512",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 146.744,
            "range": "± 3.89666",
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
        "date": 1784012609085,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.8242,
            "range": "± 381.876",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.6998,
            "range": "± 322.116",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 103.153,
            "range": "± 2.99108",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 115.055,
            "range": "± 1.2194",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.393,
            "range": "± 704.662",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 151.631,
            "range": "± 4.01187",
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
        "date": 1784099110907,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 39.2616,
            "range": "± 46.7277",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.5071,
            "range": "± 388.265",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 109.32,
            "range": "± 292.019",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.943,
            "range": "± 539.738",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.9859,
            "range": "± 151.999",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 172.798,
            "range": "± 2.59063",
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
        "date": 1784185809273,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.3402,
            "range": "± 737.173",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.8008,
            "range": "± 1.26189",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 105.788,
            "range": "± 5.7053",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 113.848,
            "range": "± 510.628",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.0374,
            "range": "± 951.891",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 146.356,
            "range": "± 3.0575",
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
        "date": 1784272025512,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 31.5412,
            "range": "± 268.346",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 64.5271,
            "range": "± 330.621",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 100.937,
            "range": "± 319.401",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 113.826,
            "range": "± 577.034",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 41.3091,
            "range": "± 286.065",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 192.854,
            "range": "± 1.93338",
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
        "date": 1784358086436,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.214,
            "range": "± 146.84",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.3634,
            "range": "± 181.337",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 110.214,
            "range": "± 974.97",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.876,
            "range": "± 3.17791",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.9774,
            "range": "± 387.114",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 158.894,
            "range": "± 2.78907",
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
        "date": 1784445142727,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.0527,
            "range": "± 172.241",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.5083,
            "range": "± 314.656",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 110.617,
            "range": "± 1.22948",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 110.703,
            "range": "± 2.13632",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.703,
            "range": "± 447.765",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 160.008,
            "range": "± 3.60128",
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
        "date": 1784532388562,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.1875,
            "range": "± 295.851",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.7304,
            "range": "± 896.372",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 110.513,
            "range": "± 343.716",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 111.257,
            "range": "± 919.698",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.1272,
            "range": "± 851.112",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 157.84,
            "range": "± 814.95",
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
        "date": 1784618103294,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.1407,
            "range": "± 1.1492",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.345,
            "range": "± 190.444",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 110.967,
            "range": "± 1.19048",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.953,
            "range": "± 326.716",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.6527,
            "range": "± 1.16562",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 157.766,
            "range": "± 1.38973",
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
        "date": 1784704529934,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 40.3052,
            "range": "± 364.733",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.2336,
            "range": "± 381.805",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 103.492,
            "range": "± 2.01149",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 116.033,
            "range": "± 361.434",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.8615,
            "range": "± 1.19073",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 147.696,
            "range": "± 4.23281",
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
        "date": 1784790819282,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.2953,
            "range": "± 323.851",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.0619,
            "range": "± 133.294",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 107.931,
            "range": "± 2.12716",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 112.536,
            "range": "± 1.1766",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.0089,
            "range": "± 422.138",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 144.705,
            "range": "± 2.84103",
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
        "date": 1784877257210,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 24.2978,
            "range": "± 2.14473",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 54.582,
            "range": "± 1.04281",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 86.4446,
            "range": "± 3.03077",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 97.0774,
            "range": "± 538.449",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 35.0411,
            "range": "± 153.58",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 166.57,
            "range": "± 3.73056",
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
        "date": 1784963235477,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.7062,
            "range": "± 363.998",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.3932,
            "range": "± 553.776",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 116.044,
            "range": "± 1.12803",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 110.151,
            "range": "± 768.843",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 47.2559,
            "range": "± 499.32",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 158.964,
            "range": "± 1.57406",
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
        "date": 1785050108187,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.0891,
            "range": "± 117.264",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.5703,
            "range": "± 691.34",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.503,
            "range": "± 1.062",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.174,
            "range": "± 559.267",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.8767,
            "range": "± 1.0347",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 158.003,
            "range": "± 2.58492",
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
        "date": 1785137494305,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.069,
            "range": "± 210.737",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.5868,
            "range": "± 366.983",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 113.289,
            "range": "± 647.214",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 110.251,
            "range": "± 1.5034",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.3982,
            "range": "± 603.979",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 158.814,
            "range": "± 3.45724",
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
        "date": 1785222987526,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 35.0094,
            "range": "± 280.307",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 67.1974,
            "range": "± 392.648",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 103.412,
            "range": "± 1.27702",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 116.841,
            "range": "± 482.51",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.842,
            "range": "± 363.592",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 200.258,
            "range": "± 1.6881",
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
        "date": 1785309459449,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.6184,
            "range": "± 162.337",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.9211,
            "range": "± 364.578",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 112.018,
            "range": "± 2.00664",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.928,
            "range": "± 1.00647",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.2878,
            "range": "± 196.67",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 159.266,
            "range": "± 2.22211",
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
          "id": "953ce2a80545e612c30b4dd98e86c10695087ee8",
          "message": "test: align and simplify unit coverage (#230)\n\nIsolate optimizer passes in tests, remove redundant oracle coverage,\nstrengthen frontend assertions, and organize tests by responsibility.\nAlso export TileAxisFusionPass from the top-level Python package.\n\nAssisted-by: ChatGPT (GPT-5.6 sol) <noreply@openai.com>",
          "timestamp": "2026-07-29T20:49:43Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/953ce2a80545e612c30b4dd98e86c10695087ee8"
        },
        "date": 1785395773019,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 38.2876,
            "range": "± 724.339",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.1605,
            "range": "± 1.46534",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 99.98,
            "range": "± 1.2196",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 110.249,
            "range": "± 802.994",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 41.1075,
            "range": "± 514.329",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 145.898,
            "range": "± 3.19876",
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
          "id": "e8f9d1b4766626abcc8e1b1798fc4957f7e15bf4",
          "message": "feat(noncomp): add leakage and loss simulation (#231)\n\nAdd five-level noncomputational models, transition annotations,\nmeasurement classifiers, and exact trajectory sampling through\nresumable SVM instruments.\n\nExpose the experimental Python API, documentation, playground guidance,\nand end-to-end validation while preserving ordinary simulation paths.\n\nAssisted-by: Claude (Fable) <noreply@anthropic.com>\nAssisted-by: Claude (Opus) <noreply@anthropic.com>\nAssisted-by: Claude (Sonnet 5) <noreply@anthropic.com>\nAssisted-by: ChatGPT (GPT-5.6 sol) <noreply@openai.com>",
          "timestamp": "2026-07-30T17:58:33Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/e8f9d1b4766626abcc8e1b1798fc4957f7e15bf4"
        },
        "date": 1785434880077,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.0939,
            "range": "± 146.892",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.6973,
            "range": "± 691.48",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 110.28,
            "range": "± 1.80426",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.55,
            "range": "± 687.343",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 45.5729,
            "range": "± 368.345",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 158.631,
            "range": "± 2.24972",
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
          "id": "e8f9d1b4766626abcc8e1b1798fc4957f7e15bf4",
          "message": "feat(noncomp): add leakage and loss simulation (#231)\n\nAdd five-level noncomputational models, transition annotations,\nmeasurement classifiers, and exact trajectory sampling through\nresumable SVM instruments.\n\nExpose the experimental Python API, documentation, playground guidance,\nand end-to-end validation while preserving ordinary simulation paths.\n\nAssisted-by: Claude (Fable) <noreply@anthropic.com>\nAssisted-by: Claude (Opus) <noreply@anthropic.com>\nAssisted-by: Claude (Sonnet 5) <noreply@anthropic.com>\nAssisted-by: ChatGPT (GPT-5.6 sol) <noreply@openai.com>",
          "timestamp": "2026-07-30T17:58:33Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/e8f9d1b4766626abcc8e1b1798fc4957f7e15bf4"
        },
        "date": 1785482460179,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 45.0505,
            "range": "± 617.554",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 64.5074,
            "range": "± 604.49",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 114.037,
            "range": "± 1.06837",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 108.394,
            "range": "± 1.74374",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 45.4547,
            "range": "± 783.308",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 158.886,
            "range": "± 988.807",
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
          "id": "27085d10a6822a00eb0bbae9ac708ec13dd4de27",
          "message": "docs: prepare v0.7.0 release (#235)",
          "timestamp": "2026-07-31T15:10:55Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/27085d10a6822a00eb0bbae9ac708ec13dd4de27"
        },
        "date": 1785568322867,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 35.5529,
            "range": "± 124.992",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.06,
            "range": "± 148.702",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 106.709,
            "range": "± 1.05945",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 107.243,
            "range": "± 165.068",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.0227,
            "range": "± 257.129",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 166.983,
            "range": "± 705.774",
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
          "id": "27085d10a6822a00eb0bbae9ac708ec13dd4de27",
          "message": "docs: prepare v0.7.0 release (#235)",
          "timestamp": "2026-07-31T15:10:55Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/27085d10a6822a00eb0bbae9ac708ec13dd4de27"
        },
        "date": 1785654811203,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.1403,
            "range": "± 208.158",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.5144,
            "range": "± 352.852",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 115.074,
            "range": "± 381.825",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.083,
            "range": "± 807.973",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.9636,
            "range": "± 639.247",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 157.823,
            "range": "± 1.05385",
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
          "id": "2c1dfa6029c4f0573c499e938e9a88106a6801b3",
          "message": "perf(noncomp): reuse noise arena for continuations (#237)",
          "timestamp": "2026-08-02T19:17:39Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/2c1dfa6029c4f0573c499e938e9a88106a6801b3"
        },
        "date": 1785742336530,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.5557,
            "range": "± 156.075",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.9336,
            "range": "± 1.47181",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 113.819,
            "range": "± 2.06915",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.242,
            "range": "± 860.936",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 46.8712,
            "range": "± 427.452",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 163.519,
            "range": "± 2.4854",
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
          "id": "db22e0309d26a3272293229b98b70d05d5ddb4a5",
          "message": "test: keep release SVM finite checks effective (#240)\n\nReplace std::isfinite assertions with the shared IEEE-754 bit-pattern\ncheck so numerical-stability coverage remains meaningful under\n-ffast-math.\n\nCloses #174\n\nAssisted-by: ChatGPT (GPT-5.6 Sol) <noreply@openai.com>",
          "timestamp": "2026-08-03T17:29:11Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/db22e0309d26a3272293229b98b70d05d5ddb4a5"
        },
        "date": 1785827761909,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 42.5779,
            "range": "± 131.895",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 64.5446,
            "range": "± 608.077",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 110.775,
            "range": "± 1.13681",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 109.346,
            "range": "± 1.29982",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.8543,
            "range": "± 333.796",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 153.365,
            "range": "± 2.04358",
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
          "id": "0e05ebf2896a25124e0509bd612997970fa7e9a2",
          "message": "fix(optimizer): eliminate terminal phases across resets (#243)\n\nAllow terminal measurement phases to cross conditional Pauli corrections when their quantum supports are disjoint and commutation/dataflow checks prove the crossing safe. Preserve barriers for overlapping feedback and instrument boundaries.\n\nAdd regression coverage for MR/MRX/MRY ordering, classical records, source maps, continuation fences, and sampled equivalence.\n\nCloses #242\n\nAssisted-by: ChatGPT (GPT-5.6 Sol) <noreply@openai.com>",
          "timestamp": "2026-08-04T12:37:03Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/0e05ebf2896a25124e0509bd612997970fa7e9a2"
        },
        "date": 1785914187882,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 37.9467,
            "range": "± 443.649",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.9351,
            "range": "± 117.668",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 104.377,
            "range": "± 1.21232",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 112.168,
            "range": "± 758.093",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.3135,
            "range": "± 280.18",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 150.598,
            "range": "± 2.39878",
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
          "id": "1f4692ee51814c2abed14f455a66505e9e964651",
          "message": "ci: cache wasm object files with ccache in the emsdk container (#254)\n\nci: cache wasm object files with ccache in the emsdk container (#254)\n\nThe wasm build cache was all-or-nothing: any change under src/clifft\ninvalidated the key and forced a full ~3 minute rebuild of every\ntranslation unit. Mount a pinned static ccache binary and a\npersistent cache directory into the emsdk container so unchanged\ntranslation units hit the object cache even when the build-directory\nkey misses; verified in CI at a 100% direct hit rate, the wasm job\ndrops to under a minute on core-touching changes. The key also\nhashes the action definition itself, so recipe changes (including\nthis one landing on main) force one rebuild that seeds the shared\ncompiler cache immediately.\n\nAssisted-by: Claude (Fable 5) <noreply@anthropic.com>",
          "timestamp": "2026-08-05T22:10:13Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/1f4692ee51814c2abed14f455a66505e9e964651"
        },
        "date": 1786000840202,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.704,
            "range": "± 354.641",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.6005,
            "range": "± 489.684",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 115.239,
            "range": "± 526.627",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 111.968,
            "range": "± 1.91394",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 46.6923,
            "range": "± 460.085",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 166.42,
            "range": "± 2.66486",
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
          "id": "6780afc49a10606c1c878481a9568c4c752e1ce6",
          "message": "feat(sampling): add scalar plan executor (#258)\n\nExecute prepared sampling plans with preallocated state, affine symbol evaluation, direct-Pauli kernels, deterministic sampling, and numerical-dust telemetry. Add differential coverage against the legacy SVM, including multi-coordinate measurements and arbitrary-angle rotations.\n\nAssisted-by: Codex (GPT-5.6) <noreply@openai.com>",
          "timestamp": "2026-08-06T15:13:53Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/6780afc49a10606c1c878481a9568c4c752e1ce6"
        },
        "date": 1786085264179,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 24.1809,
            "range": "± 57.6837",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 58.9483,
            "range": "± 188.142",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 88.0889,
            "range": "± 212.102",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 98.1043,
            "range": "± 2.80857",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 35.2768,
            "range": "± 1.51913",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 120.155,
            "range": "± 2.58394",
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
          "id": "f295361f9f42eb38d9f95eb44014518f32f54d4a",
          "message": "perf: evaluate symbolic expressions incrementally (#281)\n\nReplace repeated affine term scans in the symbolic sampling executor with\nprecomputed expression registers and a symbol-to-register dependency tape.\nPropagate true symbol contributions once so expression reads become constant\ntime, while preserving authoritative symbol values for replay and continuation\nreconstruction.\n\nAt trap boundaries, grow register storage as needed and rebuild continuation\nregisters from their constants and the live true-symbol prefix. Remove the\nretained per-expression term lists and document the register dependency\ninvariants.\n\nAdd coverage for shot reset, noisy postselection, true-prefix continuation,\nand experimental noncomputational continuations.\n\nRefs #280\n\nAssisted-by: Claude (Opus 4.6) <noreply@anthropic.com>",
          "timestamp": "2026-08-08T00:34:04Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/f295361f9f42eb38d9f95eb44014518f32f54d4a"
        },
        "date": 1786170888945,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.5116,
            "range": "± 144.51",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.9834,
            "range": "± 197.657",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 109.447,
            "range": "± 1.38837",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 101.231,
            "range": "± 839.979",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.2788,
            "range": "± 545.432",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 159.651,
            "range": "± 2.67117",
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
          "id": "b80225c911f736fef56bd2fddbac389c6621cdbf",
          "message": "perf: specialize symbolic Pauli rotations (#282)\n\nEnumerate coefficient pairs directly, derive paired phases from the Pauli phase class, and use explicit real/imaginary butterflies to reduce scalar active-state work.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-08-08T09:42:27Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/b80225c911f736fef56bd2fddbac389c6621cdbf"
        },
        "date": 1786257253379,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.7643,
            "range": "± 246.147",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.4724,
            "range": "± 524.013",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 112.567,
            "range": "± 761.149",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 100.165,
            "range": "± 206.881",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.2743,
            "range": "± 525.463",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 154.075,
            "range": "± 2.06878",
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
          "id": "cbb6a1e96337d8d098cda77adaf6d9183b1dcafc",
          "message": "refactor(sampling): separate executor responsibilities\n\nMove CPU plan lowering and multi-shot orchestration out of the mutable one-shot executor. Encapsulate optional ISA-specific fused-rotation preparation behind a portable dispatch boundary while preserving the action tape and scalar fallback.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-08-09T14:11:08Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/cbb6a1e96337d8d098cda77adaf6d9183b1dcafc"
        },
        "date": 1786344583461,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 43.8358,
            "range": "± 227.37",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 64.0586,
            "range": "± 252.315",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 112.945,
            "range": "± 1.32323",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 102.655,
            "range": "± 462.329",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.1355,
            "range": "± 1.64406",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 165.866,
            "range": "± 4.08972",
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
          "id": "550cfbb65d97e4100b9ba05e5ea023046f9eb218",
          "message": "perf(sampling): vectorize lane-paired active measurements (#292)\n\nAdd AVX-512 probability and in-place collapse kernels for active\nmeasurements whose Pauli pairs fit within a vector block. Select the\nkernel during executable-plan lowering while preserving scalar\nfallbacks for unsupported ISAs, diagonal and high-pivot shapes, and\nunprofitable width-three cases.\n\nStore the dispatch tag in existing action padding and compact\npivot-zero lane representatives directly into the state prefix. Add\nexhaustive scalar-oracle coverage for masks, pivots, phase classes,\nbranches, and ISA fallback boundaries.\n\nAssisted-by: Codex (GPT-5.6) <noreply@openai.com>",
          "timestamp": "2026-08-11T01:27:45Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/550cfbb65d97e4100b9ba05e5ea023046f9eb218"
        },
        "date": 1786430423701,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.3722,
            "range": "± 153.532",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 63.2796,
            "range": "± 138.285",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 116.18,
            "range": "± 288.449",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 101.854,
            "range": "± 1.46175",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 44.9694,
            "range": "± 2.96622",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 154.38,
            "range": "± 3.57779",
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
          "id": "7e8e301636cc2b896b7fd8641b66513361cba1a7",
          "message": "perf(sampling): add AVX2 symbolic kernels (#309)\n\nAdd runtime-dispatched AVX2 kernels for direct and fused rotations and\nactive-measurement probability and collapse. Preserve scalar and AVX-512\nfallbacks, keep ISA-specific data outside portable descriptors, and add\ndifferential, boundary, and portability coverage.\n\nCloses #285.\n\nAssisted-by: Codex (GPT-5.6) <noreply@openai.com>",
          "timestamp": "2026-08-11T21:09:17Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/7e8e301636cc2b896b7fd8641b66513361cba1a7"
        },
        "date": 1786517449952,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 41.4238,
            "range": "± 172.39",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 62.9338,
            "range": "± 271.239",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.425,
            "range": "± 2.61707",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 99.5484,
            "range": "± 631.353",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.0457,
            "range": "± 228.279",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 153.894,
            "range": "± 1.10677",
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
          "id": "73398fd5904a818f41993fc9b5011a24a8bb2ce8",
          "message": "refactor(sampling): specialize instrument actions\n\nLower planned instrument modes into concrete prepared execution forms while retaining one outer action alternative so ordinary dispatch remains unchanged. Remove hot mode and optional-field checks, preserve classical, dormant-trap, active, generic-activation, and new-X execution semantics, and keep seeded RNG draws, symbol assignment, and continuation behavior unchanged.\n\nMake lowering exhaustive and Release-safe, document the intentionally distinct activation forms, and guard the compact descriptor layout. Retain compiler-selected outlining for rare fire and trap paths after forced inlining measured worse.\n\nCloses #318.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-08-12T19:59:27Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/73398fd5904a818f41993fc9b5011a24a8bb2ce8"
        },
        "date": 1786603916316,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 45.9417,
            "range": "± 415.474",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.2916,
            "range": "± 1.89474",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 111.533,
            "range": "± 1.41792",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 102.085,
            "range": "± 2.05326",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 42.6727,
            "range": "± 266.935",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 157.723,
            "range": "± 2.23185",
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
          "id": "bac6f0b90e9db9bc8afe34ec039a757e3759c44d",
          "message": "refactor(sampling): specialize executor kernel backends\n\nResolve the scalar, AVX2, or AVX-512 backend once while lowering an executable plan, then instantiate the hot action loop for that backend. Keep action tags architecture-neutral and call separately compiled SIMD kernels directly without rediscovering the process ISA during action dispatch.\n\nConsolidate the operation-specific dispatch files into one boundary, preserve compact plan, executor, and action layouts, and validate continuation backend compatibility. Retain type-erased fused-rotation sidecars where prepared storage must remain coupled to its matching kernel.\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-08-14T00:47:35Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/bac6f0b90e9db9bc8afe34ec039a757e3759c44d"
        },
        "date": 1786690235937,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 37.9203,
            "range": "± 370.131",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 61.3032,
            "range": "± 138.27",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 100.526,
            "range": "± 1.8944",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 103.439,
            "range": "± 360.996",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 41.7578,
            "range": "± 216.794",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 150.566,
            "range": "± 2.69672",
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
          "id": "7fdaa82f74f15fe2635f2799eb260a6324e48466",
          "message": "ci: run cross-platform checks on pull requests (#328)\n\nGate pull requests on macOS and Windows, speed the Windows Python suite with an optimized assertion-enabled extension, and stabilize matrix check names.\n\nAssisted-by: Claude (Fable 5) <noreply@anthropic.com>\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-08-14T13:32:58Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/7fdaa82f74f15fe2635f2799eb260a6324e48466"
        },
        "date": 1786775514782,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 29.3511,
            "range": "± 371.644",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 47.4799,
            "range": "± 249.986",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 81.6179,
            "range": "± 272.537",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 80.2962,
            "range": "± 533.111",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 32.8409,
            "range": "± 873.364",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 116.823,
            "range": "± 1.07338",
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
          "id": "df0b811688d79378919a39fe229de7a68339bb84",
          "message": "feat(sampling): expose plan inspection provenance (#330)\n\nfeat(sampling): expose plan inspection provenance\n\nAdd opt-in circuit source-line provenance to SamplingPlan and preserve it through executable lowering and fusion.\n\n- map semantic plan actions to their source lines\n- map executable actions to half-open plan-action ranges\n- preserve provenance through filtering passes\n- isolate diagnostic formatting from ordinary production binaries\n- retain no per-action provenance storage unless requested\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-08-15T12:59:46Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/df0b811688d79378919a39fe229de7a68339bb84"
        },
        "date": 1786861979671,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 42.3627,
            "range": "± 240.076",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 65.45,
            "range": "± 186.074",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 113.375,
            "range": "± 513.109",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 103.673,
            "range": "± 437.385",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 43.9618,
            "range": "± 258.9",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 154.137,
            "range": "± 4.09991",
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
          "id": "bca72ee394ea3b257dd16f5d2b600121dec73746",
          "message": "feat(wasm): use symbolic sampling backend in playground\n\nReplace the playground's legacy SVM compilation and execution with SamplingPlan, ExecutablePlan, and the symbolic sampler.\n\n- make the semantic sampling plan the default inspection view\n- retain prepared executable actions as an advanced view\n- preserve linked circuit, HIR, plan, and executable navigation\n- separate HIR formatting from legacy VM introspection\n- bound interactive expression output and avoid provenance overhead during simulation\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-08-16T12:02:04Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/bca72ee394ea3b257dd16f5d2b600121dec73746"
        },
        "date": 1786948437967,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 44.854,
            "range": "± 893.328",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 64.5469,
            "range": "± 235.511",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 118.161,
            "range": "± 1.24407",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 101.823,
            "range": "± 311.838",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 46.8237,
            "range": "± 577.423",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 153.214,
            "range": "± 430.543",
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
          "id": "164ca9e5f016ec86981bd14dab561193125b554f",
          "message": "docs: explain the symbolic sampling architecture (#337)\n\ndocs: document the symbolic sampling architecture\n\nExplain the HIR-to-executor pipeline, planning and execution boundaries,\nruntime dispatch, validation, and testing strategy. Clarify SymFT provenance\nand standardize active width versus active-state dimension terminology.\n\nCloses #306\n\nAssisted-by: Codex (GPT-5) <noreply@openai.com>",
          "timestamp": "2026-08-17T21:59:48Z",
          "url": "https://github.com/unitaryfoundation/clifft/commit/164ca9e5f016ec86981bd14dab561193125b554f"
        },
        "date": 1787034776223,
        "tool": "catch2",
        "benches": [
          {
            "name": "QV-10 x100 shots",
            "value": 7.5146,
            "range": "± 62.5958",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "cultivation-d5 x1000 shots",
            "value": 33.274,
            "range": "± 98.6288",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d7-r7 p=1e-3 x10000 shots",
            "value": 55.219,
            "range": "± 345.357",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d5-r5 p=0.05 x10000 shots",
            "value": 64.0671,
            "range": "± 2.39447",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "surface-d11-r11 p=1e-3 x1000 shots",
            "value": 19.9102,
            "range": "± 468.283",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          },
          {
            "name": "exp-val 20q 200 probes x100k",
            "value": 96.5492,
            "range": "± 648.941",
            "unit": "ms",
            "extra": "100 samples\n1 iterations"
          }
        ]
      }
    ]
  }
}
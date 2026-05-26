window.BENCHMARK_DATA = {
  "lastUpdate": 1779780935938,
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
      }
    ]
  }
}
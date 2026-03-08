#!/usr/bin/env bash
# Cloud instance setup for UCC SOFT paper reproduction.
#
# Usage (on a fresh EC2 instance):
#   curl -sL <raw-github-url>/paper/magic/cloud_setup.sh | bash
#   # or:
#   git clone https://github.com/unitaryfoundation/ucc-next.git
#   cd ucc-next && bash paper/magic/cloud_setup.sh
#
# Tested on: Amazon Linux 2023, Ubuntu 22.04/24.04
# Target instance: c7i.24xlarge (48 vCPUs, Sapphire Rapids, AVX-512)

set -euo pipefail

echo "=== UCC Cloud Setup ==="
echo "Instance: $(uname -n)"
echo "CPUs: $(nproc)"
echo "Arch: $(uname -m)"
echo

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
echo "--- Installing system dependencies ---"
if command -v dnf &>/dev/null; then
    # Amazon Linux 2023
    sudo dnf install -y gcc gcc-c++ cmake git python3.12 python3.12-devel \
        libstdc++-devel || true
elif command -v apt-get &>/dev/null; then
    # Ubuntu
    sudo apt-get update -qq
    sudo apt-get install -y build-essential cmake git python3-dev || true
fi

# ---------------------------------------------------------------------------
# 2. Install uv (Python package manager)
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "--- Installing uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Clone repo (if not already in it)
# ---------------------------------------------------------------------------
REPO_DIR="$HOME/ucc-next"
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "--- Cloning repo ---"
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        git clone "https://${GITHUB_TOKEN}@github.com/unitaryfoundation/ucc-next.git" "$REPO_DIR"
    else
        echo "NOTE: Repo is private. Set GITHUB_TOKEN env var if clone fails."
        echo "  export GITHUB_TOKEN=github_pat_..."
        git clone https://github.com/unitaryfoundation/ucc-next.git "$REPO_DIR"
    fi
fi
cd "$REPO_DIR"
git pull --ff-only origin main || true
echo "Repo: $(git log --oneline -1)"

# ---------------------------------------------------------------------------
# 4. Build UCC with native optimizations
# ---------------------------------------------------------------------------
echo "--- Building UCC (Release, -march=native) ---"
uv venv --python 3.12 || uv venv
uv pip install -e . 2>&1 | tail -3

# Also build native profiler for quick checks
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUCC_BUILD_PROFILER=ON 2>&1 | tail -5
cmake --build build -j"$(nproc)" 2>&1 | tail -5

# ---------------------------------------------------------------------------
# 5. Verify BMI2/AVX-512 is enabled
# ---------------------------------------------------------------------------
echo
echo "--- Checking CPU features ---"
if grep -q bmi2 /proc/cpuinfo; then
    echo "BMI2: YES (PDEP optimization active)"
else
    echo "BMI2: NO (software fallback)"
fi
if grep -q avx512 /proc/cpuinfo; then
    echo "AVX-512: YES"
else
    echo "AVX-512: NO"
fi

# ---------------------------------------------------------------------------
# 6. Quick smoke test
# ---------------------------------------------------------------------------
echo
echo "--- Smoke test (1000 shots, d=5 p=0.001) ---"
uv run python -c "
import ucc
import pathlib
text = pathlib.Path('paper/magic/circuits/circuit_d5_p0.001.stim').read_text()
p = ucc.compile(text, postselection_mask=[1]*107)
stats = ucc.sample_survivors(p, 1000, keep_records=False)
print(f'  shots={stats[\"total_shots\"]}, discards={stats[\"discards\"]}, '
      f'errors={stats[\"logical_errors\"]}')
print('  Smoke test PASSED')
"

# ---------------------------------------------------------------------------
# 7. Create results directory
# ---------------------------------------------------------------------------
mkdir -p results
echo
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Run benchmark:  uv run python paper/magic/benchmark_cloud.py"
echo "  2. Set up S3 sync: bash paper/magic/setup_s3_sync.sh <bucket-name>"
echo "  3. Run production:  uv run python paper/magic/run_cloud.py --noise 0.002"

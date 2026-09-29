#!/usr/bin/env bash
# Run as root on a fresh Ubuntu 24.04 MI300X Droplet with ROCm preinstalled.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo 'Run this initial setup as root on the fresh AMD Droplet.' >&2
    exit 1
fi
if [[ $(uname -m) != x86_64 ]]; then
    echo 'This recipe targets x86_64 MI300X Droplets.' >&2
    exit 1
fi
command -v hipcc >/dev/null || {
    echo 'HIP compiler missing from PATH; use the AMD image with ROCm preinstalled.' >&2
    exit 1
}
if [[ -e "$HOME/clifft" || -e "$HOME/clifft-hip-check" ]]; then
    echo 'This fresh-Droplet setup expects ~/clifft and ~/clifft-hip-check to be absent.' >&2
    exit 1
fi

apt-get update
apt-get install -y build-essential cmake ninja-build git curl ca-certificates python3-dev python3-venv
if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/unitaryfoundation/clifft.git "$HOME/clifft"

cat > "$HOME/clifft-hip-check" <<'CHECK'
#!/usr/bin/env bash
# Run after checking out the desired Clifft commit; does not change Git state.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/clifft"
log_dir="$HOME/clifft-hip-logs/$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
mkdir -p "$log_dir"
{
    date -u
    git rev-parse HEAD
    git status --short
    cat /etc/os-release
    uname -r
    hipcc --version
    cmake --version
    uv --version

    uv sync --frozen --only-group dev --python 3.12
    CMAKE_ARGS="" CMAKE_BUILD_PARALLEL_LEVEL=2 \
    uv pip install --reinstall-package clifft --editable . \
        -Cbuild-dir=build/manual-hip-python \
        -Ccmake.build-type=Release \
        -Ccmake.define.CLIFFT_ENABLE_HIP=ON \
        -Ccmake.define.CLIFFT_ENABLE_CUDA=OFF \
        -Ccmake.define.CMAKE_HIP_ARCHITECTURES=gfx942 \
        -Ccmake.define.CLIFFT_CPU_BASELINE=x86-64-v2

    uv run --no-sync python - <<'PY'
from clifft.experimental import hip
print(hip.backend_info())
assert hip.is_built() and hip.is_available(), 'HIP build and visible AMD GPU required'
assert 'gfx942' in hip.backend_info(), 'This setup targets MI300X / gfx942'
PY

    # A portable CPU baseline allows the cached build to survive host changes.
    cmake -S . -B build/manual-hip-native -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLIFFT_ENABLE_HIP=ON \
        -DCLIFFT_ENABLE_CUDA=OFF \
        -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        -DCLIFFT_CPU_BASELINE=x86-64-v2 \
        -DCLIFFT_BUILD_TESTS=ON
    cmake --build build/manual-hip-native --target clifft_tests clifft_hip_tests --parallel 2
    ctest --test-dir build/manual-hip-native --output-on-failure --no-tests=error \
        -R HIP --output-junit "$log_dir/native.xml"
    uv run --no-sync pytest tests/python --require-gpu=hip \
        -k 'hip or gpu_reference' -v -rP --durations=20 \
        --junitxml="$log_dir/python.xml"
} 2>&1 | tee "$log_dir/run.log"
echo "Results: $log_dir"
CHECK
chmod +x "$HOME/clifft-hip-check"
"$HOME/clifft-hip-check"
echo 'Review both test summaries for failures or skips. Re-run with ~/clifft-hip-check.'

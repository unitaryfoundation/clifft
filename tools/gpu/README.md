# Manual AMD GPU testing

Use [prepare-amd.sh](prepare-amd.sh) on a fresh DigitalOcean MI300X Droplet
with the AMD AI/ML-ready Ubuntu 24.04 image (`gpu-amd-base`). The image supplies
ROCm and the GPU driver; the script adds the build and test dependencies.
See DigitalOcean's [image guide](https://docs.digitalocean.com/products/droplets/getting-started/recommended-gpu-setup/).

## Fresh installation

SSH in as root. Download the script from this branch and run it:

```bash
curl -fL \
    https://raw.githubusercontent.com/unitaryfoundation/clifft/codex/manual-amd-setup/tools/gpu/prepare-amd.sh \
    -o ~/prepare-amd.sh
bash ~/prepare-amd.sh
```

The script expects `hipcc` on `PATH` and no existing `~/clifft` checkout or
`~/clifft-hip-check` helper. It clones the default branch, installs the locked
Python development dependencies, and builds Release HIP extensions and native
tests with two build workers. It then runs the native HIP suite and the Python
HIP selection, including shared sampling cases, focused API/replay cases, and
common CPU references. Both FP64 and FP32 are covered by that selection.

The CPU baseline is `x86-64-v2`, so cached builds can be reused on another host;
the GPU target is MI300X (`gfx942`). The checkout, Python environment, and build
caches stay under the root user's home directory on the boot disk.

Review both test summaries for failures or unexpected skips. Logs under
`~/clifft-hip-logs/<timestamp>-<commit>/` include the tested commit, Git status,
toolchain and device details, build output, and test results. JUnit reports are
saved as `native.xml` and `python.xml`. Python testing requires a visible HIP
device through `--require-gpu=hip`.

If a build or test fails after the helper has been installed, fix the reported
problem and run `~/clifft-hip-check` again; the fresh-install script deliberately
refuses to overwrite an existing checkout.

## Test another revision

The installed helper rebuilds and tests the current checkout without changing
its Git revision. To test current main:

```bash
cd ~/clifft
git fetch origin
git switch --detach origin/main
~/clifft-hip-check
```

To test a pull request, replace `123` with its number:

```bash
cd ~/clifft
git fetch origin pull/123/head
git switch --detach FETCH_HEAD
~/clifft-hip-check
```

Each run synchronizes Python dependencies to the checkout's lockfile, rebuilds
using the existing caches, and writes a new log directory. The system packages
and ROCm installation are reused. Copy any results you want to retain off the
Droplet before destroying it.

## Optional snapshot

A fresh installation is sufficient for manual testing. If saving the prepared
environment is useful, first complete a passing run, then run `shutdown -h now`
and use **Backups & Snapshots > Take snapshot** in DigitalOcean. GPU snapshots
include the boot disk, where this setup stores its files, but not the scratch
disk. See the [snapshot guide](https://docs.digitalocean.com/products/snapshots/how-to/snapshot-droplets/).

Create another MI300X Droplet from the snapshot and run `~/clifft-hip-check`
once to verify the restored environment. Afterward, use the revision commands
above for later tests. Destroy unused Droplets when finished: powering them off
does not stop compute billing. Snapshot storage is billed separately; see
[Droplet pricing](https://docs.digitalocean.com/products/droplets/details/pricing/)
and [snapshot pricing](https://docs.digitalocean.com/products/snapshots/details/pricing/).

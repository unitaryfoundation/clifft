#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: install_ubuntu_clang.sh CLANG_VERSION INSTALL_ROOT" >&2
    exit 2
fi

readonly clang_version="$1"
readonly install_root="$2"

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "the native Clang CI installer supports only x86_64" >&2
    exit 1
fi
if [[ "${clang_version}" != "22.1.8.1" ]]; then
    echo "no checksum is pinned for Clang ${clang_version}" >&2
    exit 1
fi
if [[ -e "${install_root}" ]]; then
    echo "install root already exists: ${install_root}" >&2
    exit 1
fi

readonly archive_name="static-clang-linux-amd64.tar.xz"
readonly archive_sha256="6a5419dbb658dd9c379e4cddc2a130d33aef458add0c7ccd1f529f12b0a4d67a"
readonly archive_url="https://github.com/mayeut/static-clang-images/releases/download/v${clang_version}/${archive_name}"
staging="$(mktemp -d)"
readonly staging
trap 'rm -rf -- "${staging}"' EXIT

curl --fail --silent --show-error --location \
    --retry 3 --retry-delay 2 --retry-all-errors \
    --connect-timeout 15 --max-time 300 \
    --output "${staging}/${archive_name}" "${archive_url}"
echo "${archive_sha256}  ${staging}/${archive_name}" | sha256sum --check
tar -C "${staging}" -xJf "${staging}/${archive_name}"
test -x "${staging}/clang/bin/clang++"

mkdir -p "$(dirname "${install_root}")"
mv "${staging}/clang" "${install_root}"

# The static archive defaults to its build host's musl target. Match the GNU
# driver configuration applied by manylinux before using it on Ubuntu.
readonly gcc_triple="$(gcc -dumpmachine)"
readonly driver_config="ubuntu-x86_64.cfg"
printf '%s\n' \
    '-target x86_64-unknown-linux-gnu' \
    '-march=x86-64' \
    '--gcc-toolchain=/usr' \
    "--gcc-triple=${gcc_triple}" \
    >"${install_root}/bin/${driver_config}"
for driver in clang clang++ clang-cpp; do
    printf '@%s\n' "${driver_config}" >"${install_root}/bin/${driver}.cfg"
done

readonly system_openmp_include="/usr/lib/llvm-18/lib/clang/18/include"
readonly system_openmp_library="/usr/lib/llvm-18/lib/libomp.so"
test -f "${system_openmp_include}/omp.h"
test -f "${system_openmp_library}"

# Keep the older OpenMP resource directory out of CPATH: its stdint.h would
# shadow the pinned compiler's resource header. Expose only OpenMP headers.
readonly openmp_root="$(dirname "${install_root}")/clifft-openmp"
mkdir -p "${openmp_root}/include"
for header in omp.h omp-tools.h ompt.h ompt-multiplex.h ompx.h; do
    if [[ -f "${system_openmp_include}/${header}" ]]; then
        ln -s "${system_openmp_include}/${header}" "${openmp_root}/include/${header}"
    fi
done

"${install_root}/bin/clang++" --version
"${install_root}/bin/ld.lld" --version

: "${GITHUB_ENV:?GITHUB_ENV must identify the GitHub Actions environment file}"
: "${GITHUB_PATH:?GITHUB_PATH must identify the GitHub Actions path file}"
{
    echo "CC=${install_root}/bin/clang"
    echo "CXX=${install_root}/bin/clang++"
    echo "CPATH=${openmp_root}/include${CPATH:+:${CPATH}}"
    echo "LIBRARY_PATH=$(dirname "${system_openmp_library}")${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    echo "LDFLAGS=-fuse-ld=lld"
} >>"${GITHUB_ENV}"
echo "${install_root}/bin" >>"${GITHUB_PATH}"

#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: install_manylinux_clang.sh CLANG_VERSION" >&2
    exit 2
fi

readonly clang_version="$1"
readonly clang_root="/opt/clang"
readonly openmp_root="/opt/clifft-openmp"

# The static manylinux compiler intentionally reuses the image's libstdc++ and
# libgcc. Install only the matching OpenMP development/runtime package here so
# auditwheel can bundle the non-policy runtime into the repaired wheel.
dnf install -y libomp-devel
manylinux-install-clang -v "${clang_version}"

mapfile -t omp_headers < <(rpm -ql libomp-devel | awk '/\/omp\.h$/ { print }')
if [[ ${#omp_headers[@]} -ne 1 || ! -f "${omp_headers[0]}" ]]; then
    echo "libomp-devel did not provide omp.h" >&2
    exit 1
fi
readonly omp_header="${omp_headers[0]}"

# AlmaLinux installs omp.h below its system Clang resource directory. Expose a
# stable include path so a newer pinned static Clang can discover that header.
mkdir -p "${openmp_root}/include"
omp_include_dir="$(dirname "${omp_header}")"
for header in omp.h omp-tools.h ompt.h ompt-multiplex.h ompx.h; do
    if [[ -f "${omp_include_dir}/${header}" ]]; then
        ln -s "${omp_include_dir}/${header}" "${openmp_root}/include/${header}"
    fi
done

"${clang_root}/bin/clang++" --version
"${clang_root}/bin/ld.lld" --version
test -f /usr/lib64/libomp.so

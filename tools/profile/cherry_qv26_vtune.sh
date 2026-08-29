#!/usr/bin/env bash

set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
readonly clang_version="22.1.8.1"
readonly qiskit_version="2.3.1"
# The public QV campaign records this artifact identity for QV26 seed 42.
readonly qv_qasm_sha256="6ecab6f0ed2746161133a88a7afc38d61a37bb82f320d77b173c97295e1cd9da"
readonly profile_cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/clifft-vtune"
readonly clang_root="${profile_cache_root}/clang-${clang_version}"
readonly qiskit_venv="${profile_cache_root}/qiskit-${qiskit_version}"
readonly build_dir="${CLIFFT_VTUNE_BUILD_DIR:-${repo_root}/build-vtune}"
readonly qasm_fixture="${build_dir}/qv26_seed42.qasm"
readonly fixture="${build_dir}/qv26_seed42.stim"
readonly profile_binary="${build_dir}/profile_sample"

action="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ "${action}" == "-h" || "${action}" == "--help" ]]; then
    action="help"
fi

node="0"
isa="both"
port="8080"
threads="16"
cpu_list=""
output_root="${CLIFFT_VTUNE_OUTPUT_ROOT:-${HOME}/clifft-vtune-results}"
toolchain_configured="false"
profile_command=()

usage() {
    cat <<'EOF'
Usage: tools/profile/cherry_qv26_vtune.sh ACTION [OPTIONS]

Actions:
  setup      Install host packages, pinned Clang, and VTune
  build      Generate QV26 and build profile_sample with Clang ThinLTO
  check      Verify AVX-512, perf counters, VTune, and the profile binary
  collect    Capture perf and VTune results, then create a .tar.zst archive
  all        Run setup, build, check, and collect
  serve      Start VTune Profiler Server on loopback for SSH tunneling

Options:
  --threads N         Intra-shot workers (default: 16, matching clifft-bench)
  --cpus LIST         Comma-separated logical CPUs; defaults to one per core
  --node N            NUMA node for CPU selection and memory binding (default: 0)
  --isa MODE          avx512, avx2, or both (default: both)
  --output-root PATH  Result storage (default: ~/clifft-vtune-results)
  --port N            VTune Profiler Server port (default: 8080)
  -h, --help          Show this help

Recommended first run on the Cherry Ubuntu 24.04 host:

  tools/profile/cherry_qv26_vtune.sh all

Run as root, or as a sudo-capable user. Collection uses one logical CPU from
each physical core on one NUMA node. The script does not expose the VTune web
port publicly.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads)
            [[ $# -ge 2 ]] || die "--threads requires a value"
            threads="$2"
            shift 2
            ;;
        --cpus)
            [[ $# -ge 2 ]] || die "--cpus requires a value"
            cpu_list="$2"
            shift 2
            ;;
        --node)
            [[ $# -ge 2 ]] || die "--node requires a value"
            node="$2"
            shift 2
            ;;
        --isa)
            [[ $# -ge 2 ]] || die "--isa requires a value"
            isa="$2"
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || die "--output-root requires a value"
            output_root="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || die "--port requires a value"
            port="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "${isa}" in
    avx512|avx2|both) ;;
    *) die "--isa must be avx512, avx2, or both" ;;
esac
[[ "${node}" =~ ^[0-9]+$ ]] || die "--node must be a non-negative integer"
[[ "${threads}" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive integer"
[[ "${port}" =~ ^[0-9]+$ ]] && ((port > 0 && port <= 65535)) || \
    die "--port must be between 1 and 65535"

sudo_command=()
if [[ ${EUID} -ne 0 ]] && command -v sudo >/dev/null; then
    sudo_command=(sudo)
fi

run_as_root() {
    if [[ ${EUID} -ne 0 && ${#sudo_command[@]} -eq 0 ]]; then
        die "setup requires root or sudo"
    fi
    "${sudo_command[@]}" "$@"
}

configure_toolchain() {
    if [[ "${toolchain_configured}" == "true" ]]; then
        return
    fi
    [[ -x "${clang_root}/bin/clang++" ]] || die "pinned Clang is missing; run setup"
    export PATH="${clang_root}/bin:${PATH}"
    export CC="${clang_root}/bin/clang"
    export CXX="${clang_root}/bin/clang++"
    export CPATH="${profile_cache_root}/clifft-openmp/include${CPATH:+:${CPATH}}"
    export LIBRARY_PATH="/usr/lib/llvm-18/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export LDFLAGS="-fuse-ld=lld${LDFLAGS:+ ${LDFLAGS}}"
    toolchain_configured="true"
}

vtune_command() {
    if command -v vtune >/dev/null 2>&1; then
        command -v vtune
    elif [[ -x /opt/intel/oneapi/vtune/latest/bin64/vtune ]]; then
        echo /opt/intel/oneapi/vtune/latest/bin64/vtune
    else
        die "VTune is missing; run setup"
    fi
}

vtune_backend_command() {
    if command -v vtune-backend >/dev/null 2>&1; then
        command -v vtune-backend
    elif [[ -x /opt/intel/oneapi/vtune/latest/bin64/vtune-backend ]]; then
        echo /opt/intel/oneapi/vtune/latest/bin64/vtune-backend
    else
        die "VTune Profiler Server is missing; run setup"
    fi
}

install_host_packages() {
    command -v apt-get >/dev/null || die "this bootstrap currently supports Ubuntu/Debian"
    run_as_root apt-get update
    run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential cmake curl git gnupg libomp-18-dev linux-tools-common \
        linux-tools-generic ninja-build numactl python3 python3-venv wget xz-utils zstd
}

install_clang() {
    if [[ -x "${clang_root}/bin/clang++" ]]; then
        return
    fi
    mkdir -p "${profile_cache_root}"
    local github_env github_path
    github_env="$(mktemp)"
    github_path="$(mktemp)"
    GITHUB_ENV="${github_env}" GITHUB_PATH="${github_path}" \
        "${repo_root}/.github/scripts/install_ubuntu_clang.sh" \
        "${clang_version}" "${clang_root}"
    rm -f -- "${github_env}" "${github_path}"
}

install_vtune() {
    if command -v vtune >/dev/null 2>&1 || \
       [[ -x /opt/intel/oneapi/vtune/latest/bin64/vtune ]]; then
        return
    fi

    local staging
    staging="$(mktemp -d)"
    wget -qO "${staging}/intel.key" \
        https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    gpg --batch --yes --dearmor \
        --output "${staging}/oneapi-archive-keyring.gpg" "${staging}/intel.key"
    run_as_root install -m 0644 "${staging}/oneapi-archive-keyring.gpg" \
        /usr/share/keyrings/oneapi-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        >"${staging}/oneAPI.list"
    run_as_root install -m 0644 "${staging}/oneAPI.list" /etc/apt/sources.list.d/oneAPI.list
    run_as_root apt-get update
    run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y intel-oneapi-vtune
    rm -rf -- "${staging}"
}

setup_host() {
    install_host_packages
    install_clang
    install_vtune
    configure_toolchain
    echo "setup complete"
    "${CXX}" --version
    "$(vtune_command)" -version
}

generate_fixture() {
    if [[ ! -x "${qiskit_venv}/bin/python" ]]; then
        python3 -m venv "${qiskit_venv}"
        "${qiskit_venv}/bin/python" -m pip install \
            --disable-pip-version-check \
            "dill==0.4.1" \
            "numpy==2.4.4" \
            "qiskit==${qiskit_version}" \
            "rustworkx==0.17.1" \
            "scipy==1.17.1" \
            "stevedore==5.7.0" \
            "typing-extensions==4.15.0"
    fi
    "${qiskit_venv}/bin/python" "${script_dir}/generate_qv_fixture.py" \
        --width 26 --seed 42 --qasm-output "${qasm_fixture}" --output "${fixture}"
    echo "${qv_qasm_sha256}  ${qasm_fixture}" | sha256sum --check || \
        die "generated QASM does not match the published clifft-bench QV26 seed-42 circuit"
}

build_profiler() {
    configure_toolchain
    generate_fixture
    cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS_RELEASE="-O3 -g -fdebug-info-for-profiling -DNDEBUG" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -g -fdebug-info-for-profiling -DNDEBUG" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DCLIFFT_BUILD_PROFILER=ON \
        -DCLIFFT_CPU_BASELINE=x86-64-v2 \
        -DCLIFFT_OPENMP=ON
    cmake --build "${build_dir}" --target profile_sample --parallel "$(nproc)"
    grep -q -- '-flto=thin' "${build_dir}/compile_commands.json" || \
        die "build did not enable ThinLTO"
    [[ -x "${profile_binary}" ]] || die "profile_sample was not built"
    echo "built ${profile_binary} with pinned Clang and ThinLTO"
}

select_cpus() {
    if [[ -n "${cpu_list}" ]]; then
        return
    fi
    local -a node_cpus
    mapfile -t node_cpus < <(
        lscpu -p=CPU,NODE,SOCKET,CORE | awk -F, -v wanted="${node}" \
            '$1 !~ /^#/ && $2 == wanted && !seen[$3 ":" $4]++ {print $1}'
    )
    [[ ${#node_cpus[@]} -ge threads ]] || \
        die "NUMA node ${node} has only ${#node_cpus[@]} physical cores; ${threads} requested"
    local selected=("${node_cpus[@]:0:threads}")
    local IFS=,
    cpu_list="${selected[*]}"
}

ensure_cpus_match_node() {
    [[ "${cpu_list}" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
        die "--cpus must be a comma-separated list of logical CPU numbers"
    local -a selected_cpus
    IFS=, read -r -a selected_cpus <<<"${cpu_list}"
    [[ ${#selected_cpus[@]} -eq threads ]] || \
        die "--cpus has ${#selected_cpus[@]} entries; --threads requests ${threads}"
    local selected_cpu actual_node
    for selected_cpu in "${selected_cpus[@]}"; do
        actual_node="$(lscpu -p=CPU,NODE | awk -F, -v wanted="${selected_cpu}" \
            '$1 !~ /^#/ && $1 == wanted {print $2; exit}')"
        [[ -n "${actual_node}" ]] || die "CPU ${selected_cpu} is not online"
        [[ "${actual_node}" == "${node}" ]] || \
            die "CPU ${selected_cpu} belongs to NUMA node ${actual_node}, not ${node}"
    done
}

profile_environment() {
    local selected_isa="$1"
    echo "CLIFFT_CIRCUIT_FILE=${fixture}"
    echo "CLIFFT_PROFILE_API=sample"
    echo "CLIFFT_PROFILE_SHOTS=1"
    echo "CLIFFT_PROFILE_THREADS=${threads}"
    echo "CLIFFT_PROFILE_WARMUPS=1"
    echo "CLIFFT_PROFILE_REPETITIONS=3"
    echo "CLIFFT_FORCE_ISA=${selected_isa}"
    echo "OMP_NUM_THREADS=${threads}"
    echo "OMP_DYNAMIC=FALSE"
    echo "OMP_PROC_BIND=spread"
    echo "OMP_PLACES=threads"
}

make_profile_command() {
    local selected_isa="$1"
    local -a selected_environment
    mapfile -t selected_environment < <(profile_environment "${selected_isa}")
    profile_command=(
        env "${selected_environment[@]}"
        numactl --physcpubind="${cpu_list}" --membind="${node}"
        "${profile_binary}"
    )
}

check_host() {
    configure_toolchain
    [[ -x "${profile_binary}" ]] || die "profile binary is missing; run build"
    [[ -f "${fixture}" ]] || die "QV26 fixture is missing; run build"
    command -v perf >/dev/null || die "perf is missing; run setup"
    command -v numactl >/dev/null || die "numactl is missing; run setup"
    local vtune
    vtune="$(vtune_command)"

    local feature
    for feature in avx2 bmi2 fma avx512f avx512dq; do
        grep -qw "${feature}" /proc/cpuinfo || die "host does not advertise ${feature}"
    done
    select_cpus
    ensure_cpus_match_node

    perf stat -e cycles,instructions -- true
    local smoke_output
    smoke_output="$(
        env CLIFFT_PROFILE_GENERATED_WIDTH=18 CLIFFT_PROFILE_GENERATED_DEPTH=18 \
            CLIFFT_PROFILE_WARMUPS=0 CLIFFT_PROFILE_REPETITIONS=1 \
            CLIFFT_PROFILE_SHOTS=1 CLIFFT_PROFILE_THREADS="${threads}" \
            CLIFFT_FORCE_ISA=avx512 OMP_NUM_THREADS="${threads}" OMP_DYNAMIC=FALSE \
            OMP_PROC_BIND=spread OMP_PLACES=threads \
            numactl --physcpubind="${cpu_list}" --membind="${node}" "${profile_binary}"
    )"
    echo "${smoke_output}"
    grep -q "layout 1 shot x ${threads} intra-shot" <<<"${smoke_output}" || \
        die "profile_sample did not select the requested intra-shot OpenMP team"
    local vtune_help
    vtune_help="$(mktemp)"
    "${vtune}" -help collect >"${vtune_help}"
    grep -q uarch-exploration "${vtune_help}" || \
        die "VTune does not advertise uarch-exploration"
    rm -f -- "${vtune_help}"
    echo "host check passed: CPUs ${cpu_list}, NUMA node ${node}"
}

record_metadata() {
    local session_dir="$1"
    {
        echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$(git -C "${repo_root}" rev-parse HEAD)"
        echo "git_describe=$(git -C "${repo_root}" describe --always --dirty)"
        echo "threads=${threads}"
        echo "cpus=${cpu_list}"
        echo "numa_node=${node}"
        echo "isa=${isa}"
        echo "kernel_perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"
        echo "kernel_kptr_restrict=$(cat /proc/sys/kernel/kptr_restrict)"
        echo
        "${CXX}" --version
        cmake --version
        "$(vtune_command)" -version
        echo
        uname -a
        echo
        lscpu
        echo
        numactl --hardware
        echo
        cat /etc/os-release
    } >"${session_dir}/metadata.txt"
    "${qiskit_venv}/bin/python" -m pip freeze \
        >"${session_dir}/qv-generator-requirements.txt"
    cp "${qasm_fixture}" "${session_dir}/qv26_seed42.qasm"
    cp "${fixture}" "${session_dir}/qv26_seed42.stim"
    cp "${profile_binary}" "${session_dir}/profile_sample"
    ninja -C "${build_dir}" -t commands profile_sample \
        >"${session_dir}/build-commands.txt"
}

run_logged() {
    local log_file="$1"
    shift
    "$@" 2>&1 | tee "${log_file}"
}

collect_one_isa() {
    local selected_isa="$1"
    local session_dir="$2"
    local isa_dir="${session_dir}/${selected_isa}"
    local hotspots_result="${isa_dir}/vtune-hotspots"
    local uarch_result="${isa_dir}/vtune-uarch"
    local vtune
    vtune="$(vtune_command)"
    mkdir -p "${isa_dir}"
    make_profile_command "${selected_isa}"

    run_logged "${isa_dir}/smoke.txt" "${profile_command[@]}"
    run_logged "${isa_dir}/perf-stat.txt" \
        perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,\
task-clock,context-switches,cpu-migrations,page-faults -- "${profile_command[@]}"
    run_logged "${isa_dir}/perf-record.txt" \
        perf record -e cycles:u -F 999 -g --call-graph dwarf \
        -o "${isa_dir}/perf.data" -- "${profile_command[@]}"

    run_logged "${isa_dir}/vtune-hotspots-collect.txt" \
        "${vtune}" -collect hotspots -knob sampling-mode=hw \
        -knob enable-stack-collection=true -result-dir "${hotspots_result}" \
        -- "${profile_command[@]}"
    "${vtune}" -report summary -result-dir "${hotspots_result}" \
        -report-output "${isa_dir}/vtune-hotspots-summary.txt"
    "${vtune}" -report hotspots -result-dir "${hotspots_result}" \
        -format csv -csv-delimiter comma \
        -report-output "${isa_dir}/vtune-hotspots.csv"

    run_logged "${isa_dir}/vtune-uarch-collect.txt" \
        "${vtune}" -collect uarch-exploration -result-dir "${uarch_result}" \
        -- "${profile_command[@]}"
    "${vtune}" -report summary -result-dir "${uarch_result}" \
        -report-output "${isa_dir}/vtune-uarch-summary.txt"
}

collect_profiles() {
    configure_toolchain
    check_host
    mkdir -p "${output_root}"
    select_cpus
    local timestamp session_dir archive
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    session_dir="${output_root}/qv26-${timestamp}"
    archive="${session_dir}.tar.zst"
    mkdir -p "${session_dir}"
    record_metadata "${session_dir}"

    case "${isa}" in
        avx512) collect_one_isa avx512 "${session_dir}" ;;
        avx2) collect_one_isa avx2 "${session_dir}" ;;
        both)
            collect_one_isa avx512 "${session_dir}"
            collect_one_isa avx2 "${session_dir}"
            ;;
    esac

    tar --zstd -cf "${archive}" -C "${output_root}" "$(basename "${session_dir}")"
    sha256sum "${archive}" >"${archive}.sha256"
    echo
    echo "collection complete"
    echo "results: ${session_dir}"
    echo "archive: ${archive}"
    echo "checksum: ${archive}.sha256"
}

serve_results() {
    mkdir -p "${output_root}"
    local backend
    backend="$(vtune_backend_command)"
    echo "VTune will listen only on the server loopback interface."
    echo "On your Mac, run:"
    echo "  ssh -N -L 18080:127.0.0.1:${port} USER@SERVER_IP"
    echo "Then open https://127.0.0.1:18080 and use the token printed below."
    exec "${backend}" --web-port="${port}" --data-directory="${output_root}"
}

case "${action}" in
    setup) setup_host ;;
    build) build_profiler ;;
    check) check_host ;;
    collect) collect_profiles ;;
    all)
        setup_host
        build_profiler
        collect_profiles
        ;;
    serve) serve_results ;;
    help) usage ;;
    *)
        usage >&2
        die "unknown action: ${action}"
        ;;
esac

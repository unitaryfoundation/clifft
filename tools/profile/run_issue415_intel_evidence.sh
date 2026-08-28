#!/usr/bin/env bash

set -euo pipefail

readonly baseline_revision="fba10d75acc77ac559e5750e1ccb162993340723"
readonly candidate_revision="26f3ba1ab05d46f9ad7ea24d89174b9e53266532"
readonly results_branch="bench/issue-415-linux-intel-c7i"
readonly repository_url="https://github.com/unitaryfoundation/clifft.git"
readonly checkout="${CLIFFT_ISSUE415_CHECKOUT:-${PWD}/clifft-issue415-intel}"
readonly baseline_root="/tmp/clifft-issue415-intel-baseline"
readonly candidate_root="/tmp/clifft-issue415-intel-candidate"
readonly relative_evidence_dir="tools/profile/evidence/issue-415/linux-intel-c7i"

push_results=false
if [[ ${1:-} == "--push" ]]; then
    push_results=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--push]" >&2
    exit 2
fi

for command in cmake git lscpu mpstat ninja nproc python3 taskset uv; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Missing required command: ${command}" >&2
        exit 1
    fi
done

if [[ $(lscpu -p=VENDOR | awk -F, '!/^#/ { print $1; exit }') != "GenuineIntel" ]]; then
    echo "This runner requires an Intel CPU." >&2
    exit 1
fi
if (( $(nproc) < 8 )); then
    echo "This runner requires at least eight online CPUs." >&2
    exit 1
fi
if ! grep -m1 '^flags' /proc/cpuinfo | grep -qw avx2; then
    echo "The Intel host does not expose AVX2." >&2
    exit 1
fi
if ! grep -m1 '^flags' /proc/cpuinfo | grep -qw avx512f; then
    echo "The Intel host does not expose AVX-512F." >&2
    exit 1
fi

for path in "${checkout}" "${baseline_root}" "${candidate_root}"; do
    if [[ -e ${path} ]]; then
        echo "Refusing to overwrite existing path: ${path}" >&2
        exit 1
    fi
done

git_author_name=$(git config --global user.name || true)
git_author_email=$(git config --global user.email || true)
if [[ -z ${git_author_name} ]]; then
    read -r -p "Git author name: " git_author_name
    git config --global user.name "${git_author_name}"
fi
if [[ -z ${git_author_email} ]]; then
    read -r -p "Git author email: " git_author_email
    git config --global user.email "${git_author_email}"
fi

echo "Cloning the public repository into ${checkout}"
git clone "${repository_url}" "${checkout}"
git -C "${checkout}" fetch origin
git -C "${checkout}" cat-file -e "${baseline_revision}^{commit}"
git -C "${checkout}" cat-file -e "${candidate_revision}^{commit}"
git -C "${checkout}" switch -c "${results_branch}" "${candidate_revision}"

git -C "${checkout}" worktree add --detach "${baseline_root}" "${baseline_revision}"
git -C "${checkout}" worktree add --detach "${candidate_root}" "${candidate_revision}"

for root in "${baseline_root}" "${candidate_root}"; do
    cmake -S "${root}" -B "${root}/build-profile" -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCLIFFT_BUILD_PROFILER=ON
    cmake --build "${root}/build-profile" --target profile_sample -j 8
done

evidence_dir="${checkout}/${relative_evidence_dir}"
mkdir -p "${evidence_dir}"

{
    echo "machine_label: linux-intel-ec2-c7i"
    echo "instance_type: c7i.2xlarge"
    echo "region: us-east-1"
    echo "purchase: on-demand"
    echo "baseline: ${baseline_revision}"
    echo "candidate: ${candidate_revision}"
    echo
    date -u
    uname -a
    cat /etc/os-release
    lscpu
    gcc --version
    cmake --version
    python3 --version
} >"${evidence_dir}/machine.txt"

mpstat -P ALL 1 20 >"${evidence_dir}/preflight-mpstat.txt"

cat >"${evidence_dir}/protocol.txt" <<EOF
shots: 100000
threads: 1 8
warmups: 2
repetitions: 11
fixed_k: 1
apis: sample sample_survivors sample_k sample_k_survivors
keep_records: 0 1
batches: 1 auto 256 1024
postselection: none all alternating
threshold: 0.05
fixtures: active_width5_transient.stim active_width5_sustained.stim
baseline: ${baseline_revision}
candidate: ${candidate_revision}
expected_cases_per_csv: 56
EOF

for threads in 1 8; do
    if [[ ${threads} -eq 1 ]]; then
        cpu_set="0"
    else
        cpu_set="0-7"
    fi

    for fixture in active_width5_transient active_width5_sustained; do
        for label in baseline candidate; do
            if [[ ${label} == "baseline" ]]; then
                root="${baseline_root}"
            else
                root="${candidate_root}"
            fi

            echo "Running ${label} ${fixture} with ${threads} thread(s)"
            taskset -c "${cpu_set}" \
                python3 "${root}/tools/profile/run_sampling_mode_matrix.py" \
                --executable "${root}/build-profile/profile_sample" \
                --circuit "${root}/tools/profile/fixtures/${fixture}.stim" \
                --shots 100000 \
                --threads "${threads}" \
                --warmups 2 \
                --repetitions 11 \
                --fixed-k 1 \
                --apis sample sample_survivors sample_k sample_k_survivors \
                --keep-records 0 1 \
                --batches 1 auto 256 1024 \
                --postselection none all alternating \
                --threshold 0.05 \
                --output \
                "${evidence_dir}/${label}-${fixture}-t${threads}.csv"
        done
    done
done

python3 - "${evidence_dir}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
fields = (
    "api",
    "keep_records",
    "fixed_k",
    "postselection",
    "requested_batch",
    "effective_batch",
    "effective_workers",
    "shot_workers",
    "intra_shot_workers",
    "batch_lane_work",
    "passed_shots",
    "survival",
    "retained_rows",
    "checksum",
)

baseline_paths = sorted(root.glob("baseline-*.csv"))
assert len(baseline_paths) == 4, baseline_paths

for baseline_path in baseline_paths:
    candidate_path = root / baseline_path.name.replace("baseline-", "candidate-", 1)
    with baseline_path.open() as file:
        baseline_rows = list(csv.DictReader(file))
    with candidate_path.open() as file:
        candidate_rows = list(csv.DictReader(file))

    assert len(baseline_rows) == 56, baseline_path
    assert len(candidate_rows) == 56, candidate_path

    baseline = {
        (row["mode"], row["requested_batch"]): row for row in baseline_rows
    }
    candidate = {
        (row["mode"], row["requested_batch"]): row for row in candidate_rows
    }
    assert baseline.keys() == candidate.keys()

    expected_threads = "8" if "-t8.csv" in baseline_path.name else "1"
    for key in baseline:
        for field in fields:
            assert baseline[key][field] == candidate[key][field], (
                baseline_path.name,
                key,
                field,
                baseline[key][field],
                candidate[key][field],
            )
        assert baseline[key]["shot_workers"] == expected_threads, (
            baseline_path.name,
            key,
            baseline[key]["shot_workers"],
        )
        assert baseline[key]["intra_shot_workers"] == "1", (
            baseline_path.name,
            key,
            baseline[key]["intra_shot_workers"],
        )

print("Validated eight CSVs and matching baseline/candidate outputs.")
PY

git -C "${checkout}" add "${relative_evidence_dir}"
(
    cd "${checkout}"
    uv run --frozen --only-group dev \
        pre-commit run --all-files --show-diff-on-failure
    git diff --check
    git add "${relative_evidence_dir}"
    git commit \
        -m "bench(sampling): record issue 415 results on Intel C7i" \
        -m "Assisted-by: Codex (GPT-5) <noreply@openai.com>"
)

echo
echo "Evidence commit: $(git -C "${checkout}" rev-parse HEAD)"
if [[ ${push_results} == true ]]; then
    echo "Pushing ${results_branch}. Use your GitHub username and fine-grained token when asked."
    git -C "${checkout}" push -u origin "${results_branch}"
else
    echo "Results are committed but not pushed. To push without storing the token, run:"
    echo "  git -C '${checkout}' push -u origin '${results_branch}'"
    echo "Enter your GitHub username, then paste the fine-grained token as the password."
fi

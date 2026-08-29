# Performance Benchmarks

pytest-benchmark tests for tracking Clifft performance over time.

## Running

```bash
just bench
```

Or directly:

```bash
uv run pytest tools/bench/ --benchmark-sort=name --benchmark-columns=Mean,StdDev,Ops
```

## Benchmarks

| File | Circuit | What it measures |
|------|---------|-----------------|
| `test_bench_qec.py` | d=3 surface code (`tests/fixtures/target_qec.stim`) | Compile and sample latency vs Stim |
| `test_bench_deep_clifford.py` | 50-qubit, 5000 random Cliffords | Pure Clifford compile/sample throughput |
| `test_bench_qv.py` | 20-qubit Quantum Volume (`fixtures/qv20_seed42.stim`) | Large statevector (peak active width 20) per-shot throughput |
| `test_bench_noncomp.py` | d=17, r=5 repetition-code memory with a hooked leak/loss layer | Noncomputational pipeline overhead and trap/continuation cost vs plain sampling |

## Fixtures

Pre-generated circuit files live in `fixtures/`:

- **`qv20_seed42.stim`** — 20-qubit Quantum Volume circuit (seed=42) in Stim-superset
  format. Peak active width 20 (2^20 = 1M complex amplitudes, 16 MB statevector).
  Useful for profiling dense active-state kernels.

## EC2 compiler matrix

The issue 317 EC2 collector compares a GCC 13 control against the Linux-wheel
candidate: pinned static Clang 22.1.8.1, ThinLTO, and `lld`. It covers QV-20
and QEC circuits with peak active ranks 0, 4, 5, and 10. The collector pins
the public `clifft-bench` QEC corpus by commit and SHA-256 digest, alternates
configuration order between repetitions, records EC2 and CPU metadata, and
checks scalar, AVX2, and AVX-512 output checksums before collection.

Use the Intel `c8i.8xlarge` reference host running Ubuntu 24.04. Clone the
benchmark branch, then install the pinned dependencies and checksum-verified
static Clang archive:

```bash
git clone --branch codex/issue-317-clang22-ec2-benchmark \
  https://github.com/unitaryfoundation/clifft.git
cd clifft
./tools/bench/ec2_compiler_matrix.py install-deps
```

Run the matrix inside `tmux`. The default takes several minutes and uses one
pinned logical CPU. It requires OpenMP in every build so the configurations
match the Linux release feature set. Keep the selected CPU's SMT sibling idle
during collection.

```bash
EXECUTION_ID="c8i-clang22-$(date -u +%Y%m%d)"
./tools/bench/ec2_compiler_matrix.py run --execution-id "$EXECUTION_ID"
```

Inspect the generated `tools/bench/ec2-results/issue-317/$EXECUTION_ID`
directory. Publishing is deliberately separate and refuses unrelated worktree
or staged changes:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
./tools/bench/ec2_compiler_matrix.py publish \
  --execution-id "$EXECUTION_ID" \
  --push
```

The push uses the checkout's existing Git credentials and creates a normal
commit on `codex/issue-317-clang22-ec2-benchmark`. Configure `user.name` and
`user.email` before publishing if the EC2 checkout does not already have an
author identity. The first publish also resolves the locked development tools
and runs the repository's required pre-commit suite. Stop the instance after
the push completes.

For a short harness check without collecting reportable numbers, add `--quick`.

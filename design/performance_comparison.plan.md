# **UCC Implementation Plan: The "Three Exponential Walls" Benchmark**

## **Executive Summary & Constraints**

Every exact quantum simulator eventually hits an exponential wall. The computational complexity of simulating a generic quantum circuit is governed by three primary variables:

1. **$N$**: Total physical qubits.
2. **$t$**: Total non-Clifford gate count.
3. **$k_{\max}$**: Peak simultaneous non-Clifford entanglement (Active Rank).

Standard statevector (SV) simulators scale as $\mathcal{O}(2^N)$. ZX-calculus Stabilizer Rank simulators (like tsim) scale as $\mathcal{O}(2^{\alpha t})$. UCC’s Factored State Architecture scales as $\mathcal{O}(2^{k_{\max}})$.

To mathematically prove UCC's architectural superiority in the Fault-Tolerant Quantum Computing (FTQC) regime, this benchmark suite holds two variables constant while sweeping the third, forcing each simulator into its theoretical limit.

**Strict Constraints:**

1. **The "David vs. Goliath" Hardware Rule:** UCC, Qiskit, and Qulacs MUST be strictly pinned to **a single physical CPU core**. tsim MUST be allowed to utilize an **NVIDIA GPU** (via jax[cuda13]). We are proving that UCC's $\mathcal{O}(1)$ algorithmic elegance on a CPU can outpace GPU brute-force stabilizer decomposition.
2. **Correctness First:** Before capturing any timing metrics, every benchmark circuit must be mathematically verified against Qiskit Aer (fidelity $> 0.9999$) or Stim (exact measurement distributions) at small scales ($N \le 8$).
3. **Subprocess Isolation:** To prevent Python garbage collection drift and inaccurate memory profiling, the benchmarking harness must execute every single simulation run in a fresh, isolated subprocess.
4. **Reproducibility:** All generation, execution, and plotting must be self-contained Python scripts within the paper/benchmark/ directory.

## ---

**Tool Participation Matrix**

| Benchmark Panel | Qiskit / Qulacs | tsim (GPU) | UCC (1 CPU) | Why are tools included/excluded? |
| :---- | :---- | :---- | :---- | :---- |
| **A: Physical Wall ($N$)** | ✅ | ✅ | ✅ | Shows Qiskit/Qulacs hitting the $\mathcal{O}(2^N)$ wall at $N \approx 30$. Both tsim and UCC easily survive. |
| **B: Active Rank Wall ($k$)** | ❌ | ✅ | ✅ | Qiskit/Qulacs are excluded (automatic OOM at $N=50$). tsim survives easily. UCC hits its theoretical $\mathcal{O}(2^k)$ OOM limit at $k \approx 30$. |
| **C: Stabilizer Wall ($t$)** | ❌ | ✅ | ✅ | Qiskit/Qulacs are excluded (automatic OOM at $N=50$). tsim hits its theoretical $\mathcal{O}(2^{\alpha t})$ timeout limit at $t \approx 60$. UCC survives infinitely. |
| **D: FTQC Crucible (MSD)** | ❌ | ✅ | ✅ | The practical tie-breaker. Evaluates raw shots/second on an 85-qubit Magic State Distillation protocol. Qiskit/Qulacs cannot run 85 qubits. |

*(Note: stim is used exclusively in Phase 1 as an exact mathematical oracle to verify classical parity distributions on pure-Clifford subsets. It is not included in the timing plots because the benchmark circuits require exact non-Clifford representations).*

## ---

**Phase 1: Verification & The Universal Generator**

**Goal:** Build a single parameterized circuit generator capable of navigating the $(N, t, k)$ phase space, and rigorously prove its mathematical correctness.

* **Task 1.1 (The Phase Space Generator):** Write paper/benchmark/generator.py. Implement a function generate_boundary_circuit(N, t, k_target) that outputs a valid .stim file and an equivalent OpenQASM 2.0 file (for Qiskit/Qulacs in Panel A).
  * *To grow $k$ without growing $t$:* Apply $t$ T-gates early, then apply deep, global Clifford mixing (CNOTs, H) without any measurements.
  * *To grow $t$ without growing $k$:* Inject T-states on a few qubits, briefly entangle them, and immediately measure them out to force UCC's array compaction. Repeat $t$ times.
* **Task 1.2 (Statevector Equivalence Fuzzing):** Write paper/benchmark/verify_sv.py. Generate boundary circuits at $N \in [4, 8, 12]$. Extract the dense complex statevector from Qiskit Aer. Run the circuit in UCC, call ucc.get_statevector(), and assert absolute fidelity $> 0.9999$.
* **DoD:** The generator can produce circuits that cleanly isolate $t$ from $k$, and the compiler’s geometric rewinding is proven 100% mathematically correct before benchmarking begins.

## **Phase 2: The Unified Profiling Harness**

**Goal:** Build an automated runner that safely captures Peak Memory and distinct Compilation vs. Execution times.

* **Task 2.1 (Subprocess Runner):** Write paper/benchmark/runner.py. The runner accepts a .stim or .qasm file and a --target (qiskit, qulacs, tsim, ucc).
* **Task 2.2 (Resource Tracking):** Inside the runner, use resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss to capture the absolute peak physical RAM utilized during the subprocess.
* **Task 2.3 (Strict Timing Separation):**
  * For **UCC**, time ucc.compile() separately from sampler.sample().
  * For **tsim**, time the circuit.compile_sampler() phase (the ZX-graph reduction) separately from the sampler.sample() phase (the JAX GPU execution).
* **Task 2.4 (Data Export):** Append results to a master results.csv containing: Panel, N, t, k, Target, Peak_Mem_MB, Compile_s, Exec_s.

## **Phase 3: Executing The "Three Walls"**

**Goal:** Execute the parameter sweeps that force each simulator to collapse. Set a hard system memory limit (e.g., 32GB) and a timeout of 120 seconds.

* **Task 3.1 (Panel A - The Physical Qubit Wall):**
  * *Parameters:* Lock $t=20$, $k=15$. Sweep $N \in [20, 30, 40, 60, 80, 100]$.
  * *Expected Outcome:* Qiskit and Qulacs execution times scale exponentially, crashing with Out-Of-Memory (OOM) at $N \approx 30$. tsim and UCC execution times remain flat near zero.
* **Task 3.2 (Panel B - The Active Rank Wall):**
  * *Parameters:* Lock $N=50$, $t=30$. Sweep $k_{\max} \in [10, 15, 20, 25, 30, 35]$. No mid-circuit measurements.
  * *Expected Outcome:* This defines the Factored State limit. UCC will scale exponentially in memory and time, OOM-crashing at $k_{\max} \approx 30$ (~16 GB RAM). tsim handles $t=30$ trivially and remains perfectly flat.
* **Task 3.3 (Panel C - The Stabilizer Rank Wall):**
  * *Parameters:* Lock $N=50$, $k_{\max}=10$. Sweep $t \in [10, 30, 50, 80, 120]$.
  * *Expected Outcome:* This defines the ZX Stabilizer limit. Despite having a GPU, tsim's compilation/evaluation will blow up exponentially due to generating trillions of stabilizer terms, timing out or crashing at $t \approx 60$. UCC executes in flat/linear $\mathcal{O}(t)$ time because its active array safely breathes at $2^{10}$.

## **Phase 4: The FTQC Crucible (Panel D)**

**Goal:** Benchmark raw throughput on a practical, real-world fault-tolerant algorithm.

* **Task 4.1 (The MSD Shootout):** Write paper/benchmark/run_msd.py. Utilize the 85-qubit 17-to-1 Color Code magic state distillation protocol directly from tsim's own tutorials.
* **Task 4.2 (Execution):** Request $100,000$ shots from both tsim and UCC.
  * Allow tsim to use max batch sizes to heavily saturate the GPU.
  * Configure UCC to use "Dense Survivor Sampling" (passing the OP_POSTSELECT mask to abort doomed shots instantly in the VM).
* **Task 4.3 (Metric):** Calculate raw throughput: **Shots per Second**.
* **Expected Outcome:** A bar chart proving that UCC on a single CPU core is highly competitive with, or significantly outperforms, a GPU running tsim in the FTQC regime (where $N$ and $t$ are high, but $k$ is tightly bounded by QEC limits).

## **Phase 5: Data Visualization**

**Goal:** Generate the definitive 4-panel dashboard for the paper.

* **Task 5.1 (Plotting Script):** Write paper/benchmark/plot_walls.py using matplotlib and seaborn.
* **Task 5.2 (Layout):**
  * **Top Left (Panel A):** Time vs $N$. Log-Y axis. Shows the Qiskit/Qulacs SV wall.
  * **Top Right (Panel B):** Time vs $k_{\max}$. Log-Y axis. Shows the UCC Factored State wall.
  * **Bottom Left (Panel C):** Time vs $t$. Log-Y axis. Shows the tsim Stabilizer Rank wall.
  * **Bottom Right (Panel D):** Bar Chart. "Throughput (Shots/sec) on 85-qubit Magic State Distillation."
* **DoD:** A single, publication-ready .pdf or .png file that immediately communicates the distinct mathematical capabilities of statevector, stabilizer-rank, and factored-state architectures.

Here is the formatted section, written in the exact style of your design documents. You can copy and paste this directly at the bottom of your `performance_comparison.plan.md` document.

---

## Appendix: Safe EC2 Benchmarking & OOM Capture

Relying on a system to natively hit an Out-Of-Memory (OOM) threshold is dangerous on cloud infrastructure. If a statevector simulator attempts to allocate 128 GB of RAM on a 32 GB EC2 instance, Linux will often attempt to page memory to the EBS volume (thrashing) or trigger the global OOM-killer, which can blindly assassinate the SSH daemon and permanently lock you out of the instance.

To safely induce, catch, and plot OOM crashes and timeouts during the parameter sweeps without breaking the AWS instance, the benchmarking harness must enforce strict boundary-boxing.

### 1. Disable Swap Space (Pre-Requisite)

By default, some AWS AMIs (like Amazon Linux or Ubuntu) enable a swap file. If swap is active, exceeding physical RAM will not instantly crash the process; it will spill over to the SSD, dropping execution speed to zero and freezing the machine for hours.

Before executing the benchmark harness, SSH into the EC2 instance and disable swap entirely to force immediate OOM kills.

```bash
# Check if swap is active
free -h

# Turn off all swap immediately
sudo swapoff -a

```

### 2. The "Honest Math" Short-Circuit (Recommended & Safest)

For Panel A (Qiskit/Qulacs) and Panel B (UCC), memory requirements are mathematically absolute. A dense statevector of double-precision complex numbers requires exactly $16$ bytes per element.

* $N=30 \implies 2^{30} \times 16 \text{ bytes} \approx 17.1 \text{ GB}$
* $N=32 \implies 2^{32} \times 16 \text{ bytes} \approx 68.7 \text{ GB}$
* $N=35 \implies 2^{35} \times 16 \text{ bytes} \approx 549.7 \text{ GB}$

To guarantee machine stability and save AWS compute credits, the runner should mathematically preempt allocations that are physically impossible on the host hardware.

```python
def check_theoretical_oom(tool, N, k, max_ram_gb=30):
    """Short-circuits simulations that mathematically exceed physical RAM."""
    # Leave a 2GB buffer for OS/Python overhead on a 32GB instance
    max_array_elements = (max_ram_gb * 1024**3) // 16

    if tool in ["qiskit", "qulacs"]:
        if (2 ** N) > max_array_elements:
            return True

    if tool == "ucc":
        if (2 ** k) > max_array_elements:
            return True

    return False

# In the runner loop:
if check_theoretical_oom(target_tool, N, k):
    print(f"[{target_tool}] SKIPPED: Theoretical OOM limit reached (N={N}, k={k}).")
    log_result(target_tool, status="OOM", N=N, k=k, peak_mem_mb=None, time_s=None)
    continue

```

*(Note: This is a widely accepted methodology in computer science papers. You simply state in your methodology section: "Runs mathematically requiring >30 GB of RAM were theoretically bounded and recorded as OOM to prevent hardware instability.")*

### 3. OS-Level Sandboxing via `cgroups` (For Forced Execution)

To enforce boundaries on execution time (for `tsim`'s stabilizer wall) or to catch empirical OOMs, wrap the actual subprocess execution in a temporary Linux `cgroup` using `systemd-run`.

*Note: Do not use Python's built-in `resource.setrlimit(RLIMIT_AS)`. This limits Virtual Memory, which will instantly crash JAX/`tsim` upon initialization before the benchmark even begins.*

```python
import subprocess
import time

def run_sandboxed_simulation(target, script_args, max_ram_gb=30, timeout_s=120):
    start_time = time.time()

    # Wraps the execution in a memory-capped cgroup
    command = [
        "systemd-run",
        "--user",             # Run as current user (requires active SSH session)
        "--scope",            # Run in foreground, don't detach
        f"-p", f"MemoryMax={max_ram_gb}G", # Hard physical RAM limit
        "python", "execute_single_run.py"
    ] + script_args

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s
        )
        exec_time = time.time() - start_time

        # systemd-run returns 137 or -9 if the cgroup OOM killer assassinates the process.
        # It returns MemoryError or bad_alloc if Python/C++ caught the denied allocation itself.
        if result.returncode in [-9, 137] or 'MemoryError' in result.stderr or 'bad_alloc' in result.stderr:
            return {"status": "OOM", "time_s": None}

        return {"status": "SUCCESS", "time_s": exec_time}

    except subprocess.TimeoutExpired:
        # Crucial for catching tsim hitting the Stabilizer Wall in Panel C
        return {"status": "TIMEOUT", "time_s": timeout_s}

```

*(Note: If you run this script via an automated AWS SSM agent rather than standard SSH, `--user` might throw a "Failed to connect to bus" error. If that happens, run your runner script with `sudo` and change the command to: `["systemd-run", "--scope", f"-p", f"MemoryMax={max_ram_gb}G", "sudo", "-u", "ubuntu", ...]`).*

### 4. Managing `tsim` (JAX GPU VRAM)

When a GPU runs out of VRAM, it does not crash the EC2 instance; the CUDA driver simply denies the allocation and throws a Python exception (`jax.errors.OutOfMemoryError`).

However, JAX defaults to greedily pre-allocating 90% of GPU memory on boot, which can break isolated subprocess benchmarking. Inject this environment variable at the top of the `tsim` worker script to force JAX to allocate memory dynamically:

```python
import os
# Prevent JAX from hogging VRAM across successive benchmark runs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import tsim

```

### 5. Visualizing the Walls

When plotting the final data in `plot_walls.py`, runs logged as `"OOM"` or `"TIMEOUT"` must not be omitted or plotted as zeros.

They should be plotted at the absolute ceiling of the graph's Y-axis (e.g., $Y = 120\text{ seconds}$ or $Y = 32\text{ GB}$) using a highly visible marker (e.g., a large red "X" or Star). Drop a vertical dashed line from the marker down to the X-axis to visually communicate the exact mathematical threshold where the architecture collapsed.

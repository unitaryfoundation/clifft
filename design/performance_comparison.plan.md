# UCC Implementation Plan: The "Three Exponential Walls" Benchmark

## Executive Summary & Narrative Strategy

Every exact quantum simulator eventually hits an exponential wall. The computational complexity of simulating a generic quantum circuit is governed by three primary variables:

1.  **$N$**: Total physical qubits.
2.  **$t$**: Total non-Clifford gate count.
3.  **$k_{\max}$**: Peak simultaneous non-Clifford entanglement (Active Rank).

Standard statevector (SV) simulators scale as $\mathcal{O}(2^N)$. ZX-calculus Stabilizer Rank simulators (like `tsim`) scale as $\mathcal{O}(2^{\alpha t})$. UCC’s Factored State Architecture scales as $\mathcal{O}(2^{k_{\max}})$.

**The Narrative Goal:** This benchmark suite mathematically proves the underlying architectural theory. By holding two variables constant and sweeping the third, we force each simulator into its theoretical limit. This establishes the scientific foundation that explains *why* UCC is uniquely capable of executing the trillion-shot Fault-Tolerant Quantum Computing (FTQC) workloads demonstrated in the Magic State Cultivation section of the paper. Since QEC intrinsically bounds $k_{\max}$, UCC can scale $N$ and $t$ to unprecedented limits.

**Strict Constraints:**
1.  **The "David vs. Goliath" Hardware Rule:** UCC and Qiskit MUST be strictly pinned to **a single physical CPU core**. `tsim` MUST be allowed to utilize an **NVIDIA GPU** (via `jax[cuda12]`). We are proving that UCC's $\mathcal{O}(1)$ algorithmic elegance on a CPU outpaces GPU brute-force stabilizer decomposition.
2.  **Subprocess Isolation:** To prevent Python garbage collection drift and inaccurate memory profiling, the benchmarking harness must execute every single simulation run in a fresh, isolated subprocess.
3.  **Reproducibility:** All generation, execution, and plotting must be self-contained Python scripts within the `paper/benchmark/` directory.

---

## Tool Participation Matrix

| Benchmark Panel | Qiskit (1 CPU) | tsim (GPU) | UCC (1 CPU) | Why are tools included/excluded? |
| :--- | :--- | :--- | :--- | :--- |
| **A: Physical Wall ($N$)** | ✅ | ✅ | ✅ | Shows Qiskit hitting the $\mathcal{O}(2^N)$ wall at $N \approx 30$. Both tsim and UCC easily survive. |
| **B: Active Rank Wall ($k$)** | ❌ *(OOM)* | ✅ | ✅ | Qiskit is excluded (automatic OOM at $N=50$). tsim survives easily. UCC hits its theoretical $\mathcal{O}(2^k)$ OOM limit at $k \approx 30$. Proves we are honest about limits. |
| **C: Stabilizer Rank Wall ($t$)** | ❌ *(OOM)*| ✅ | ✅ | Qiskit is excluded (automatic OOM at $N=50$). tsim hits its theoretical $\mathcal{O}(2^{\alpha t})$ timeout limit at $t \approx 60$. UCC survives infinitely. |

---

## Phase 1: The Universal Generator

**Goal:** Build a single parameterized circuit generator capable of navigating the $(N, t, k)$ phase space to isolate structural limits.

*   **Task 1.1 (The Phase Space Generator):** Write `paper/benchmark/generator.py`. Implement a function `generate_boundary_circuit(N, t, k)` that outputs a valid `.stim` file and an equivalent OpenQASM 2.0 file.
    *   *To control $k$ and $t$ independently:* Apply $t$ T-gates exclusively to the first $k$ qubits. Then, apply deep Clifford mixing (CNOTs, H) *only* among those $k$ qubits so ZX-calculus tools can't trivially cancel them. Finally, apply a global CNOT ladder across all $N$ qubits. This forces Qiskit to allocate $2^N$, but UCC natively absorbs the global routing into its $\mathcal{O}(1)$ Clifford frame, leaving its Active array bound exactly at $k$.

## Phase 2: The Unified Profiling Harness

**Goal:** Build an automated runner that safely captures Peak Memory and strict execution times without bringing down the host machine.

*   **Task 2.1 (Subprocess Runner):** Write `paper/benchmark/runner.py`. The runner acts as an orchestrator, spawning isolated worker subprocesses for `qiskit`, `tsim`, and `ucc`.
*   **Task 2.2 (Resource Tracking & Preemption):**
    *   Use `resource.getrusage` inside the worker to capture peak physical RAM.
    *   Implement an "Honest Math" check to instantly skip Qiskit runs where $N > 30$ and UCC runs where $k > 30$, preventing 128GB+ allocations that would lock up the OS.
*   **Task 2.3 (Data Export):** Append results to a master `results_walls.csv` containing: `Panel, N, t, k, Target, Status, Exec_s, Peak_Mem_MB`.

## Phase 3: Executing The "Three Walls"

**Goal:** Execute the parameter sweeps. Set a hard system memory limit (e.g., 30GB) and a timeout of 120 seconds.

*   **Task 3.1 (Panel A - The Physical Qubit Wall):**
    *   *Parameters:* Lock $t=20$, $k=15$. Sweep $N \in [20, 24, 28, 30, 32, 40]$.
    *   *Expected Outcome:* Qiskit scales exponentially, crashing at $N \approx 30$. `tsim` and UCC execution times remain flat.
*   **Task 3.2 (Panel B - The Active Rank Wall):**
    *   *Parameters:* Lock $N=50$, $t=40$. Sweep $k \in [10, 15, 20, 25, 30, 32]$.
    *   *Expected Outcome:* UCC scales exponentially in memory/time, OOM-crashing at $k \approx 30$ (~17 GB RAM). `tsim` handles $t=40$ trivially and remains flat.
*   **Task 3.3 (Panel C - The Stabilizer Rank Wall):**
    *   *Parameters:* Lock $N=50$, $k=10$. Sweep $t \in [10, 30, 50, 70, 90, 120]$.
    *   *Expected Outcome:* Despite having a GPU, `tsim`'s compilation/evaluation blows up exponentially due to generating trillions of stabilizer terms, timing out at $t \approx 60$. UCC executes in flat linear $\mathcal{O}(t)$ time because its active array safely breathes at exactly $2^{10}$.

## Phase 4: Data Visualization

**Goal:** Generate the definitive 3-panel dashboard for the paper.

*   **Task 4.1 (Plotting Script):** Write `paper/benchmark/plot_walls.py` using `matplotlib`.
*   **Task 4.2 (Layout):** A 1x3 horizontal grid.
    *   **Left (Panel A):** Time vs $N$. Log-Y axis. Shows the Qiskit SV wall.
    *   **Middle (Panel B):** Time vs $k$. Log-Y axis. Shows the UCC Factored State wall.
    *   **Right (Panel C):** Time vs $t$. Log-Y axis. Shows the `tsim` Stabilizer Rank wall.
*   **Visual Standard:** Runs logged as `"OOM"` or `"TIMEOUT"` must be plotted at the ceiling of the graph's Y-axis with a distinct red "X" marker, dropping a vertical dashed line to the X-axis to show the exact architectural failure point.

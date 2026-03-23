# UCC -- Importance Sampling via Forced k-Faults

## 1. Motivation

Estimating logical error rates of quantum error-correcting codes at
physical error rates near or below threshold requires sampling
exponentially rare events. With standard Monte Carlo sampling, the
number of shots needed to resolve a logical error rate `p_L` scales as
`O(1 / p_L)`, making direct simulation at `p_L ~ 1e-10` or below
completely impractical.

**Stratified importance sampling** solves this by conditioning on the
number of physical faults `k` that occur in a single shot. For a
distance-`d` code, at least `ceil(d/2)` faults are required to cause a
logical error. By running dedicated shot batches at each fault count
`k = ceil(d/2), ceil(d/2)+1, ...` and weighting the results by the
exact probability `P(K = k)`, we can estimate `p_L` with dramatically
fewer total shots.

## 2. Mathematical Foundation

### 2.1 The Poisson-Binomial Distribution

A circuit with `N` fault locations (quantum noise sites + readout noise
entries), each with distinct probability `p_i`, produces a total fault
count `K` that follows the **Poisson-Binomial distribution**:

    P(K = k) = sum over all size-k subsets S of: prod_{i in S} p_i * prod_{j not in S} (1 - p_j)

This is computed exactly via dynamic programming in `O(N * k_max)` time.

### 2.2 Conditional Poisson Sampling

Given that exactly `k` faults occur, the probability of selecting a
specific subset `S` of size `k` is:

    P(S | K = k) = prod_{i in S} p_i * prod_{j not in S} (1 - p_j) / P(K = k)

This is proportional to `prod_{i in S} w_i` where `w_i = p_i / (1 - p_i)`
is the odds ratio for site `i`. We sample from this distribution exactly
using a DP table sweep (Section 4.2).

### 2.3 Stratified Estimator

The overall logical error rate is:

    p_L = sum_{k=0}^{k_max} P(K = k) * p_L(k)

where `p_L(k)` is the conditional logical error rate given `k` faults,
estimated by running `n_k` shots with exactly `k` forced faults:

    p_L(k) = (logical errors in stratum k) / (passed shots in stratum k)

The variance of the stratified estimator is:

    Var(p_L) = sum_k P(K = k)^2 * Var(p_L(k))
             = sum_k P(K = k)^2 * p_L(k) * (1 - p_L(k)) / n_k

For strata where `P(K = k)` is tiny, even crude estimates of `p_L(k)`
contribute negligible variance to the total.

## 3. Unified Fault Site Model

Rather than treating quantum noise and readout noise as separate
subsystems, we unify them into a single contiguous fault site array of
length `N = N_q + N_r`:

```
    Index:    0          1         ...    N_q-1     N_q     N_q+1    ...   N-1
    Type:  [quantum]  [quantum]         [quantum] [readout] [readout]     [readout]
```

- **Quantum sites** (`i < N_q`): Each is a `NoiseSite` with one or more
  `NoiseChannel` entries. The total probability for site `i` is
  `sum(ch.prob for ch in site.channels)`. When forced to fire, the VM
  still rolls the PRNG to select which specific Pauli channel activates
  (proportional to channel probabilities within the site).

- **Readout sites** (`i >= N_q`): Each is a `ReadoutNoiseEntry` with a
  single probability `prob`. When forced to fire, the corresponding
  measurement record bit is flipped unconditionally.

The DP algorithm selects `k` indices from this unified `[0, N)` array.
After selection, indices are partitioned into sorted quantum and readout
vectors for the VM's two-pointer consumption during execution.

## 4. C++ Implementation Plan

### 4.1 New State: `ForcedFaults` Struct

**File:** `src/ucc/svm/svm.h`

Add a grouped struct inside `SchrodingerState`:

```cpp
struct ForcedFaults {
    bool active = false;
    std::vector<uint32_t> noise_indices;    // Sorted quantum site indices to force
    std::vector<uint32_t> readout_indices;  // Sorted readout entry indices to force
    uint32_t noise_pos = 0;                 // Two-pointer cursor for noise
    uint32_t readout_pos = 0;               // Two-pointer cursor for readout
} forced_faults;
```

Add a helper method to advance the noise cursor:

```cpp
void advance_forced_noise() {
    auto& ff = forced_faults;
    if (ff.noise_pos < ff.noise_indices.size()) {
        next_noise_idx = ff.noise_indices[ff.noise_pos++];
    } else {
        next_noise_idx = static_cast<uint32_t>(-1);  // Sentinel: no more faults
    }
}
```

In `SchrodingerState::reset()`, add:

```cpp
forced_faults.noise_pos = 0;
forced_faults.readout_pos = 0;
```

### 4.2 Precomputation: DP Table and Odds Ratios

**File:** `src/ucc/svm/svm.cc`

Before the shots loop in both `sample()` and `sample_survivors()`, when
`force_k_faults` is provided:

1. **Build the odds-ratio vector** `w[i] = p_i / (1 - p_i)` for all `N`
   unified sites. Clamp `p_i` to `[0, 1 - 1e-15]` to prevent division
   by zero.

2. **Detect uniform mode:** If all `p_i` are identical within `1e-9`
   tolerance, set a flag and skip the DP table entirely.

3. **Build a flat DP table** of size `(N + 1) * (k + 1)` using a single
   `std::vector<double>`. Entry `W[i][j]` (accessed as
   `dp[i * (k + 1) + j]`) stores the sum of products of odds ratios
   over all size-`j` subsets drawn from suffix `[i, N)`:

```cpp
// Base case: empty subset has weight 1
for (uint32_t i = 0; i <= n_L; ++i)
    dp[i * stride + 0] = 1.0;

// Fill bottom-up
for (int i = static_cast<int>(n_L) - 1; i >= 0; --i) {
    uint32_t max_j = std::min(n_L - static_cast<uint32_t>(i), target_k);
    for (uint32_t j = 1; j <= max_j; ++j) {
        dp[i * stride + j] = dp[(i + 1) * stride + j]
                            + w[i] * dp[(i + 1) * stride + (j - 1)];
    }
}
```

4. **Uniform mode precomputation:** Allocate a persistent
   `std::vector<uint32_t> uniform_pool(n_L)` initialized with
   `std::iota`. This pool is allocated once and reused across all shots
   via partial Fisher-Yates shuffles (Section 4.3).

### 4.3 Per-Shot Fault Sampling

Inside the shots loop, after `state.reset()` (or before the first shot):

#### Non-Uniform Path (DP Sweep)

Sweep the DP table from `i = 0` to `n_L - 1`, deciding inclusion of
each site:

```cpp
uint32_t needed = target_k;
for (uint32_t i = 0; i < n_L && needed > 0; ++i) {
    double prob_include;
    if (needed == n_L - i) {
        prob_include = 1.0;  // Must include all remaining
    } else {
        prob_include = (w[i] * dp[(i + 1) * stride + (needed - 1)])
                     / dp[i * stride + needed];
    }
    if (state.random_double() < prob_include) {
        // Partition into quantum vs readout
        if (i < N_q) ff.noise_indices.push_back(i);
        else ff.readout_indices.push_back(i - N_q);
        needed--;
    }
}
```

This produces a perfectly sorted subset (indices are visited in order).

#### Uniform Path (Partial Fisher-Yates)

Perform `k` Fisher-Yates swaps at the front of `uniform_pool`:

```cpp
for (uint32_t j = 0; j < target_k; ++j) {
    uint32_t remaining = n_L - j;
    uint32_t pick = j + static_cast<uint32_t>(state.random_double() * remaining);
    std::swap(uniform_pool[j], uniform_pool[pick]);
}
```

Copy the first `k` elements to a temporary buffer, sort them, then
partition into `noise_indices` / `readout_indices`.

**Key property:** The pool does not need to be reset between shots.
Partial shuffling of an already-permuted array still produces uniformly
random subsets.

### 4.4 Kernel Interception

**File:** `src/ucc/svm/svm_kernels.inl`

#### `exec_noise()`

Replace the gap-advance tail with:

```cpp
if (state.forced_faults.active) {
    state.advance_forced_noise();
} else {
    state.next_noise_idx++;
    state.draw_next_noise(pool.noise_hazards);
}
```

The body of `exec_noise` (channel selection via `random_double() * prob_sum`)
is unchanged. The forced-fault mechanism only controls *whether* a site
fires; the specific Pauli channel is still selected by the existing PRNG
roll within the site's channel distribution.

#### `exec_readout_noise()`

Replace the probability check with a two-pointer match:

```cpp
bool fire = false;
if (state.forced_faults.active) {
    auto& ff = state.forced_faults;
    if (ff.readout_pos < ff.readout_indices.size() &&
        ff.readout_indices[ff.readout_pos] == entry_idx) {
        fire = true;
        ff.readout_pos++;
    }
} else {
    fire = (state.random_double() < entry.prob);
}
if (fire) {
    assert(entry.meas_idx < state.meas_record.size());
    state.meas_record[entry.meas_idx] ^= 1;
}
```

The two-pointer comparison is safe because:
- The backend emits `OP_READOUT_NOISE` with strictly increasing `entry_idx`
- `readout_indices` is sorted at construction time

#### `exec_noise_block()`

No changes required. The existing `while` loop checks
`state.next_noise_idx` against the block range, which is correctly
updated by `advance_forced_noise()` through `exec_noise`.

### 4.5 Impact on Post-Selection and Normalization

Zero impact. The forced-fault mechanism only intercepts PRNG decisions
("does this site fire?"). All downstream VM behavior is identical:

- **Normalization (`FLAG_EXPECTED_ONE`):** Expected parities are baked
  into opcodes during compilation from the noiseless Clifford frame.
  Forced faults inject errors into the dynamic Pauli frame `p_x/p_z`.
  `OP_DETECTOR` and `OP_OBSERVABLE` faithfully XOR the noisy measurement
  record against baked-in parities.

- **Post-selection (`OP_POSTSELECT`):** If forced faults trigger a
  detection event that fails a postselect check, `state.discarded = true`
  and the VM aborts early. The shot is correctly rejected, and
  `sample_survivors` draws a new fault set for the next attempt.

### 4.6 API Changes

Forced k-fault sampling is exposed as **separate functions** rather than
additional arguments on the existing `sample` / `sample_survivors`. This
keeps the API semantically clear: results from `sample_k` /
`sample_k_survivors` are *not* directly interpretable as regular samples
and must be weighted by the Poisson-Binomial PMF `P(K = k)`. Separate
function names make this obvious and prevent accidental misuse.

#### C++ Headers (`src/ucc/svm/svm.h`)

Add two new functions (existing `sample` / `sample_survivors` are unchanged):

```cpp
/// Sample with exactly k forced faults per shot.
/// Sites are sampled from the exact conditional Poisson-Binomial
/// distribution. When all site probabilities are uniform, an O(k)
/// Fisher-Yates sampler is used automatically.
SampleResult sample_k(const CompiledModule& program, uint32_t shots,
                      uint32_t k,
                      std::optional<uint64_t> seed = std::nullopt);

/// Sample survivors with exactly k forced faults per shot.
SurvivorResult sample_k_survivors(const CompiledModule& program, uint32_t shots,
                                  uint32_t k,
                                  std::optional<uint64_t> seed = std::nullopt,
                                  bool keep_records = false);
```

Internally, `sample_k` and `sample_k_survivors` share the same
`execute()` dispatch path as their non-forced counterparts. The only
difference is the DP precomputation before the shots loop and the
per-shot fault injection within it. The core VM code is not duplicated.

#### Python Bindings (`src/python/bindings.cc`)

Bind `sample_k` and `sample_k_survivors` as new module-level functions.
Docstrings must clearly state that results require weighting by `P(K=k)`:

```python
ucc.sample_k(program, shots, k, seed=None)
# -> tuple of (measurements, detectors, observables) numpy arrays

ucc.sample_k_survivors(program, shots, k, seed=None, keep_records=False)
# -> dict with total_shots, passed_shots, discards, logical_errors, ...
```

Add a read-only property on `Program`:

```python
prog.noise_site_probabilities  # 1D numpy array: [p_0, ..., p_{N_q-1}, p_{N_q}, ..., p_{N-1}]
```

This returns the total probability for each quantum noise site (sum of
channel probs) followed by each readout noise entry's probability.

## 5. Python Usage Pattern

```python
import numpy as np
import ucc

prog = ucc.compile(stim_text, normalize_syndromes=True,
                   hir_passes=ucc.default_hir_pass_manager(),
                   bytecode_passes=ucc.default_bytecode_pass_manager())

# Extract per-site probabilities from the compiled program
site_probs = prog.noise_site_probabilities

# Compute exact Poisson-Binomial PMF via DP (in Python)
def poisson_binomial_pmf(probs: np.ndarray, max_k: int) -> np.ndarray:
    dp = np.zeros(max_k + 1)
    dp[0] = 1.0
    for p in probs:
        for k in range(max_k, 0, -1):
            dp[k] = dp[k] * (1 - p) + dp[k - 1] * p
        dp[0] *= 1 - p
    return dp

d = 5  # Code distance
max_k = 15
P_K = poisson_binomial_pmf(site_probs, max_k)

p_fail_total = 0.0
var_total = 0.0

for k in range(d, max_k + 1):
    if P_K[k] < 1e-15:
        continue

    shots_for_k = max(10_000, int(10_000_000 * P_K[k]))

    # Dedicated forced-fault function -- results must be weighted by P_K[k]
    result = ucc.sample_k_survivors(prog, shots=shots_for_k, k=k)

    passed = result["passed_shots"]
    if passed == 0:
        continue

    p_fail_given_k = result["logical_errors"] / passed
    p_fail_total += P_K[k] * p_fail_given_k

    var_given_k = (p_fail_given_k * (1.0 - p_fail_given_k)) / passed
    var_total += P_K[k] ** 2 * var_given_k

print(f"Error Rate: {p_fail_total:.3e} +/- {np.sqrt(var_total):.3e}")
```

## 6. Implementation Task Breakdown

### Task 1: ForcedFaults struct and state wiring
- Add `ForcedFaults` struct to `SchrodingerState` in `svm.h`
- Add `advance_forced_noise()` method
- Update `reset()` to zero the cursors
- **DoD:** Compiles cleanly, existing tests pass

### Task 2: Kernel interception
- Modify `exec_noise()` tail in `svm_kernels.inl`
- Modify `exec_readout_noise()` in `svm_kernels.inl`
- **DoD:** Compiles cleanly, existing tests pass (no behavioral change
  when `forced_faults.active == false`)

### Task 3: DP precomputation and per-shot sampling
- Implement `sample_k()` and `sample_k_survivors()` in `svm.cc`
- Implement DP table construction (flat vector, single allocation)
- Implement non-uniform conditional sampling sweep
- Implement uniform Fisher-Yates fallback (persistent pool, O(k) per shot)
- Wire `forced_faults` initialization into the shots loop
- **DoD:** C++ unit tests for both uniform and non-uniform sampling
  verify correct subset distributions

### Task 4: Python bindings
- Bind `sample_k` and `sample_k_survivors` as new module-level functions
- Add `noise_site_probabilities` read-only property on `Program`
- Update docstrings
- **DoD:** Python integration tests verify forced-fault mode produces
  correct observable error rates against known circuits

### Task 5: Tests
- C++ Catch2 test: verify DP table values against known small examples
- C++ Catch2 test: verify subset sampling distribution (chi-squared or
  exact enumeration for small N, k)
- C++ Catch2 test: uniform fallback produces uniform subsets
- C++ Catch2 test: forced faults with readout noise
- Python test: forced k-faults on a repetition code, verify that
  `p_L(k=0) == 0` and `p_L(k >= d) > 0`
- Python test: round-trip `noise_site_probabilities` against known circuit
- **DoD:** All new and existing tests pass

### Task 6: Documentation
- Update `docs/reference/` with API docs for `sample_k`,
  `sample_k_survivors`, and `noise_site_probabilities`
- Update `docs/guide/simulation.md` with a section on importance sampling
- **DoD:** New functions are documented with signatures, parameters, and
  usage notes

## 7. Complexity Analysis

| Phase | Time | Memory | Per-Shot? |
|---|---|---|---|
| Odds ratio computation | O(N) | O(N) | No |
| DP table construction | O(N * k) | O(N * k) | No |
| Non-uniform sampling | O(N) | O(1) | Yes |
| Uniform sampling (Fisher-Yates) | O(k) | O(N) pool | Yes |
| Kernel interception | O(1) per site | O(1) | Yes |

For typical QEC circuits: N ~ 10,000, k <= 50. The DP table is ~4 MB
(510K doubles) and built once in sub-millisecond time. Per-shot overhead
is negligible compared to VM execution.

## 8. Tutorial: Importance Sampling of Magic State Cultivation Circuits

After the core implementation is complete, a tutorial should be added at
`docs/guide/importance-sampling.md` demonstrating end-to-end importance
sampling on the magic state cultivation circuits in `paper/magic/circuits/`.

This is directly inspired by Tuloup & Ayral (2026), "Computing logical
error thresholds with the Pauli Frame Sparse Representation"
(arXiv:2603.14670), which introduced subset importance sampling to
estimate the logical error rate of Gidney's T-state cultivation protocol
at distance d=5. Their key insight: for a fault-tolerant post-selected
protocol at distance d, `p_fail|k = 0` for all `k < d`, so the logical
error probability is supported only on subsets with `k >= d`. This
concentrates computational effort on a narrow range of k values and
reduces sampling cost by orders of magnitude -- they achieved accurate
estimates with a few billion shots where brute force would require >10^13.

The tutorial should cover:

1. **Loading a cultivation circuit** from `paper/magic/circuits/` (e.g.,
   `circuit_d5_p0.001.stim`) and compiling it with UCC.

2. **Extracting `noise_site_probabilities`** and computing the
   Poisson-Binomial PMF. For the cultivation circuits, all fault sites
   share the same physical error rate `p`, so the PMF reduces to the
   standard Binomial distribution and UCC automatically engages the
   O(k) Fisher-Yates uniform sampler.

3. **Running stratified sampling** using `sample_k_survivors` for
   `k = d, d+1, ..., k_max`, demonstrating that `p_fail|k = 0` for
   `k < d` and computing the weighted sum.

4. **Sweeping physical error rates** without re-simulating: since
   `p_fail|k` is independent of `p`, the same simulation results can be
   reweighted with different Binomial PMFs `P(K=k; p, N)` to produce
   full error-rate curves over a range of `p` values from a single run.

5. **Comparing T-state vs S-state injection** error rates, reproducing
   the key finding from Tuloup & Ayral that the multiplicative factor
   between T and S injection error rates grows from ~2 at d=3 to ~7 at
   d=5.

### Task 7: Importance sampling tutorial
- Add `docs/guide/importance-sampling.md`
- Include runnable Python code using UCC's `sample_k_survivors`
- Demonstrate the full workflow on `paper/magic/circuits/` cultivation
  circuits
- Show error-rate sweep reweighting trick
- **DoD:** Tutorial runs end-to-end and produces error rate estimates
  consistent with published results

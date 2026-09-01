# Importance Sampling

Importance sampling estimates rare outcomes by deliberately sampling fault
counts that ordinary Monte Carlo would visit infrequently. Use it when the
circuit contains independent Pauli or readout noise sites and the quantity of
interest is dominated by rare faults.

!!! warning "Conditional samples require weights"
    `sample_k()` and `sample_k_survivors()` condition every shot on exactly
    `k` faults. A result from one stratum is not an unconditional error-rate
    estimate. Combine strata with their fault-count probabilities
    $P(K = k)$.

## Choose a function

- Use `clifft.sample_k()` when the program has no post-selection mask and each
  conditional shot should return a row.
- Use `clifft.sample_k_survivors()` when the program has a post-selection mask.
  It reports aggregate survivor and logical-error counts and can optionally
  retain survivor rows.

Both functions accept the same CPU execution controls as ordinary fixed-plan
sampling. See [CPU Execution and Tuning](cpu-execution.md) after choosing the
right result contract.

## Build the strata

Compile the noisy circuit, inspect `program.noise_site_probabilities`, and
compute the probability mass $P(K=k)$ for every sampled stratum. For uniform
site probability $p$ across $N$ sites, $K$ is binomial:

$$
P(K = k) = \binom{N}{k}p^k(1-p)^{N-k}.
$$

For unequal site probabilities, use the corresponding Poisson-binomial mass.
Clifft uses the exact conditional distribution internally when choosing which
sites fault.

This minimal uniform-noise example estimates the probability of measuring
`1`. The circuit has one noise site, so both strata are included explicitly:

```python
import clifft

p = 0.01
program = clifft.compile(f"X_ERROR({p}) 0\nM 0")

strata = []
for k, weight in [(0, 1 - p), (1, p)]:
    result = clifft.sample_k(program, shots=1_000, k=k, seed=42 + k)
    conditional_one_rate = result.measurements[:, 0].mean()
    strata.append(weight * conditional_one_rate)

estimated_one_rate = sum(strata)
assert abs(estimated_one_rate - p) < 1e-12
```

In a post-selected workflow, weight the error numerator and survival
denominator separately:

$$
p_{\text{fail}} =
\frac{\sum_k P(K=k)\,\hat p_{\text{fail}|k}}
     {\sum_k P(K=k)\,\hat p_{\text{surv}|k}}.
$$

Here, each conditional rate uses the total attempted shots in that stratum.
Include enough strata that the omitted tail probability is negligible for the
target precision.

## Result and error checks

The forced-fault functions raise `ValueError` when the requested stratum has
zero probability mass, such as when `k` exceeds the number of sites with
nonzero fault probability.

`sample_k()` returns the usual fixed-row `SampleResult`.
`sample_k_survivors()` adds `total_shots`, `passed_shots`, `discards`,
`logical_errors`, and `observable_ones`; pass `keep_records=True` only when the
individual survivor rows are needed.

For a full worked analysis, including post-selection, confidence intervals,
physical-error-rate reweighting, and cultivation circuits, continue to
[Magic State Cultivation with Importance Sampling](importance-sampling-tutorial.md).

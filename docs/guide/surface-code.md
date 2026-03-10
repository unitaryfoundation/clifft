<!--pytest-codeblocks:skipfile-->

# Tutorial: Surface Code Error Suppression

This tutorial demonstrates using UCC as a sampling backend for quantum error correction experiments. We reproduce a surface code error suppression plot similar to Stim's [getting started notebook](https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb), using UCC for circuit compilation and sampling, and [PyMatching](https://github.com/oscarhiggott/PyMatching) for minimum-weight perfect matching (MWPM) decoding.

![Surface Code Error Suppression](images/surface_code_plot.png)

## Pipeline

The experiment follows four steps:

1. **Generate** a rotated surface code memory-Z circuit with `stim.Circuit.generated()`
2. **Compile & sample** detector and observable data with `ucc.compile()` and `ucc.sample()`
3. **Decode** detector syndromes with PyMatching's MWPM decoder
4. **Compute** the logical error rate as the fraction of mispredicted observables

## Prerequisites

```bash
pip install ucc stim pymatching matplotlib numpy
```

## Full Code

```python
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim

import ucc


def generate_surface_code_circuit(
    distance: int,
    rounds: int,
    phys_error_rate: float,
) -> stim.Circuit:
    """Generate a rotated surface code memory-Z experiment."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=phys_error_rate,
        after_reset_flip_probability=phys_error_rate,
        before_measure_flip_probability=phys_error_rate,
        before_round_data_depolarization=phys_error_rate,
    )


def estimate_logical_error_rate(
    circuit: stim.Circuit,
    shots: int,
) -> float:
    """Compile with UCC, sample, decode with PyMatching."""
    # UCC compile & sample
    prog = ucc.compile(str(circuit))
    _meas, det, obs = ucc.sample(prog, shots)

    # PyMatching decode
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predictions = matching.decode_batch(det)

    # Logical error = prediction XOR actual observable
    num_errors = int(np.sum(predictions[:, 0] ^ obs[:, 0]))
    return num_errors / shots


# Experiment parameters
distances = [3, 5]
phys_error_rates = [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03]
shots = 20_000

# Collect data
results = {}
for d in distances:
    results[d] = []
    for p in phys_error_rates:
        circuit = generate_surface_code_circuit(d, rounds=d, phys_error_rate=p)
        logical_err = estimate_logical_error_rate(circuit, shots)
        results[d].append(logical_err)
        print(f"d={d}  p={p:.4f}  logical_err={logical_err:.5f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
for d in distances:
    ax.plot(phys_error_rates, results[d], "o-", label=f"d = {d}", markersize=7)

ax.plot(phys_error_rates, phys_error_rates, "k--", alpha=0.3, label="logical = physical")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Physical Error Rate")
ax.set_ylabel("Logical Error Rate")
ax.set_title("Surface Code Error Suppression")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("surface_code.png", dpi=150)
plt.show()
```

## How It Works

### Circuit Generation

Stim generates the full noisy surface code circuit including:

- Data qubit initialization and measurement
- Stabilizer measurement rounds with ancilla qubits
- Depolarizing noise after Clifford gates, reset, and measurement
- `DETECTOR` annotations marking parity checks
- `OBSERVABLE_INCLUDE` annotations for logical observable tracking

For distance $d$, the rotated surface code uses $d^2 + (d-1)^2$ total qubits. Distance 3 uses 26 qubits; distance 5 uses 64.

### UCC Compilation & Sampling

`ucc.compile()` accepts the full Stim circuit text — including `REPEAT` blocks, noise channels, `DETECTOR` and `OBSERVABLE_INCLUDE` annotations. The compiled program is then sampled with `ucc.sample()`, which returns three arrays:

- **measurements**: raw measurement outcomes
- **detectors**: detector values (syndrome bits)
- **observables**: logical observable outcomes

### Decoding

PyMatching constructs a matching graph from Stim's detector error model and performs MWPM decoding on the detector syndromes. The logical error rate is the fraction of shots where the decoder's prediction disagrees with the actual observable value.

### Results

The plot shows the classic threshold behavior of the surface code:

- **Below threshold** (~$p = 0.005$): increasing distance $d$ exponentially suppresses logical errors. At $p = 0.001$, distance 5 achieves ~$10\times$ lower logical error rate than distance 3.
- **Above threshold**: larger codes perform *worse* because they have more opportunities for errors to accumulate.
- The crossover point is the **threshold** of the surface code under this noise model.

!!! note "Arbitrary qubit scaling"
    By default, UCC supports up to 64 qubits. To simulate larger circuits (e.g., distance 7 requires 118 qubits), change `UCC_MAX_QUBITS` in `pyproject.toml` and rebuild. See [Custom Qubit Width](../../DEVELOPMENT.md#custom-qubit-width) for details.

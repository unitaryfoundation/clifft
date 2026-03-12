"""Surface code error suppression tutorial script.

Runs both UCC and Stim backends, compares logical error rates,
and reports compile/sample timing.
"""

import time

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


def run_ucc(circuit: stim.Circuit, shots: int) -> tuple[float, float, float]:
    """Compile and sample with UCC. Returns (logical_err, compile_s, sample_s)."""
    stim_text = str(circuit)
    hir_pm = ucc.default_hir_pass_manager()
    bpm = ucc.default_bytecode_pass_manager()

    t0 = time.perf_counter()
    prog = ucc.compile(stim_text, hir_passes=hir_pm, bytecode_passes=bpm)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _meas, det, obs = ucc.sample(prog, shots)
    sample_s = time.perf_counter() - t0

    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predictions = matching.decode_batch(det)
    num_errors = int(np.sum(predictions[:, 0] ^ obs[:, 0]))
    return num_errors / shots, compile_s, sample_s


def run_stim(circuit: stim.Circuit, shots: int) -> tuple[float, float, float]:
    """Compile and sample with Stim. Returns (logical_err, compile_s, sample_s)."""
    t0 = time.perf_counter()
    sampler = circuit.compile_detector_sampler()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    det, obs = sampler.sample(shots, separate_observables=True)
    sample_s = time.perf_counter() - t0

    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predictions = matching.decode_batch(det)
    num_errors = int(np.sum(predictions[:, 0] ^ obs[:, 0]))
    return num_errors / shots, compile_s, sample_s


def main() -> None:
    distances = [3, 5]
    phys_error_rates = [
        0.001,
        0.002,
        0.003,
        0.005,
        0.007,
        0.01,
        0.015,
        0.02,
        0.03,
    ]
    shots = 20_000

    ucc_results: dict[int, list[float]] = {}
    stim_results: dict[int, list[float]] = {}
    timing_rows: list[tuple[int, float, float, float, float, float]] = []

    for d in distances:
        ucc_results[d] = []
        stim_results[d] = []
        for p in phys_error_rates:
            circuit = generate_surface_code_circuit(d, rounds=d, phys_error_rate=p)

            ucc_err, ucc_comp, ucc_samp = run_ucc(circuit, shots)
            stim_err, stim_comp, stim_samp = run_stim(circuit, shots)

            ucc_results[d].append(ucc_err)
            stim_results[d].append(stim_err)
            timing_rows.append((d, p, ucc_comp, ucc_samp, stim_comp, stim_samp))

            print(
                f"d={d}  p={p:.4f}  "
                f"UCC={ucc_err:.5f} ({ucc_comp*1e3:.1f}ms + {ucc_samp:.3f}s)  "
                f"Stim={stim_err:.5f} ({stim_comp*1e3:.1f}ms + {stim_samp:.3f}s)"
            )

    # Print timing summary
    print("\n--- Timing Summary ---")
    hdr = (
        f"{'d':>3} {'p':>7} "
        f"{'UCC compile':>13} {'UCC sample':>12} "
        f"{'Stim compile':>13} {'Stim sample':>12}"
    )
    print(hdr)
    for d, p, uc, us, sc, ss in timing_rows:
        print(f"{d:>3} {p:>7.4f} {uc*1e3:>11.1f}ms {us:>10.3f}s {sc*1e3:>11.1f}ms {ss:>10.3f}s")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e"]
    for i, d in enumerate(distances):
        ax.plot(
            phys_error_rates,
            ucc_results[d],
            "o-",
            color=colors[i],
            label=f"UCC d={d}",
            markersize=7,
        )
        ax.plot(
            phys_error_rates,
            stim_results[d],
            "x--",
            color=colors[i],
            label=f"Stim d={d}",
            markersize=7,
        )

    ax.plot(
        phys_error_rates,
        phys_error_rates,
        "k--",
        alpha=0.3,
        label="logical = physical",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate")
    ax.set_title("Surface Code Error Suppression: UCC vs Stim")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/guide/images/surface_code_plot.png", dpi=150)
    print("\nPlot saved to docs/guide/images/surface_code_plot.png")


if __name__ == "__main__":
    main()

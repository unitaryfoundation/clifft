"""Regenerate the surface-code circuits used by the C++ benchmarks."""

from pathlib import Path

import stim

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CASES = (
    ("surface_d7_r7_p001.stim", 7, 7, 1e-3),
    ("surface_d5_r5_p05.stim", 5, 5, 0.05),
    ("surface_d11_r11_p001.stim", 11, 11, 1e-3),
)


def generate(distance: int, rounds: int, probability: float) -> str:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=probability,
        before_measure_flip_probability=probability,
        after_clifford_depolarization=probability,
        after_reset_flip_probability=probability,
    )
    return f"{circuit}\n"


def main() -> None:
    for name, distance, rounds, probability in CASES:
        (FIXTURES / name).write_text(generate(distance, rounds, probability), encoding="ascii")


if __name__ == "__main__":
    main()

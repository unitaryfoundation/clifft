"""Independent checks of the study's explicit measurement expansion."""

import pytest


@pytest.mark.parametrize(
    "text,expected",
    [
        ("R 0\nM(1) 0\nM 0", "01"),
        ("RX 0\nMX(1) 0\nMX 0", "01"),
        ("RX 0 1\nMPP(1) X0*X1\nMPP X0*X1", "01"),
        ("R 0 1\nX_ERROR(1) 0\nMPP Z0*Z1", "1"),
    ],
)
def test_readout_flips_preserve_the_true_postmeasurement_state(text, expected):
    aer = pytest.importorskip("qiskit_aer")
    from validate_cultivation_aer import to_aer

    circuit = to_aer(text, 0.001)
    result = (
        aer.AerSimulator(method="matrix_product_state", max_parallel_threads=1)
        .run(circuit, shots=16, memory=True, seed_simulator=82)
        .result()
    )
    assert result.success
    assert set(result.get_memory()) == {expected}

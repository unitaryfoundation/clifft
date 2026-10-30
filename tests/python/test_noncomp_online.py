"""Independent references and resource contracts for online execution."""

import numpy as np
import pytest
import stim
from utils_conformance import assert_joint_distribution
from utils_qiskit import qiskit_statevector, stim_to_qiskit_noiseless

from clifft import noncomp


@pytest.mark.parametrize("seed", range(12))
def test_online_rotations_and_compaction_match_aer(seed):
    rng = np.random.default_rng(seed + 713)
    n = 1 + seed % 4
    lines = [f"H {q}" for q in range(n)] + [f"T {q}" for q in range(n)]
    for _ in range(30):
        if n > 1 and rng.random() < 0.4:
            a, b = rng.choice(n, 2, replace=False)
            name = rng.choice(["CX", "CY", "CZ", "R_XX", "R_YY", "R_ZZ"])
            args = f"({rng.uniform(-2, 2)})" if name.startswith("R_") else ""
            lines.append(f"{name}{args} {a} {b}")
        else:
            q = rng.integers(n)
            name = rng.choice(["H", "S", "S_DAG", "T", "T_DAG", "R_X", "R_Y", "R_Z"])
            args = f"({rng.uniform(-2, 2)})" if name.startswith("R_") else ""
            lines.append(f"{name}{args} {q}")
    text = "\n".join(lines)
    expected = abs(qiskit_statevector(stim_to_qiskit_noiseless(text))) ** 2
    # Sequential measurements exercise removal of active coordinates in different orders.
    order = rng.permutation(n)
    result = noncomp.sample(
        text + "\nM " + " ".join(map(str, order)),
        noncomp.Model(),
        shots=20000,
        seed=seed + 101,
        execution="online",
        collect_stats=True,
    )
    assert result.stats is not None
    assert result.stats.peak_active_width == n
    assert result.stats.online_state_growths == n
    assert_joint_distribution(result.measurements[:, np.argsort(order)], expected)


CLIFFORDS = [
    name
    for name, data in stim.gate_data().items()
    if data.is_unitary
    and (data.is_single_qubit_gate or data.is_two_qubit_gate)
    and name not in ("I", "II")
]


@pytest.mark.parametrize("gate", CLIFFORDS)
def test_online_named_cliffords_match_stim(gate):
    n = len(stim.gate_data(gate).tableau)
    for basis in ("X", "Y", "Z"):
        text = f"H 0\nS 0\n{gate} " + " ".join(map(str, range(n)))
        if n == 2:
            text = "H 1\n" + text + "\nCX 1 0\nS 1"
        if basis in ("X", "Y"):
            if basis == "Y":
                text += "\nS_DAG " + " ".join(map(str, range(n)))
            text += "\nH " + " ".join(map(str, range(n)))
        reference = stim.TableauSimulator()
        reference.do(stim.Circuit(text))
        expected = abs(reference.state_vector(endian="little").astype(np.complex128)) ** 2
        expected /= expected.sum()
        result = noncomp.sample(
            text + "\nM " + " ".join(map(str, range(n))),
            noncomp.Model(),
            shots=4096,
            seed=53,
            execution="online",
            max_active_width=0,
        )
        assert_joint_distribution(result.measurements, expected)


@pytest.mark.parametrize("n", [63, 64, 65, 129])
def test_online_dormant_measurements_and_reuse_cross_word_boundaries(n):
    text = (
        f"H {n - 1}\nCX {n - 1} 0\nMY {n - 1}\n"
        f"S 0\nCZ rec[-1] 0\nMX 0\nR {n - 1}\n"
        f"R_X(0.5) {n - 1}\nR_PAULI(-0.5) X{n - 1}\nM {n - 1}"
    )
    result = noncomp.sample(
        text,
        noncomp.Model(),
        shots=256,
        seed=12,
        execution="online",
        max_active_width=0,
        collect_stats=True,
    )
    reference = (
        stim.Circuit(
            text.replace("R_X(0.5)", "SQRT_X").replace(
                f"R_PAULI(-0.5) X{n - 1}", f"SQRT_X_DAG {n - 1}"
            )
        )
        .compile_sampler(seed=32)
        .sample(256)
    )
    # The random first record is fair; both remaining records are deterministic.
    np.testing.assert_array_equal(result.measurements[:, 1:], reference[:, 1:])
    assert 70 < result.measurements[:, 0].sum() < 186
    assert result.stats is not None
    assert result.stats.peak_active_width == result.stats.online_state_growths == 0


@pytest.mark.parametrize("gate", ["MPP X0", "U3(0.1,0.2,0.3) 0"])
def test_online_rejects_unsupported_operations_before_sampling(gate):
    with pytest.raises(ValueError, match="online execution does not support"):
        noncomp.compile(gate, noncomp.Model(), execution="online")


@pytest.mark.parametrize("batch_size", [65, "tune"])
def test_online_rejects_packing_and_tuning_even_for_zero_shots(batch_size):
    program = noncomp.compile("M 0", noncomp.Model(), execution="online")
    assert program.strategy == "online"
    assert "no continuation compilation" in program.explain()
    with pytest.raises(ValueError, match="selected strategy is online"):
        program.sample(0, batch_size=batch_size)


def test_online_three_qubit_depolarization_has_the_complete_joint_law():
    result = noncomp.sample(
        "DEPOLARIZE3(0.63) 0 1 2\nM 0 1 2",
        noncomp.Model(),
        shots=20000,
        seed=91,
        execution="online",
    )
    assert_joint_distribution(result.measurements, [0.44] + [0.08] * 7)


@pytest.mark.parametrize(
    "gate,angle",
    [("SPP", 0.5), ("SPP_DAG", -0.5), ("TPP", 0.25), ("TPP_DAG", -0.25)]
    + [(f"R_PAULI({a})", a) for a in (-1.5, -1, 0, 0.5, 1, 2)],
)
def test_online_product_phases_match_aer_on_an_active_state(gate, angle):
    prefix = "H 0 1 2\nT 0 1\nS 2\n"
    suffix = "\nH 0 1 2"
    product = " X0*Y1*Z2"
    oracle = prefix + f"R_PAULI({angle})" + product + suffix
    expected = abs(qiskit_statevector(stim_to_qiskit_noiseless(oracle))) ** 2
    result = noncomp.sample(
        prefix + gate + product + suffix + "\nM 0 1 2",
        noncomp.Model(),
        shots=12000,
        seed=29,
        execution="online",
    )
    assert_joint_distribution(result.measurements, expected)

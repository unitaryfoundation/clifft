"""Native prefix handoff and fresh-noise sampling against full physical replay."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import stim


@pytest.fixture
def compiler():
    pytest.importorskip("cirq")
    import compile_terminal_gadget

    return compile_terminal_gadget


def small_source():
    rows = [0b0001111, 0b0110011, 0b1010101]
    checks = [
        stim.PauliString("".join(axis if row >> q & 1 else "I" for q in range(7)))
        for axis in "XZ"
        for row in rows
    ]
    encoder = stim.Tableau.from_stabilizers(checks + [stim.PauliString("Z" * 7)]).to_circuit()
    out = "H 6\nT 6\nH 6\nT 6\n" + str(encoder) + "\n"
    out += "DEPOLARIZE1(0.2) 0 1 2 3 4 5 6\n"
    out += "T 0 2 4 6\nT_DAG 1 3 5\nDEPOLARIZE1(0.1) 0 1 2 3 4 5 6\n"
    out += "CX 0 1 0 2 0 3 0 4 0 5 0 6\nDEPOLARIZE2(0.15) 0 1 2 3\n"
    out += "MX(0.2) 0\nY_ERROR(0.2) 0\nRX 0\nDEPOLARIZE1(0.1) 0 2 5\n"
    out += "CX 0 6 0 5 0 4 0 3 0 2 0 1\nDEPOLARIZE2(0.1) 0 6 2 5\n"
    out += "T_DAG 0 2 4 6\nT 1 3 5\nDEPOLARIZE1(0.1) 0 1 2 3 4 5 6\n"
    out += (
        "MPP(0.2) "
        + " ".join(
            "*".join(f"{'IXYZ'[axis]}{q}" for q, axis in enumerate(p) if axis)
            for p in [stim.PauliString("Y" * 7)] + checks
        )
        + "\n"
    )
    out += "DETECTOR rec[-6]\nOBSERVABLE_INCLUDE(0) rec[-7]\n"
    return out


def test_native_complete_attempts_with_fresh_faults(compiler, tmp_path):
    from benchmark_native_terminal import validate_shot

    native = Path("build-study/sample_terminal_gadget").resolve()
    reference = Path("build-study/replay_cultivation").resolve()
    if not native.is_file() or not reference.is_file():
        pytest.skip("build native terminal and replay tools")
    source = small_source()
    metadata = compiler.compile_bundle(source, tmp_path)
    command = [str(native), str(tmp_path), "4", "24", "812"]
    first = [json.loads(line) for line in subprocess.check_output(command, text=True).splitlines()]
    second = [json.loads(line) for line in subprocess.check_output(command, text=True).splitlines()]
    assert first[:-1] == second[:-1]
    assert first[-1]["max_normalization_error"] < 1e-12
    assert any(shot["prefix_faults"] for shot in first[:-1])
    assert any(shot["suffix_faults"] for shot in first[:-1])
    for shot in first[:-1]:
        assert (
            validate_shot(source, metadata, shot, reference, tmp_path)["absolute_log_error"] < 1e-12
        )


def test_native_compiler_declines_non_code_boundary(compiler, tmp_path):
    native = Path("build-study/sample_terminal_gadget").resolve()
    if not native.is_file():
        pytest.skip("build native terminal tool")
    source = small_source().replace("T 0 2 4 6", "H 0\nT 0 2 4 6", 1)
    compiler.compile_bundle(source, tmp_path)
    result = subprocess.run(
        [str(native), str(tmp_path), "1", "0", "0"], text=True, capture_output=True
    )
    assert result.returncode == 2
    assert "cannot certify the signed CSS code space" in result.stderr


def test_compiler_declines_nonterminal_suffix(compiler, tmp_path):
    with pytest.raises(ValueError, match="unsupported terminal suffix"):
        compiler.compile_bundle(small_source() + "H 0\n", tmp_path)


@pytest.mark.parametrize("geometric_family", [False, True])
def test_native_distribution_against_dense_physical_projectors(
    compiler, tmp_path, geometric_family
):
    pytest.importorskip("qiskit")
    from clifford_gadget import atoms_from_text, bind_faults
    from soft_cultivation_study import zero_noise
    from test_clifford_gadget import matrix, pauli_matrix

    native = Path("build-study/sample_terminal_gadget").resolve()
    if not native.is_file():
        pytest.skip("build native terminal tool")
    if geometric_family:
        from gadget_family import make_circuit

        source = make_circuit(3, 0)
    else:
        source = zero_noise(small_source())
    compiler.compile_bundle(source, tmp_path)
    clean, _ = bind_faults(source, 0, 0)
    _, _, _, atoms = atoms_from_text(clean)
    mx = next(i for i, a in enumerate(atoms) if a.name == "MX")
    reset = next(i for i, a in enumerate(atoms) if a.name == "RX")
    final = next(i for i, a in enumerate(atoms) if a.name == "MPP")
    before = matrix(atoms[:mx], 7)[:, 0]
    after = matrix(atoms[reset + 1 : final], 7)
    x0, z0 = stim.PauliString(7), stim.PauliString(7)
    x0[0], z0[0] = "X", "Z"
    px, pz = pauli_matrix(x0), pauli_matrix(z0)
    checks = [pauli_matrix(a.pauli) for a in atoms[final:] if a.name == "MPP"]
    expected = np.zeros(256)
    for m in (0, 1):
        state = (before + (-1) ** m * (px @ before)) / 2
        if m:
            state = pz @ state
        state = after @ state
        for outcome in range(128):
            projected = state.copy()
            for i, p in enumerate(checks):
                projected = (projected + (-1) ** ((outcome >> i) & 1) * (p @ projected)) / 2
            expected[m | (outcome << 1)] = np.vdot(projected, projected).real
    np.testing.assert_allclose(expected.sum(), 1, atol=1e-13)
    shots = 2048
    output = subprocess.check_output(
        [str(native), str(tmp_path), "1", str(shots), "631"], text=True
    )
    counts = np.zeros(256, dtype=int)
    for line in output.splitlines():
        row = json.loads(line)
        if row["kind"] == "sample":
            outcome = sum(int(bit) << i for i, bit in enumerate(row["records"][:8]))
            counts[outcome] += 1
    assert counts.sum() == shots
    assert not np.any(counts[expected < 1e-14])
    # A conservative simultaneous binomial envelope avoids depending on the
    # native sampler's seed-to-record mapping while detecting biased draws.
    deviation = abs(counts - shots * expected)
    tolerance = 7 * np.sqrt(shots * expected * (1 - expected)) + 4
    assert np.all(deviation <= tolerance)

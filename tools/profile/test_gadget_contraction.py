"""Independent dense and brute-force checks of fixed contraction schedules."""

import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import stim


@pytest.fixture
def compiled():
    pytest.importorskip("cirq")
    import compiled_gadget_contraction

    return compiled_gadget_contraction


def test_parity_contraction_matches_brute_force(compiled):
    rng = np.random.default_rng(713)
    for rank in range(1, 9):
        masks = list(map(int, rng.integers(0, 1 << rank, size=12))) + [1 << i for i in range(rank)]
        plan = compiled.ParityContraction(masks, rank)
        for _ in range(4):
            local = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(len(masks), 2)))
            expected = sum(
                np.prod([local[q, (mask & x).bit_count() % 2] for q, mask in enumerate(masks)])
                for x in range(1 << rank)
            ) / (1 << rank)
            np.testing.assert_allclose(plan.evaluate(local), expected, atol=1e-13)


def test_contraction_declines_oversized_or_invalid_masks(compiled):
    with pytest.raises(ValueError, match="storage limit"):
        compiled.ParityContraction([(1 << 24) - 1], 24)
    with pytest.raises(ValueError, match="invalid variable"):
        compiled.ParityContraction([8], 3)


@pytest.mark.parametrize("faulty", [False, True])
def test_terminal_probabilities_and_marginals_against_dense_physical_gates(compiled, faulty):
    pytest.importorskip("qiskit")
    from clifford_gadget import Atom, Gadget
    from test_clifford_gadget import matrix, pauli_matrix

    width = 7
    rows = [0b0001111, 0b0110011, 0b1010101]
    forward = [Atom("T" if q % 2 else "T_DAG", (q,)) for q in range(width)]
    backward = [Atom("T_DAG" if q % 2 else "T", (q,)) for q in range(width)]
    fan = [Atom("CX", (0, q)) for q in range(1, width)]
    lead = fan + ([Atom("Z", (2,))] if faulty else [])
    middle = [Atom("Y", (0,))] if faulty else []
    tail = ([Atom("X", (3,))] if faulty else []) + list(reversed(fan))
    gadget = Gadget(
        width,
        forward,
        lead + [Atom("MX", (0,), record=0)] + middle + [Atom("RX", (0,), record=8)] + tail,
        backward,
    )
    checks = [stim.PauliString("Y" * width)]
    for axis in "XZ":
        for row in rows:
            checks.append(
                stim.PauliString("".join(axis if row >> q & 1 else "I" for q in range(width)))
            )
    suffix = [gadget] + [Atom("MPP", record=i + 1, pauli=p) for i, p in enumerate(checks)]
    code = compiled.TerminalCode(gadget, suffix)
    marginal = compiled.TerminalMarginals(code)
    logical = np.array([np.sqrt(0.3), np.sqrt(0.7) * np.exp(0.37j)])
    boundary = compiled.Boundary(
        offset=0b1000011 if faulty else 0,
        x_signs=0b101 if faulty else 0,
        logical=logical,
        ancillas={},
    )
    choices = code.sampling_inputs(boundary, suffix, 9)
    initial = np.zeros(1 << width, dtype=complex)
    for logical_bit, x in itertools.product(range(2), range(8)):
        bits = boundary.offset ^ (127 if logical_bit else 0)
        for i, row in enumerate(rows):
            if x >> i & 1:
                bits ^= row
        initial[bits] = (
            logical[logical_bit] * (-1) ** ((x & boundary.x_signs).bit_count() % 2) / np.sqrt(8)
        )
    # Dense physical projectors surround the actual measurement and reset;
    # this reference does not call the reduced two-branch operator formula.
    x0, z0 = stim.PauliString(width), stim.PauliString(width)
    x0[0], z0[0] = "X", "Z"
    px, pz = pauli_matrix(x0), pauli_matrix(z0)
    identity = np.eye(1 << width)
    u, a, b, c, v = [matrix(atoms, width) for atoms in [forward, lead, middle, tail, backward]]
    projectors = [pauli_matrix(p) for p in checks]
    probabilities = {}
    for m in (0, 1):
        records, _ = choices[m]
        reset = records[8]
        state = a @ u @ initial
        state = (state + (-1) ** m * (px @ state)) / 2
        state = b @ state
        state = (state + (-1) ** reset * (px @ state)) / 2
        state = (pz if reset else identity) @ state
        state = v @ c @ state
        for outcome in range(128):
            forced = records.copy()
            projected = state.copy()
            for i, p in enumerate(projectors):
                bit = (outcome >> i) & 1
                forced[i + 1] = bit
                projected = (projected + (-1) ** bit * (p @ projected)) / 2
            expected = float(np.vdot(projected, projected).real)
            actual = code.probability(code.bind(boundary, suffix, forced))
            np.testing.assert_allclose(actual, expected, atol=2e-14)
            probabilities[m, outcome] = expected
    np.testing.assert_allclose(sum(probabilities.values()), 1, atol=2e-14)
    for m, y, measured in itertools.product(range(2), range(2), range(4)):
        for prefix in range(1 << measured):
            expected = sum(
                p
                for (branch, outcome), p in probabilities.items()
                if branch == m
                and outcome & 1 == y
                and (outcome >> 1) & ((1 << measured) - 1) == prefix
            )
            actual = marginal.probability(choices[m][1], measured, prefix, y)
            np.testing.assert_allclose(actual, expected, atol=3e-14)
    for seed in range(16):
        observed = marginal.sample(choices, seed)
        bits = observed["records"]
        outcome = sum(bits[i + 1] << i for i in range(7))
        np.testing.assert_allclose(
            np.exp(observed["log_probability"]), probabilities[bits[0], outcome], atol=2e-14
        )


def test_native_fixed_kernel_matches_complex_contraction(compiled, tmp_path):
    native = Path("build-study/profile_gadget_contraction").resolve()
    if not native.is_file():
        pytest.skip("build profile_gadget_contraction")
    plan = compiled.ParityContraction([1, 2, 3, 6, 4], 3)
    rng = np.random.default_rng(514)
    terms = [
        (0.3 + 0.2j, np.exp(1j * rng.uniform(-np.pi, np.pi, (5, 2)))),
        (-0.4 + 0.1j, np.exp(1j * rng.uniform(-np.pi, np.pi, (5, 2)))),
    ]
    path = tmp_path / "plan.txt"
    for marginal in (False, True):
        compiled.write_native_plan(path, plan, [terms, []], marginal=marginal)
        result = json.loads(subprocess.check_output([str(native), str(path), "3"], text=True))
        total = sum(a * plan.evaluate(local) for a, local in terms)
        expected = total.real if marginal else abs(total) ** 2
        np.testing.assert_allclose(result["probabilities"], [expected, 0], atol=1e-14)

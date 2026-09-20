"""Check source geometry, CSS algebra, and bounded native benchmark behavior."""

import itertools
import json
import subprocess
from pathlib import Path

import pytest
import stim
from cultivation_study import CORPUS
from gadget_family import geometry, make_circuit


@pytest.mark.parametrize("distance", [3, 5, 7, 9])
def test_geometric_family_css_code(distance):
    coordinates, rows, signs, _ = geometry(distance)
    n = len(coordinates)
    assert n == (3 * distance * distance + 1) // 4
    assert len(rows) == (n - 1) // 2
    assert all((a & b).bit_count() % 2 == 0 for a, b in itertools.product(rows, repeat=2))
    assert all(sum(signs[q] for q in range(n) if row >> q & 1) % 4 == 0 for row in rows)
    checks = [
        stim.PauliString("".join(axis if row >> q & 1 else "I" for q in range(n)))
        for axis in "XZ"
        for row in rows
    ]
    stim.Tableau.from_stabilizers(checks + [stim.PauliString("Z" * n)])
    if distance <= 7:
        # Exhaust the logical coset for these sizes instead of inferring
        # distance merely from the geometry parameter or noiseless outputs.
        word, minimum = (1 << n) - 1, n
        for i in range(1 << len(rows)):
            if i:
                word ^= rows[(i & -i).bit_length() - 1]
            minimum = min(minimum, word.bit_count())
        assert minimum == distance


@pytest.mark.parametrize("distance,ancillas", [(3, 6), (5, 19)])
def test_geometry_matches_pinned_author_checks(distance, ancillas):
    source = stim.Circuit(
        (CORPUS / f"d{distance}a{ancillas}_inject+cultivate_p1e-3.stim").read_text()
    )
    coordinates = source.get_final_qubit_coordinates()
    expected = set()
    for op in source:
        if op.name == "MPP":
            for group in op.target_groups():
                if len(group) in {4, 6} and all(t.is_x_target for t in group):
                    expected.add(frozenset(tuple(coordinates[t.value]) for t in group))
    points, rows, _, _ = geometry(distance)
    actual = {frozenset(points[q] for q in range(len(points)) if row >> q & 1) for row in rows}
    assert actual == expected


def test_native_baseline_limit_preserves_sampling(tmp_path):
    pytest.importorskip("cirq")
    from compile_terminal_gadget import compile_bundle

    native = Path("build-study/sample_terminal_gadget").resolve()
    if not native.is_file():
        pytest.skip("build native terminal tool")
    compile_bundle(make_circuit(3, 0.1), tmp_path)
    results = []
    for limit in (0, 24):
        output = subprocess.check_output(
            [str(native), str(tmp_path), "4", "3", "17", str(limit)], text=True
        )
        results.append([json.loads(line) for line in output.splitlines()])
    assert results[0][:-1] == results[1][:-1]
    assert results[0][-1]["ordinary_seconds"] == []
    assert results[0][-1]["baseline_enabled"] is False
    assert results[1][-1]["baseline_enabled"] is True

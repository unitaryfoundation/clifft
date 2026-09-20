"""Check compact native lookup tables against expanded NumPy contractions."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is needed for the contraction probe")
    assert compiler is not None
    directory = tmp_path_factory.mktemp("contraction-probe")
    source = directory / "probe.cpp"
    source.write_text(
        r"""
#include "gadget_contraction_kernel.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    using namespace gadget_study;
    try {
        std::ifstream input(argv[1]);
        Reader reader{input};
        const auto rank = std::stoull(argv[2]), leaves = std::stoull(argv[3]);
        Contraction plan(reader, rank, leaves, 4);
        ContractionWorkspace workspace;
        workspace.prepare(plan.scratch_size(), plan.product_size());
        const auto cases = reader.size();
        std::vector<Complex> local(4 * leaves);
        std::cout << plan.lookup_bytes() << '\n' << std::setprecision(17);
        for (size_t i = 0; i < cases; ++i) {
            for (auto& value : local) {
                const auto re = reader.number(), im = reader.number();
                value = {re, im};
            }
            const auto value = plan.evaluate(local.data(), workspace);
            std::cout << value.real() << ' ' << value.imag() << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
"""
    )
    binary = directory / "probe"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I",
            str(root / "src"),
            "-I",
            str(root / "tools/profile"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


@pytest.mark.parametrize("rank", [3, 9, 12])
@pytest.mark.parametrize("irregular", [False, True])
def test_native_workspace_matches_expanded_contractions(probe, tmp_path, rank, irregular):
    pytest.importorskip("cirq")
    from compiled_gadget_contraction import write_native_plan
    from folded_check_contraction import FactorPlan

    rng = np.random.default_rng(8176)
    masks = [((1 << rank) - 1, (1 << rank) // 3)] + [(1 << i,) for i in range(rank)]
    plan = FactorPlan(masks, rank)
    if irregular:
        # Arbitrary valid gathers must retain an expanded fallback.
        rng.shuffle(plan.steps[0][2][0])
    path = tmp_path / "plan.txt"
    write_native_plan(path, plan, [], marginal=True)
    cases = [np.exp(1j * rng.uniform(-np.pi, np.pi, size=(len(masks), 4))) for _ in range(5)]
    with path.open("a") as f:
        f.write(str(len(cases)) + "\n")
        for local in cases:
            for value in local.flat:
                f.write(f"{value.real} {value.imag}\n")
    command = [str(probe), str(path), str(rank), str(len(masks))]
    result = subprocess.check_output(command, text=True).splitlines()
    actual = [complex(*map(float, line.split())) for line in result[1:]]
    expected = [plan.evaluate(local) for local in cases]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    if rank == 12 and not irregular:
        assert int(result[0]) < 4 * plan.gather_entries

    # A wide invalid label must be rejected before narrowing to a byte.
    tokens = path.read_text().split()
    tokens[6] = "256"
    path.write_text("\n".join(tokens) + "\n")
    bad = subprocess.run(command, text=True, capture_output=True)
    assert bad.returncode == 2
    assert "invalid marginal parity" in bad.stderr

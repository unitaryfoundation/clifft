"""Numerical Schmidt-rank diagnostics, never used to truncate execution states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from factored_state import analyze


def singular_values(state: np.ndarray, width: int, left: list[int]) -> np.ndarray:
    # Coefficient bit q is tensor axis width - 1 - q in NumPy's C order.
    order = [width - 1 - q for q in left]
    order += [q for q in range(width) if q not in order]
    matrix = state.reshape([2] * width).transpose(order).reshape(1 << len(left), -1)
    return np.linalg.svd(matrix, compute_uv=False)


def inspect_snapshots(directory: Path, plan: dict) -> dict:
    metadata = json.loads((directory / "snapshots.json").read_text())
    factors = {step["action"]: step["coordinates"] for step in analyze(plan)["trace"]}
    thresholds = [1e-8, 1e-10, 1e-12, 1e-14]
    dtype = "<f8" if metadata["byte_order"] == "little" else ">f8"
    results = []
    for item in metadata["snapshots"]:
        width = item["width"]
        raw = np.fromfile(directory / item["file"], dtype=dtype)
        if raw.size != 2 << width:
            raise ValueError("snapshot size disagrees with metadata")
        state = raw[: 1 << width] + 1j * raw[1 << width :]
        norm = float(np.vdot(state, state).real)
        if not np.isfinite(norm) or abs(norm - 1) > 1e-9:
            raise ValueError(f"snapshot is not normalized: {norm}")
        cuts = [singular_values(state, width, list(range(cut))) for cut in range(1, width)]
        ranks = {
            str(tol): [int(np.count_nonzero(s > tol * s[0])) for s in cuts] for tol in thresholds
        }
        factor_error = 0.0
        for component in factors[item["action"]]:
            if len(component) < width:
                values = singular_values(state, width, component)
                factor_error = max(factor_error, float(np.linalg.norm(values[1:])))
        if factor_error > 1e-10:
            raise ValueError(f"predicted product factor is entangled: {factor_error}")
        results.append(
            {
                **item,
                "norm_squared": norm,
                "ranks_by_relative_threshold": ranks,
                "max_factor_tail_norm": factor_error,
            }
        )
    return {
        "sampling": metadata["sampling"],
        "ordering": "current active coordinates, ascending coefficient bit",
        "snapshots": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_snapshots(args.directory, json.loads(args.plan.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Run one fresh-process timing sample and emit one JSON object."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from cases import CASE_BY_ID, Case, load_circuit, sha256_text

import clifft
from clifft import experimental, noncomp


def peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage * 1024)


def array_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    contiguous = values if values.flags.c_contiguous else np.ascontiguousarray(values)
    digest_source = b"" if contiguous.nbytes == 0 else memoryview(contiguous).cast("B")
    summary: dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "bytes": int(values.nbytes),
        "nonzero": int(np.count_nonzero(values)),
        "blake2b_64": hashlib.blake2b(digest_source, digest_size=8).hexdigest(),
    }
    if np.issubdtype(values.dtype, np.floating):
        finite = np.isfinite(values)
        summary["finite"] = int(np.count_nonzero(finite))
        summary["sum"] = float(np.sum(values[finite], dtype=np.float64))
    else:
        summary["sum"] = int(np.sum(values, dtype=np.uint64))
    return summary


def summarize_sample_result(result: Any, requested_shots: int) -> dict[str, Any]:
    arrays = {
        "measurements": array_summary(result.measurements),
        "detectors": array_summary(result.detectors),
        "observables": array_summary(result.observables),
        "exp_vals": array_summary(result.exp_vals),
    }
    if result.total_shots is None:
        attempted = requested_shots
        accepted = requested_shots
        logical_errors = int(
            np.count_nonzero(np.any(result.observables != 0, axis=1))
            if result.observables.shape[1]
            else 0
        )
        observable_ones = np.sum(result.observables, axis=0, dtype=np.uint64).astype(int).tolist()
    else:
        attempted = int(result.total_shots)
        accepted = int(result.passed_shots)
        logical_errors = int(result.logical_errors)
        observable_ones = result.observable_ones.astype(int).tolist()
    return {
        "attempted_shots": attempted,
        "accepted_shots": accepted,
        "discarded_shots": attempted - accepted,
        "discard_fraction": (attempted - accepted) / attempted if attempted else 0.0,
        "logical_errors": logical_errors,
        "observable_ones": observable_ones,
        "arrays": arrays,
    }


def summarize_noncomp_result(result: noncomp.NonComputationalSample) -> dict[str, Any]:
    arrays = {
        "measurements": array_summary(result.measurements),
        "detectors": array_summary(result.detectors),
        "observables": array_summary(result.observables),
        "final_status": array_summary(result.final_status),
        "heralds": array_summary(result.heralds),
    }
    logical_errors = int(
        np.count_nonzero(np.any(result.observables != 0, axis=1))
        if result.observables.shape[1]
        else 0
    )
    return {
        "attempted_shots": result.shots,
        "accepted_shots": result.shots,
        "discarded_shots": 0,
        "discard_fraction": 0.0,
        "logical_errors": logical_errors,
        "observable_ones": (
            np.sum(result.observables, axis=0, dtype=np.uint64).astype(int).tolist()
        ),
        "arrays": arrays,
    }


def hir_metadata(text: str) -> tuple[Any, dict[str, Any]]:
    hir = clifft.trace(clifft.parse(text))
    passes = clifft.default_hir_pass_manager()
    passes.run(hir)
    operation_counts = Counter(str(op.as_dict()["op_type"]) for op in hir)
    return hir, {
        "num_qubits": int(hir.num_qubits),
        "num_measurements": int(hir.num_measurements),
        "num_detectors": int(hir.num_detectors),
        "num_observables": int(hir.num_observables),
        "num_exp_vals": int(hir.num_exp_vals),
        "num_hir_ops": len(hir),
        "hir_operation_counts": dict(sorted(operation_counts.items())),
    }


def compile_program(case: Case, text: str, backend: str, num_detectors: int) -> tuple[Any, float]:
    postselection_mask = [1] * num_detectors if case.postselection == "all_detectors" else []
    compile_kwargs = {
        "postselection_mask": postselection_mask,
        "normalize_syndromes": True,
        "hir_passes": clifft.default_hir_pass_manager(),
    }
    start = time.perf_counter()
    if backend == "legacy":
        program = clifft.compile(
            text,
            **compile_kwargs,
            bytecode_passes=clifft.default_bytecode_pass_manager(),
        )
    else:
        program = experimental.compile(text, **compile_kwargs)
    return program, time.perf_counter() - start


def program_metadata(program: Any, backend: str) -> dict[str, Any]:
    common = {
        "num_qubits": int(program.num_qubits),
        "num_measurements": int(program.num_measurements),
        "num_detectors": int(program.num_detectors),
        "num_observables": int(program.num_observables),
        "num_exp_vals": int(program.num_exp_vals),
        "noise_site_count": len(program.noise_site_probabilities),
    }
    if backend == "legacy":
        opcode_counts = Counter(str(instruction.as_dict()["opcode"]) for instruction in program)
        common.update(
            {
                "peak_rank": int(program.peak_rank),
                "num_instructions": int(program.num_instructions),
                "active_k_history": [int(width) for width in program.active_k_history],
                "opcode_counts": dict(sorted(opcode_counts.items())),
            }
        )
    else:
        common.update({"num_actions": int(program.num_actions)})
    return common


def build_noncomp_model(probability: float) -> noncomp.Model:
    matrix = [[0.0] * 5 for _ in range(5)]
    if probability:
        matrix[noncomp.Level.LEAK_E][noncomp.Level.E] = 0.8 * probability
        matrix[noncomp.Level.LOST][noncomp.Level.E] = 0.2 * probability
    classifier_matrix = [[0.0] * 5 for _ in range(3)]
    classifier_matrix[0][noncomp.Level.G] = 1.0
    classifier_matrix[0][noncomp.Level.LEAK_G] = 1.0
    classifier_matrix[1][noncomp.Level.E] = 1.0
    classifier_matrix[1][noncomp.Level.LEAK_E] = 1.0
    classifier_matrix[2][noncomp.Level.LOST] = 1.0
    return noncomp.Model(
        initial_state=[1.0, 0.0, 0.0, 0.0, 0.0],
        transitions={"S": matrix} if probability else {},
        classifier=noncomp.Classifier(classifier_matrix),
    )


def sample_program(case: Case, program: Any, backend: str, shots: int, seed: int) -> Any:
    api = clifft if backend == "legacy" else experimental
    if case.kind == "importance":
        assert case.forced_k is not None
        return api.sample_k_survivors(program, shots, case.forced_k, seed=seed, keep_records=False)
    if case.output_mode == "aggregate":
        return api.sample_survivors(program, shots, seed=seed, keep_records=False)
    return api.sample(program, shots, seed=seed)


def run_noncomp(case: Case, text: str, backend: str, shots: int, seed: int) -> Any:
    assert case.noncomp_probability is not None
    circuit = clifft.parse(text)
    model = build_noncomp_model(case.noncomp_probability)
    if backend == "legacy":
        return noncomp.sample(circuit, model, shots=shots, seed=seed)
    return experimental.sample_noncomputational(circuit, model, shots=shots, seed=seed)


def validate_nominal_width(case: Case, metadata: dict[str, Any], backend: str) -> None:
    if case.nominal_k is None or backend != "legacy":
        return
    actual = int(metadata["peak_rank"])
    if actual != case.nominal_k:
        raise ValueError(f"{case.case_id}: legacy peak rank {actual} != {case.nominal_k}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASE_BY_ID))
    parser.add_argument("--backend", required=True, choices=("legacy", "symbolic"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--paper-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    clifft.set_num_threads(1)
    case = CASE_BY_ID[args.case]
    text, source = load_circuit(case, args.repo_root, args.paper_dir)
    _, static_metadata = hir_metadata(text)

    rss_imported = peak_rss_bytes()
    compile_seconds: float | None = None
    compile_in_sample = case.kind == "noncomp"
    program_meta: dict[str, Any] = {}

    if case.kind == "noncomp":
        warmup = run_noncomp(case, text, args.backend, min(case.shots, 2), args.seed ^ 1)
    else:
        program, compile_seconds = compile_program(
            case, text, args.backend, static_metadata["num_detectors"]
        )
        program_meta = program_metadata(program, args.backend)
        validate_nominal_width(case, program_meta, args.backend)
        warmup = sample_program(case, program, args.backend, min(case.shots, 2), args.seed ^ 1)
    rss_after_compile_and_warmup = peak_rss_bytes()
    del warmup
    gc.collect()

    start = time.perf_counter()
    if case.kind == "noncomp":
        result = run_noncomp(case, text, args.backend, case.shots, args.seed)
    else:
        result = sample_program(case, program, args.backend, case.shots, args.seed)
    sample_seconds = time.perf_counter() - start
    rss_after_sample = peak_rss_bytes()

    consume_start = time.perf_counter()
    if case.kind == "noncomp":
        outcomes = summarize_noncomp_result(result)
    else:
        outcomes = summarize_sample_result(result, case.shots)
    consume_seconds = time.perf_counter() - consume_start
    rss_after_consume = peak_rss_bytes()

    attempted = int(outcomes["attempted_shots"])
    accepted = int(outcomes["accepted_shots"])
    record = {
        "schema": "clifft_backend_comparison_sample_v1",
        "status": "success",
        "case": case.as_dict(),
        "source": source,
        "circuit_sha256": sha256_text(text),
        "backend": args.backend,
        "seed": args.seed,
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "thread_count": clifft.get_num_threads(),
        "clifft_version": getattr(clifft, "__version__", "unknown"),
        "svm_isa": clifft.svm_backend(),
        "compile_seconds": compile_seconds,
        "compile_in_sample": compile_in_sample,
        "sample_seconds": sample_seconds,
        "consume_seconds": consume_seconds,
        "attempted_throughput": attempted / sample_seconds,
        "accepted_throughput": accepted / sample_seconds,
        "outcomes": outcomes,
        "static_metadata": static_metadata,
        "program_metadata": program_meta,
        "whole_process_peak_rss_bytes": {
            "after_import_and_static_prep": rss_imported,
            "after_compile_and_warmup": rss_after_compile_and_warmup,
            "after_sample": rss_after_sample,
            "after_consume": rss_after_consume,
        },
    }
    if not math.isfinite(record["sample_seconds"]) or record["sample_seconds"] <= 0:
        raise RuntimeError("invalid sample duration")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()

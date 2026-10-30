"""Pinned workload generation and a file-based bridge to the native experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = {
    "initial_state": [1, 0, 0, 0, 0],
    "classifier": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 1]],
    "reset_restores_lost": True,
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def serialize_model(spec, path):
    lines = []

    def values(items):
        lines.append(" ".join(str(float(x)) for x in items))

    values(spec.get("initial_state", [1, 0, 0, 0, 0]))
    transitions = spec.get("transitions", {})
    restore = int(spec.get("reset_restores_lost", False))
    neglect = int(spec.get("damping", "exact") == "neglect")
    lines.append(f"{restore} {neglect} {len(transitions)}")
    for key, matrix in transitions.items():
        lines.append(json.dumps(key))
        for row in matrix:
            values(row)
    classifier = spec.get("classifier", [])
    lines.append(str(len(classifier)))
    for row in classifier:
        values(row)
    for key in ("leaked", "lost"):
        effect = spec.get("gate_partner_effects", {}).get(key)
        lines.append(str(int(effect is not None)))
        if effect is not None:
            values([*effect.get("pauli", [0, 0, 0]), effect.get("spread_probability", 0)])
    rules = spec.get("interactions", [])
    lines.append(str(len(rules)))
    for rule in rules:
        lost = int(rule["source_status"] == "lost")
        lines.append(f"{json.dumps(rule['gate'])} {rule['source_operand']} {lost}")
        effect = rule["effect"]
        values([*effect.get("pauli", [0, 0, 0]), effect.get("spread_probability", 0)])
    Path(path).write_text("\n".join(lines) + "\n")


def python_model(spec):
    from clifft import noncomp

    kwargs = dict(spec)
    if "classifier" in kwargs:
        kwargs["classifier"] = noncomp.Classifier(kwargs["classifier"])
    if "gate_partner_effects" in kwargs:
        kwargs["gate_partner_effects"] = {
            k: noncomp.PartnerEffect(**v) for k, v in kwargs["gate_partner_effects"].items()
        }
    if "interactions" in kwargs:
        kwargs["interactions"] = [
            noncomp.InteractionRule(**{**r, "effect": noncomp.PartnerEffect(**r["effect"])})
            for r in kwargs["interactions"]
        ]
    return noncomp.Model(**kwargs)


def run(
    binary,
    directory,
    text,
    model,
    shots,
    *,
    mode="hybrid",
    seed=42,
    capacity=16,
    outputs=True,
    timeout=180,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "circuit.stim").write_text(text)
    serialize_model(model, directory / "model.txt")
    args = [
        str(binary),
        "--circuit",
        str(directory / "circuit.stim"),
        "--model",
        str(directory / "model.txt"),
        "--shots",
        str(shots),
        "--seed",
        str(seed),
        "--capacity",
        str(capacity),
        "--mode",
        mode,
    ]
    if outputs:
        args += ["--output", str(directory / "sample"), "--final-status", "1"]
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    (directory / "stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{directory}: {result.stderr}")
    info = json.loads(result.stdout)
    info["phases"] = {
        line.split()[1]: {"seconds": float(line.split()[2]), "calls": int(line.split()[3])}
        for line in result.stderr.splitlines()
        if line.startswith("PHASE ")
    }
    (directory / "result.json").write_text(json.dumps(info, indent=2) + "\n")
    arrays = {}
    if outputs:
        for key, size in [
            ("measurements", "measurements"),
            ("heralds", "measurements"),
            ("detectors", "detectors"),
            ("observables", "observables"),
            ("status", "qubits"),
        ]:
            arrays[key] = np.fromfile(directory / f"sample.{key}", dtype=np.uint8).reshape(
                shots, info[size]
            )
    return info, arrays


def workloads(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reproduction = module("qdk_skip_loss", ROOT / "benchmarks/qdk_skip_loss.py")
    neutral = module(
        "neutral_leakage", ROOT / "docs/guide/scripts/neutral_atom_leakage_tutorial.py"
    )
    cases = []
    for distance in (5, 7, 9, 11):
        for rate in (0, 1e-4, 1e-3):
            text, _ = reproduction.make_circuit(distance, "matched-loss", rate)
            cases.append(
                (
                    f"surface_d{distance}_loss{rate:g}",
                    text,
                    DEFAULT_MODEL,
                    "matched QDK loss; Pauli rates zero",
                )
            )
    for distance, path in [
        (3, ROOT / "docs/guide/circuits/circuit_d3_t_gate_p0.001.stim"),
        (5, ROOT / "tests/fixtures/cultivation_d5.stim"),
    ]:
        # Use the same physical Pauli rate at both distances so d5 has a
        # measurable survivor population at practical validation shot counts.
        original = expand_parity_readout(path.read_text().replace("(0.005)", "(0.001)"))
        for rate in (0, 1e-3):
            lines = []
            for line in original.splitlines():
                lines.append(line)
                words = line.split()
                if (
                    rate
                    and words
                    and words[0]
                    in {"H", "S", "S_DAG", "X", "Y", "Z", "CX", "CY", "CZ", "T", "T_DAG"}
                    and not any("rec[" in w for w in words)
                ):
                    lines.append(f"LOSS({rate}) " + " ".join(words[1:]))
            cases.append(
                (
                    f"cultivation_d{distance}_loss{rate:g}",
                    "\n".join(lines) + "\n",
                    DEFAULT_MODEL,
                    "explicit reusable ancilla parity readout; source noise channels at p=0.001; "
                    "added loss after listed physical unitaries; Clifft semantics",
                )
            )
        depol = {**DEFAULT_MODEL, "gate_partner_effects": {"lost": {"pauli": [0.25] * 3}}}
        cases.append(
            (f"cultivation_d{distance}_depol_partner", cases[-1][1], depol, "fixed packed control")
        )
    for name in ("two_row", "two_row_ldu"):
        for kind in ("matched", "exact"):
            transitions = {
                "CZ": neutral.cz_jump_matrix(),
                "RZ_TRANSITION": np.asarray(neutral.RZ_LEVEL_TRANSITIONS),
            }
            if kind == "matched":
                transitions = {
                    k: neutral.equalize_computational_jump_rates(v) for k, v in transitions.items()
                }
            model = {
                "initial_state": list(neutral.INITIAL_LEVELS),
                "classifier": neutral.classifier_matrix().tolist(),
                "transitions": {k: v.tolist() for k, v in transitions.items()},
                "reset_restores_lost": False,
                "damping": "neglect" if kind == "matched" else "exact",
            }
            text = (neutral.CIRCUIT_DIR / neutral.CIRCUIT_FILES[name]).read_text()
            cases.append(
                (
                    f"neutral_{name}_{kind}",
                    text,
                    model,
                    "published schedule and repository five-level model",
                )
            )
    manifest = []
    for name, text, spec, description in cases:
        case = directory / name
        case.mkdir(exist_ok=True)
        (case / "circuit.stim").write_text(text)
        (case / "model.json").write_text(json.dumps(spec, indent=2) + "\n")
        manifest.append(
            {
                "name": name,
                "description": description,
                "circuit_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "model_sha256": hashlib.sha256(
                    json.dumps(spec, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return cases


def expand_parity_readout(text):
    """Give parity readout an explicit physical circuit before adding loss.

    Noiseless ancilla couplings preserve the source instrument when loss is
    absent. Under loss, this defines the benchmark's physical readout model.
    """
    qubits = [
        int(q)
        for line in text.splitlines()
        if not line.lstrip().startswith("#")
        for q in re.findall(r"(?<![\w.\-])(?:[XYZ])?(\d+)(?![\w.])", line.split(" ", 1)[-1])
    ]
    ancilla = max(qubits) + 1
    result = []
    for line in text.splitlines():
        words = line.split()
        if not words or not words[0].startswith("MPP"):
            result.append(line)
            continue
        suffix = words[0][3:]
        for product in words[1:]:
            inverted = False
            result.append(f"RX {ancilla}")
            for factor in product.split("*"):
                if factor.startswith("!"):
                    inverted = not inverted
                    factor = factor[1:]
                result.append(f"C{factor[0]} {ancilla} {factor[1:]}")
            result.append(f"MX{suffix} {'!' if inverted else ''}{ancilla}")
    return "\n".join(result) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Wrote {len(workloads(args.output))} workloads to {args.output}")

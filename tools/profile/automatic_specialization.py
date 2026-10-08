"""Research host for fault and initial Clifford-prefix specialization.

All stochastic selection, state reconstruction, and compilation happen outside
Clifft execution. The optimizer is the existing algebraic HIR pipeline; this
module has no knowledge of encoders, logical gates, or verification tails.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Any

import stim

import clifft

History = tuple[tuple[int, int], ...]
MEASUREMENTS = {"M", "MX", "MY", "MPP", "MPAD"}
ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"}


def instruction_text(node: Any, records: int) -> str:
    gate = str(node.gate.name)
    targets = [f"rec[{t.value - records}]" if t.is_rec else repr(t) for t in node.targets]
    args = [] if gate in MEASUREMENTS else list(node.args)
    suffix = "(" + ",".join(format(x, ".17g") for x in args) + ")" if args else ""
    separator = "*" if gate in {"MPP", "R_PAULI", "EXP_VAL"} else " "
    return gate + suffix + (" " + separator.join(targets) if targets else "")


def record_count(node: Any) -> int:
    return (
        (1 if node.gate.name == "MPP" else len(node.targets))
        if (node.gate.name in MEASUREMENTS)
        else 0
    )


@dataclass(frozen=True)
class FaultSite:
    line: int
    gate: str
    probabilities: tuple[float, ...]
    replacements: tuple[str, ...]


class FaultModel:
    """Independent categorical Pauli locations, with heterogeneous probabilities.

    A two-qubit channel is one categorical event, not independent X/Y/Z draws.
    Readout faults invert the reported record, leaving quantum collapse intact.
    Inter-location correlations and conditional channels are rejected.
    """

    def __init__(self, source: str):
        self.source = source
        parsed = clifft.parse(source)
        self.num_qubits = parsed.num_qubits
        self.num_records = parsed.num_measurements
        self.lines: list[str] = []
        self.sites: list[FaultSite] = []
        records = 0
        for node in parsed.nodes:
            gate = node.gate.name
            if node.tag:
                raise ValueError("Instruction tags are outside this research host")
            if gate == "READOUT_NOISE":
                if not self.lines or node.targets[0].value != records - 1:
                    raise ValueError("Expected readout noise on the most recent measurement")
                original = self.lines[-1]
                measured_gate, measured_targets = original.split(" ", 1)
                if measured_gate not in MEASUREMENTS:
                    raise ValueError("Expected a measurement before readout noise")
                if measured_gate == "MPAD":
                    flipped = str(1 - int(measured_targets))
                else:
                    flipped = (
                        measured_targets[1:]
                        if measured_targets.startswith("!")
                        else "!" + measured_targets
                    )
                self.sites.append(
                    FaultSite(
                        len(self.lines) - 1,
                        gate,
                        (1 - node.arg, node.arg),
                        (original, measured_gate + " " + flipped),
                    )
                )
                continue
            if gate in {
                "X_ERROR",
                "Y_ERROR",
                "Z_ERROR",
                "DEPOLARIZE1",
                "DEPOLARIZE2",
                "PAULI_CHANNEL_1",
                "PAULI_CHANNEL_2",
            }:
                arity = 2 if gate.endswith("2") else 1
                paulis = tuple(
                    "".join(p) for p in product("IXYZ", repeat=arity) if any(x != "I" for x in p)
                )
                if gate.endswith("ERROR"):
                    paulis = (gate[0],)
                probabilities = (
                    tuple(node.args)
                    if gate.startswith("PAULI_CHANNEL")
                    else (node.arg / len(paulis),) * len(paulis)
                )
                if len(probabilities) != len(paulis):
                    raise ValueError("Unexpected Pauli-channel arity")
                probabilities = (max(0.0, 1 - sum(probabilities)), *probabilities)
                for first in range(0, len(node.targets), arity):
                    targets = node.targets[first : first + arity]
                    replacements = ("",) + tuple(
                        "\n".join(f"{p} {q.value}" for p, q in zip(pauli, targets) if p != "I")
                        for pauli in paulis
                    )
                    self.sites.append(FaultSite(len(self.lines), gate, probabilities, replacements))
                    self.lines.append("")
                continue
            if any(word in gate for word in ("ERROR", "CHANNEL", "DEPOLARIZE", "HERALDED")):
                raise ValueError(f"Unsupported stochastic instruction: {gate}")
            self.lines.append(instruction_text(node, records))
            records += record_count(node)
        if records != self.num_records:
            raise ValueError("Unsupported record-producing instruction")

    def render(self, history: History) -> str:
        lines = self.lines.copy()
        seen: set[int] = set()
        for site_id, outcome in history:
            if site_id in seen or not 0 <= site_id < len(self.sites):
                raise ValueError("Invalid or repeated fault location")
            seen.add(site_id)
            site = self.sites[site_id]
            if not 0 < outcome < len(site.replacements):
                raise ValueError("Invalid nonidentity outcome")
            lines[site.line] = site.replacements[outcome]
        return "\n".join(line for line in lines if line) + "\n"

    def draw(self, rng: random.Random) -> History:
        result = []
        for index, site in enumerate(self.sites):
            value = rng.random()
            for outcome, probability in enumerate(site.probabilities):
                value -= probability
                if value < 0:
                    if outcome:
                        result.append((index, outcome))
                    break
            else:
                # Rounding a cumulative sum must not silently choose identity.
                outcome = max(i for i, p in enumerate(site.probabilities) if p)
                if outcome:
                    result.append((index, outcome))
        return tuple(result)


def specialize_prefix(source: str, seed: int) -> tuple[str, dict[str, Any]]:
    """Sample the maximal initial Stim-compatible prefix and retain its records.

    This supports one boundary before the first unsupported (usually T) gate.
    It does not resume arbitrary non-Clifford states at later measurements.
    Reset-induced hidden outcomes are sampled by the stabilizer simulator too.
    """
    parsed = clifft.parse(source)
    prefix = stim.Circuit()
    retained: list[str | int] = []
    suffix = []
    records = 0
    boundary = None
    for node in parsed.nodes:
        text = instruction_text(node, records)
        if boundary is None:
            try:
                data = stim.gate_data(node.gate.name)
            except IndexError:
                boundary = node.gate.name
            else:
                if data.is_noisy_gate and node.gate.name not in MEASUREMENTS:
                    raise ValueError("Prefix specialization requires fixed faults")
                prefix += stim.Circuit(text)
                count = record_count(node)
                if count:
                    retained.extend(range(records, records + count))
                elif node.gate.name in ANNOTATIONS:
                    retained.append(text)
        if boundary is not None:
            suffix.append(text)
        records += record_count(node)
    if boundary is None or prefix.num_measurements == 0:
        return source, {"status": "no_measured_prefix", "boundary": boundary, "records": 0}
    simulator = stim.TableauSimulator(seed=seed)
    simulator.set_num_qubits(parsed.num_qubits)
    simulator.do(prefix)
    measured = simulator.current_measurement_record()
    preparation = str(simulator.current_inverse_tableau().inverse().to_circuit())
    classical = [f"MPAD {int(measured[x])}" if isinstance(x, int) else x for x in retained]
    specialized = "\n".join([preparation, *classical, *suffix]) + "\n"
    if clifft.parse(specialized).num_measurements != parsed.num_measurements:
        raise AssertionError("Prefix specialization changed record numbering")
    return specialized, {
        "status": "specialized",
        "boundary": boundary,
        "records": len(measured),
        "outcomes": list(map(int, measured)),
        "preparation_instructions": len(preparation.splitlines()),
    }


def analyze(source: str) -> tuple[Any, dict[str, Any]]:
    started = perf_counter()
    hir = clifft.trace(clifft.parse(source))
    traced = perf_counter()
    input_t = hir.num_t_gates
    phase = clifft.PhasePolynomialPass()
    manager = clifft.HirPassManager()
    for pass_ in (
        clifft.PeepholeFusionPass(),
        phase,
        clifft.RotationSimplificationPass(),
        clifft.StatevectorSqueezePass(),
    ):
        manager.add(pass_)
    manager.run(hir)
    optimized = perf_counter()
    width = clifft.active_width_trace(hir)
    return hir, {
        "input_t": input_t,
        "output_t": hir.num_t_gates,
        "peak_width": width["peak"],
        "phase": {
            name: getattr(phase, name)
            for name in ("blocks_examined", "blocks_reduced", "blocks_capped", "blocks_expanded")
        },
        "parse_trace_seconds": traced - started,
        "optimizer_seconds": optimized - traced,
        "width_inspection_seconds": perf_counter() - optimized,
    }


def exact_record_probabilities(program: Any, records: int) -> list[float]:
    """Exhaustive small-case reference, including reset-induced hidden records."""
    import clifft._clifft_core as core

    result = [0.0] * (1 << records)
    # The replay API supplies its required count in the validation error.
    try:
        core._replay_record(program, [0] * records)
        hidden = 0
    except ValueError as error:
        message = str(error)
        if "expected " not in message:
            raise
        total = int(message.split("expected ")[1].split(",")[0])
        hidden = total - records
    if records + hidden > 16:
        raise ValueError("Exact record enumeration exceeds the study budget")
    for outcome in range(1 << (records + hidden)):
        bits = [(outcome >> i) & 1 for i in range(records + hidden)]
        replay = core._replay_record(program, bits)
        if replay["reachable"]:
            result[outcome % (1 << records)] += math.exp(replay["log_probability"])
    if abs(sum(result) - 1) > 1e-10:
        raise AssertionError("Exact record distribution is not normalized")
    return result

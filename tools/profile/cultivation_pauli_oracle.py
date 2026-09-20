"""Exact noiseless record-parity expectations by backwards Pauli propagation.

Coefficients are integer pairs (a, b) representing (a + b*sqrt(2))/2**power.
This independent reference is deliberately bounded and is not a sampling path.
Stim parses the Clifford proxy; its S gates stand for physical T gates here.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import stim


def record_parity(proxy, selected, term_limit=262144):
    circuit = stim.Circuit(proxy).flattened()
    terms = {(0, 0): (1, 0)}
    power = 0
    peak = 1
    record = circuit.num_measurements

    def add(out, key, value, sign=1):
        a, b = out.get(key, (0, 0))
        value = (a + sign * value[0], b + sign * value[1])
        if value == (0, 0):
            out.pop(key, None)
        else:
            out[key] = value

    def measurement(x, z, counted, reset=False):
        nonlocal terms
        out: dict[tuple[int, int], tuple[int, int]] = {}
        for (u, v), value in terms.items():
            if ((u & z).bit_count() + (v & x).bit_count()) % 2:
                continue
            if reset:
                # Reset gates here have one target; the surviving I or measured
                # axis is evaluated in the prepared positive eigenstate.
                add(out, (u & ~(x | z), v & ~(x | z)), value)
            elif counted:
                phase = (
                    (u & v).bit_count()
                    + (x & z).bit_count()
                    - ((u ^ x) & (v ^ z)).bit_count()
                    + 2 * (v & x).bit_count()
                ) % 4
                assert phase in (0, 2)
                add(out, (u ^ x, v ^ z), value, 1 if phase == 0 else -1)
            else:
                add(out, (u, v), value)
        terms = out

    for op in reversed(list(circuit)):
        name = op.name
        targets = op.targets_copy()
        args = op.gate_args_copy()
        if name in {"QUBIT_COORDS", "SHIFT_COORDS", "TICK", "DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        if name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
            if any(args):
                raise ValueError("exact reference requires zero noise")
            continue
        if name in {"M", "MX", "R", "RX", "MPP"}:
            if any(args):
                raise ValueError("exact reference requires zero readout noise")
            groups = op.target_groups() if name == "MPP" else [[t] for t in targets]
            for group in reversed(groups):
                x = z = 0
                for target in group:
                    if target.is_inverted_result_target or target.is_measurement_record_target:
                        raise ValueError("unsupported measurement target")
                    if name in {"MX", "RX"} or target.is_x_target or target.is_y_target:
                        x ^= 1 << target.value
                    if name in {"M", "R"} or target.is_z_target or target.is_y_target:
                        z ^= 1 << target.value
                reset = name in {"R", "RX"}
                if not reset:
                    record -= 1
                measurement(x, z, not reset and record in selected, reset)
        elif name == "CX":
            for control, target in reversed(list(zip(targets[::2], targets[1::2], strict=True))):
                if control.is_measurement_record_target:
                    raise ValueError("feedback is unsupported by this reference")
                c, t = 1 << control.value, 1 << target.value
                out: dict[tuple[int, int], tuple[int, int]] = {}
                for (x, z), value in terms.items():
                    sign = -1 if x & c and z & t and bool(x & t) == bool(z & c) else 1
                    add(out, (x ^ (t if x & c else 0), z ^ (c if z & t else 0)), value, sign)
                terms = out
        elif name in {"S", "S_DAG"}:
            for target in reversed(targets):
                mask = 1 << target.value
                out = {}
                for (x, z), (a, b) in terms.items():
                    if not x & mask:
                        add(out, (x, z), (2 * a, 2 * b))
                    else:
                        value = (2 * b, a)
                        add(out, (x, z), value)
                        sign = (1 if z & mask else -1) * (1 if name == "S" else -1)
                        add(out, (x, z ^ mask), value, sign)
                power += 1
                terms = out
                peak = max(peak, len(terms))
                if peak > term_limit:
                    raise ValueError("Pauli term budget exceeded")
        else:
            raise ValueError(f"unsupported reference gate: {name}")
    assert record == 0
    a = sum(value[0] for (x, _), value in terms.items() if x == 0)
    b = sum(value[1] for (x, _), value in terms.items() if x == 0)
    while power and a % 2 == 0 and b % 2 == 0:
        a //= 2
        b //= 2
        power -= 1
    return dict(
        rational_numerator=a,
        sqrt2_numerator=b,
        denominator_power=power,
        expectation=(a + b * math.sqrt(2)) / 2**power,
        peak_terms=peak,
    )


def detector_expectation(proxy, detector, term_limit=262144):
    prefix = stim.Circuit()
    index = 0
    for op in stim.Circuit(proxy).flattened():
        if op.name == "DETECTOR":
            if index == detector:
                selected: set[int] = set()
                for target in op.targets_copy():
                    slot = prefix.num_measurements + target.value
                    selected.symmetric_difference_update({slot})
                return record_parity(str(prefix), selected, term_limit)
            index += 1
        prefix.append(op)
    raise ValueError("detector index out of range")


if __name__ == "__main__":
    from soft_cultivation_study import clifford_proxy, source_text, zero_noise

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=int, default=69)
    parser.add_argument("--term-limit", type=int, default=1048576)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, _ = source_text()
    result = detector_expectation(
        clifford_proxy(zero_noise(source)), args.detector, args.term_limit
    )
    result.update(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        detector=args.detector,
        probability_one=(1 - result["expectation"]) / 2,
        arithmetic="integer pairs in Q(sqrt(2)); no coefficient truncation",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

"""Compile an experimental native Clifft prefix plus a physical terminal block.

Stim is used only while preparing the fixed fault maps and contractions. The
native executable certifies the handoff using Clifft's own sampling planner.
"""

import argparse
import itertools
import json
from pathlib import Path

import stim
from clifford_gadget import Gadget, atoms_from_text, bind_faults, compile_gadgets, require_t_dialect
from compiled_gadget_contraction import TerminalCode, TerminalMarginals, write_kernel_plan
from soft_cultivation_study import clifford_proxy


def physical_text(circuit):
    return (
        "\n".join(
            "T" + line[1:] if line.split()[0] in {"S", "S_DAG"} else line
            for line in str(circuit).splitlines()
        )
        + "\n"
    )


def pauli_text(p):
    return "*".join(f"{'IXYZ'[axis]}{q}" for q, axis in enumerate(p) if axis)


def compile_bundle(source, directory):
    require_t_dialect(source)
    circuit = stim.Circuit(clifford_proxy(source)).flattened()
    ops = list(circuit)
    groups: list[list[int]] = []
    for i, op in enumerate(ops):
        if op.name in {"S", "S_DAG"}:
            if groups and groups[-1][1] == i:
                groups[-1][1] = i + 1
            else:
                groups.append([i, i + 1])
    if len(groups) < 2:
        raise ValueError("no terminal T-conjugated gadget")
    (begin, forward_end), (inverse_begin, end) = groups[-2:]
    clean, _ = bind_faults(source, 0, 0)
    width, visible, hidden, atoms = atoms_from_text(clean)
    program = compile_gadgets(width, atoms)
    position = max(i for i, a in enumerate(program) if isinstance(a, Gadget))
    suffix = program[position:]
    gadget = suffix[0]
    identity = stim.PauliString(width)
    if any(frame != identity for frame in (gadget.pre, gadget.mid, gadget.post)):
        raise ValueError("fixed Pauli gates inside the terminal gadget are unsupported")
    code = TerminalCode(gadget, suffix)
    if len(code.data) > 63 or width - len(code.data) > 63:
        raise ValueError("native experiment supports at most 63 data and 63 spectators")
    prefix_circuit = circuit[:begin]
    prefix_text = physical_text(prefix_circuit)
    prefix_clean, _ = bind_faults(prefix_text, 0, 0)
    _, prefix_visible, prefix_hidden, _ = atoms_from_text(prefix_clean)
    if hidden != prefix_hidden + 1:
        raise ValueError("terminal block must have exactly one hidden reset")
    measurements = [i for i in range(forward_end, inverse_begin) if ops[i].name == "MX"]
    resets = [i for i in range(forward_end, inverse_begin) if ops[i].name == "RX"]
    if len(measurements) != 1 or len(resets) != 1 or measurements[0] >= resets[0]:
        raise ValueError("unsupported gadget measurement/reset structure")
    measure, reset = measurements[0], resets[0]
    spectators = [q for q in range(width) if q not in code.signs]
    spectator_index = {q: i for i, q in enumerate(spectators)}
    future_mx = {}
    records_by_op = {}
    detectors: list[list[int]] = []
    observables: list[list[int]] = [[] for _ in range(circuit.num_observables)]
    total_record = 0
    for i, op in enumerate(ops):
        targets = op.targets_copy()
        if op.name in {"M", "MX", "MPP"}:
            count = len(op.target_groups()) if op.name == "MPP" else len(targets)
            records_by_op[i] = list(range(total_record, total_record + count))
            total_record += count
        elif op.name == "DETECTOR":
            detectors.append([total_record + t.value for t in targets])
        elif op.name == "OBSERVABLE_INCLUDE":
            observables[int(op.gate_args_copy()[0])].extend(total_record + t.value for t in targets)
        if i >= end and op.name == "MX":
            for target, slot in zip(targets, records_by_op[i], strict=True):
                if target.value not in spectator_index or target.value in future_mx:
                    raise ValueError("terminal MX must measure a spectator exactly once")
                future_mx[target.value] = (i, slot)
    if total_record != visible:
        raise AssertionError("record enumeration differs")
    for op in ops[end:]:
        if op.name not in {
            "MX",
            "MPP",
            "DEPOLARIZE1",
            "DEPOLARIZE2",
            "X_ERROR",
            "Y_ERROR",
            "Z_ERROR",
            "TICK",
            "DETECTOR",
            "OBSERVABLE_INCLUDE",
            "SHIFT_COORDS",
            "QUBIT_COORDS",
        }:
            raise ValueError(f"unsupported terminal suffix operation {op.name}")
    first_mpp = next(i for i in range(end, len(ops)) if ops[i].name == "MPP")
    if any(
        op.name.endswith("ERROR") or op.name.startswith("DEPOLARIZE") for op in ops[first_mpp + 1 :]
    ):
        raise ValueError("faults between terminal CSS measurements are unsupported")

    def reduce_mask(p, axes):
        return sum((p[q] in axes) << i for i, q in enumerate(code.data))

    def frame_effect(p):
        return [
            reduce_mask(p, (1, 2)),
            reduce_mask(p, (2, 3)),
            sum((p[q] in (2, 3)) << i for i, q in enumerate(spectators)),
        ]

    correction = stim.PauliString(width)
    correction[gadget.qubit] = "Z"
    reset_effect = frame_effect(gadget.inverse(correction))
    sites = []
    descriptions = []
    prefix_sites = []
    cached = {}

    def effect_at(i, p):
        if i not in cached and i < inverse_begin:
            stop = measure if i < measure else inverse_begin
            remaining = stim.Circuit()
            for op in ops[i + 1 : stop]:
                if op.name == "CX":
                    remaining.append(op)
            remaining.append("I", [width - 1])
            cached[i] = stim.Tableau.from_circuit(remaining)
        effect = [0] * 8
        if i < measure:
            propagated = cached[i](p)
            effect[0], effect[1], effect[4] = frame_effect(gadget.inverse(propagated))
            effect[5] = int(propagated[gadget.qubit] in (2, 3))
        elif i < reset:
            effect[0], effect[1], effect[4] = frame_effect(gadget.inverse(p))
            effect[6] = int(p[gadget.qubit] in (2, 3))
        elif i < inverse_begin:
            effect[0], effect[1], effect[4] = frame_effect(cached[i](p))
        elif i >= end:
            effect[2], effect[3] = reduce_mask(p, (1, 2)), reduce_mask(p, (2, 3))
            effect[4] = sum(
                (p[q] in (2, 3) and i < future_mx[q][0]) << spectator_index[q] for q in future_mx
            )
        else:
            raise ValueError("fault inserted inside a T layer")
        return effect

    for i, op in enumerate(ops):
        name, targets, args = op.name, op.targets_copy(), op.gate_args_copy()
        if name in {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
            size = 2 if name == "DEPOLARIZE2" else 1
            labels = (
                ["".join(x) for x in itertools.product("IXYZ", repeat=size)][1:]
                if name.startswith("DEPOLARIZE")
                else [name[0]]
            )
            for start in range(0, len(targets), size):
                qubits = [t.value for t in targets[start : start + size]]
                description = dict(op=i, group=start // size, targets=qubits, labels=labels)
                if i < begin:
                    prefix_sites.append(description)
                    continue
                effects = []
                for label in labels:
                    p = stim.PauliString(width)
                    for q, axis in zip(qubits, label, strict=True):
                        p[q] = axis
                    effects.append(effect_at(i, p))
                sites.append((args[0], effects))
                descriptions.append(description)
        elif name in {"MX", "M", "MPP"} and args and i >= begin:
            for group, slot in enumerate(records_by_op[i]):
                effect = [0] * 8
                effect[7] = slot + 1
                sites.append((args[0], [effect]))
                descriptions.append(dict(op=i, group=group, record=slot))

    probes = [a.pauli for a in code.x_checks + code.z_checks]
    for axis in "XYZ":
        p = stim.PauliString(width)
        for q in code.data:
            p[q] = axis
        probes.append(p)
    for q in spectators:
        for axis in "XZ":
            p = stim.PauliString(width)
            p[q] = axis
            probes.append(p)
    prefix_with_probes = prefix_text + "".join("EXP_VAL " + pauli_text(p) + "\n" for p in probes)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prefix.stim").write_text(prefix_with_probes)
    (directory / "full.stim").write_text(source)
    marginals = TerminalMarginals(code)
    for i, plan in enumerate(marginals.plans):
        write_kernel_plan(directory / f"marginal_{i}.txt", plan)
    words: list[str] = []

    def emit(*values):
        words.extend(map(str, values))

    emit(len(code.data), code.rank, len(spectators), prefix_visible, prefix_hidden, visible, hidden)
    emit(gadget.measurement.record, gadget.reset.record, code.logical_measurement.record)
    emit(int(gadget.parity.sign == -1), *reset_effect)
    for q in code.data:
        emit(int(code.signs[q] == -1))
    for row in code.x_rows + code.z_rows + code.z_duals:
        emit(sum(((row >> q) & 1) << j for j, q in enumerate(code.data)))
    for check in code.x_checks + code.z_checks:
        emit(check.record)
    for q in spectators:
        emit(gadget.parity[q], future_mx[q][1] + 1 if q in future_mx else 0)
    emit(len(sites))
    for probability, effects in sites:
        emit(probability, len(effects))
        for effect in effects:
            emit(*effect)
    for parities in (detectors, observables):
        emit(len(parities))
        for parity in parities:
            emit(len(parity), *parity)
    (directory / "model.txt").write_text("\n".join(words) + "\n")
    metadata = dict(
        begin=begin,
        end=end,
        prefix_visible=prefix_visible,
        prefix_hidden=prefix_hidden,
        visible=visible,
        hidden=hidden,
        data=list(code.data),
        spectators=spectators,
        prefix_noise_sites=prefix_sites,
        suffix_noise_sites=descriptions,
        contractions=[
            dict(
                rank=plan.rank,
                peak_scope=plan.peak_scope,
                storage_complex_entries=plan.storage,
                gather_entries=plan.gather_entries,
                leaf_parity_entries=sum(len(parity) for _, parity in plan.leaves),
                output_entries=len(plan.outputs),
                elimination_work=plan.work,
            )
            for plan in marginals.plans
        ],
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    compile_bundle(args.source.read_text(), args.directory)


if __name__ == "__main__":
    main()

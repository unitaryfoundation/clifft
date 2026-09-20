"""Offline coherent-Clifford reference for bound physical fold-transversal MSC.

Cat computational branches turn each noisy T-conjugated CX into a Clifford
and each CCZ into a conditional CZ. This reference uses dynamic Stim/Cirq
tableaux and is deliberately outside production Clifft execution.
"""

import math

import stim
from clifford_gadget import Atom, CoherentState


def z_pauli(width, qubit):
    result = stim.PauliString(width)
    result[qubit] = "Z"
    return result


def split_cat(state, qubit, width):
    expanded = []
    for term in state.terms:
        for bit in (0, 1):
            branch = term.copy()
            if branch.project(z_pauli(width, qubit), bit):
                expanded.append(branch)
    state.terms = expanded
    state.peak_terms = max(state.peak_terms, len(expanded))


def computational_bit(term, qubit):
    expectation = term.tableau.peek_z(qubit)
    if not expectation:
        raise ValueError("cat control is not computational in a coherent branch")
    return int(expectation == -1)


def conjugated_cx(state, operations, width):
    first, last = operations[0], operations[-1]
    q = first.qubits[0]
    if last.qubits != (q,) or {first.name, last.name} != {"T", "T_DAG"}:
        raise ValueError("expected inverse T gates on one data qubit")
    controls = [op.qubits[0] for op in operations if op.name == "CX"]
    if len(controls) != 1:
        raise ValueError("expected exactly one controlled X inside the conjugation")
    ancilla = controls[0]
    for term in state.terms:
        product = stim.PauliString(width)
        for op in operations[1:-1]:
            if op.name == "CX":
                if op.qubits != (ancilla, q):
                    raise ValueError("controlled factor has unexpected support")
                if computational_bit(term, ancilla):
                    p = stim.PauliString(width)
                    p[q] = "X"
                    product = p * product
            elif op.name in {"X", "Y", "Z"} and op.qubits == (q,):
                p = stim.PauliString(width)
                p[q] = op.name
                product = p * product
            elif op.name in {"X", "Y", "Z"} and op.qubits == (ancilla,):
                term.gate(op.name, op.qubits)
            else:
                raise ValueError("unexpected operation inside a physical T-conjugated CX")
        term.conjugated_pauli(product, {q: 1 if first.name == "T" else -1})


def evaluate(circuit, records):
    if any(op.probability is not None for op in circuit.operations):
        raise ValueError("bind all physical faults before reference evaluation")
    visible = len(circuit.measurements)
    hidden = sum(op.name == "R" for op in circuit.operations)
    if len(records) != visible + hidden or any(bit not in "01" for bit in records):
        raise ValueError("expected visible then hidden binary measurement records")
    width = circuit.manifest()["num_qubits"]
    state = CoherentState(width)
    m, r, i = 0, visible, 0
    split_stages = set()
    stage_terms: dict[str, int] = {}
    ops = circuit.operations
    while i < len(ops):
        op = ops[i]
        if op.name in {"M", "R"}:
            index = m if op.name == "M" else r
            atom = Atom(op.name, op.qubits, index, pauli=z_pauli(width, op.qubits[0]))
            state.project(atom, int(records[index]))
            state.merge()
            if op.name == "M":
                m += 1
            else:
                r += 1
        elif op.name in {"T", "T_DAG"} and "logical" in op.stage:
            end = i + 1
            while end < len(ops) and ops[end].name not in {"T", "T_DAG"}:
                if ops[end].stage != op.stage:
                    raise ValueError("unclosed controlled factor")
                end += 1
            if end == len(ops):
                raise ValueError("unclosed controlled factor")
            block = ops[i : end + 1]
            if op.stage not in split_stages:
                controls = [gate.qubits[0] for gate in block if gate.name == "CX"]
                if len(controls) != 1:
                    raise ValueError("cannot identify the first cat control")
                split_cat(state, controls[0], width)
                split_stages.add(op.stage)
            conjugated_cx(state, block, width)
            i = end
        elif op.name == "CCZ":
            ancilla, a, b = op.qubits
            for term in state.terms:
                if computational_bit(term, ancilla):
                    term.gate("CZ", (a, b))
        else:
            state.gate(Atom(op.name, op.qubits))
        stage_terms[op.stage] = max(stage_terms.get(op.stage, 0), len(state.terms))
        if not state.terms:
            return dict(
                probability=0.0,
                log_probability=None,
                peak_terms=state.peak_terms,
                stage_terms=stage_terms,
            )
        i += 1
    state.merge()
    probability = state.norm()
    return dict(
        probability=probability,
        log_probability=math.log(probability) if probability else None,
        peak_terms=state.peak_terms,
        stage_terms=stage_terms,
    )

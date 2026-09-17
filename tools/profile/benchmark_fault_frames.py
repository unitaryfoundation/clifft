"""Export validated native update probes, without a tensor-state executor."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from clifford_branches import materialize
from export_fold_blocks import Exporter
from fold_blocks import Fold, NoiseLayout, Protocol
from fold_cultivation import Operation
from fold_fault_frame import FramePlan
from study_fault_frames import cases


def frame_initializer(exporter, frame):
    words = [(frame.edges >> (64 * k)) & ((1 << 64) - 1) for k in range(4)]
    return (
        "{"
        + exporter.mask(frame.flips)
        + ","
        + str(frame.phase)
        + ",{"
        + ",".join(map(str, frame.linear))
        + "},{"
        + ",".join(f"{w}ULL" for w in words)
        + "}}"
    )


def export_plan(exporter, plan, offset, data):
    if plan.layout.width > 128 or len(plan.edges) > 256 or offset + len(plan.layout.slots) > 8192:
        raise ValueError("native research capacity exceeded")
    updates: list[str] = []
    actions = []
    for action in plan.actions:
        start = len(updates)
        updates.extend("{" + f"{a},{b}" + "}" for a, b in action.updates)
        targets = list(action.targets) + [0] * (3 - len(action.targets))
        edges = list(action.edges) + [0] * (3 - len(action.edges))
        fields = [{"fault": 0, "T": 2, "T_DAG": 3, "CX": 4, "CCZ": 5}[action.kind]]
        fields += targets + edges + [start, len(action.updates)]
        fields += [v + offset if v >= 0 else -1 for v in (action.x, action.z)]
        actions.append("{" + ",".join(map(str, fields)) + "}")
    nondata = sum(1 << e for e, pair in enumerate(plan.edges) if not set(pair) <= data)
    return exporter.obj(
        "fault_frame::Plan",
        [
            plan.layout.width,
            exporter.array("fault_frame::Action", actions),
            exporter.array("fault_frame::Update", updates),
            "{" + ",".join(f"{(nondata >> (64 * k)) & ((1 << 64) - 1)}ULL" for k in range(4)) + "}",
            exporter.mask(sum(1 << q for q in range(plan.layout.width) if q not in data)),
        ],
    )


def fixture(exporter, plan, faults, initial, offset, data):
    shifted = faults << offset
    words = [(shifted >> (64 * k)) & ((1 << 64) - 1) for k in range(128)]
    expected = plan.evaluate(faults, initial)
    return (
        "{{"
        + ",".join(f"{w}ULL" for w in words)
        + "},"
        + frame_initializer(exporter, initial)
        + ","
        + frame_initializer(exporter, expected)
        + ","
        + str(int(plan.data_only(expected, data)))
        + "}"
    )


def export(path):
    protocol = Protocol(7)
    r = protocol.reconstruction
    fold, stage = next(
        (p, g[1][0]) for _, p, g in protocol.groups if isinstance(p, Fold) and p.plan.distance == 7
    )
    plan = FramePlan(
        fold.core,
        [
            (min(fold.mapping[a], fold.mapping[b]), max(fold.mapping[a], fold.mapping[b]))
            for a, b in fold.plan.edges
        ],
    )
    exporter = Exporter(protocol)
    header = Path(__file__).with_name("fold_fault_frame_native.h").resolve()
    exporter.lines += [
        f'#include "{header}"',
        "struct FrameFixture { History h; fault_frame::Frame initial, expected; bool eligible; };",
    ]
    native_fold = exporter.fold(fold)
    offset = exporter.offsets[id(fold.core)]
    data = set(range(r.ancilla))
    native_plan = export_plan(exporter, plan, offset, data)
    histories = [fold.core.encode(history[stage]) for _, history in cases(protocol)]
    histories += [
        fold.core.encode(materialize(r, 0.001, seed)[0][stage]) for seed in range(3000, 3256)
    ]
    rng = random.Random(84920)
    histories += [rng.getrandbits(len(fold.core.slots)) for _ in range(128)]
    singles = [
        fault
        for boundary in fold.core.boundaries
        for _, x, z in boundary
        for fault in (x, z, x | z)
    ]
    histories += singles
    fixtures = exporter.array(
        "FrameFixture", [fixture(exporter, plan, f, plan.empty(), offset, data) for f in histories]
    )
    branch_fixtures = []
    for faults in histories:
        terms = []
        for weight, op in fold.branches(0, faults, 0):
            edges = sum(1 << fold.plan.edges.index(pair) for pair in op.edges)
            mono = (
                "{"
                + f"{exporter.mask(op.flips)},{exporter.mask(edges)},{op.global_phase},"
                + "{"
                + ",".join(map(str, op.linear))
                + "}}"
            )
            if complex(weight).imag:
                raise ValueError("native branch weight must be real")
            terms.append("{" + f"{complex(weight).real:.17g}," + mono + "}")
        branch_fixtures.append("{" + str(len(terms)) + ",{{" + ",".join(terms) + "}}}")
    exporter.lines.append("struct BranchFixture { unsigned count; std::array<Term,2> terms; };")
    expected_branches = exporter.array("BranchFixture", branch_fixtures)

    operations = []
    for _ in range(36):
        name = rng.choice(["T", "T_DAG", "CX", "CCZ"])
        operations.append(
            Operation(
                name, tuple(rng.sample(range(6), {"T": 1, "T_DAG": 1, "CX": 2, "CCZ": 3}[name]))
            )
        )
    small = FramePlan(
        NoiseLayout(operations, [True] * len(operations), 6),
        [(a, b) for a in range(6) for b in range(a + 1, 6)],
    )
    small_data = {1, 2, 3, 4, 5}
    small_plan = export_plan(exporter, small, 0, small_data)
    small_fixtures = []
    for _ in range(128):
        initial = small.empty()
        initial.flips = rng.getrandbits(6)
        initial.phase = rng.randrange(8)
        initial.linear = [2 * rng.randrange(4) for _ in range(6)]
        initial.edges = rng.getrandbits(15)
        small_fixtures.append(
            fixture(
                exporter, small, rng.getrandbits(len(small.layout.slots)), initial, 0, small_data
            )
        )
    small_data_name = exporter.array("FrameFixture", small_fixtures)
    exporter.lines.append(r"""
static uint64_t sink = 0;
bool validate(const fault_frame::Plan& p, std::span<const FrameFixture> fixtures) {
    for (const auto& fixture : fixtures) {
        auto actual = fixture.initial;
        fault_frame::evaluate(p, fixture.h, actual);
        if (actual != fixture.expected || fault_frame::data_only(actual,p) != fixture.eligible)
            return false;
    }
    return true;
}
double frame_time(const fault_frame::Plan& p, std::span<const FrameFixture> fixtures,
                  size_t count) {
    fault_frame::Frame output;
    auto start = std::chrono::steady_clock::now();
    for (size_t j=0; j<count; ++j) {
        const auto& f = fixtures[j % fixtures.size()];
        output = f.initial;
        fault_frame::evaluate(p, f.h, output);
        sink += fault_frame::data_only(output,p);
        asm volatile("" : : "g"(&output) : "memory");
    }
    auto elapsed = std::chrono::steady_clock::now()-start;
    return std::chrono::duration<double,std::nano>(elapsed).count()/count;
}
double block_time(const Fold& fold, std::span<const FrameFixture> fixtures, size_t count) {
    Executor executor;
    std::array<Term,2> output;
    auto start = std::chrono::steady_clock::now();
    for (size_t j=0; j<count; ++j) {
        sink += executor.branches(fold,fixtures[j % fixtures.size()].h,output);
        asm volatile("" : : "g"(&output) : "memory");
    }
    auto elapsed = std::chrono::steady_clock::now()-start;
    return std::chrono::duration<double,std::nano>(elapsed).count()/count;
}
""")
    exporter.lines.append(f"""
int main(int argc, char** argv) {{
    size_t count = argc > 1 ? std::stoull(argv[1]) : 100000;
    if (!count) return 2;
    if (!validate({native_plan},{fixtures}) || !validate({small_plan},{small_data_name})) return 3;
    Executor executor;
    for (size_t j=0; j<{fixtures}.size(); ++j) {{
        std::array<Term,2> output;
        unsigned count = executor.branches({native_fold},{fixtures}[j].h,output);
        const auto& expected = {expected_branches}[j];
        if (count != expected.count) return 4;
        for (unsigned k=0; k<count; ++k) {{
            const auto& a = output[k]; const auto& b = expected.terms[k];
            if (a.weight != b.weight || a.op.flips != b.op.flips || a.op.edges != b.op.edges ||
                a.op.phase != b.op.phase || a.op.linear != b.op.linear) return 5;
        }}
    }}
    std::array<double,7> frames, blocks, natural_frames, natural_blocks;
    auto natural = std::span<const FrameFixture>({fixtures}).subspan(62,256);
    frame_time({native_plan},{fixtures},1000);
    block_time({native_fold},{fixtures},1000);
    for (unsigned j=0; j<7; ++j) {{
        if (j % 2) {{
            blocks[j] = block_time({native_fold},{fixtures},count);
            frames[j] = frame_time({native_plan},{fixtures},count);
            natural_blocks[j] = block_time({native_fold},natural,count);
            natural_frames[j] = frame_time({native_plan},natural,count);
        }} else {{
            frames[j] = frame_time({native_plan},{fixtures},count);
            blocks[j] = block_time({native_fold},{fixtures},count);
            natural_frames[j] = frame_time({native_plan},natural,count);
            natural_blocks[j] = block_time({native_fold},natural,count);
        }}
    }}
    std::cout << R"({{"frame_ns":[)";
    for (unsigned j=0; j<7; ++j) std::cout << (j ? "," : "") << frames[j];
    std::cout << R"(],"block_branches_ns":[)";
    for (unsigned j=0; j<7; ++j) std::cout << (j ? "," : "") << blocks[j];
    std::cout << R"(],"natural_frame_ns":[)";
    for (unsigned j=0; j<7; ++j) std::cout << (j ? "," : "") << natural_frames[j];
    std::cout << R"(],"natural_block_branches_ns":[)";
    for (unsigned j=0; j<7; ++j) std::cout << (j ? "," : "") << natural_blocks[j];
    std::cout << R"(],"frame_bytes":)" << sizeof(fault_frame::Frame)
              << R"(,"fold_fixtures":{len(histories)},"generic_fixtures":128,"iterations":)"
              << count
              << R"(,"checksum":)" << sink << "}}" << std::endl;
}}
""")
    path.write_text("\n".join(exporter.lines) + "\n")
    return {
        "fold_fixtures": len(histories),
        "generic_fixtures": 128,
        "edge_slots": len(plan.edges),
        "frame_actions": len(plan.actions),
        "block_actions": len(fold.actions) + sum(map(len, fold.fault_actions)),
        "single_pauli_patterns": len(singles),
        "single_pauli_data_only": sum(plan.data_only(plan.evaluate(f), data) for f in singles),
        "natural_core_samples": 256,
        "natural_core_data_only": sum(
            plan.data_only(plan.evaluate(f), data) for f in histories[62:318]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))


if __name__ == "__main__":
    main()

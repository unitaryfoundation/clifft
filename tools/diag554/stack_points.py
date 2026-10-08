"""Repeat selected stack positions on the aliasing page, one process per run.

usage: stack_points.py <steer.so> <cpu> <out dir> <heap addr hex> <positions> <repeats>
                       <name>=<binary> ...

Positions are line indices on the first page below the probed frame whose
micro-tag matches libm's log table page (as in stack_sweep.py). Runs are
interleaved across repeats, variants, and positions so that interference from
the rest of the runner does not land on one configuration.
"""

import json
import pathlib
import sys

import stack_collision as sc


def main():
    steer, cpu, out_arg, heap_addr, positions_arg, repeats_arg = sys.argv[1:7]
    variants = dict(arg.split("=", 1) for arg in sys.argv[7:])
    positions = [int(p) for p in positions_arg.split(",")]
    out_dir = pathlib.Path(out_arg)
    (out_dir / "single").mkdir(parents=True, exist_ok=True)
    steer_env = [
        f"LD_PRELOAD={steer}",
        "STEER_SIZE=000792",
        "STEER_SKIP=0000",
        f"STEER_ADDR={int(heap_addr, 16):012x}",
        "STEER_REPORT=1",
    ]
    plan = {}
    for name, binary in variants.items():
        ret_off = sc.log_return_offset(binary)
        rsp0, libm, _ = sc.gdb_probe(binary, steer_env + sc.pad_env(0), cpu, ret_off)
        want = sc.utag(libm + sc.LOG_TABLE_OFFSET)
        page = (rsp0 >> 12) - 2
        while sc.utag(page << 12) != want:
            page -= 1
        extras = {j: rsp0 - ((page << 12) + j * 64 + (rsp0 & 0x3F)) for j in positions}
        check = positions[0]
        rsp, _, _ = sc.gdb_probe(binary, steer_env + sc.pad_env(extras[check]), cpu, ret_off)
        expected = (page << 12) + check * 64 + (rsp0 & 0x3F)
        plan[name] = {
            "binary": binary,
            "extras": extras,
            "rsp0": hex(rsp0),
            "check": rsp == expected,
        }
    for r in range(int(repeats_arg)):
        for name, info in plan.items():
            for j in positions:
                sc.timed(
                    info["binary"],
                    steer_env + sc.pad_env(info["extras"][j]),
                    cpu,
                    out_dir / "single" / f"stackpoint_{name}_{j}_{r}.json",
                )
    record = {n: {k: v for k, v in i.items() if k != "extras"} for n, i in plan.items()}
    (out_dir / "stackpoints.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()

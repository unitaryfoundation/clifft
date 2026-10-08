"""Sweep the noise loop's stack frame across all 64 lines of an aliasing page.

usage: stack_sweep.py <name> <binary> <steer.so> <cpu> <out dir> <heap addr hex>

The stack pointer at the noise loop moves in 64-byte steps with environment
length (a frame realigns the stack), so after one gdb probe every position is
reached by padding alone. Positions 0, 31 and 63 are re-probed to confirm.
"""

import json
import pathlib
import sys

import stack_collision as sc


def main():
    name, binary, steer, cpu, out_arg, heap_addr = sys.argv[1:7]
    out_dir = pathlib.Path(out_arg)
    (out_dir / "single").mkdir(parents=True, exist_ok=True)
    steer_env = [
        f"LD_PRELOAD={steer}",
        "STEER_SIZE=000792",
        "STEER_SKIP=0000",
        f"STEER_ADDR={int(heap_addr, 16):012x}",
        "STEER_REPORT=1",
    ]
    ret_off = sc.log_return_offset(binary)
    rsp0, libm, _ = sc.gdb_probe(binary, steer_env + sc.pad_env(0), cpu, ret_off)
    want = sc.utag(libm + sc.LOG_TABLE_OFFSET)
    page = (rsp0 >> 12) - 2
    while sc.utag(page << 12) != want:
        page -= 1
    record: dict[str, object] = {
        "name": name,
        "rsp0": hex(rsp0),
        "page": hex(page << 12),
        "line_offset": hex(rsp0 & 0x3F),
        "checks": {},
    }

    def extra_for(page_base, j):
        return rsp0 - (page_base + j * 64 + (rsp0 & 0x3F))

    checks = {}
    for j in (0, 31, 63):
        rsp, _, _ = sc.gdb_probe(
            binary, steer_env + sc.pad_env(extra_for(page << 12, j)), cpu, ret_off
        )
        checks[j] = {"rsp": hex(rsp), "expected": hex((page << 12) + j * 64 + (rsp0 & 0x3F))}
    record["checks"] = checks
    for j in range(64):
        sc.timed(
            binary,
            steer_env + sc.pad_env(extra_for(page << 12, j)),
            cpu,
            out_dir / "single" / f"stacksweep_bad_{name}_{j}.json",
        )
    good = (page - 1) << 12
    assert sc.utag(good) != want
    for j in range(0, 64, 8):
        sc.timed(
            binary,
            steer_env + sc.pad_env(extra_for(good, j)),
            cpu,
            out_dir / "single" / f"stacksweep_good_{name}_{j}.json",
        )
    (out_dir / f"stacksweep_{name}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()

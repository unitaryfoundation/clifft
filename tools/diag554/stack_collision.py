"""Place the noise loop's stack frame on a page that aliases libm's log table.

usage: stack_collision.py <name> <binary> <steer.so> <cpu> <out dir> <heap addr hex>
                          [rsp offset hex]

The benchmark disables ASLR, so the stack depends only on the environment and
argument strings. gdb measures the stack pointer at the return address of the
noise loop's first log call; environment padding then moves that frame onto a
page whose L1D micro-tag equals the micro-tag of libm's log table page, with
the frame's lines in sets 16-18. The favorable control adds one more page of
padding, which keeps the in-page offset and changes the micro-tag.
"""

import json
import pathlib
import re
import subprocess
import sys

PAIRS = [(12, 27), (13, 26), (14, 25), (15, 20), (16, 21), (17, 22), (18, 23), (19, 24)]
PIE_BASE = 0x555555554000
LOG_TABLE_OFFSET = 0xB5000
FILTER = "sample_surface_d5_r5_high_noise_10000_shots"
PAD_VARS = 10
PAD_MAX = 131071


def utag(va):
    return tuple(((va >> a) ^ (va >> b)) & 1 for a, b in PAIRS)


def log_return_offset(binary):
    nm = subprocess.run(["nm", "-S", binary], capture_output=True, text=True, check=True)
    for line in nm.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and "BatchExecutor23sample_presampled_noise" in parts[3]:
            start, size = int(parts[0], 16), int(parts[1], 16)
            break
    else:
        raise SystemExit("noise function not found")
    dis = subprocess.run(
        [
            "objdump",
            "-d",
            "--no-show-raw-insn",
            f"--start-address={start}",
            f"--stop-address={start + size}",
            binary,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    for i, line in enumerate(dis):
        if "call" in line and "<log@plt>" in line:
            for nxt in dis[i + 1 :]:
                m = re.match(r"\s*([0-9a-f]+):", nxt)
                if m:
                    return int(m.group(1), 16)
    raise SystemExit("log call not found")


def pad_env(extra):
    """PAD_VARS fixed variables whose total length is PAD_VARS + extra."""
    base = 1 + extra // PAD_VARS
    lengths = [base] * PAD_VARS
    for i in range(extra % PAD_VARS):
        lengths[i] += 1
    assert max(lengths) <= PAD_MAX, "padding exceeds one environment string"
    return [f"DIAG_SP{i}={'x' * n}" for i, n in enumerate(lengths)]


def bench_args(binary, min_time, reps):
    return [
        binary,
        f"--benchmark_filter={FILTER}",
        f"--benchmark_min_time={min_time}s",
        "--benchmark_min_warmup_time=0.2",
        f"--benchmark_repetitions={reps}",
        "--benchmark_out=/tmp/sampling-canary.json",
        "--benchmark_out_format=json",
    ]


def gdb_probe(binary, env, cpu, ret_off):
    cmds = [
        "set startup-with-shell off",
        "set disable-randomization on",
        "unset environment LINES",
        "unset environment COLUMNS",
        f"break *{PIE_BASE + ret_off:#x}",
        "run",
        'printf "RSP %lx\\n", $rsp',
        "info proc mappings",
        "delete",
        "continue",
    ]
    argv = ["env", *env, "taskset", "-c", cpu, "gdb", "-nx", "-batch"]
    for c in cmds:
        argv += ["-ex", c]
    argv += ["--args", *bench_args(binary, 0.1, 1)]
    out = subprocess.run(argv, capture_output=True, text=True)
    text = out.stdout + out.stderr
    found = re.search(r"RSP ([0-9a-f]+)", text)
    if found is None:
        raise SystemExit(f"gdb did not reach the noise loop:\n{text[-2000:]}")
    rsp = int(found.group(1), 16)
    libm = min(
        int(m.group(1), 16) for m in re.finditer(r"^\s*0x([0-9a-f]+)\s.*libm\.so", text, re.M)
    )
    frame = re.search(r"frame=(\d+)", text)
    return rsp, libm, int(frame.group(1)) if frame else None


def timed(binary, env, cpu, out_json):
    out = subprocess.run(
        ["env", *env, "taskset", "-c", cpu, *bench_args(binary, 0.5, 3)],
        capture_output=True,
        text=True,
    )
    pathlib.Path("/tmp/sampling-canary.json").rename(out_json)
    frame = re.search(r"frame=(\d+)", out.stderr)
    return int(frame.group(1)) if frame else None


def main():
    name, binary, steer, cpu, out_arg, heap_addr = sys.argv[1:7]
    rsp_offset = int(sys.argv[7], 16) if len(sys.argv) > 7 else 0x400
    tag = f"{name}_{rsp_offset:03x}"
    out_dir = pathlib.Path(out_arg)
    (out_dir / "single").mkdir(parents=True, exist_ok=True)
    steer_env = [
        f"LD_PRELOAD={steer}",
        "STEER_SIZE=000792",
        "STEER_SKIP=0000",
        f"STEER_ADDR={int(heap_addr, 16):012x}",
        "STEER_REPORT=1",
    ]
    ret_off = log_return_offset(binary)
    record: dict[str, object] = {"name": name, "log_return_offset": hex(ret_off)}

    rsp0, libm, frame0 = gdb_probe(binary, steer_env + pad_env(0), cpu, ret_off)
    target_tag = utag(libm + LOG_TABLE_OFFSET)
    record.update(rsp0=hex(rsp0), libm=hex(libm), table_utag=target_tag)
    page = (rsp0 >> 12) - 1
    while utag(page << 12) != target_tag:
        page -= 1
    target = (page << 12) + rsp_offset + (rsp0 & 0xF)
    extra = rsp0 - target
    for _ in range(3):
        rsp, _, frame = gdb_probe(binary, steer_env + pad_env(extra), cpu, ret_off)
        if rsp == target:
            break
        extra += rsp - target
    record.update(
        target=hex(target),
        unfav_extra=extra,
        unfav_rsp=hex(rsp),
        unfav_hit=rsp == target,
        unfav_rsp_utag=utag(rsp),
    )
    fav_rsp, _, _ = gdb_probe(binary, steer_env + pad_env(extra + 4096), cpu, ret_off)
    record.update(fav_rsp=hex(fav_rsp), fav_rsp_utag=utag(fav_rsp))

    frames = {}
    for i in range(3):
        for kind, ext in (("fav", extra + 4096), ("unfav", extra)):
            frames[f"{kind}{i}"] = timed(
                binary,
                steer_env + pad_env(ext),
                cpu,
                out_dir / "single" / f"stack_{tag}_{kind}_{i}.json",
            )
    _, _, gdb_frame_unfav = gdb_probe(binary, steer_env + pad_env(extra), cpu, ret_off)
    record.update(timed_frames=frames, gdb_frame_unfav=gdb_frame_unfav)
    record["name"] = tag
    (out_dir / f"stack_{tag}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()

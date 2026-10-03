"""Move one already-compiled copy loop within existing padding, then measure."""

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

OUT = Path("diagnostic-results").resolve()
OUT.mkdir(exist_ok=True)
with (OUT / "original.zip").open("wb") as output:
    subprocess.run(
        ["gh", "api", "repos/unitaryfoundation/clifft/actions/artifacts/11280345806/zip"],
        stdout=output,
        check=True,
    )
with zipfile.ZipFile(OUT / "original.zip") as archive:
    archive.extractall(OUT / "original")

original = (OUT / "original/head").read_bytes()
# The ELF text segment's file offset is 4096 bytes below its virtual address.
loop = 0xC3080 - 0x1000
end = 0xC30BB - 0x1000
padding = loop - 13
body = original[loop:end]
assert body.hex().startswith("f2430f100402f20f1144dfe8")
assert body[-2:] == bytes.fromhex("75c5")
assert original[padding:loop] == bytes.fromhex("666666662e0f1f840000000000")

for name in ["base", "head"]:
    shutil.copy2(OUT / "original" / name, OUT / name)
    (OUT / name).chmod(0o755)
for shift in [1, 2, 4, 8, 12, 13]:
    data = bytearray(original)
    # There are no incoming branches except the loop's own relative backedge.
    # Its instructions and displacement stay identical; only addresses change.
    data[padding:end] = b"\x90" * (13 - shift) + body + b"\x90" * shift
    path = OUT / ("head-shift-" + str(shift))
    path.write_bytes(data)
    path.chmod(0o755)

core = str(min(os.sched_getaffinity(0)))
subprocess.run(["lscpu"], stdout=(OUT / "cpu.txt").open("w"), check=True)
for name in [
    "base",
    "head",
    "head-shift-1",
    "head-shift-2",
    "head-shift-4",
    "head-shift-8",
    "head-shift-12",
    "head-shift-13",
    "head",
    "base",
]:
    index = len(list(OUT.glob("run-*.json")))
    filename = OUT / ("run-" + str(index) + "-" + name + ".json")
    args = [
        "taskset",
        "-c",
        core,
        str(OUT / name),
        "--benchmark_filter=^sample_exp_val_20q_200_probes_100000_shots$",
        "--benchmark_min_time=0.5s",
        "--benchmark_min_warmup_time=0.2",
        "--benchmark_repetitions=3",
        "--benchmark_out=" + str(filename),
    ]
    with filename.with_suffix(".log").open("w") as output:
        subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, check=True)
    rows = json.loads(filename.read_text())["benchmarks"]
    median = next(row["cpu_time"] for row in rows if row.get("aggregate_name") == "median")
    print(name, median, flush=True)

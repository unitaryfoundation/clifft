"""Summarize issue 554 executor-variant runs as Markdown."""

import json
import math
import os
import pathlib
import re
import statistics
import sys

VARIANTS = os.environ.get("VARIANTS", "control local compact").split()
OTHERS = [v for v in VARIANTS if v != "control"]


def iterations(path):
    data = json.loads(path.read_text())
    return [b["cpu_time"] / 1e6 for b in data["benchmarks"] if b.get("run_type") == "iteration"]


def med(path):
    return statistics.median(iterations(path))


def balanced(directory):
    out = {}
    for v in VARIANTS:
        first, second = directory / f"{v}-first.json", directory / f"{v}-second.json"
        if first.exists() and second.exists():
            out[v] = math.sqrt(med(first) * med(second))
    return out


def main():
    out = pathlib.Path(sys.argv[1])
    env = out / "env.txt"
    cpu = "unknown"
    if env.exists():
        m = re.search(r"Model name:\s*(.*)", env.read_text())
        cpu = m.group(1).strip() if m else cpu
    print(f"### Executor variants on {cpu}\n")
    pages = out / "pages.txt"
    if pages.exists():
        print(f"`{pages.read_text().strip()}`\n")

    for name in ["natural", "natural2"]:
        d = out / name
        if d.exists():
            r = balanced(d)
            if "control" in r:
                text = ", ".join(
                    f"{v} {r[v]:.2f} ms ({100 * (r[v] / r['control'] - 1):+.1f}%)"
                    for v in VARIANTS
                    if v in r
                )
                print(f"- {name} placement: {text}")
    single = out / "single"

    def runs(prefix):
        return sorted(single.glob(f"{prefix}_*.json"))

    if runs("fav_control"):
        print("\n| Variant | favorable ms | unfavorable ms | penalty |")
        print("|---|---:|---:|---:|")
        for v in VARIANTS:
            fav = statistics.median(med(p) for p in runs(f"fav_{v}"))
            unfav = statistics.median(med(p) for p in runs(f"unfav_{v}"))
            print(f"| {v} | {fav:.2f} | {unfav:.2f} | {100 * (unfav / fav - 1):+.1f}% |")

    if (single / "sweep_bad_control_0.json").exists():
        print(
            "\n| Variant | aliasing-page sweep min / median / max ms"
            " | positions >3% over favorable | favorable-page sweep min / max ms |"
        )
        print("|---|---|---:|---|")
        for v in VARIANTS:
            bad = [
                med(single / f"sweep_bad_{v}_{j}.json")
                for j in range(64)
                if (single / f"sweep_bad_{v}_{j}.json").exists()
            ]
            good = [
                med(single / f"sweep_good_{v}_{j}.json")
                for j in range(0, 64, 8)
                if (single / f"sweep_good_{v}_{j}.json").exists()
            ]
            ref = statistics.median(good) if good else min(bad)
            slow = [
                j
                for j in range(64)
                if (single / f"sweep_bad_{v}_{j}.json").exists()
                and med(single / f"sweep_bad_{v}_{j}.json") > 1.03 * ref
            ]
            print(
                f"| {v} | {min(bad):.2f} / {statistics.median(bad):.2f} / {max(bad):.2f} | "
                f"{len(slow)} ({_ranges(slow)}) | {min(good):.2f} / {max(good):.2f} |"
            )
        print("\n<details><summary>Sweep by line position (aliasing page)</summary>\n")
        print("| j | " + " | ".join(VARIANTS) + " |")
        print("|---:|" + "---:|" * len(VARIANTS))
        for j in range(64):
            row: list[str] = []
            for v in VARIANTS:
                p = single / f"sweep_bad_{v}_{j}.json"
                row.append(f"{med(p):.2f}" if p.exists() else "")
            print(f"| {j} | " + " | ".join(row) + " |")
        print("\n</details>")

    stacks = sorted(out.glob("stack_*.json"))
    if stacks:
        print(
            "\n| Variant | stack favorable ms | stack aliasing ms | penalty"
            " | aliasing rsp | gdb frame matches |"
        )
        print("|---|---:|---:|---:|---|---|")
        for p in stacks:
            rec = json.loads(p.read_text())
            v = rec["name"]
            fav = statistics.median(med(x) for x in runs(f"stack_{v}_fav"))
            unfav = statistics.median(med(x) for x in runs(f"stack_{v}_unfav"))
            frames = {k: f for k, f in rec["timed_frames"].items() if k.startswith("unfav")}
            match = all(f == rec["gdb_frame_unfav"] for f in frames.values())
            print(
                f"| {v} | {fav:.2f} | {unfav:.2f} | {100 * (unfav / fav - 1):+.1f}% | "
                f"{rec['unfav_rsp']} | {match} |"
            )

    wide = out / "wide"
    if wide.exists():
        print("\n| Benchmark | control ms | " + " | ".join(OTHERS) + " |")
        print("|---|---:|" + "---:|" * len(OTHERS))
        for d in sorted(wide.iterdir()):
            r = balanced(d)
            if "control" not in r:
                continue
            cells: list[str] = [f"{r['control']:.3f}"]
            for v in OTHERS:
                cells.append(f"{100 * (r[v] / r['control'] - 1):+.1f}%" if v in r else "")
            print(f"| {d.name} | " + " | ".join(cells) + " |")


def stacksweep(out):
    single = out / "single"
    if not (single / "stacksweep_bad_control_0.json").exists():
        return
    print(
        "\n| Variant | stack sweep on aliasing page min / median / max ms"
        " | positions >3% over favorable | favorable page min / max ms |"
    )
    print("|---|---|---:|---|")
    table = {}
    for v in VARIANTS:
        bad = [med(single / f"stacksweep_bad_{v}_{j}.json") for j in range(64)]
        good = [med(single / f"stacksweep_good_{v}_{j}.json") for j in range(0, 64, 8)]
        table[v] = bad
        ref = statistics.median(good)
        slow = [j for j, t in enumerate(bad) if t > 1.03 * ref]
        print(
            f"| {v} | {min(bad):.2f} / {statistics.median(bad):.2f} / {max(bad):.2f} | "
            f"{len(slow)} ({_ranges(slow)}) | {min(good):.2f} / {max(good):.2f} |"
        )
    print("\n<details><summary>Stack sweep by line position</summary>\n")
    print("| j | " + " | ".join(VARIANTS) + " |")
    print("|---:|" + "---:|" * len(VARIANTS))
    for j in range(64):
        print(f"| {j} | " + " | ".join(f"{table[v][j]:.2f}" for v in VARIANTS) + " |")
    print("\n</details>")


def stackpoints(out):
    single = out / "single"
    files = sorted(single.glob("stackpoint_*.json"))
    if not files:
        return
    table: dict[tuple[str, int], list[float]] = {}
    for p in files:
        _, name, j, _r = p.stem.split("_")
        table.setdefault((name, int(j)), []).append(med(p))
    positions = sorted({j for _, j in table})
    print(
        "\n| Variant | "
        + " | ".join(f"stack position {j}: median (min-max) ms" for j in positions)
        + " |"
    )
    print("|---|" + "---|" * len(positions))
    for v in VARIANTS:
        cells = []
        for j in positions:
            xs = table.get((v, j), [])
            cells.append(f"{statistics.median(xs):.2f} ({min(xs):.2f}-{max(xs):.2f})" if xs else "")
        print(f"| {v} | " + " | ".join(cells) + " |")


def layouts(out):
    root = out / "layouts"
    if not root.exists():
        return
    print("\n| Benchmark | control median (min-max) ms | " + " | ".join(OTHERS) + " |")
    print("|---|---|" + "---:|" * len(OTHERS))
    for d in sorted(root.iterdir()):
        by = {v: [med(p) for p in sorted(d.glob(f"pad*_{v}.json"))] for v in VARIANTS}
        if not by["control"]:
            continue
        c = statistics.median(by["control"])
        cells = [f"{c:.3f} ({min(by['control']):.3f}-{max(by['control']):.3f})"]
        for v in OTHERS:
            if by[v]:
                m = statistics.median(by[v])
                cells.append(f"{100 * (m / c - 1):+.1f}% ({min(by[v]):.3f}-{max(by[v]):.3f})")
            else:
                cells.append("")
        print(f"| {d.name} | " + " | ".join(cells) + " |")


def _ranges(js):
    if not js:
        return "none"
    parts: list[str] = []
    start = prev = js[0]
    for j in js[1:] + [None]:
        if j is not None and j == prev + 1:
            prev = j
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if j is not None:
            start = prev = j
    return ", ".join(parts)


if __name__ == "__main__":
    main()
    stacksweep(pathlib.Path(sys.argv[1]))
    stackpoints(pathlib.Path(sys.argv[1]))
    layouts(pathlib.Path(sys.argv[1]))

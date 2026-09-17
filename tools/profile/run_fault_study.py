"""Run fault specialization on named local circuits; retain hashes and raw timings."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--circuit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patterns", type=int, default=32)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--k", type=int, nargs="+", default=[-1, 0, 1, 2])
    parser.add_argument("--postselect", type=int, choices=[0, 1], default=1)
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--latency-shots", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "schema": 1,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "machine": platform.uname()._asdict(),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "runs": [],
    }
    if args.latency:
        if args.postselect != 1 or args.repeats < 1:
            parser.error("latency requires postselection and positive repeats")
        for path in args.circuit:
            for shots in args.latency_shots:
                print(path.stem, shots, flush=True)
                for repeat in range(args.repeats):
                    order = ["shared", "specialized"]
                    if repeat % 2:
                        order.reverse()
                    for strategy in order:
                        invocation = subprocess.run(
                            [
                                str(args.binary.resolve()),
                                "--latency",
                                strategy,
                                str(path.resolve()),
                                str(shots),
                                "--postselect-all",
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=180,
                        )
                        run = json.loads(invocation.stdout)
                        run.update(
                            id=path.stem,
                            repeat=repeat,
                            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        )
                        result["runs"].append(run)
        (args.output / "latency.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    for path in args.circuit:
        for k in args.k:
            name = path.stem + f"-k{k}"
            print(name, flush=True)
            directory = args.output / name
            invocation = subprocess.run(
                [
                    str(args.binary.resolve()),
                    str(path.resolve()),
                    str(args.patterns),
                    str(args.shots),
                    str(k),
                    str(directory),
                    str(args.postselect),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            run = json.loads(invocation.stdout)
            run["id"] = name
            run["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            run["specialized_sha256"] = [
                hashlib.sha256((directory / f"{i}.stim").read_bytes()).hexdigest()
                for i in range(args.patterns)
            ]
            result["runs"].append(run)
            # Preserve completed experiments if a later invocation fails.
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

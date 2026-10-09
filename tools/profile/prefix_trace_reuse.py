"""Reuse preparation serialization and compare native fragment compilation."""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from automatic_specialization import History
from compiled_prefix_reuse import ReusablePhase
from conditional_phase_frontend import ConditionalResult
from shared_phase_specialization import SharedPhase, parity


class PreparedPhase(ReusablePhase):
    """Emit the optimized prefix directly, preserving the preceding interface."""

    def __init__(self, source: str, binary: Path):
        super().__init__(source, binary)
        self.fixed_gate_count = 0
        self.physical: dict[int, int] = {}
        self.step: dict[str, Any] = {}
        if self.reuse_info["eligible"]:
            assert self.first is not None
            region, shared = self.first.region, self.first.region.shared
            self.fixed_gate_count = len(shared.preparation) + len(shared.nonclifford)
            self.physical = {compact: q for q, compact in shared.quantum_map.items()}
            self.step = {
                "residual_t": shared.metadata()["residual_t"],
                "support_variables": shared.variables,
                "boundary_gate": region.boundary_gate,
                "prefix_records": shared.prefix_records,
                "moved_operations": self.first.audit["moved_operations"],
            }

    def rewrite(
        self,
        history: History,
        seed: int,
        *,
        choose_prefix: Callable[[SharedPhase, int], int] | None = None,
        decisions: tuple[int, ...] | None = None,
    ) -> ConditionalResult:
        if not self.reuse_info["eligible"]:
            return super().rewrite(history, seed, choose_prefix=choose_prefix, decisions=decisions)
        assert self.first is not None
        region, shared = self.first.region, self.first.region.shared
        prefix, tail = region.split_history(self.first.map_history(history))
        if choose_prefix is None:
            sampler = shared.prefix.compile_sampler(seed=random.Random(seed).getrandbits(64))
            ideal_bits = int.from_bytes(sampler.sample(1, bit_packed=True)[0].tobytes(), "little")
        else:
            ideal_bits = choose_prefix(shared, 0)
        controls = shared.controls(prefix, ideal_bits)
        lines = []
        # The remaining gates are Clifford, so they cannot change the already
        # known residual T count or select a different frontend route.
        for line in shared.quantum_lines(controls)[self.fixed_gate_count :]:
            gate, *targets = line.split()
            lines.append(gate + " " + " ".join(str(self.physical[int(q)]) for q in targets))
        lines += [
            f"X {q}" for q, offset in enumerate(shared.exit_offsets) if parity(offset, controls)
        ]
        for output in shared.output:
            if isinstance(output, str):
                lines.append(output)
            elif isinstance(output, int):
                lines.append(f"MPAD {(controls >> (1 + output)) & 1}")
            else:
                raise AssertionError("State-preserving output contains a deferred readout")
        state = self.optimized_prefix + ("\n".join(lines) + "\n" if lines else "")
        values = (controls >> 1) & ((1 << shared.prefix_records) - 1)
        return ConditionalResult(
            state + "\n".join(tail) + "\n",
            [{**self.step, "record_values": values}],
            "residual_magic",
        )


class TraceWorker:
    def __init__(self, binary: Path, prefix: str, *, max_width: int, phase: bool):
        self.directory = tempfile.TemporaryDirectory(prefix="clifft-trace-reuse-")
        path = Path(self.directory.name) / "prefix.stim"
        path.write_text(prefix)
        self.process = subprocess.Popen(
            [str(binary.resolve()), str(path), str(max_width), str(int(phase))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            self.setup = self.read()
        except Exception:
            self.close()
            raise

    def read(self) -> dict[str, Any]:
        assert self.process.stdout is not None and self.process.stderr is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Native fragment worker failed: " + self.process.stderr.read())
        return dict(json.loads(line))

    def run(self, mode: str, tail: str, seed: int, *, check: bool = True) -> dict[str, Any]:
        start = perf_counter()
        lines = tail.splitlines()
        assert self.process.stdin is not None
        self.process.stdin.write(
            f"{mode} {seed} {len(lines)} {int(check)}\n" + "".join(line + "\n" for line in lines)
        )
        self.process.stdin.flush()
        result = self.read()
        result["roundtrip_seconds"] = perf_counter() - start
        return result

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.directory.cleanup()

    def __enter__(self) -> TraceWorker:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

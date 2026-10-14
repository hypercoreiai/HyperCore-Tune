"""HyperCore-Tune/engine_runner.py - boot the engine once, measure one request.

The engine is not an HTTP server: with --serve it reads "GEN <max_new> <token ids>" on
stdin and answers on stdout (src/program/generate.cpp:9998). The telemetry we need is in
two lines it prints:

    INFO logit_bias=1 context=... kv=... pool_workers=... pcie_frac=...   what it applied
    DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <drafts accepted>
         <drafts offered> <reused> <hits> <lookups> <ram blobs> <file blobs> <file MB>
         <prompt tokens read> <offloaded>                        src/program/generate.cpp:12205

Everything is parsed with an explicit field table rather than a loose regex, so a change in
the DONE line raises instead of silently returning a wrong number.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DONE_MIN_FIELDS = 6

# index -> name, for the fields we read. Positions past 15 are the --lookup-chain tail.
DONE_FIELDS: dict[int, str] = {
    1: "generated", 2: "prompt_tokens", 3: "prompt_ms", 4: "decode_ms", 5: "finish",
    6: "drafts_accepted", 7: "drafts_offered", 8: "reused", 14: "prompt_read",
}


class EngineError(RuntimeError):
    """The engine could not be started, or produced nothing we can measure."""


@dataclass
class Metrics:
    """One measured configuration. Rates are tok/s; latency is the request wall time."""

    status: str = "FAIL"
    gen_tok_s: float = 0.0
    prefill_tok_s: float = 0.0
    latency_ms: float = 0.0
    prompt_ms: float = 0.0
    decode_ms: float = 0.0
    prompt_tokens: int = 0
    prompt_read: int = 0
    generated: int = 0
    finish: str = ""
    effective: dict[str, str] = field(default_factory=dict)
    log_path: str = ""
    wall_ms: float = 0.0
    note: str = ""

    def ok(self) -> bool:
        return self.status == "PASS"


def parse_done(line: str) -> dict[str, Any]:
    """Split a DONE line into named fields. Raises EngineError on a malformed line."""
    parts = line.split()
    if len(parts) < DONE_MIN_FIELDS:
        raise EngineError(f"malformed DONE line ({len(parts)} fields): {line.strip()}")
    out: dict[str, Any] = {}
    for idx, name in DONE_FIELDS.items():
        if idx < len(parts):
            out[name] = parts[idx]
    try:
        out["generated"] = int(out["generated"])
        out["prompt_tokens"] = int(out["prompt_tokens"])
        out["prompt_ms"] = float(out["prompt_ms"])
        out["decode_ms"] = float(out["decode_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EngineError(f"unreadable DONE line: {line.strip()}") from exc
    return out


def parse_info(line: str) -> dict[str, str]:
    """key=value pairs from the INFO line, so a degraded arm is visible in the record."""
    return {k.split("=")[0]: k.split("=", 1)[1] for k in line.split()[1:] if "=" in k}


def failure_reason(log_text: str) -> str:
    """The engine's own last complaint, used to label a failed row."""
    for needle, label in (
        ("does not fit", "NO_VRAM_FOR_MTP: the draft head does not fit at this --vram-reserve-mib"),
        ("out of memory", "NO_VRAM: a device allocation failed"),
        ("mlock failed", "PLE table could not be pinned; the ram arm degrades to mmap"),
    ):
        if needle in log_text:
            return label
    for line in reversed(log_text.splitlines()):
        if line.startswith(("hypercore ", "error")):
            return line.strip()
    return "the engine exited with no message"


def build_prompt(prompt_file: str, tokens: int) -> list[int]:
    """A deterministic prompt of `tokens` ids, cycled from the bench token file."""
    ids = json.loads(Path(prompt_file).read_text(encoding="utf-8"))["input"]
    if not ids:
        raise EngineError(f"prompt file {prompt_file} carries no token ids")
    out: list[int] = []
    while len(out) < tokens:
        out.extend(ids)
    return out[:tokens]


def run_config(
    engine: str,
    args: list[str],
    prompt_ids: list[int],
    max_new: int,
    log_path: Path,
    boot_timeout: int = 300,
    run_timeout: int = 180,
    extra_env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> Metrics:
    """Boot the engine with `args`, send one GEN request, return its metrics.

    A boot that never reaches "session is up", a process that exits, a request that produces
    no tokens, and a timeout are each recorded as their own status rather than as a crash:
    the optimizer needs a value for every point it tried. `engine` and `log_path` are
    resolved against the current directory before the process starts, so a relative value
    still works when `cwd` is the Strata root.
    """
    engine = str(Path(engine).expanduser().resolve())
    log_path = Path(log_path).expanduser().resolve()
    metrics = Metrics(log_path=str(log_path))
    t_start = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    env = dict(extra_env or {})
    cmd = [engine] + args
    # A .py stand-in is launched with this interpreter: Windows cannot exec a shebang.
    if engine.lower().endswith(".py"):
        cmd = [sys.executable, engine] + args
    popen_kw: dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "bufsize": 0,
        "env": env or None,
    }
    if cwd is not None:
        popen_kw["cwd"] = str(Path(cwd).expanduser().resolve())
    try:
        # Binary pipes on purpose. bufsize=0 gives an unbuffered raw handle for stdin, so
        # there is no TextIOWrapper whose flush-at-GC can raise BrokenPipeError after the
        # engine has exited - the "Exception ignored while finalizing file" noise.
        proc = subprocess.Popen(cmd, **popen_kw)
    except OSError as exc:
        metrics.status = "FAIL_EXEC"
        metrics.note = f"could not start {engine}: {exc}"
        return metrics

    raw_out, raw_in = proc.stdout, proc.stdin
    if raw_out is None:
        metrics.status = "FAIL_EXEC"
        metrics.note = "the engine has no stdout to read"
        return metrics

    def send(line: str) -> bool:
        """Write one request line to the engine. False if its stdin is already gone."""
        if raw_in is None:
            return False
        try:
            raw_in.write(line.encode("utf-8"))
            raw_in.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    # A reader thread, not select(): Windows select() only accepts sockets, not pipes.
    fd = raw_out.fileno()
    chunks: queue.Queue[bytes | None] = queue.Queue()

    def reader() -> None:
        try:
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                chunks.put(chunk)
        except OSError:
            pass
        finally:
            chunks.put(None)

    thread = threading.Thread(target=reader, name="engine-stdout", daemon=True)
    thread.start()
    pending = b""
    lines: list[str] = []
    done: dict[str, Any] | None = None
    boot_deadline = t_start + boot_timeout
    run_deadline = boot_deadline + run_timeout
    sent = False
    eof = False

    def handle(line: str) -> None:
        nonlocal sent, done
        lines.append(line)
        if line.startswith("INFO "):
            metrics.effective = parse_info(line)
        if not sent and "session is up" in line:
            sent = send(f"GEN {max_new} {','.join(map(str, prompt_ids))}\n")
        if line.startswith("DONE "):
            done = parse_done(line)

    try:
        while time.monotonic() < run_deadline and not eof:
            try:
                chunk = chunks.get(timeout=0.5)
            except queue.Empty:
                if not sent and time.monotonic() > boot_deadline:
                    break
                continue
            if chunk is None:
                eof = True
                break
            pending += chunk
            while b"\n" in pending:
                raw_line, pending = pending.split(b"\n", 1)
                handle(raw_line.decode("utf-8", "replace"))
                if done is not None:
                    break
            if done is not None:
                break
    except (BrokenPipeError, ValueError) as exc:
        metrics.note = f"the engine closed its output: {exc}"
    finally:
        # QUIT only while the engine is alive; a dead engine's stdin raises EPIPE.
        if proc.poll() is None:
            send("QUIT\n")
        for pipe in (raw_out, raw_in):
            if pipe is not None:
                try:
                    pipe.close()
                except (BrokenPipeError, ValueError, OSError):
                    pass
        if proc.poll() is None:
            proc.terminate()
            deadline = time.monotonic() + 15
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.5)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        thread.join(timeout=1)

    log_text = "\n".join(lines)
    log_path.write_text(log_text, encoding="utf-8")
    metrics.wall_ms = (time.monotonic() - t_start) * 1000.0

    if done is None:
        metrics.status = "FAIL_BOOT" if not sent else "TIMEOUT"
        metrics.note = failure_reason(log_text)
        return metrics

    prompt_ms = float(done["prompt_ms"])
    decode_ms = float(done["decode_ms"])
    generated = int(done["generated"])
    prompt_tokens = int(done["prompt_tokens"])
    read_n = int(done.get("prompt_read") or 0)
    if read_n <= 0:
        read_n = max(0, prompt_tokens - int(done.get("reused") or 0))

    if generated <= 0:
        metrics.status = "NO_TOKENS"
        metrics.note = failure_reason(log_text)
        metrics.finish = str(done.get("finish", ""))
        return metrics

    metrics.status = "PASS"
    metrics.generated = generated
    metrics.prompt_tokens = prompt_tokens
    metrics.prompt_read = read_n
    metrics.prompt_ms = prompt_ms
    metrics.decode_ms = decode_ms
    metrics.finish = str(done.get("finish", ""))
    metrics.gen_tok_s = 1000.0 * generated / decode_ms if decode_ms > 0 else 0.0
    metrics.prefill_tok_s = 1000.0 * read_n / prompt_ms if prompt_ms > 0 else 0.0
    metrics.latency_ms = prompt_ms + decode_ms
    return metrics

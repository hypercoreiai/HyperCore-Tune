#!/usr/bin/env python3
"""HyperCore-Tune/stub_engine.py - a deterministic stand-in for engine/hypercore.

Speaks the same protocol as the real engine with --serve: it prints an INFO line, says
"session is up", reads "GEN <max_new> <token ids>" on stdin, answers with a DONE line,
and stays up until QUIT. The numbers are a smooth function of the swept flags with a
single interior optimum, which is what a Bayesian sampler needs to show convergence.

    ./HyperCore-Tune/stub_engine.py --serve --max-context 131072 --kv q4_0 --pool-workers 8
"""
from __future__ import annotations

import sys
import time


def parse_flags(argv: list[str]) -> dict[str, str]:
    flags: dict[str, str] = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            flags[argv[i]] = argv[i + 1]
            i += 2
        elif argv[i].startswith("--"):
            flags[argv[i]] = "1"
            i += 1
        else:
            i += 1
    return flags


def main() -> int:
    flags = parse_flags(sys.argv[1:])
    delay = float(flags.get("--delay", "0.02"))
    context = int(flags.get("--max-context", "4096"))
    vram = int(flags.get("--vram-reserve-mib", "512"))
    pcie = float(flags.get("--pcie-frac", "0.95"))
    kv = flags.get("--kv", "q4_0")
    ple = flags.get("--ple-io", "direct")
    host = flags.get("--host-core", "first")
    affinity = flags.get("--pool-affinity", "all")
    workers = int(flags.get("--pool-workers", "0")) or 16
    kv_grow = flags.get("--kv-grow", "0")
    tail = flags.get("--prompt-cache-tail", "0")
    draft = float(flags.get("--spec-min-p", "0.8"))

    if vram < 256 or vram > 2048:
        print("hypercore error: vram reserve out of range")
        return 1

    kv_gain = {"k8v4": 1.0, "q4_0": 0.85, "int8": 0.6, "fp16": 0.3}.get(kv, 0.5)
    gen = 110.0
    gen += 45.0 * max(0.0, min(1.0, pcie))
    gen += 22.0 * kv_gain
    gen += 6.0 * min(workers, 16)
    if host == "first":
        gen -= 18.0
    if affinity == "auto":
        gen += 4.0
    if kv_grow == "1":
        gen += 3.0
    if vram < 512:
        gen -= 25.0
    gen += 8.0 * max(0.0, 1.0 - abs(draft - 0.5) / 0.5)
    gen = max(20.0, gen)

    prefill = 900.0 * (1.15 if ple == "ram" else 1.0 if ple == "direct" else 0.7)
    prefill *= 1.0 + 0.05 * min(context, 131072) / 131072
    if tail == "1":
        prefill *= 1.08

    print(f"INFO context={context} kv={kv} pool_workers={workers} pcie_frac={pcie:.2f} "
          f"vram_free_mib={24000 - vram} engine=stub")
    print("session is up")
    sys.stdout.flush()

    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] != "GEN":
            continue
        max_new = int(parts[1]) if len(parts) > 1 else 256
        prompt_tokens = len(parts[2].split(",")) if len(parts) > 2 else 0
        time.sleep(delay)
        prompt_ms = 1000.0 * prompt_tokens / prefill
        decode_ms = 1000.0 * max_new / gen
        print(f"DONE {max_new} {prompt_tokens} {prompt_ms:.1f} {decode_ms:.1f} length "
              f"0 0 0 0 0 0 0 0.0 {prompt_tokens} 0")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

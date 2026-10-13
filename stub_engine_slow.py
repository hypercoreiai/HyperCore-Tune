#!/usr/bin/env python3
"""HyperCore-Tune/stub_engine_slow.py - a stub that stays up, like the real --serve engine.

Reads "GEN <max_new> <ids>" on stdin and answers with a DONE line, taking --delay seconds
per request. Used by the tests that need a run long enough to interrupt, and by the test
that starts, status-checks and stops the generated launcher.
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    argv = sys.argv[1:]
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

    delay = float(flags.get("--delay", "0.5"))
    context = flags.get("--max-context", "4096")
    print(f"INFO context={context} kv=k8v4 pool_workers=8 engine=stub-slow")
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
        max_new = int(parts[1])
        prompt_tokens = len(parts[2].split(",")) if len(parts) > 2 else 0
        time.sleep(delay)
        decode_ms = 1000.0 * max_new / 150.0
        print(f"DONE {max_new} {prompt_tokens} {1000.0 * prompt_tokens / 1000.0:.1f} "
              f"{decode_ms:.1f} length 0 0 0 0 0 0 0 0.0 {prompt_tokens} 0")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

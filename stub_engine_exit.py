#!/usr/bin/env python3
"""optuna_tune/stub_engine_exit.py - prints INFO, then exits without ever serving.

This is the shape of a real boot failure (the draft head does not fit, an allocation
fails), and it is the case that used to raise BrokenPipeError when the request pipe was
finalised after the engine had gone.
"""
import sys

flags = sys.argv[1:]
print("INFO context=4096 kv=q4_0 engine=stub-exit")
print("hypercore error: the draft head does not fit")
sys.stdout.flush()
raise SystemExit(1)

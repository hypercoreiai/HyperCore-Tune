#!/usr/bin/env python3
"""HyperCore-Tune/strata_env.py - where this sweep finds the Strata project it tunes.

HyperCore-Tune is a standalone repository that is cloned either inside a Strata checkout
(<Strata>/HyperCore-Tune) or next to one (<home>/HyperCore-Tune beside <home>/Strata).
Every path handed to the engine, the server or the deliverables is absolute, so a sweep
resolves the same tree the installer used on that machine, whatever the working directory
was when it started.

Resolution order for the project root:

    1. --strata-root, $STRATA_ROOT, $HYPERCORE_ROOT
    2. the nearest ancestor of this repository that is a Strata checkout
    3. a sibling of this repository named Strata / HyperCore / strata

A directory counts as a Strata root when it has serve/server.py and one of setup.py,
CMakeLists.txt or src/program/generate.cpp. Nothing here probes the network or the GPU.
"""
from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Names a Strata checkout uses. The old HyperCore names are still accepted so a clone
# beside an older install keeps working.
ROOT_NAMES: tuple[str, ...] = ("Strata", "HyperCore", "hypercore", "strata")
ROOT_MARKERS: tuple[str, ...] = ("serve/server.py",)
ROOT_HINTS: tuple[str, ...] = ("setup.py", "CMakeLists.txt", "src/program/generate.cpp")

# Engine binaries, newest project name first. A ready-made or CUDA12 build is a valid
# target for a sweep; a build directory is only used when it holds an executable.
ENGINE_CANDIDATES: tuple[str, ...] = (
    "engine/strata", "engine/strata.exe", "engine/hypercore", "engine/hypercore.exe",
    "engine-cuda12/strata", "engine-cuda12/hypercore",
    "build/strata", "build/hypercore", "build-release/strata", "dist/strata",
)

SETTINGS_DIRS: tuple[str, ...] = ("strata", "hypercore")
DATA_ENV: tuple[str, ...] = ("STRATA_DATA", "HYPERCORE_DATA")
SERVER_ENGINE_ENV: tuple[str, ...] = ("STRATA_SERVER_ENGINE", "HYPERCORE_SERVER_ENGINE")
ROOT_ENV: tuple[str, ...] = ("STRATA_ROOT", "HYPERCORE_ROOT", "TUNE_ROOT")
ENGINE_ENV: tuple[str, ...] = ("STRATA_ENGINE", "HYPERCORE_ENGINE", "HC_ENGINE")
HARDWARE_ENV: tuple[str, ...] = ("TUNE_HARDWARE", "HC_TUNE_HARDWARE")


def _is_strata_root(path: Path) -> bool:
    """A directory is the project root when the server is in it and the source is beside it."""
    if not path.is_dir():
        return False
    if not any((path / marker).is_file() for marker in ROOT_MARKERS):
        return False
    return any((path / hint).is_file() for hint in ROOT_HINTS)


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p is not None and _is_strata_root(p):
            return p
    return None


def candidate_roots() -> list[Path]:
    """Every root this repository could reasonably mean, best first. Never raises."""
    env = [os.environ.get(name) for name in ROOT_ENV]
    pinned = [Path(v) for v in env if v]

    ancestors: list[Path] = list(REPO.parents)  # <Strata>/HyperCore-Tune -> <Strata>

    siblings: list[Path] = []
    for base in dict.fromkeys((REPO.parent, REPO.parent.parent)):
        try:
            entries = sorted(base.iterdir())
        except OSError:
            continue
        for d in entries:
            if d.name in ROOT_NAMES or d.name.lower() in {n.lower() for n in ROOT_NAMES}:
                siblings.append(d)

    ordered: list[Path] = []
    for p in pinned + ancestors + siblings:
        if p is None:
            continue
        p = p.resolve()
        if p not in ordered:
            ordered.append(p)
    return ordered


@lru_cache(maxsize=1)
def strata_root() -> Path:
    """The absolute Strata root this sweep runs against. Raises with what it looked for."""
    root = _first_existing(candidate_roots())
    if root is None:
        looked = ", ".join(str(p) for p in candidate_roots()) or "nothing"
        raise SystemExit(
            "HyperCore-Tune: no Strata checkout found. Clone this repository inside a "
            "Strata tree, or point at one with --strata-root or STRATA_ROOT. "
            f"Looked in: {looked}"
        )
    return root


def engine_binary(root: Path | None = None) -> str:
    """The absolute path of the engine this project built. "" when there is no engine yet."""
    root = root or strata_root()
    pinned = next((os.environ.get(name) for name in ENGINE_ENV if os.environ.get(name)), "")
    if pinned:
        return str(Path(pinned).resolve())
    for rel in ENGINE_CANDIDATES:
        cand = root / rel
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return ""


def server_script(root: Path | None = None) -> str:
    """serve/server.py of the project, as an absolute path."""
    root = root or strata_root()
    return str(root / "serve" / "server.py")


def venv_python(root: Path | None = None) -> str:
    """The project venv's interpreter, or "" if it has not been created."""
    root = root or strata_root()
    for rel in (".venv/bin/python", ".venv/bin/python3", "venv/bin/python",
                ".venv/Scripts/python.exe"):
        cand = root / rel
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return ""


@lru_cache(maxsize=8)
def server_flags(root: Path) -> frozenset[str]:
    """The --flags serve/server.py defines, read from its own argparse block.

    Empty when the file cannot be read. Cached per root; the server is not going to
    change while a sweep is running.
    """
    try:
        text = (root / "serve" / "server.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return frozenset()
    return frozenset(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9-]+)"', text))


def server_supports(flag: str, root: Path | None = None) -> bool:
    """True when this project's server accepts the flag.

    A deliverable must not carry a key or a flag the local server does not know: the
    launcher would fail at startup. Feature names move between releases, so the server's
    own source is the only reliable answer.
    """
    root = root or strata_root()
    return flag in server_flags(root)


def server_engine_name(root: Path | None = None) -> str:
    """The backend name serve/server.py accepts for the real engine.

    Read from the server's own --engine choices, so a project that renames its backend is
    followed rather than assumed. Falls back to "strata" when the server cannot be read.
    """
    root = root or strata_root()
    pinned = next((os.environ.get(name) for name in SERVER_ENGINE_ENV if os.environ.get(name)), "")
    if pinned:
        return pinned
    try:
        text = (root / "serve" / "server.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "strata"
    for line in text.splitlines():
        if '"--engine"' in line and "choices" in line:
            names = re.findall(r'["\']([^"\']+)["\']', line.split("choices", 1)[1])
            real = [n for n in names if n and n != "mock"]
            if real:
                return real[0]
    return "strata"


def settings_path() -> Path:
    """The installer's settings file: ~/.config/strata/settings.json, then hypercore's."""
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    for name in SETTINGS_DIRS:
        p = base / name / "settings.json"
        if p.is_file():
            return p
    return base / SETTINGS_DIRS[0] / "settings.json"


def describe() -> dict[str, str]:
    """A small, printable summary of what was resolved. Used by --show-project and the tests."""
    root = strata_root()
    return {
        "repo": str(REPO),
        "root": str(root),
        "engine": engine_binary(root),
        "server": server_script(root),
        "server_engine": server_engine_name(root),
        "server_flags": sorted(server_flags(root)),
        "venv": venv_python(root),
        "settings": str(settings_path()),
    }


if __name__ == "__main__":  # pragma: no cover - a one-line diagnostic
    import json

    print(json.dumps(describe(), indent=1))
    sys.exit(0)

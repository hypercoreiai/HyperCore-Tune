#!/usr/bin/env python3
"""HyperCore-Tune/strata_env.py - where this sweep finds the Strata project it tunes.

HyperCore-Tune is a standalone app (also cloned as HyperCore-CPP) that is compatible with
Strata and has no other association with it. It is cloned either inside a Strata checkout
(<Strata>/HyperCore-Tune) or on its own (<home>/HyperCore-CPP beside <home>/Strata).
Every path handed to the engine, the server or the deliverables is absolute, so a sweep
resolves the same tree on Windows 10/11 and on Linux, whatever the working directory was
when it started. ~ and %VAR% in a user path are expanded first.

Resolution order for the Strata checkout this app talks to:

    1. --strata-root, $STRATA_ROOT, $HYPERCORE_ROOT, $TUNE_ROOT
    2. the nearest ancestor of this repository that is a Strata checkout
    3. a sibling of this repository named Strata / HyperCore / strata
    4. any other sibling that looks like a Strata checkout

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

# Names a Strata checkout uses. The old HyperCore name is still accepted so a clone
# beside an older install keeps working. HyperCore-CPP is this app, not a Strata root.
ROOT_NAMES: tuple[str, ...] = ("Strata", "HyperCore", "hypercore", "strata")
ROOT_MARKERS: tuple[str, ...] = ("serve/server.py",)
ROOT_HINTS: tuple[str, ...] = ("setup.py", "CMakeLists.txt", "src/program/generate.cpp")

# Engine binaries, newest project name first. A ready-made or CUDA12 build is a valid
# target for a sweep; a build directory is only used when it holds an executable.
ENGINE_CANDIDATES: tuple[str, ...] = (
    "engine/strata", "engine/strata.exe", "engine/hypercore", "engine/hypercore.exe",
    "engine-cuda12/strata", "engine-cuda12/strata.exe",
    "engine-cuda12/hypercore", "engine-cuda12/hypercore.exe",
    "build/strata", "build/strata.exe", "build/hypercore", "build/hypercore.exe",
    "build-release/strata", "build-release/strata.exe",
    "dist/strata", "dist/strata.exe",
)
_VENV_PYTHONS: tuple[str, ...] = (
    ".venv/Scripts/python.exe", "venv/Scripts/python.exe",
    ".venv/bin/python", ".venv/bin/python3", "venv/bin/python",
)

SETTINGS_DIRS: tuple[str, ...] = ("strata", "hypercore")
DATA_ENV: tuple[str, ...] = ("STRATA_DATA", "HYPERCORE_DATA")
SERVER_ENGINE_ENV: tuple[str, ...] = ("STRATA_SERVER_ENGINE", "HYPERCORE_SERVER_ENGINE")
ROOT_ENV: tuple[str, ...] = ("STRATA_ROOT", "HYPERCORE_ROOT", "TUNE_ROOT")
ENGINE_ENV: tuple[str, ...] = ("STRATA_ENGINE", "HYPERCORE_ENGINE", "HC_ENGINE")
HARDWARE_ENV: tuple[str, ...] = ("TUNE_HARDWARE", "HC_TUNE_HARDWARE")


def abs_path(path: str | Path | None) -> Path | None:
    """An absolute path, with ~ and %VAR% expanded. None when the value is missing or blank.

    Path.resolve() does not expand ~, so '~/HyperCore-CPP' would otherwise be read as
    a directory named '~' under the current working directory. On Windows, %USERPROFILE%
    and %STRATA_ROOT% are expanded the same way. Every caller that takes a user, env or
    settings path goes through here.
    """
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    text = os.path.expandvars(text)
    text = re.sub(r"%([A-Za-z0-9_]+)%",
                  lambda m: os.environ.get(m.group(1), m.group(0)), text)
    # A Windows path copied into a Linux test (or a .bat) uses backslashes.
    text = text.replace("\\", "/")
    return Path(text).expanduser().resolve()


def _is_runnable(path: Path) -> bool:
    """True when path is a file this process can launch. Windows has no Unix execute bit."""
    if not path.is_file():
        return False
    if os.name == "nt":
        return True
    return os.access(path, os.X_OK)


def config_bases() -> list[Path]:
    """Folders that may hold strata/settings.json: $XDG_CONFIG_HOME, %APPDATA%, ~/.config."""
    ordered: list[Path] = []
    for raw in (os.environ.get("XDG_CONFIG_HOME"), os.environ.get("APPDATA")):
        p = abs_path(raw)
        if p is not None and p not in ordered:
            ordered.append(p)
    fallback = Path.home() / ".config"
    if fallback not in ordered:
        ordered.append(fallback)
    return ordered


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


def _sibling_dirs(base: Path) -> list[Path]:
    try:
        return sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return []


def candidate_roots() -> list[Path]:
    """Every root this repository could reasonably mean, best first. Never raises."""
    pinned = [abs_path(os.environ.get(name)) for name in ROOT_ENV]

    ancestors: list[Path] = list(REPO.parents)  # <Strata>/HyperCore-Tune -> <Strata>

    named: list[Path] = []
    markers: list[Path] = []
    names = {n.lower() for n in ROOT_NAMES}
    parent, grand = REPO.parent, REPO.parent.parent
    for d in _sibling_dirs(parent):
        if d.resolve() == REPO:
            continue
        if d.name.lower() in names:
            named.append(d)
        elif _is_strata_root(d):
            markers.append(d)
    for d in _sibling_dirs(grand):
        if d.name.lower() in names:
            named.append(d)

    ordered: list[Path] = []
    for p in pinned + ancestors + named + markers:
        if p is None:
            continue
        p = abs_path(p) or p.resolve()
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
        found = abs_path(pinned)
        return str(found) if found is not None else str(Path(os.path.expandvars(pinned)).expanduser())
    for rel in ENGINE_CANDIDATES:
        cand = root / rel
        if _is_runnable(cand):
            return str(cand)
    return ""


def server_script(root: Path | None = None) -> str:
    """serve/server.py of the project, as an absolute path."""
    root = root or strata_root()
    return str(root / "serve" / "server.py")


def venv_python(root: Path | None = None) -> str:
    """The project venv's interpreter, or "" if it has not been created."""
    root = root or strata_root()
    for rel in _VENV_PYTHONS:
        cand = root / rel
        if _is_runnable(cand):
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
    """The installer's settings file.

    Linux: ~/.config/strata/settings.json (or $XDG_CONFIG_HOME). Windows: the same
    folder under %APPDATA%, then ~/.config. The older hypercore name is tried second.
    """
    bases = config_bases()
    for base in bases:
        for name in SETTINGS_DIRS:
            p = base / name / "settings.json"
            if p.is_file():
                return p
    return bases[0] / SETTINGS_DIRS[0] / "settings.json"


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

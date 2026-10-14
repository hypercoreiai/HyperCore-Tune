"""HyperCore-Tune/run_config.py - the sweep's winner as a run config the server can read.

Deliverables, in the project's own format (the same pair setup.py writes for a model):

    max_run_<model>.json   a run config: the keys setup.py owns (exe, args, cwd, tokenizer,
                           model_name, log, lib_dirs, port, gpu, host, api_key) with the
                           winning engine arguments in "args", plus turn_tail_check only
                           when this project's server has that flag
    max_run_<model>.sh     the launcher setup.py writes: serve/server.py with that config
    max_run_<model>.bat    the same launcher for a Windows install of Strata

Only "args" is replaced. Everything the model's existing config carries that setup does not
own (sampling, reasoning, mcp_servers, ...) is kept, so the deliverable is a drop-in startup
pair rather than a tuning dump. The full trial record - every configuration measured and its
metrics - stays in HyperCore-Tune/results/max_run_<model>.record.json.

setup.py is imported when it is available so the defaults, the run flags and the atomic write
are the ones the real installer uses; a stand-in is used when it cannot be imported. The
project root comes from strata_env, so every path written here is absolute and belongs to
the Strata checkout on the machine running the sweep.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strata_env import (
    REPO,
    abs_path,
    engine_binary,
    server_engine_name,
    server_script,
    server_supports,
    strata_root,
    venv_python,
)

# Flags that exist only because the sweep measures one request in one process. They are not
# part of a production run config: the server owns the process, the request length is
# whatever the client asks for, and --prompt-cache-tail is not a flag the default Strata
# interface accepts. --spec-min-p (draft-floor) stays: Strata does accept that one.
SWEEP_ONLY: tuple[tuple[str, int], ...] = (
    ("--serve", 0),
    ("--max-new", 1),
    ("--expert-profile-save-every", 1),
    # Not a flag the default Strata server/run-config accepts.
    ("--prompt-cache-tail", 0),
)

# Keys a file must have to be treated as a run config rather than a tuning record.
RUN_CONFIG_MARKERS = ("exe", "args", "cwd")


def _setup_module() -> Any | None:
    """The project's setup.py, or None if it cannot be imported."""
    root = str(strata_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import setup  # the installer module, resolved at call time
    except Exception as exc:  # noqa: BLE001 - a broken or partial install must not lose a finished sweep
        print(f"note: setup.py not importable ({exc}); using built-in defaults", file=sys.stderr)
        return None
    return setup


def _cuda_lib_dirs() -> list[str]:
    """The CUDA bin/lib directories actually present, the way the run configs list them."""
    out: list[str] = []
    seen: set[str] = set()

    def add(bin_dir: Path, lib_dir: Path) -> None:
        if not lib_dir.is_dir():
            return
        for p in (bin_dir, lib_dir):
            s = str(p)
            if s not in seen:
                seen.add(s)
                out.append(s)

    for cand in sorted(Path("/usr/local").glob("cuda*")):
        add(cand / "bin", cand / "lib64")
    cuda_root = abs_path(os.environ.get("CUDA_PATH"))
    if cuda_root is not None:
        add(cuda_root / "bin", cuda_root / "lib" / "x64")
        add(cuda_root / "bin", cuda_root / "lib64")
    toolkit = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit.is_dir():
        for cand in sorted(toolkit.glob("v*")):
            add(cand / "bin", cand / "lib" / "x64")
    return out


def clean_args(args: list[str]) -> list[str]:
    """The engine arguments without the sweep-only flags, order preserved."""
    out: list[str] = []
    i = 0
    while i < len(args):
        flag = args[i]
        skip = next((n for name, n in SWEEP_ONLY if name == flag), None)
        if skip is not None:
            i += 1 + skip
            continue
        out.append(flag)
        i += 1
    return out


def find_base_config(model: str, root: Path) -> dict[str, Any] | None:
    """The model's current run config, if there is one.

    Preferred order: the installed model config, then a deliverable from an earlier sweep.
    A file is only accepted when it has the run-config shape, so a tuning record is never
    mistaken for a config.
    """
    quant = model.lower()
    for name in (f"strata-{quant}.json", f"ni_hypercore-{quant}.json",
                 f"hypercore-{quant}.json", f"max_run_{quant}.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict) and all(k in cfg for k in RUN_CONFIG_MARKERS):
            return cfg
    return None


def build_run_config(
    model: str,
    paths: dict[str, str],
    args: list[str],
    record: dict[str, Any],
    base: dict[str, Any] | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    """The run config for the winning configuration.

    `base` is the model's existing config; its non-args keys survive untouched. `paths` is
    ModelPaths.as_dict(), used only when the base has nothing to say about a path.
    """
    cfg: dict[str, Any] = dict(base) if base else {}
    setup = _setup_module()
    root = strata_root()

    engine = str(record.get("engine") or engine_binary(root) or root / "engine" / "strata")
    cfg["exe"] = str(abs_path(engine) or Path(engine).expanduser().resolve())
    cfg["args"] = clean_args(args)
    cfg["cwd"] = str(root)

    # The tokenizer has to match the pack the args point at. A base config left over from
    # another account's data folder is rewritten when this sweep's pack carries its own.
    tokenizer = str(cfg.get("tokenizer") or "")
    beside_pack = Path(paths["pack"]) / "tokenizer" if paths.get("pack") else None
    if beside_pack is not None and (beside_pack.is_dir() or beside_pack.is_file()):
        if tokenizer and Path(tokenizer).expanduser().resolve() != beside_pack.resolve():
            print(f"note: {tokenizer} is not beside the pack in use ({paths['pack']}); "
                  f"using {beside_pack}", file=sys.stderr)
        tokenizer = str(beside_pack.resolve())
    elif tokenizer:
        found = abs_path(tokenizer)
        tokenizer = str(found) if found is not None else tokenizer
    if tokenizer:
        cfg["tokenizer"] = tokenizer

    if not cfg.get("model_name"):
        cfg["model_name"] = f"strata-{model.lower()}"

    cfg["log"] = str(root / f"max_run_{model.lower()}.log")

    if not cfg.get("lib_dirs"):
        libs = _cuda_lib_dirs()
        if libs:
            cfg["lib_dirs"] = libs

    if port is not None:
        cfg["port"] = int(port)
    # A key the local server does not read is not a default, it is a dead key: the
    # launcher would pass a flag argparse rejects. turn_tail_check is one of these - it
    # belongs to the older HyperCore server, and Strata's server has no such flag.
    tail_flag = "--turn-tail-check"
    tail_supported = server_supports(tail_flag, root)
    if hasattr(setup, "apply_server_defaults"):
        # The installer's own precedence: what this run asked for, then the model's config,
        # then the default. A key someone set by hand is never replaced.
        setup.apply_server_defaults(cfg, base)
    else:
        def default(name: str, fallback: str) -> str:
            for env in (f"STRATA_DEFAULT_{name}", f"HYPERCORE_DEFAULT_{name}"):
                if os.environ.get(env):
                    return os.environ[env]
            return fallback
        fallback: dict[str, Any] = {
            "host": default("HOST", "0.0.0.0"),
            "api_key": default("API_KEY", "0000"),
            "port": int(default("PORT", "8081")),
        }
        if tail_supported:
            fallback["turn_tail_check"] = default("TURN_TAIL_CHECK", "1") not in ("0", "", "false")
        for key, default in fallback.items():
            if cfg.get(key) is None:
                cfg[key] = default
    cfg["port"] = int(cfg["port"])
    if tail_supported:
        cfg["turn_tail_check"] = bool(cfg.get("turn_tail_check", False))
    else:
        cfg.pop("turn_tail_check", None)
    cfg.setdefault("gpu", 0)

    cfg["tuned_by"] = {
        "schema": "hypercore.max_run/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "objective": record.get("objective"),
        "context": record.get("context"),
        "hardware": record.get("hardware", {}),
        "hardware_slug": record.get("hardware_slug", ""),
        "best": record.get("best", {}),
        "verified": bool(record.get("verified", False)),
        # history | dry_run | study: who produced the winning numbers.
        "source": record.get("source", "study"),
        "metrics": record.get("metrics", {}),
        "study": record.get("study", {}),
        "history": record.get("history", {}),
        "csv": record.get("csv"),
    }
    return cfg


def _write_json(path: Path, cfg: dict[str, Any]) -> Path:
    """Whole or not at all, the way setup.py writes a run config."""
    setup = _setup_module()
    if setup is not None:
        setup.write_config(path, cfg)
        return path
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def serve_command(
    cfg_path: Path,
    engine: str,
    port: int,
    open_browser: bool,
    turn_tail_check: bool,
    root: Path | None = None,
) -> list[str]:
    """The server invocation, built from absolute paths of the project in front of us.

    The interpreter is the project venv when it exists, so the launcher runs with the
    packages setup.py installed rather than whatever happened to start the sweep.
    """
    root = root or strata_root()
    python = venv_python(root) or sys.executable
    return [python, server_script(root), "--engine", engine,
            "--config", str(cfg_path), "--port", str(port)] + \
        (["--open"] if open_browser else []) + \
        (["--turn-tail-check"] if turn_tail_check else [])


def launcher_text(
    cfg_path: Path,
    engine: str,
    port: int,
    open_browser: bool,
    turn_tail_check: bool,
    note: tuple[str, ...],
) -> str:
    """The launcher body: the server with this config, exactly as setup.py writes it."""
    serve = serve_command(cfg_path, engine, port, open_browser, turn_tail_check)
    lines = ["#!/bin/sh"]
    lines += [f"# {line}" for line in note]
    lines.append(f'cd "{strata_root()}"')
    lines.append("exec " + " ".join(f'"{x}"' for x in serve))
    return "\n".join(lines) + "\n"


def launcher_bat(
    cfg_path: Path,
    engine: str,
    port: int,
    open_browser: bool,
    turn_tail_check: bool,
    note: tuple[str, ...],
) -> str:
    """The same launcher for a Windows install of Strata, in setup.py's .bat shape."""
    serve = serve_command(cfg_path, engine, port, open_browser, turn_tail_check)
    lines = ["@echo off", f"title Strata {cfg_path.stem}"]
    lines += [f"rem {line}" for line in note]
    lines.append('cd /d "' + str(strata_root()) + '"')
    lines.append(" ".join(f'"{x}"' for x in serve))
    lines.append("if errorlevel 1 pause")
    return "\r\n".join(lines) + "\r\n"


def _emit_pair(
    model: str,
    paths: dict[str, str],
    args: list[str],
    record: dict[str, Any],
    deliverable_dir: Path,
    port: int | None,
    open_browser: bool,
) -> tuple[Path, Path]:
    """Write the pair into deliverable_dir; raises OSError if that directory cannot be written."""
    quant = model.lower()
    root = strata_root()
    base = find_base_config(quant, root)
    cfg = build_run_config(quant, paths, args, record, base=base, port=port)
    js = _write_json(deliverable_dir / f"max_run_{quant}.json", cfg)

    note = (
        (f"max_run_{quant}.sh - generated {cfg['tuned_by']['generated_utc']} "
         f"by HyperCore-Tune/max_tune.sh"),
        (f"Bayesian (Optuna TPE) winner for the {cfg['tuned_by']['objective']} objective "
         f"at --max-context {cfg['tuned_by']['context']}"),
        (f"gen {cfg['tuned_by']['metrics'].get('gen_tok_s', 0):.2f} tok/s, "
         f"prefill {cfg['tuned_by']['metrics'].get('prefill_tok_s', 0):.2f} tok/s, "
         f"latency {cfg['tuned_by']['metrics'].get('latency_ms', 0):.1f} ms"),
        f"tuned on {cfg['tuned_by'].get('hardware_slug', 'unknown machine')}",
        ("the trial record is "
         f"{REPO / 'results' / f'max_run_{quant}.record.json'}"),
    )
    engine_name = server_engine_name(root)
    tail_on = bool(cfg.get("turn_tail_check", False))
    sh = deliverable_dir / f"max_run_{quant}.sh"
    # "--engine" is the server's backend selector (mock | strata), not the binary path:
    # the binary the worker runs is the "exe" key in the config beside the script.
    sh.write_text(
        launcher_text(js, engine_name, cfg["port"], open_browser, tail_on, note),
        encoding="utf-8",
    )
    sh.chmod(0o755)
    (deliverable_dir / f"max_run_{quant}.bat").write_text(
        launcher_bat(js, engine_name, cfg["port"], open_browser, tail_on, note),
        encoding="utf-8",
    )
    return sh, js


def write_deliverables(
    model: str,
    paths: dict[str, str],
    args: list[str],
    record: dict[str, Any],
    deliverable_dir: Path,
    port: int | None = None,
    open_browser: bool = True,
    fallback_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write max_run_<model>.sh and max_run_<model>.json; returns (script, config).

    A launcher left by an earlier sweep can belong to another user. A PermissionError on
    the deliverable directory must not throw away a finished sweep, so the pair is written
    to fallback_dir instead and the fallback is printed.
    """
    try:
        return _emit_pair(model, paths, args, record, deliverable_dir, port, open_browser)
    except OSError as exc:
        fb = Path(fallback_dir) if fallback_dir else None
        if fb is None or fb == Path(deliverable_dir):
            raise
        fb.mkdir(parents=True, exist_ok=True)
        print(f"note: cannot write {deliverable_dir} ({exc}); the deliverables go to {fb}",
              file=sys.stderr)
        return _emit_pair(model, paths, args, record, fb, port, open_browser)

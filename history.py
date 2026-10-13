"""HyperCore-Tune/history.py - previous sweeps as priors, keyed to the machine they ran on.

One file per (machine, model, objective, context):

    HyperCore-Tune/history/<CPU>_<GPU>_<RAM>_<PCIe>x<width>_<model>_<objective>_<cntx>.json

The file is a merge of every sweep that ran on that machine: the trials it measured,
deduplicated by configuration signature, plus a short list of the runs that produced
them. Loading it does two things:

  1. the configurations already measured on this machine are not booted again;
  2. they are added to the Optuna study as completed trials, so TPE starts from them
     instead of from a cold prior.

A record from a different CPU, GPU, RAM or PCIe generation is not loaded unless
--allow-hardware-mismatch is passed, because a pcie-frac or pool-workers winner on one
host is not a winner on another.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hardware import Hardware, differences, hardware_from, slug
from search_space import Config, Param, signature

SCHEMA = "hypercore.tuning_history/1"
CFG_KEYS: tuple[str, ...] = (
    "vram", "pcie", "ple_io", "kv", "kvgrow", "host_core", "affinity", "worker", "tail",
)


def history_path(history_dir: Path, model: str, objective: str, context: int,
                 hw: Hardware) -> Path:
    """The data file for this machine, model, objective and context length."""
    return history_dir / f"{slug(hw)}_{model.lower()}_{objective}_{context}.json"


def load(path: Path) -> dict[str, Any] | None:
    """A record, or None if the file is missing, unreadable or not a history file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        print(f"note: cannot read {path} ({exc})", file=sys.stderr)
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("trials"), list):
        return None
    return raw


def find_history(history_dir: Path, model: str, objective: str, context: int,
                 hw: Hardware, allow_mismatch: bool = False) -> list[Path]:
    """Every history file for this model/objective/context whose machine matches."""
    if not history_dir.is_dir():
        return []
    needle = f"_{model.lower()}_{objective}_{context}.json"
    found: list[Path] = []
    for path in sorted(history_dir.glob("*" + needle)):
        record = load(path)
        if record is None:
            continue
        diffs = differences(hw, hardware_from(record))
        if not diffs:
            found.append(path)
        elif allow_mismatch:
            print(f"note: {path.name} was measured on another machine "
                f"({'; '.join(diffs)}); loaded because --allow-hardware-mismatch",
                file=sys.stderr)
            found.append(path)
    return found


def normalise_trial(row: dict[str, Any], max_new: int, spec_min_p: float) -> dict[str, Any] | None:
    """A stored trial, or None if it is missing the fields a prior needs."""
    cfg_raw = row.get("cfg")
    if not isinstance(cfg_raw, dict):
        cfg_raw = {k: row[k] for k in CFG_KEYS if k in row}
    cfg: Config = {}
    for key in CFG_KEYS:
        if key not in cfg_raw:
            return None
        cfg[key] = cfg_raw[key]
    try:
        score = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        return None
    metrics = row.get("metrics") or {k: row.get(k) for k in
                                    ("gen_tok_s", "prefill_tok_s", "latency_ms",
                                     "prompt_ms", "decode_ms")}
    return {
        "cfg": cfg,
        "signature": list(signature(cfg)),
        "score": score,
        "status": str(row.get("status", "UNKNOWN")),
        "layer": str(row.get("layer", "history")),
        "metrics": {k: float(v) if v is not None else 0.0 for k, v in metrics.items()},
        "max_new": int(row.get("max_new", max_new)),
        "spec_min_p": float(row.get("spec_min_p", spec_min_p)),
        "measured_utc": str(row.get("measured_utc", "")),
        "hardware_slug": str(row.get("hardware_slug", "")),
        "measured": int(row.get("measured", 1)),
    }


def usable_trials(record: dict[str, Any], max_new: int, spec_min_p: float) -> list[dict[str, Any]]:
    """Trials whose request shape matches the sweep about to run.

    Metrics only compare against each other when the prompt and the generation length are
    the same, so a trial measured with a different --max-new or --spec-min-p is dropped
    rather than silently mixed in.
    """
    out: list[dict[str, Any]] = []
    for row in record.get("trials", []):
        if not isinstance(row, dict):
            continue
        trial = normalise_trial(row, max_new, spec_min_p)
        if trial is None:
            continue
        if trial["max_new"] != int(max_new) or abs(trial["spec_min_p"] - float(spec_min_p)) > 1e-9:
            continue
        out.append(trial)
    return out


def merge(existing: dict[str, Any] | None, record: dict[str, Any]) -> dict[str, Any]:
    """Fold one sweep into the history file: best score wins for each configuration."""
    base = existing or {
        "schema": SCHEMA,
        "model": record["model"],
        "objective": record["objective"],
        "context": record["context"],
        "hardware": record["hardware"],
        "hardware_slug": record["hardware_slug"],
        "runs": [],
        "trials": [],
    }
    best: dict[tuple[Any, ...], dict[str, Any]] = {
        tuple(t["signature"]): t for t in base.get("trials", []) if isinstance(t, dict)
    }
    for row in record.get("trials", []):
        trial = normalise_trial(row, record.get("max_new", 0), record.get("spec_min_p", 0.0))
        if trial is None:
            continue
        key = tuple(trial["signature"])
        prior = best.get(key)
        if prior is None or trial["score"] > prior["score"]:
            trial["measured"] = (prior.get("measured", 0) + 1) if prior else 1
            best[key] = trial
        else:
            prior["measured"] = prior.get("measured", 1) + 1
    base["trials"] = sorted(best.values(), key=lambda t: (-t["score"], t["signature"]))
    base["hardware"] = record["hardware"]
    base["hardware_slug"] = record["hardware_slug"]
    base["updated_utc"] = datetime.now(timezone.utc).isoformat()
    base.setdefault("runs", []).append({
        "generated_utc": record.get("generated_utc"),
        "trials": len(record.get("trials", [])),
        "measured": record.get("study", {}).get("measured", 0),
        "best": record.get("best", {}),
        "metrics": record.get("metrics", {}),
        "csv": record.get("csv"),
        "loaded_priors": record.get("history", {}).get("trials_loaded", 0),
    })
    base["runs"] = base["runs"][-50:]
    return base


def save(path: Path, record: dict[str, Any]) -> Path:
    """Write the merged history. Whole file or no file: a half-written prior is worse than none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = merge(load(path), record)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _int_distribution(low: int, high: int, step: int):
    """The integer distribution, named differently across Optuna releases."""
    import optuna.distributions as dist

    cls = (getattr(dist, "IntDistribution", None)
           or getattr(dist, "UniformIntDistribution", None)
           or getattr(dist, "IntUniformDistribution", None))
    if cls is None:  # pragma: no cover - one of these exists in every supported Optuna
        raise RuntimeError("this Optuna build has no integer distribution")
    return cls(low, high, step=step)


def _float_distribution(low: float, high: float):
    """The continuous distribution, named differently across Optuna releases."""
    import optuna.distributions as dist

    cls = (getattr(dist, "FloatDistribution", None)
           or getattr(dist, "UniformFloatDistribution", None)
           or getattr(dist, "UniformDistribution", None))
    if cls is None:  # pragma: no cover - one of these exists in every supported Optuna
        raise RuntimeError("this Optuna build has no continuous distribution")
    return cls(low, high)


def distributions_for(params: list[Param], cfg: Config) -> dict[str, Any]:
    """The Optuna distributions a stored configuration must sit inside."""
    import optuna.distributions as dist

    out: dict[str, Any] = {}
    for p in params:
        if p.key not in cfg:
            continue
        value: Any = cfg[p.key]
        if p.kind == "choice":
            if value not in p.choices:
                raise ValueError(f"{p.key}={value!r} is outside the current grid")
            out[p.key] = dist.CategoricalDistribution(choices=list(p.choices))
        elif p.kind == "int":
            low, high, step = int(p.low), int(p.high), int(p.step) or 1
            if not (low <= int(value) <= high):
                raise ValueError(f"{p.key}={value!r} is outside [{low}, {high}]")
            out[p.key] = _int_distribution(low, high, step)
        else:
            flo, fhi = float(p.low), float(p.high)
            if not (flo <= float(value) <= fhi):
                raise ValueError(f"{p.key}={value!r} is outside [{flo}, {fhi}]")
            out[p.key] = _float_distribution(flo, fhi)
    return out

#!/usr/bin/env python3
"""HyperCore-Tune/bayesian_tune.py - Bayesian (TPE) tuning of the Strata engine.

Searches the ten dimensions the brief names - vram, pcie, ple-io, kv, kvgrow, hostcore,
affinity, worker, tail - against one of three user-selected objectives:

    --prefill   maximise prompt tokens/s
    --gen       maximise generated tokens/s
    --latency   minimise prompt_ms + decode_ms for the request

Two search modes:

    joint    one study over the whole space (the Bayesian sweep proper)
    staged   one study per layer, other layers frozen at the incumbent best. `layer` in the
             CSV is then the layer under test, which is how tune/max_tune.sh rows are read.

Deliverables, written at the Strata project root as requested:

    max_run_<model>.sh     the production launcher for the winning configuration
    max_run_<model>.json   the winning configuration, its metrics, and every trial

Run data from previous sweeps is kept in HyperCore-Tune/history/, one file per machine,
model, objective and context:

    <CPU>_<GPU>_<RAM>_<PCIe>x<width>_<model>_<objective>_<cntx>.json

A sweep on the same machine loads it, never re-boots a configuration it has already
measured, and seeds the TPE sampler with those trials. --no-load starts cold; a record
from different hardware is ignored unless --allow-hardware-mismatch is given.

Run it through the project venv:  ./HyperCore-Tune/max_tune.sh --model q2_0 --gen --cntx 131072
The Strata root is found from this repository's location, or pinned with --strata-root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import RandomSampler, TPESampler

try:
    from optuna.samplers import GPSampler
except ImportError:  # GPSampler needs scipy; it is optional here.
    GPSampler = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_runner import Metrics, build_prompt, run_config
from hardware import Hardware, fingerprint, hardware_from
from history import distributions_for, find_history, history_path, usable_trials
from history import load as load_history
from history import save as save_history
from run_config import write_deliverables as write_run_pair
from search_space import (
    LAYERS,
    ROW_COLUMNS,
    Config,
    ModelPaths,
    Param,
    build_space,
    engine_args,
    physical_cores,
    resolve_model_paths,
    sample,
    signature,
)
from strata_env import engine_binary, strata_root

OBJECTIVES = ("gen", "prefill", "latency")
PENALTY = {"gen": 0.0, "prefill": 0.0, "latency": -1e9}  # every score is "higher is better"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def signed_score(m: Metrics, objective: str) -> float:
    """One number to maximise. Failed boots score at the bottom of the range."""
    if not m.ok():
        return PENALTY[objective]
    if objective == "gen":
        return m.gen_tok_s
    if objective == "prefill":
        return m.prefill_tok_s
    return -m.latency_ms


def make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    """TPE is the Bayesian sampler; GP is the Gaussian-process one (needs scipy)."""
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "gp":
        if GPSampler is None:
            print("note: --sampler gp needs scipy; falling back to TPE", file=sys.stderr)
            return TPESampler(seed=seed, n_startup_trials=10)
        return GPSampler(seed=seed)
    return TPESampler(seed=seed, n_startup_trials=10, multivariate=True)


@dataclass
class TrialRow:
    layer: str
    cfg: Config
    metrics: Metrics
    score: float
    from_history: bool = False

    def csv_row(self) -> dict[str, object]:
        m = self.metrics
        row: dict[str, object] = {
            "layer": self.layer,
            "vram": self.cfg["vram"],
            "pcie": self.cfg["pcie"],
            "ple_io": self.cfg["ple_io"],
            "kv": self.cfg["kv"],
            "kvgrow": self.cfg["kvgrow"],
            "hostcore": self.cfg["host_core"],
            "affinity": self.cfg["affinity"],
            "worker": self.cfg["worker"],
            "tail": self.cfg["tail"],
            "status": m.status,
        }
        row.update({
            "gen_tok_s": f"{m.gen_tok_s:.2f}",
            "prefill_tok_s": f"{m.prefill_tok_s:.2f}",
            "latency_ms": f"{m.latency_ms:.1f}",
            "prompt_ms": f"{m.prompt_ms:.1f}",
            "decode_ms": f"{m.decode_ms:.1f}",
            "generated": m.generated,
            "prompt_tokens": m.prompt_tokens,
            "score": f"{self.score:.4f}",
            "note": m.note,
        })
        return row

    def to_dict(self) -> dict[str, Any]:
        """The CSV row plus the configuration as a dict, so a record can be replayed."""
        row = self.csv_row()
        row["cfg"] = dict(self.cfg)
        row["signature"] = list(signature(self.cfg))
        row["from_history"] = self.from_history
        return row


class Sweeper:
    """Runs the study, keeps the incumbent best, and writes the deliverables."""

    def __init__(self, args: argparse.Namespace, root: Path, params: list[Param], paths,
                 hw: Hardware, priors: list[dict[str, Any]]) -> None:
        self.args = args
        self.root = root
        self.params = params
        self.paths = paths
        self.hw = hw
        self.priors = priors
        self.rows: list[TrialRow] = []
        self.seen: set[tuple[object, ...]] = set()
        self.prior_rows: dict[tuple[object, ...], TrialRow] = {}
        self.best_row: TrialRow | None = None
        self.priors_skipped = 0
        self.prompt_ids = build_prompt(args.prompt_tokens_file, args.prompt_tokens)
        self.model = args.model
        self._seed_from_history()

    def _seed_from_history(self) -> None:
        """Turn every prior measured on this machine into a row that never needs a boot."""
        for prior in self.priors:
            stored = prior["metrics"]
            metrics = Metrics(status="CACHED")
            metrics.gen_tok_s = float(stored.get("gen_tok_s", 0.0) or 0.0)
            metrics.prefill_tok_s = float(stored.get("prefill_tok_s", 0.0) or 0.0)
            metrics.latency_ms = float(stored.get("latency_ms", 0.0) or 0.0)
            metrics.prompt_ms = float(stored.get("prompt_ms", 0.0) or 0.0)
            metrics.decode_ms = float(stored.get("decode_ms", 0.0) or 0.0)
            metrics.generated = int(stored.get("generated", 0) or 0)
            metrics.prompt_tokens = int(stored.get("prompt_tokens", 0) or 0)
            row = TrialRow(layer="history", cfg=dict(prior["cfg"]),
                          metrics=metrics, score=float(prior["score"]),
                          from_history=True)
            sig = tuple(prior["signature"])
            self.seen.add(sig)
            self.prior_rows[sig] = row
            if self.best_row is None or row.score > self.best_row.score:
                self.best_row = row

    # ---- one configuration ----------------------------------------------------
    def evaluate(self, layer: str, cfg: Config) -> TrialRow:
        sig = signature(cfg)
        if sig in self.seen:
            # A duplicate point costs a full boot. Reuse the measurement we already have,
            # which may come from a previous sweep on this machine.
            prior = self.prior_rows.get(sig) or self.best_row
            if prior is None:
                row = TrialRow(layer, cfg, Metrics(status="DUPLICATE"),
                              PENALTY[self.args.objective])
            else:
                row = TrialRow(layer, cfg, prior.metrics, prior.score,
                              from_history=prior.from_history)
            self.rows.append(row)
            if self.best_row is None or row.score > self.best_row.score:
                self.best_row = row
            return row
        self.seen.add(sig)

        args = engine_args(cfg, self.paths.as_dict(), self.args.cntx,
                           self.args.max_new, self.args.spec_min_p)
        if self.args.dry_run:
            m = Metrics(status="DRY")
            m.note = " ".join(args)
            row = TrialRow(layer, cfg, m, PENALTY[self.args.objective])
            self.rows.append(row)
            if self.best_row is None:
                self.best_row = row
            return row

        runs: list[Metrics] = []
        for i in range(self.args.repeat):
            log = self.args.results_dir / f"run-{utc_stamp()}-{layer}-{i}.log"
            runs.append(run_config(
                engine=self.args.engine, args=args, prompt_ids=self.prompt_ids,
                max_new=self.args.max_new, log_path=log,
                boot_timeout=self.args.boot_timeout, run_timeout=self.args.run_timeout,
            ))
        # Median of the repeats: one boot is noisy enough to flip a ranking.
        best = max(runs, key=lambda m: signed_score(m, self.args.objective))
        if self.args.repeat > 1 and best.ok():
            best.gen_tok_s = statistics.median(r.gen_tok_s for r in runs if r.ok())
            best.prefill_tok_s = statistics.median(r.prefill_tok_s for r in runs if r.ok())
            best.latency_ms = statistics.median(r.latency_ms for r in runs if r.ok())

        row = TrialRow(layer, cfg, best, signed_score(best, self.args.objective))
        self.rows.append(row)
        if self.best_row is None or row.score > self.best_row.score:
            self.best_row = row
        m = row.metrics
        print(f"  trial {len(self.rows):>3} layer={layer} "
              f"vram={cfg['vram']} pcie={cfg['pcie']} ple-io={cfg['ple_io']} kv={cfg['kv']} "
              f"kvgrow={cfg['kvgrow']} hostcore={cfg['host_core']} affinity={cfg['affinity']} "
              f"worker={cfg['worker']} tail={cfg['tail']} -> {m.status} "
              f"gen {m.gen_tok_s:.2f} tok/s prefill {m.prefill_tok_s:.2f} tok/s "
              f"latency {m.latency_ms:.0f} ms" + (f"  [{m.note}]" if m.note else ""))
        sys.stdout.flush()
        return row

    # ---- the study -------------------------------------------------------------
    def run(self) -> bool:
        """Run the study. Returns False if it was interrupted by Ctrl-C."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = make_sampler(self.args.sampler, self.args.seed)
        interrupted = False

        if self.args.mode == "joint":
            study = optuna.create_study(direction="maximize", sampler=sampler)
            self._warm_study(study, self.params)
            frozen: dict[str, object] = {}

            def objective(trial) -> float:
                cfg = sample(trial, self.params, frozen)
                incumbent = self.best_row.cfg if self.best_row else {}
                changed = sorted({p.layer for p in self.params
                                 if p.key in cfg and cfg[p.key] != incumbent.get(p.key, cfg[p.key])})
                layer = "joint" if not self.best_row else (",".join(changed) or "joint")
                return self.evaluate(layer, cfg).score

            try:
                study.optimize(objective, n_trials=self.args.trials, show_progress_bar=False)
            except KeyboardInterrupt:
                interrupted = True
                print("\ninterrupted: keeping the best configuration measured so far")
        else:
            frozen = {}
            for layer in self.layers():
                layer_params = [p for p in self.params if p.layer == layer]
                if not layer_params:
                    continue
                study = optuna.create_study(direction="maximize",
                                          sampler=make_sampler(self.args.sampler, self.args.seed))
                self._warm_study(study, layer_params)

                try:
                    study.optimize(self._layer_objective(layer, layer_params, frozen),
                                   n_trials=self.args.trials_per_layer,
                                   show_progress_bar=False)
                except KeyboardInterrupt:
                    interrupted = True
                    print(f"\ninterrupted during layer {layer}; keeping what has been measured")
                    break
                if study.best_trial is not None and study.best_trial.value > PENALTY[self.args.objective]:
                    for p in layer_params:
                        frozen[p.key] = study.best_trial.params[p.key]
                    print(f"--- layer {layer}: locked {frozen} at {study.best_trial.value:.4f}")
        return interrupted

    def expected_trials(self) -> int:
        if self.args.mode == "joint":
            return self.args.trials
        return self.args.trials_per_layer * len(self.layers())

    def layers(self) -> Sequence[str]:
        wanted = self.args.layers or LAYERS
        return [l for l in wanted if l in LAYERS]

    def _layer_objective(self, layer: str, layer_params: list[Param],
                        frozen: dict[str, object]):
        """A closure that binds the layer and its parameters for one study."""
        def objective(trial) -> float:
            cfg = sample(trial, layer_params, frozen)
            return self.evaluate(layer, cfg).score
        return objective

    def _warm_study(self, study, params: list[Param]) -> int:
        """Hand the study the configurations already measured on this machine.

        TPE then samples from a posterior instead of a cold prior. A prior outside the
        grid the current run is searching (--vram-max, --max-workers, ...) is skipped and
        counted, never silently clamped.
        """
        added = 0
        for prior in self.priors:
            try:
                dists = distributions_for(params, prior["cfg"])
            except ValueError:
                self.priors_skipped += 1
                continue
            if not dists:
                continue
            try:
                study.add_trial(optuna.trial.create_trial(
                    params={k: prior["cfg"][k] for k in dists},
                    distributions=dists,
                    value=float(prior["score"]),
                ))
            except Exception as exc:  # noqa: BLE001 - a bad prior must not stop the sweep
                self.priors_skipped += 1
                print(f"note: prior {prior['cfg']} cannot seed the study ({exc})",
                      file=sys.stderr)
                continue
            added += 1
        return added

    # ---- the outputs -----------------------------------------------------------
    def write_csv(self) -> Path:
        self.args.results_dir.mkdir(parents=True, exist_ok=True)
        out = self.args.results_dir / (
            f"bayesian-{self.hw.slug()}-{self.model}-{self.args.objective}-{utc_stamp()}.csv")
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(ROW_COLUMNS) + [
                "gen_tok_s", "prefill_tok_s", "latency_ms", "prompt_ms", "decode_ms",
                "generated", "prompt_tokens", "score", "note"])
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.csv_row())
        return out

    def winner_args(self) -> list[str]:
        if not self.best_row:
            return []
        return engine_args(self.best_row.cfg, self.paths.as_dict(), self.args.cntx,
                           self.args.max_new, self.args.spec_min_p)

    def record(self, csv_path: Path) -> dict[str, Any]:
        best = self.best_row
        measured = sum(1 for r in self.rows if r.metrics.ok())
        failed = sum(1 for r in self.rows if not r.metrics.ok() and r.metrics.status != "DRY")
        return {
            "schema": "hypercore.max_run/1",
            "model": self.model,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "engine": str(self.args.engine),
            "hardware": self.hw.to_dict(),
            "hardware_slug": self.hw.slug(),
            "objective": self.args.objective,
            "context": self.args.cntx,
            "max_new": self.args.max_new,
            "spec_min_p": self.args.spec_min_p,
            "best": dict(best.cfg) if best else {},
            "metrics": {
                "gen_tok_s": best.metrics.gen_tok_s if best else 0.0,
                "prefill_tok_s": best.metrics.prefill_tok_s if best else 0.0,
                "latency_ms": best.metrics.latency_ms if best else 0.0,
                "prompt_ms": best.metrics.prompt_ms if best else 0.0,
                "decode_ms": best.metrics.decode_ms if best else 0.0,
            },
            "effective": best.metrics.effective if best else {},
            # A winner is only a winner if a boot produced tokens. A run that booted
            # nothing (GPU busy, --dry-run) has a best_row but no measurement; a prior
            # carried in from history is a measurement taken by an earlier sweep.
            "verified": bool(best and (best.metrics.ok() or best.from_history)),
            # Where the winning numbers came from: this sweep, a prior recorded on
            # this machine, or a stored record replayed with --replay/--from-records.
            "source": ("history" if best and best.from_history else
                       "dry_run" if self.args.dry_run else "study"),
            "args": self.winner_args(),
            "study": {
                "mode": self.args.mode,
                "sampler": self.args.sampler,
                "seed": self.args.seed,
                "trials": len(self.rows),
                "measured": measured,
                "failed": failed,
                "repeat": self.args.repeat,
                "complete": not self.args.trials or len(self.rows) >= self.expected_trials(),
            },
            "history": {
                "file": str(self.args.history_file or history_path(
                    self.args.history_dir, self.model, self.args.objective,
                    self.args.cntx, self.hw)),
                "trials_loaded": len(self.priors),
                "priors_skipped": self.priors_skipped,
                "reused": sum(1 for r in self.rows if r.from_history),
            },
            "trials": [r.to_dict() for r in self.rows],
            "csv": str(csv_path),
        }

    def write_record(self, record: dict[str, Any]) -> Path:
        """The full trial record: every configuration measured, with its metrics."""
        out = self.args.results_dir / f"max_run_{self.model}.record.json"
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return out

    def write_deliverables(self, record: dict[str, Any]) -> tuple[Path, Path]:
        """Write max_run_<model>.sh and max_run_<model>.json in the project's run-config format.

        A root-level launcher left by an earlier sweep may belong to another user; a
        PermissionError there must not throw away a finished sweep, so the pair falls
        back to the results directory and the fallback is printed.
        """
        return write_run_pair(
            self.model, self.paths.as_dict(), self.winner_args(), record,
            deliverable_dir=self.args.deliverable_dir, port=self.args.port,
            open_browser=not self.args.no_browser,
            fallback_dir=self.args.results_dir,
        )

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian (Optuna) tuning of the HyperCore engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="q2_0", help="quant name; names the deliverables")
    metric = p.add_mutually_exclusive_group()
    metric.add_argument("--gen", action="store_const", const="gen", dest="objective",
                       help="maximise generation tokens/s")
    metric.add_argument("--prefill", action="store_const", const="prefill", dest="objective",
                       help="maximise prefill tokens/s")
    metric.add_argument("--latency", action="store_const", const="latency", dest="objective",
                       help="minimise prompt_ms + decode_ms")
    p.add_argument("--objective", choices=OBJECTIVES, default=None,
                   help="same as the metric flag above; the flag wins")
    p.add_argument("--cntx", "--context", type=int, default=131072, dest="cntx",
                   help="context length the sweep is tuned at")
    p.add_argument("--trials", type=int, default=40, help="configurations to evaluate")
    p.add_argument("--mode", choices=("joint", "staged"), default="joint")
    p.add_argument("--layers", default=None, help="comma separated layers for staged mode")
    p.add_argument("--sampler", choices=("tpe", "gp", "random"), default="tpe")
    p.add_argument("--seed", type=int, default=20261012, help="reproducible search")
    p.add_argument("--repeat", type=int, default=1, help="boots per configuration (median)")
    p.add_argument("--max-new", type=int, default=512, help="tokens generated per request")
    p.add_argument("--prompt-tokens", type=int, default=10240,
                   help="prompt length; >8192 is needed for the tail arm to differ")
    p.add_argument("--prompt-tokens-file", default=None, help="file of token ids to cycle")
    p.add_argument("--spec-min-p", type=float, default=0.8, help="speculative draft floor")
    p.add_argument("--vram-min", type=int, default=256)
    p.add_argument("--vram-max", type=int, default=1536)
    p.add_argument("--vram-step", type=int, default=256)
    p.add_argument("--max-workers", type=int, default=None, help="cap the pool-workers grid")
    p.add_argument("--engine", default=None,
                   help="the engine binary to boot; default the one this project built "
                        "(engine/strata, engine/hypercore, build/...)")
    p.add_argument("--strata-root", default=None,
                   help="the Strata checkout to tune and to write the deliverables into; "
                        "default the repository this file is inside, or $STRATA_ROOT")
    p.add_argument("--pack", default=None)
    p.add_argument("--native", default=None)
    p.add_argument("--ple-gguf", default=None)
    p.add_argument("--mtp", default=None)
    p.add_argument("--data-dir", default=None,
                   help="folder holding packs/models/mtp; default the one the installer "
                        "recorded in the project settings file, then a data folder next to "
                        "the Strata checkout")
    p.add_argument("--boot-timeout", type=int, default=300)
    p.add_argument("--run-timeout", type=int, default=180)
    p.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--history-dir", default=str(Path(__file__).resolve().parent / "history"),
                   help="where run data keyed to this machine is kept")
    p.add_argument("--replay", default=None,
                   help="re-emit max_run_<model>.sh/.json from a stored record without "
                        "booting the engine (HyperCore-Tune/results/max_run_<model>.record.json)")
    p.add_argument("--from-records", action="store_true",
                   help="rebuild max_run_<model>.sh/.json from the best measured trial in "
                        "the CSVs under --results-dir, without running a sweep")
    p.add_argument("--history-file", default=None,
                   help="load run data from this file instead of the machine-matched one")
    p.add_argument("--no-load", action="store_true",
                   help="ignore previous runs; start the study cold")
    p.add_argument("--allow-hardware-mismatch", action="store_true",
                   help="load run data even when it was measured on different hardware")
    p.add_argument("--hardware", default=None,
                   help="pin the machine fingerprint to this JSON object or file instead of "
                        "probing the host (for tests, containers, and a farm of boxes)")
    p.add_argument("--dry-run", action="store_true", help="print the command lines, launch nothing")
    p.add_argument("--no-deliverables", action="store_true",
                   help="write only the CSV; skip max_run_<model>.sh/.json")
    p.add_argument("--deliverable-dir", default=None,
                   help="where max_run_<model>.sh/.json go; default the Strata root")
    p.add_argument("--port", type=int, default=None,
                   help="port the generated launcher uses; default the model config's, else 8081")
    p.add_argument("--no-browser", action="store_true",
                   help="the generated launcher leaves out --open")
    args = p.parse_args(argv)
    if args.strata_root:
        os.environ["STRATA_ROOT"] = str(Path(args.strata_root).expanduser().resolve())
        strata_root.cache_clear()
    root = strata_root()
    args.root = root
    if not args.engine:
        args.engine = engine_binary(root) or str(root / "engine" / "strata")
    if not args.objective:
        args.objective = "gen"
    args.layers = [x.strip() for x in args.layers.split(",")] if args.layers else list(LAYERS)
    args.results_dir = Path(args.results_dir)
    args.history_dir = Path(args.history_dir)
    args.deliverable_dir = Path(args.deliverable_dir) if args.deliverable_dir else root
    if args.trials < 1:
        p.error("--trials must be at least 1")
    args.trials_per_layer = max(2, args.trials // max(1, len(args.layers)))
    return args


def _hardware(spec: str | None) -> Hardware:
    """The machine this sweep ran on: a pinned fingerprint if given, otherwise the host."""
    if not spec:
        return fingerprint()
    try:
        raw = json.loads(Path(spec).read_text(encoding="utf-8")) if Path(spec).is_file() \
            else json.loads(spec)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"HyperCore-Tune: --hardware is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("HyperCore-Tune: --hardware must be a JSON object")
    return hardware_from(raw)


def best_from_records(results_dir: Path, model: str, objective: str) -> TrialRow | None:
    """The highest-scoring measured row across every CSV this tool has written for a model.

    The CSV rows are the measurements; a record's `best` can come from a run that never
    booted the engine. Rebuilding from the rows is the only way to recover a winner that
    is backed by a number.
    """
    best: TrialRow | None = None
    pattern = f"bayesian-*{model}-{objective}-*.csv"
    for csv_path in sorted(results_dir.glob(pattern)):
        try:
            rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        except OSError as exc:
            print(f"note: cannot read {csv_path}: {exc}", file=sys.stderr)
            continue
        for raw in rows:
            if raw.get("status") != "PASS":
                continue
            try:
                metrics = Metrics(
                    status="PASS",
                    gen_tok_s=float(raw.get("gen_tok_s") or 0.0),
                    prefill_tok_s=float(raw.get("prefill_tok_s") or 0.0),
                    latency_ms=float(raw.get("latency_ms") or 0.0),
                    prompt_ms=float(raw.get("prompt_ms") or 0.0),
                    decode_ms=float(raw.get("decode_ms") or 0.0),
                    generated=int(raw.get("generated") or 0),
                    prompt_tokens=int(raw.get("prompt_tokens") or 0),
                )
                cfg: Config = {
                    "vram": int(raw["vram"]),
                    "pcie": float(raw["pcie"]),
                    "ple_io": raw["ple_io"],
                    "kv": raw["kv"],
                    "kvgrow": raw["kvgrow"],
                    "host_core": raw["hostcore"],
                    "affinity": raw["affinity"],
                    "worker": int(raw["worker"]),
                    "tail": raw["tail"],
                }
            except (KeyError, ValueError, TypeError):
                continue
            if not metrics.ok():
                continue
            row = TrialRow("records", cfg, metrics, signed_score(metrics, objective))
            if best is None or row.score > best.score:
                best = row
    return best


def rebuild_from_records(args: argparse.Namespace, paths: ModelPaths, hw: Hardware) -> int:
    """Write the deliverable pair from measured trials already on disk. No engine boots."""
    row = best_from_records(Path(args.results_dir), args.model, args.objective)
    if row is None:
        raise SystemExit(
            f"HyperCore-Tune: no measured (status PASS) trial for model {args.model} "
            f"objective {args.objective} in {args.results_dir}"
        )
    engine = engine_args(row.cfg, paths.as_dict(), args.cntx, args.max_new, args.spec_min_p)
    record = {
        "schema": "hypercore.max_run/1",
        "model": args.model,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "engine": str(args.engine),
        "hardware": hw.to_dict(),
        "hardware_slug": hw.slug(),
        "objective": args.objective,
        "context": args.cntx,
        "max_new": args.max_new,
        "spec_min_p": args.spec_min_p,
        "best": dict(row.cfg),
        "metrics": {
            "gen_tok_s": row.metrics.gen_tok_s,
            "prefill_tok_s": row.metrics.prefill_tok_s,
            "latency_ms": row.metrics.latency_ms,
            "prompt_ms": row.metrics.prompt_ms,
            "decode_ms": row.metrics.decode_ms,
        },
        "effective": row.metrics.effective,
        "verified": True,
        "args": engine,
        "study": {"mode": "records", "sampler": "none", "seed": args.seed,
                  "trials": 0, "measured": 1, "failed": 0, "repeat": args.repeat,
                  "complete": False},
        "source": "best measured row in the CSVs under " + str(args.results_dir),
        "trials": [],
    }
    args.deliverable_dir.mkdir(parents=True, exist_ok=True)
    sh, cfg = write_run_pair(
        args.model, paths.as_dict(), engine, record,
        deliverable_dir=args.deliverable_dir, port=args.port,
        open_browser=not args.no_browser,
        fallback_dir=args.results_dir,
    )
    print(f"rebuilt from measured records -> {sh}")
    print(f"                             {cfg}")
    print(f"  winner {json.dumps(row.cfg)}")
    print(f"  gen {row.metrics.gen_tok_s:.2f} tok/s, prefill {row.metrics.prefill_tok_s:.2f} "
          f"tok/s, latency {row.metrics.latency_ms:.1f} ms ({args.objective})")
    return 0


def replay(record_path: Path, args: argparse.Namespace, paths: ModelPaths,
           hw: Hardware) -> int:
    """Re-emit the deliverables from a stored record, without booting the engine.

    A sweep that finished before the run-config deliverables existed, or one whose
    launcher was written to a read-only root, can be rebuilt from its record. The
    hardware block is refreshed to the machine doing the replay, so the pair always
    states where the numbers came from.
    """
    record = load_history(record_path)
    if record is None:
        raise SystemExit(f"HyperCore-Tune: {record_path} is not a readable tuning record")
    if not record.get("best") or not record.get("args"):
        raise SystemExit(f"HyperCore-Tune: {record_path} has no winning configuration to replay")
    if not record.get("verified", any(t.get("status") == "PASS" for t in record.get("trials", []))):
        raise SystemExit(
            f"HyperCore-Tune: {record_path} has no measured trial behind its best; "
            "use --from-records to rebuild from the CSVs instead"
        )
    record.setdefault("hardware", hw.to_dict())
    record["hardware_slug"] = hw.slug()
    record["model"] = args.model
    record["objective"] = record.get("objective") or args.objective
    record["context"] = record.get("context") or args.cntx
    sh, cfg = write_run_pair(
        args.model, paths.as_dict(), list(record["args"]), record,
        deliverable_dir=args.deliverable_dir, port=args.port,
        open_browser=not args.no_browser,
        fallback_dir=args.results_dir,
    )
    print(f"replayed {record_path} -> {sh}")
    print(f"                        {cfg}")
    print(f"  objective {record['objective']} at --max-context {record['context']}, "
          f"machine {hw.slug()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root

    if args.pack and args.native:
        paths = ModelPaths(pack=args.pack, native=args.native, ple_gguf=args.ple_gguf or "",
                          mtp=args.mtp or "")
    else:
        paths = resolve_model_paths(root, args.model, args.data_dir)
    if args.pack:
        paths.pack = args.pack
    if args.native:
        paths.native = args.native
    if args.ple_gguf:
        paths.ple_gguf = args.ple_gguf
    if args.mtp:
        paths.mtp = args.mtp
    if args.prompt_tokens_file:
        paths.prompt_tokens = args.prompt_tokens_file
    if not paths.prompt_tokens:
        raise SystemExit(
            "HyperCore-Tune: no prompt token file found. Pass --prompt-tokens-file <json with an 'input' list> "
            "(bench/results/exchange-rotation-token-ids.json is the usual one)."
        )
    args.prompt_tokens_file = paths.prompt_tokens

    params = build_space(vram_low=args.vram_min, vram_high=args.vram_max, vram_step=args.vram_step,
                        max_workers=args.max_workers)

    hw = _hardware(args.hardware)
    history_file = Path(args.history_file) if args.history_file else history_path(
        args.history_dir, args.model, args.objective, args.cntx, hw)

    if args.replay:
        return replay(Path(args.replay), args, paths, hw)

    if args.from_records:
        return rebuild_from_records(args, paths, hw)

    priors: list[dict[str, Any]] = []
    sources: list[Path] = []
    if args.no_load:
        print("history: --no-load, starting cold")
    else:
        sources = [history_file] if args.history_file else find_history(
            args.history_dir, args.model, args.objective, args.cntx, hw,
            allow_mismatch=args.allow_hardware_mismatch)
        for source in sources:
            record = load_history(source)
            if record:
                priors.extend(usable_trials(record, args.max_new, args.spec_min_p))
        if priors:
            print(f"history: {len(priors)} configurations from "
                  f"{', '.join(str(s) for s in sources)}")
        else:
            print(f"history: nothing for {hw.slug()} / {args.model} / {args.objective} "
                  f"/ {args.cntx}; starting cold")

    print(f"optuna {optuna.__version__} | objective {args.objective} | mode {args.mode} "
          f"| sampler {args.sampler} | seed {args.seed} | context {args.cntx} "
          f"| trials {args.trials} | physical cores {physical_cores()}")
    print(f"machine: {hw.slug()}")
    print(f"engine: {args.engine}")
    print(f"pack:   {paths.pack}")

    sweeper = Sweeper(args, root, params, paths, hw, priors)
    t0 = time.monotonic()
    interrupted = sweeper.run()
    elapsed = time.monotonic() - t0

    csv_path = sweeper.write_csv()
    record = sweeper.record(csv_path)
    record["elapsed_s"] = round(elapsed, 1)
    record["interrupted"] = interrupted

    record_path = sweeper.write_record(record)
    print(f"record: {record_path}")
    hist_path = save_history(history_file, record)
    print(f"history: {hist_path} ({record['history']['trials_loaded']} priors loaded, "
          f"{record['history']['reused']} configurations reused)")

    if not args.no_deliverables:
        if not sweeper.best_row:
            print("no configuration was evaluated; nothing written", file=sys.stderr)
            return 1
        if not record["verified"]:
            if args.dry_run:
                print("dry run: nothing was measured, so no deliverable was written; "
                      "the command lines are in the CSV")
                return 0
            print("no configuration produced tokens in this run, so the deliverables were "
                  "not written. Free the GPU (./restart.sh stop) and re-run, or use "
                  "--from-records to rebuild them from measured trials.", file=sys.stderr)
            return 1
        args.deliverable_dir.mkdir(parents=True, exist_ok=True)
        sh_path, json_path = sweeper.write_deliverables(record)
        print(f"\nwinner: {json.dumps(sweeper.best_row.cfg)}")
        print(f"score:  {sweeper.best_row.score:.4f} ({args.objective}) "
              f"gen {sweeper.best_row.metrics.gen_tok_s:.2f} tok/s, "
              f"prefill {sweeper.best_row.metrics.prefill_tok_s:.2f} tok/s, "
              f"latency {sweeper.best_row.metrics.latency_ms:.1f} ms")
        print(f"wrote  {sh_path}")
        print(f"wrote  {json_path}")

    print(f"csv:    {csv_path}")
    print(f"trials: {len(sweeper.rows)} evaluated, {record['study']['measured']} measured, "
          f"{record['study']['failed']} failed, in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

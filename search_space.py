"""HyperCore-Tune/search_space.py - the parameter space the Bayesian sweep searches.

Every trial writes one row with exactly the columns the brief asks for:

    layer  vram  pcie  ple-io  kv  kvgrow  hostcore  affinity  worker  tail  status

`layer` is the group a trial is perturbing. In staged mode it is the layer the study is
currently frozen on; in joint mode it is the group whose value differs from the incumbent
best ("joint" for the first trial, which has no incumbent). `status` is not a parameter:
it is what the engine did with the configuration (PASS, FAIL_BOOT, NO_TOKENS, TIMEOUT).

Determinism: every parameter is sampled from a closed grid, so a trial's configuration is
reproducible from (seed, trial index) and the signature of a configuration is a plain
tuple. Nothing here reads the environment except the CPU count, which bounds the worker
grid and is overridable with --max-workers.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

try:  # psutil is in .venv (requirements.txt); the fallback keeps this importable anywhere.
    import psutil
except ImportError:  # pragma: no cover - only hit on a bare interpreter
    psutil = None  # type: ignore[assignment]

LAYERS: tuple[str, ...] = (
    "topology", "ple-io", "kv", "kv-grow", "host-core", "affinity", "worker", "tail",
)

# Column order of the CSV, exactly as specified in the brief.
ROW_COLUMNS: tuple[str, ...] = (
    "layer", "vram", "pcie", "ple_io", "kv", "kvgrow", "hostcore",
    "affinity", "worker", "tail", "status",
)

Config = dict[str, object]


@dataclass(frozen=True)
class Param:
    """One dimension of the search space.

    kind   "int" | "float" | "choice"
    flag   the engine flag this parameter writes
    mode   "value" -> "--flag <value>"; "on" -> bare "--flag" when value is truthy
    step   grid step for int/float sampling (keeps the space finite and reproducible)
    """

    key: str
    layer: str
    kind: str
    flag: str
    mode: str = "value"
    low: float = 0.0
    high: float = 1.0
    step: float = 1.0
    choices: tuple[str, ...] = ()

    def grid(self) -> list[object]:
        if self.kind == "choice":
            return list(self.choices)
        if self.kind == "int":
            return list(range(int(self.low), int(self.high) + 1, int(self.step)))
        steps = round((self.high - self.low) / self.step)
        return [round(self.low + i * self.step, 4) for i in range(int(steps) + 1)]


def physical_cores() -> int:
    """Physical cores the pool can actually be spread over; logical cores as a fallback."""
    n = 0
    if psutil is not None:
        try:
            n = int(psutil.cpu_count(logical=False) or 0)
        except Exception:  # noqa: BLE001 - a container that reports nothing must not stop the sweep
            n = 0
    if n > 0:
        return n
    return max(1, os.cpu_count() or 1)


def build_space(
    vram_low: int = 256,
    vram_high: int = 1536,
    vram_step: int = 256,
    pcie_low: float = 0.0,
    pcie_high: float = 1.0,
    pcie_step: float = 0.05,
    max_workers: int | None = None,
    kv_modes: Sequence[str] = ("q4_0", "k8v4", "int8", "fp16"),
    ple_modes: Sequence[str] = ("direct", "mmap", "ram"),
) -> list[Param]:
    """The ten dimensions of the sweep, with the ranges the engine accepts.

    --pool-workers 0 is not "no workers": it means "every physical core except the host"
    (src/program/generate.cpp:409), so 0 is a legitimate arm and the grid is capped at the
    physical core count rather than the logical one.
    """
    cores = max_workers if max_workers is not None else physical_cores()
    return [
        Param("vram", "topology", "int", "--vram-reserve-mib", "value",
              low=vram_low, high=vram_high, step=vram_step),
        Param("pcie", "topology", "float", "--pcie-frac", "value",
              low=pcie_low, high=pcie_high, step=pcie_step),
        Param("ple_io", "ple-io", "choice", "--ple-io", "value", choices=tuple(ple_modes)),
        Param("kv", "kv", "choice", "--kv", "value", choices=tuple(kv_modes)),
        Param("kvgrow", "kv-grow", "choice", "--kv-grow", "on", choices=("0", "1")),
        Param("host_core", "host-core", "choice", "--host-core", "value",
             choices=("first", "last")),
        Param("affinity", "affinity", "choice", "--pool-affinity", "value",
             choices=("all", "p-cores", "auto")),
        Param("worker", "worker", "int", "--pool-workers", "value",
             low=0, high=max(0, cores - 1), step=1),
        Param("tail", "tail", "choice", "--prompt-cache-tail", "on", choices=("0", "1")),
    ]


DEFAULT_CONFIG: Config = {
    "vram": 512,
    "pcie": 0.95,
    "ple_io": "direct",
    "kv": "q4_0",
    "kvgrow": "0",
    "host_core": "last",
    "affinity": "all",
    "worker": 0,
    "tail": "0",
}


def signature(cfg: Config) -> tuple[object, ...]:
    """A hashable, order-stable identity for a configuration, used to skip repeats."""
    return tuple(str(cfg[k]) for k in sorted(cfg))


def sample(trial, params: Sequence[Param], frozen: dict[str, object]) -> Config:
    """Ask Optuna for one point in the space, leaving `frozen` dimensions untouched."""
    cfg: Config = dict(DEFAULT_CONFIG)
    for p in params:
        if p.key in frozen:
            cfg[p.key] = frozen[p.key]
            continue
        if p.kind == "choice":
            cfg[p.key] = trial.suggest_categorical(p.key, list(p.choices))
        elif p.kind == "int":
            cfg[p.key] = trial.suggest_int(p.key, int(p.low), int(p.high), step=int(p.step))
        else:
            cfg[p.key] = round(trial.suggest_float(p.key, float(p.low), float(p.high)), 4)
    return cfg


def engine_args(
    cfg: Config,
    paths: dict[str, str],
    context: int,
    max_new: int,
    spec_min_p: float = 0.8,
) -> list[str]:
    """The engine command line for one configuration.

    The base block is the production one (the same flags tune/max_tune.sh boots every
    arm with); only the swept dimensions come from cfg.
    """
    args: list[str] = ["--serve"]
    for key, flag in (("pack", "--pack"), ("native", "--native"), ("ple_gguf", "--ple-gguf")):
        if paths.get(key):
            args += [flag, paths[key]]
    if paths.get("mtp"):
        args += ["--mtp", paths["mtp"]]
    if paths.get("expert_profile") and Path(paths["expert_profile"]).exists():
        args += ["--expert-profile", paths["expert_profile"]]
    if paths.get("cvec_scaled"):
        args += ["--control-vector-scaled", paths["cvec_scaled"]]
        args += ["--control-vector-layer-range", "4", "44", "--cvec-mode", "project",
                "--cvec-dir", "per-layer"]
    args += ["--prefill", "auto", "--spec", "4", "--expert-cache", "auto",
            "--max-context", str(context), "--max-new", str(max_new),
            "--spec-min-p", str(spec_min_p), "--mmap-experts",
            "--expert-profile-save-every", "0"]

    for p in build_space():
        value = cfg.get(p.key)
        if value is None:
            continue
        if p.mode == "on":
            if str(value) not in ("0", "false", "off", ""):
                args.append(p.flag)
        else:
            args += [p.flag, str(value)]
    return args


@dataclass
class ModelPaths:
    """Where the weights live. Every field is a path string; empty means "not found"."""

    pack: str = ""
    native: str = ""
    ple_gguf: str = ""
    mtp: str = ""
    cvec_scaled: str = ""
    expert_profile: str = ""
    prompt_tokens: str = ""
    searched: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, str]:
        return {
            "pack": self.pack, "native": self.native, "ple_gguf": self.ple_gguf,
            "mtp": self.mtp, "cvec_scaled": self.cvec_scaled,
            "expert_profile": self.expert_profile, "prompt_tokens": self.prompt_tokens,
        }


# The items a data folder must hold to count as one, same as setup.py's DATA_ITEMS.
DATA_ITEMS: tuple[str, ...] = ("models", "packs", "mtp")


def settings_data_dir() -> str:
    """The data folder the installer remembered in the user's settings, or "".

    Mirrors setup.py: `~/.config/strata/settings.json` (or `$XDG_CONFIG_HOME`), falling
    back to the older `hypercore` folder. A missing or unreadable file is not an error, it
    just means no remembered folder.
    """
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    for name in ("strata", "hypercore"):
        path = base / name / "settings.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("data_dir"):
            return str(data["data_dir"])
    return ""


def data_roots(root: Path) -> tuple[str, ...]:
    """Data folders in the order the installer would use them.

    Order: $STRATA_DATA (or $HYPERCORE_DATA), the `data_dir` recorded in the user's
    settings, a data folder next to the Strata checkout, then any sibling data folder.
    Only folders that exist and hold at least one of models/packs/mtp are returned. Among
    the discovered ones, folders inside the running user's home come first, so a sweep
    reads that user's own pack rather than another account's copy. Never raises.
    """
    pinned = [os.environ.get("STRATA_DATA", ""), os.environ.get("HYPERCORE_DATA", ""),
              settings_data_dir()]
    discovered: list[str] = []
    for base in dict.fromkeys((root.parent, root.parent.parent)):
        try:
            entries = list(base.iterdir())
        except OSError:
            continue
        for d in entries:
            name = d.name.lower()
            if d.is_dir() and (name.startswith(("strata-data", "hypercore-data"))
                               or name.endswith("-data")):
                discovered.append(str(d))
    home = str(Path.home())
    discovered.sort(key=lambda r: (not r.startswith(home), r))

    ordered: list[str] = []
    for raw in pinned + discovered:
        if not raw:
            continue
        p = Path(raw)
        if not p.is_dir() or not any((p / item).is_dir() for item in DATA_ITEMS):
            continue
        if str(p) not in ordered:
            ordered.append(str(p))
    return tuple(ordered)


def resolve_model_paths(root: Path, model: str, data_dir: str | None = None) -> ModelPaths:
    """Find the pack, the two GGUFs and the MTP draft for a quant.

    Searches the repo first, then the data roots the setup script uses (data_dir wins
    over them when given). A single base is
    preferred when it carries both the pack and the weights, so a run never mixes a pack
    from one tree with weights from another. Raises FileNotFoundError with the list of
    places it looked at, so a miss is a path problem, not a crash.
    """
    quant = model.lower()
    model_dir = quant.upper()
    candidates = [str(root)] + ([data_dir] if data_dir else []) + list(data_roots(root))
    found: list[ModelPaths] = []

    for base in candidates:
        b = Path(base)
        if not b.is_dir():
            continue
        paths = ModelPaths()
        for cand in (b / "packs" / quant, b / "packs" / model_dir, b / quant):
            if (cand / "experts.bin").exists() or (cand / "dense.bin").exists():
                paths.pack = str(cand)
                break
        ggufs: list[Path] = []
        for sub in (b / "models" / model_dir, b / model_dir):
            if sub.is_dir():
                ggufs = sorted(g for g in sub.glob("*.gguf") if not g.name.endswith(".done"))
                if ggufs:
                    break
        if ggufs:
            paths.native = str(ggufs[0])
            rest = [g for g in ggufs if "of-00002" in g.name and g != ggufs[0]]
            paths.ple_gguf = str(rest[0]) if rest else ""
        for cand in (b / "mtp" / "rt", b / "mtp"):
            if cand.is_dir():
                paths.mtp = str(cand)
                break
        cvec = b / "data" / "experimental-speed-projection"
        hit = next(iter(cvec.glob("*.gguf")), None) if cvec.is_dir() else None
        if hit:
            paths.cvec_scaled = f"{hit}:1.0"
        prof = b / "data" / "expert-profile.bin"
        if prof.exists():
            paths.expert_profile = str(prof)
        prompt = b / "bench" / "results" / "exchange-rotation-token-ids.json"
        if prompt.exists():
            paths.prompt_tokens = str(prompt)
        found.append(paths)

    # Prefer the base that has both the pack and the weights; otherwise merge what exists.
    best = max(found, key=lambda p: (bool(p.pack) + bool(p.native), bool(p.mtp))) if found else ModelPaths()
    paths = ModelPaths()
    for key in ("pack", "native", "ple_gguf", "mtp", "cvec_scaled", "expert_profile", "prompt_tokens"):
        for p in [best] + found:
            if getattr(p, key):
                setattr(paths, key, getattr(p, key))
                break
    paths.searched = candidates

    missing = [k for k in ("pack", "native") if not getattr(paths, k)]
    if missing:
        raise FileNotFoundError(
            "HyperCore-Tune: could not find " + ", ".join(missing) + f" for model '{model}'. "
            "Looked in: " + ", ".join(candidates) + ". Pass --pack/--native explicitly."
        )
    return paths

"""HyperCore-Tune/test_bayesian_tune.py - the sweep, end to end, without a GPU.

    pytest HyperCore-Tune/test_bayesian_tune.py

Every test uses stub_engine.py, which speaks the real protocol (INFO, "session is up",
DONE) and computes its numbers from the flags it was given. Nothing here touches the
engine or the card, and the project root is a stand-in tree so the suite also runs from a
standalone clone of this repository.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import strata_env
from bayesian_tune import main, signed_score
from engine_runner import (
    EngineError,
    build_prompt,
    parse_done,
    parse_info,
    run_config,
)
from run_config import clean_args
from search_space import (
    ROW_COLUMNS,
    build_space,
    data_roots,
    engine_args,
    resolve_model_paths,
)
from strata_env import candidate_roots, engine_binary, server_engine_name, strata_root

STUB = HERE / "stub_engine.py"

# A stand-in project tree: the two files a root is recognised by, and an engine binary.
# It keeps the suite runnable wherever this repository happens to be cloned.
FAKE_ROOT = Path(tempfile.mkdtemp(prefix="hypercore-tune-fake-strata-"))
(FAKE_ROOT / "serve").mkdir(parents=True, exist_ok=True)
(FAKE_ROOT / "serve" / "server.py").write_text(
    'ap.add_argument("--engine", choices=["mock", "strata"], default="mock")\n',
    encoding="utf-8",
)
(FAKE_ROOT / "CMakeLists.txt").write_text("# stand-in for the real project\n", encoding="utf-8")
(FAKE_ROOT / "setup.py").write_text(
    """# A stand-in for the installer: the sweep uses its write_config and defaults.
import json
import os


def write_config(path, cfg):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def apply_server_defaults(cfg, base):
    cfg.setdefault("host", "0.0.0.0")
    cfg.setdefault("api_key", "0000")
    cfg.setdefault("port", 8081)
    cfg.setdefault("turn_tail_check", True)
""",
    encoding="utf-8",
)
(FAKE_ROOT / "engine").mkdir(parents=True, exist_ok=True)
FAKE_ENGINE = FAKE_ROOT / "engine" / "strata"
FAKE_ENGINE.write_text("#!/bin/sh\n", encoding="utf-8")
FAKE_ENGINE.chmod(0o755)
FAKE_PYTHON = FAKE_ROOT / ".venv" / "bin" / "python"
FAKE_PYTHON.parent.mkdir(parents=True, exist_ok=True)
FAKE_PYTHON.symlink_to(sys.executable)
os.environ["STRATA_ROOT"] = str(FAKE_ROOT)

# A DONE line copied from the shape src/program/generate.cpp:12205 prints.
REAL_DONE = (
    "DONE 512 10240 1380.4 3050.2 length 412 512 0 0 120 300 12 41.5 10240 88\n"
)


def test_parse_done_reads_the_real_field_table():
    d = parse_done(REAL_DONE)
    assert d["generated"] == 512
    assert d["prompt_tokens"] == 10240
    assert d["prompt_ms"] == 1380.4
    assert d["decode_ms"] == 3050.2
    assert d["finish"] == "length"
    assert d["prompt_read"] == "10240"


def test_parse_done_rejects_a_truncated_line():
    with pytest.raises(EngineError):
        parse_done("DONE 512 10240 1380.4")


def test_parse_info_extracts_what_the_engine_applied():
    info = parse_info("INFO context=131072 kv=k8v4 pool_workers=16 pcie_frac=0.95")
    assert info == {"context": "131072", "kv": "k8v4", "pool_workers": "16", "pcie_frac": "0.95"}


def test_engine_args_covers_every_swept_dimension():
    cfg = {"vram": 768, "pcie": 0.9, "ple_io": "ram", "kv": "k8v4", "kvgrow": "1",
           "host_core": "first", "affinity": "auto", "worker": 12, "tail": "1"}
    args = engine_args(cfg, {"pack": "/p", "native": "/n"}, 131072, 512)
    for flag, value in (("--vram-reserve-mib", "768"), ("--pcie-frac", "0.9"),
                        ("--ple-io", "ram"), ("--kv", "k8v4"), ("--host-core", "first"),
                        ("--pool-affinity", "auto"), ("--pool-workers", "12"),
                        ("--max-context", "131072")):
        assert flag in args, f"{flag} missing"
        assert args[args.index(flag) + 1] == value
    assert "--kv-grow" in args and "--prompt-cache-tail" in args
    # Off switches must not appear at all.
    off = engine_args({**cfg, "kvgrow": "0", "tail": "0"}, {"pack": "/p", "native": "/n"}, 4096, 256)
    assert "--kv-grow" not in off and "--prompt-cache-tail" not in off


def test_the_space_is_the_ten_columns_the_brief_names():
    params = build_space(max_workers=8)
    keys = {p.key for p in params}
    assert keys == {"vram", "pcie", "ple_io", "kv", "kvgrow", "host_core", "affinity", "worker", "tail"}
    assert ROW_COLUMNS[:10] == ("layer", "vram", "pcie", "ple_io", "kv", "kvgrow",
                               "hostcore", "affinity", "worker", "tail")
    assert ROW_COLUMNS[10] == "status"
    # The worker grid is capped at physical cores, and 0 (the engine's "all cores") is a legal arm.
    worker = next(p for p in params if p.key == "worker")
    assert worker.grid()[0] == 0 and worker.grid()[-1] == 7


def test_stub_engine_measures():
    ids = build_prompt(str(HERE / "fixtures" / "prompt.json"), 1024)
    args = ["--serve", "--max-context", "131072", "--kv", "k8v4",
            "--pool-workers", "8", "--max-new", "256"]
    m = run_config(str(STUB), args, ids, 256, Path("/tmp/hc-stub-run.log"),
                   boot_timeout=20, run_timeout=20)
    if m.status == "FAIL_BOOT" and not m.effective:
        # A subprocess that never got scheduled under load: one retry, then report it.
        m = run_config(str(STUB), args, ids, 256, Path("/tmp/hc-stub-run.log"),
                       boot_timeout=20, run_timeout=20)
    assert m.status == "PASS", m.note
    assert m.gen_tok_s > 0 and m.prefill_tok_s > 0
    assert abs(m.latency_ms - (m.prompt_ms + m.decode_ms)) < 1e-6


def test_a_bad_config_is_recorded_not_crashed():
    m = run_config(str(STUB), ["--serve", "--vram-reserve-mib", "9999"], [1, 2, 3], 8,
                   Path("/tmp/hc-stub-fail.log"), boot_timeout=10, run_timeout=10)
    assert m.status in ("FAIL_BOOT", "TIMEOUT")
    assert m.note


def test_objective_direction():
    from engine_runner import Metrics
    m = Metrics(status="PASS", gen_tok_s=150.0, prefill_tok_s=900.0, latency_ms=4000.0)
    assert signed_score(m, "gen") == 150.0
    assert signed_score(m, "prefill") == 900.0
    assert signed_score(m, "latency") == -4000.0
    failed = Metrics(status="FAIL_BOOT")
    assert signed_score(failed, "gen") == 0.0
    assert signed_score(failed, "latency") < signed_score(m, "latency")


def _run(extra, out: Path = Path("/tmp/hc-optuna-test"), fresh: bool = True):
    """One sweep against the stub engine, in a directory of its own."""
    if fresh:
        shutil.rmtree(out, ignore_errors=True)
    argv = ["--model", "q2_0", "--engine", str(STUB), "--strata-root", str(FAKE_ROOT),
            "--pack", str(HERE / "fixtures"), "--native", str(HERE / "fixtures" / "tiny.gguf"),
            "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
            "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
            "--history-dir", str(out / "history"),
            "--max-workers", "8", "--trials", "12", "--seed", "7"] + extra
    return main(argv), out


def _record(out: Path) -> dict:
    """The trial record: every configuration measured, written beside the CSV."""
    return json.loads((out / "results" / "max_run_q2_0.record.json").read_text(encoding="utf-8"))


def test_the_sweep_runs_and_writes_both_deliverables():
    code, out = _run(["--gen"])
    assert code == 0
    sh = out / "max_run_q2_0.sh"
    js = out / "max_run_q2_0.json"
    assert sh.exists() and js.exists()
    cfg = json.loads(js.read_text(encoding="utf-8"))
    # The deliverable is a run config the server can read, not a tuning dump.
    for key in ("exe", "args", "cwd", "tokenizer", "model_name", "log", "port",
                "host", "api_key", "turn_tail_check"):
        assert key in cfg, f"run config missing {key}"
    assert "--serve" not in cfg["args"] and "--max-new" not in cfg["args"]
    assert cfg["args"].count("--max-context") == 1
    assert cfg["args"][cfg["args"].index("--max-context") + 1] == "131072"
    tuned = cfg["tuned_by"]
    assert tuned["objective"] == "gen"
    assert tuned["context"] == 131072
    for key in ("vram", "pcie", "ple_io", "kv", "kvgrow", "host_core", "affinity", "worker", "tail"):
        assert key in tuned["best"]
    assert tuned["metrics"]["gen_tok_s"] > 0
    record = _record(out)
    assert record["schema"] == "hypercore.max_run/1"
    assert record["study"]["measured"] == 12
    assert len(record["trials"]) == 12
    # The launcher must be valid shell and executable.
    subprocess.run(["bash", "-n", str(sh)], check=True)
    assert os.access(sh, os.X_OK)
    csvs = list((out / "results").glob("bayesian-*.csv"))
    assert csvs, "no CSV written"
    header = csvs[0].read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[:11] == list(ROW_COLUMNS)


def test_the_search_is_deterministic_for_a_seed():
    _run(["--gen"])
    first = _record(Path("/tmp/hc-optuna-test"))
    _run(["--gen"])
    second = _record(Path("/tmp/hc-optuna-test"))
    assert first["best"] == second["best"]
    assert first["metrics"]["gen_tok_s"] == second["metrics"]["gen_tok_s"]


def test_latency_objective_picks_the_short_request():
    code, out = _run(["--latency"])
    assert code == 0
    record = _record(out)
    assert record["objective"] == "latency"
    assert record["metrics"]["latency_ms"] > 0
    rows = record["trials"]
    assert all(r["status"] == "PASS" for r in rows)
    assert abs(record["metrics"]["latency_ms"] - min(float(r["latency_ms"]) for r in rows)) < 1e-6


def test_staged_mode_labels_rows_by_layer():
    code, out = _run(["--mode", "staged", "--layers", "kv,worker"])
    assert code == 0
    record = _record(out)
    layers = {r["layer"] for r in record["trials"]}
    assert layers == {"kv", "worker"}


def test_dry_run_launches_nothing():
    code, out = _run(["--dry-run"])
    assert code == 0
    record = _record(out)
    assert record["study"]["measured"] == 0
    assert all(r["status"] == "DRY" for r in record["trials"])
    assert all("--max-context 131072" in r["note"] for r in record["trials"])


def test_a_blocked_deliverable_path_falls_back_instead_of_crashing():
    """A root-level launcher owned by another user must not lose the whole sweep."""
    blocked_dir = Path("/tmp/hc-optuna-blocked")
    blocked_dir.mkdir(parents=True, exist_ok=True)
    locked = blocked_dir / "max_run_q2_0.sh"
    locked.chmod(0o644) if locked.exists() else None
    locked.write_text("# an older launcher, not ours to overwrite\n", encoding="utf-8")
    locked.chmod(0o444)
    out = Path("/tmp/hc-optuna-fallback")
    code = main([
        "--model", "q2_0", "--engine", str(STUB),
        "--pack", str(HERE / "fixtures"), "--native", str(HERE / "fixtures" / "tiny.gguf"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(blocked_dir),
        "--results-dir", str(out / "results"),
        "--history-dir", str(out / "history"),
        "--max-workers", "8", "--trials", "4", "--seed", "3", "--gen",
    ])
    assert code == 0
    assert locked.read_text().startswith("# an older launcher")
    assert (out / "results" / "max_run_q2_0.sh").exists()
    assert (out / "results" / "max_run_q2_0.json").exists()


def test_a_dead_engine_leaves_no_broken_pipe_at_finalisation():
    """The engine exiting at boot used to raise BrokenPipeError when its request pipe
    was finalised. It must now be recorded as FAIL_BOOT and stay silent."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, {root!r});"
         "from pathlib import Path;"
         "from engine_runner import run_config;"
         "m = run_config({engine!r}, ['--serve'], [1, 2, 3], 8, Path('/tmp/hc-exit.log'), "
         "boot_timeout=5, run_timeout=5);"
         "print(m.status)".format(root=str(HERE), engine=str(HERE / "stub_engine_exit.py"))],
        capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == "FAIL_BOOT"
    assert "BrokenPipeError" not in proc.stderr
    assert "Exception ignored" not in proc.stderr


def test_ctrl_c_keeps_the_best_configuration_measured_so_far():
    out = Path("/tmp/hc-optuna-interrupt")
    argv = [
        sys.executable, str(HERE / "bayesian_tune.py"),
        "--model", "q2_0", "--engine", str(HERE / "stub_engine_slow.py"),
        "--strata-root", str(FAKE_ROOT),
        "--pack", str(HERE / "fixtures"), "--native", str(HERE / "fixtures" / "tiny.gguf"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
        "--max-workers", "8", "--trials", "20", "--seed", "5", "--gen",
        "--history-dir", str(out / "history"),
    ]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(3.0)          # a few trials have completed by now
    proc.send_signal(signal.SIGINT)
    stdout, stderr = proc.communicate(timeout=30)
    assert proc.returncode == 0, stdout + stderr
    assert "interrupted" in stdout
    record = _record(out)
    assert record["interrupted"] is True
    assert record["study"]["measured"] >= 1
    assert (out / "max_run_q2_0.sh").exists()
    subprocess.run(["bash", "-n", str(out / "max_run_q2_0.sh")], check=True)


def test_the_generated_launcher_is_the_project_startup_pair():
    """max_run_<model>.sh is the server startup script setup.py writes for a model."""
    out = Path("/tmp/hc-optuna-launcher")
    main([
        "--model", "q2_0", "--engine", str(STUB), "--strata-root", str(FAKE_ROOT),
        "--pack", str(HERE / "fixtures"), "--native", str(HERE / "fixtures" / "tiny.gguf"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
        "--max-workers", "8", "--trials", "3", "--seed", "2", "--gen", "--port", "8090",
    ])
    launcher = out / "max_run_q2_0.sh"
    cfg_path = out / "max_run_q2_0.json"
    lines = launcher.read_text().splitlines()
    assert lines[0] == "#!/bin/sh"
    assert lines[-2] == f'cd "{FAKE_ROOT}"', lines[-2]
    assert lines[-1].startswith("exec ")
    assert '"--engine" "strata"' in lines[-1]
    assert f'"{FAKE_PYTHON}"' in lines[-1], "the launcher must use the project venv"
    assert f'"--config" "{cfg_path}"' in lines[-1]
    assert '"--port" "8090"' in lines[-1]
    assert "--turn-tail-check" in lines[-1]
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    assert os.access(launcher, os.X_OK)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["port"] == 8090
    assert cfg["exe"] == str(STUB)
    assert cfg["cwd"] == str(FAKE_ROOT)
    # A Windows install of Strata gets the same launcher in .bat form.
    bat = (out / "max_run_q2_0.bat").read_text(encoding="utf-8")
    assert bat.startswith("@echo off") and "cd /d " in bat and '"--engine" "strata"' in bat


def test_no_browser_leaves_open_out_of_the_launcher():
    out = Path("/tmp/hc-optuna-no-browser")
    main([
        "--model", "q2_0", "--engine", str(STUB), "--strata-root", str(FAKE_ROOT),
        "--pack", str(HERE / "fixtures"), "--native", str(HERE / "fixtures" / "tiny.gguf"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
        "--max-workers", "8", "--trials", "2", "--seed", "1", "--gen", "--no-browser",
    ])
    text = (out / "max_run_q2_0.sh").read_text()
    assert "--open" not in text
    assert "--turn-tail-check" in text


def test_the_run_config_keeps_what_the_model_config_already_has():
    """Only "args" is replaced; everything setup does not own survives the sweep."""
    from run_config import build_run_config
    base = {"exe": "old-exe", "args": ["--pack", "old"], "cwd": "/old", "tokenizer": "/tok",
            "model_name": "real-name", "sampling": {"top_p": 0.95}, "mcp_servers": ["a"]}
    record = {"engine": "/engine/hypercore", "objective": "gen", "context": 8192,
              "best": {}, "metrics": {}, "study": {}, "csv": ""}
    args = ["--serve", "--pack", "/pack", "--max-new", "64", "--max-context", "8192",
            "--expert-profile-save-every", "0", "--kv", "int8"]
    cfg = build_run_config("q2_0", {"pack": "/pack"}, args, record, base=base)
    assert cfg["args"] == ["--pack", "/pack", "--max-context", "8192", "--kv", "int8"]
    assert cfg["exe"] == "/engine/hypercore"
    assert cfg["model_name"] == "real-name"
    assert cfg["tokenizer"] == "/tok"
    assert cfg["sampling"] == {"top_p": 0.95}
    assert cfg["mcp_servers"] == ["a"]
    assert cfg["tuned_by"]["objective"] == "gen"


def test_the_entry_point_is_executable_and_the_tree_is_gitignored():
    sh = HERE / "max_tune.sh"
    assert sh.exists() and os.access(sh, os.X_OK)
    subprocess.run(["bash", "-n", str(sh)], check=True)
    own = (HERE / ".gitignore").read_text(encoding="utf-8")
    for rule in ("results/", "history/", "max_run_*.sh", "max_run_*.json", "max_run_*.bat"):
        assert rule in own, f"missing ignore rule in the repository: {rule}"
    project = Path(os.environ["STRATA_ROOT"]) / ".gitignore"
    if project.is_file():
        text = project.read_text(encoding="utf-8")
        for rule in ("/max_run_*.sh", "/max_run_*.json", "/max_run_*.bat"):
            assert rule in text, f"the project .gitignore does not ignore the deliverables: {rule}"


# ---- where the project is, and how absolute the written paths are ---------------


def _clear_root_cache() -> None:
    strata_root.cache_clear()
    os.environ["STRATA_ROOT"] = str(FAKE_ROOT)


def test_a_nested_clone_resolves_its_own_checkout(monkeypatch):
    """<Strata>/HyperCore-Tune is found by walking up, with no environment set."""
    tmp = Path(tempfile.mkdtemp(prefix="hc-nested-clone-"))
    project = tmp / "Strata"
    (project / "serve").mkdir(parents=True)
    (project / "serve" / "server.py").write_text("", encoding="utf-8")
    (project / "CMakeLists.txt").write_text("", encoding="utf-8")
    clone = project / "HyperCore-Tune"
    clone.mkdir()

    strata_root.cache_clear()
    monkeypatch.delenv("STRATA_ROOT", raising=False)
    monkeypatch.delenv("HYPERCORE_ROOT", raising=False)
    monkeypatch.delenv("TUNE_ROOT", raising=False)
    monkeypatch.setattr(strata_env, "REPO", clone)
    try:
        assert strata_root() == project
    finally:
        strata_root.cache_clear()
        os.environ["STRATA_ROOT"] = str(FAKE_ROOT)


def test_a_sibling_checkout_is_found_when_the_clone_is_alone(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="hc-sibling-clone-"))
    project = tmp / "Strata"
    (project / "serve").mkdir(parents=True)
    (project / "serve" / "server.py").write_text("", encoding="utf-8")
    (project / "setup.py").write_text("", encoding="utf-8")
    clone = tmp / "HyperCore-Tune"
    clone.mkdir()

    strata_root.cache_clear()
    monkeypatch.delenv("STRATA_ROOT", raising=False)
    monkeypatch.delenv("HYPERCORE_ROOT", raising=False)
    monkeypatch.delenv("TUNE_ROOT", raising=False)
    monkeypatch.setattr(strata_env, "REPO", clone)
    try:
        assert strata_root() == project
    finally:
        strata_root.cache_clear()
        os.environ["STRATA_ROOT"] = str(FAKE_ROOT)


def test_a_clone_with_no_project_nearby_says_so(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="hc-lone-clone-"))
    clone = tmp / "HyperCore-Tune"
    clone.mkdir()
    strata_root.cache_clear()
    monkeypatch.delenv("STRATA_ROOT", raising=False)
    monkeypatch.delenv("HYPERCORE_ROOT", raising=False)
    monkeypatch.delenv("TUNE_ROOT", raising=False)
    monkeypatch.setattr(strata_env, "REPO", clone)
    try:
        with pytest.raises(SystemExit) as exc:
            strata_root()
        assert "--strata-root" in str(exc.value)
        assert candidate_roots(), "the error should still name what was searched"
    finally:
        strata_root.cache_clear()
        os.environ["STRATA_ROOT"] = str(FAKE_ROOT)


def test_the_engine_binary_is_the_one_this_project_built():
    assert engine_binary(FAKE_ROOT) == str(FAKE_ENGINE)
    assert engine_binary(Path(tempfile.mkdtemp(prefix="hc-no-engine-"))) == ""
    assert engine_binary(FAKE_ROOT).startswith(str(FAKE_ROOT))


def test_the_launcher_runs_the_project_venv_not_whatever_started_the_sweep():
    from strata_env import venv_python

    assert venv_python(FAKE_ROOT) == str(FAKE_PYTHON)


def test_the_launcher_uses_the_backend_the_local_server_accepts():
    assert server_engine_name(FAKE_ROOT) == "strata"


def test_every_path_the_sweep_writes_is_absolute():
    """A deliverable is only drop-in if it does not depend on the current directory."""
    code, out = _run(["--gen"])
    assert code == 0
    cfg = json.loads((out / "max_run_q2_0.json").read_text(encoding="utf-8"))
    for key in ("exe", "cwd", "log", "tokenizer", "model_name"):
        assert cfg.get(key), f"{key} is empty"
    for key in ("exe", "cwd", "log", "tokenizer"):
        assert Path(cfg[key]).is_absolute(), f"{key} is not absolute: {cfg[key]}"
    for flag in ("--pack", "--native"):
        if flag in cfg["args"]:
            value = cfg["args"][cfg["args"].index(flag) + 1]
            assert Path(value).is_absolute(), f"{flag} is relative: {value}"
    launcher = (out / "max_run_q2_0.sh").read_text(encoding="utf-8")
    assert f'cd "{FAKE_ROOT}"' in launcher


# ---- run data from previous sweeps, keyed to the machine ---------------------

PINNED = json.dumps({
    "cpu": "Test CPU X1", "physical_cores": 8, "logical_cores": 16, "ram_gib": 64,
    "gpus": ["Test GPU 12GB"], "gpu_mem_mib": 12288, "pci_gen": 4, "pci_width": 16,
})
PINNED_SLUG = "Test-CPU-X1_Test-GPU-12GB_64GB_PCIe4x16"
OTHER = json.dumps({
    "cpu": "Other CPU Y2", "physical_cores": 4, "logical_cores": 8, "ram_gib": 16,
    "gpus": ["Other GPU 8GB"], "gpu_mem_mib": 8192, "pci_gen": 3, "pci_width": 8,
})


def test_the_machine_slug_is_the_four_fields_the_data_file_is_keyed_on():
    from hardware import differences, hardware_from, slug
    hw = hardware_from(json.loads(PINNED))
    assert slug(hw) == PINNED_SLUG
    assert hw.machine_key() == ("test cpu x1", ("test gpu 12gb",), 64, 4)
    assert differences(hw, hw) == []
    assert differences(hw, hardware_from(json.loads(OTHER))) == [
        "cpu Test CPU X1 vs Other CPU Y2",
        "gpu Test GPU 12GB vs Other GPU 8GB",
        "ram 64 GiB vs 16 GiB",
        "PCIe gen 4 vs 3",
    ]


def test_the_host_probe_returns_a_usable_fingerprint():
    from hardware import detect
    hw = detect()
    assert hw.ram_gib > 0 and hw.slug()
    assert all(ch not in " ()" for ch in hw.slug())


def test_pci_generation_is_read_from_sysfs_when_lspci_is_absent():
    from hardware import pci_from_sysfs
    fake = Path("/tmp/hc-pci-fake/0000:01:00.0")
    fake.mkdir(parents=True, exist_ok=True)
    fake.joinpath("max_link_speed").write_text("16.0 GT/s PCIe\n", encoding="utf-8")
    fake.joinpath("max_link_width").write_text("16\n", encoding="utf-8")
    assert pci_from_sysfs(fake.parent) == (4, 16)
    assert pci_from_sysfs("/no/such/dir") == (0, 0)


def test_history_merge_keeps_the_best_score_for_each_configuration():
    from hardware import hardware_from
    from history import merge
    cfg = {"vram": 512, "pcie": 0.5, "ple_io": "ram", "kv": "int8", "kvgrow": "0",
           "host_core": "first", "affinity": "all", "worker": 4, "tail": "0"}

    def record(score):
        return {"model": "q2_0", "objective": "gen", "context": 131072, "max_new": 512,
                "spec_min_p": 0.8, "generated_utc": "2026-10-12T00:00:00+00:00",
                "hardware": hardware_from(json.loads(PINNED)).to_dict(),
                "hardware_slug": PINNED_SLUG,
                "trials": [{"cfg": cfg, "score": score, "status": "PASS", "layer": "joint",
                            "metrics": {"gen_tok_s": score}}]}

    merged = merge(None, record(10.0))
    merged = merge(merged, record(12.0))
    merged = merge(merged, record(8.0))
    assert len(merged["trials"]) == 1
    assert merged["trials"][0]["score"] == 12.0
    assert merged["trials"][0]["measured"] == 3
    assert len(merged["runs"]) == 3


def test_a_sweep_writes_run_data_named_after_the_machine():
    out = Path("/tmp/hc-optuna-history")
    code, out = _run(["--hardware", PINNED, "--trials", "3", "--seed", "3"], out)
    assert code == 0
    files = list((out / "history").glob("*.json"))
    assert [f.name for f in files] == [f"{PINNED_SLUG}_q2_0_gen_131072.json"]
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["schema"] == "hypercore.tuning_history/1"
    assert data["hardware_slug"] == PINNED_SLUG
    assert data["hardware"]["cpu"] == "Test CPU X1"
    assert data["context"] == 131072 and data["objective"] == "gen"
    assert len(data["trials"]) == 3
    assert all(t["cfg"].keys() == {"vram", "pcie", "ple_io", "kv", "kvgrow",
                                   "host_core", "affinity", "worker", "tail"}
               for t in data["trials"])
    csvs = list((out / "results").glob("bayesian-*.csv"))
    assert any(PINNED_SLUG in c.name for c in csvs)


def test_a_second_sweep_reuses_the_configurations_it_already_measured():
    out = Path("/tmp/hc-optuna-reuse")
    argv = ["--hardware", PINNED, "--sampler", "random", "--trials", "6", "--seed", "11"]
    _run(argv, out)
    boots = len(list((out / "results").glob("run-*.log")))
    assert boots == 6
    code, _ = _run(argv, out, fresh=False)
    assert code == 0
    record = _record(out)
    assert record["history"]["trials_loaded"] == 6
    assert record["history"]["reused"] == 6
    assert record["study"]["measured"] == 0
    assert len(list((out / "results").glob("run-*.log"))) == boots
    assert record["best"], "the winner must still come from the stored data"
    assert (out / "max_run_q2_0.sh").exists() and (out / "max_run_q2_0.json").exists()


def test_run_data_from_another_machine_is_not_used():
    out = Path("/tmp/hc-optuna-mismatch")
    _run(["--hardware", PINNED, "--trials", "3", "--seed", "4"], out)
    _run(["--hardware", OTHER, "--trials", "3", "--seed", "4"], out, fresh=False)
    assert len(list((out / "history").glob("*.json"))) == 2

    _run(["--hardware", PINNED, "--sampler", "random", "--trials", "3", "--seed", "4"],
         out, fresh=False)
    assert _record(out)["history"]["trials_loaded"] == 3

    _run(["--hardware", PINNED, "--sampler", "random", "--trials", "3", "--seed", "4",
          "--allow-hardware-mismatch"], out, fresh=False)
    assert _record(out)["history"]["trials_loaded"] == 6


def test_a_prior_outside_the_current_grid_is_skipped_not_clamped():
    out = Path("/tmp/hc-optuna-grid")
    _run(["--hardware", PINNED, "--trials", "6", "--seed", "9", "--vram-max", "1536"], out)
    stored = json.loads((out / "history" / f"{PINNED_SLUG}_q2_0_gen_131072.json")
                        .read_text(encoding="utf-8"))
    outside = sum(1 for t in stored["trials"] if t["cfg"]["vram"] > 256)
    assert outside > 0, "the fixture must produce a prior the narrower grid cannot hold"
    _run(["--hardware", PINNED, "--trials", "1", "--seed", "9", "--vram-max", "256",
          "--no-deliverables"], out, fresh=False)
    record = _record(out)
    assert record["history"]["trials_loaded"] == len(stored["trials"])
    assert record["history"]["priors_skipped"] == outside


def test_the_deliverable_says_which_machine_it_was_tuned_on():
    out = Path("/tmp/hc-optuna-machine")
    _run(["--hardware", PINNED, "--trials", "2", "--seed", "1"], out)
    cfg = json.loads((out / "max_run_q2_0.json").read_text(encoding="utf-8"))
    assert cfg["tuned_by"]["hardware_slug"] == PINNED_SLUG
    assert cfg["tuned_by"]["hardware"]["ram_gib"] == 64
    assert cfg["tuned_by"]["history"]["file"].endswith(f"{PINNED_SLUG}_q2_0_gen_131072.json")
    assert PINNED_SLUG in (out / "max_run_q2_0.sh").read_text(encoding="utf-8")


def test_replaying_a_record_rewrites_the_deliverables_without_booting_anything():
    out = Path("/tmp/hc-optuna-replay")
    _run(["--hardware", PINNED, "--trials", "4", "--seed", "3"], out)
    record = _record(out)
    best = record["best"]
    csvs_before = len(list((out / "results").glob("bayesian-*.csv")))

    code, _ = _run(["--hardware", PINNED, "--replay", str(out / "results" / "max_run_q2_0.record.json")],
                   out, fresh=False)
    assert code == 0
    cfg = json.loads((out / "max_run_q2_0.json").read_text(encoding="utf-8"))
    sh = (out / "max_run_q2_0.sh").read_text(encoding="utf-8")
    assert cfg["tuned_by"]["best"] == best
    assert cfg["tuned_by"]["hardware_slug"] == PINNED_SLUG
    assert cfg["args"] == clean_args(record["args"])
    assert PINNED_SLUG in sh
    assert len(list((out / "results").glob("bayesian-*.csv"))) == csvs_before, "a replay launches nothing"


def test_replay_refuses_a_record_with_no_winner():
    out = Path("/tmp/hc-optuna-replay-bad")
    out.mkdir(parents=True, exist_ok=True)
    bad = out / "empty.record.json"
    bad.write_text(json.dumps({"schema": "hypercore.tuning_history/1", "trials": []}),
                   encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--model", "q2_0", "--engine", str(STUB), "--pack", str(HERE / "fixtures"),
             "--native", str(HERE / "fixtures" / "tiny.gguf"),
             "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
             "--replay", str(bad)])


def test_a_dry_run_writes_no_deliverables():
    """Nothing booted, so there is no winner to hand to the launcher."""
    out = Path("/tmp/hc-optuna-dry")
    code, _ = _run(["--gen", "--dry-run"], out)
    assert code == 0
    assert not (out / "max_run_q2_0.sh").exists()
    assert not (out / "max_run_q2_0.json").exists()
    assert _record(out)["verified"] is False


def test_a_measured_run_marks_its_record_verified():
    code, out = _run(["--gen"])
    assert code == 0
    record = _record(out)
    assert record["verified"] is True
    assert record["metrics"]["gen_tok_s"] > 0
    assert record["source"] == "study"


def test_the_deliverable_says_where_the_winner_came_from():
    """A prior carried in from history is labelled as such, not as a fresh boot."""
    out = Path("/tmp/hc-optuna-source")
    assert _run(["--gen"], out)[0] == 0

    code, _ = _run(["--gen", "--dry-run"], out, fresh=False)
    assert code == 0
    record = _record(out)
    assert record["source"] == "history"
    assert record["verified"] is True
    assert record["study"]["measured"] == 0
    cfg = json.loads((out / "max_run_q2_0.json").read_text(encoding="utf-8"))
    assert cfg["tuned_by"]["source"] == "history"
    assert cfg["tuned_by"]["metrics"]["gen_tok_s"] > 0


def test_from_records_rebuilds_the_pair_from_the_best_measured_row():
    out = Path("/tmp/hc-optuna-records")
    _run(["--gen"], out)
    (out / "max_run_q2_0.sh").unlink()
    (out / "max_run_q2_0.json").unlink()

    rows = [r for csv_path in (out / "results").glob("bayesian-*q2_0-gen-*.csv")
            for r in csv.DictReader(csv_path.open(encoding="utf-8"))
            if r["status"] == "PASS"]
    expected = max(rows, key=lambda r: float(r["gen_tok_s"]))

    code, _ = _run(["--gen", "--from-records"], out, fresh=False)
    assert code == 0
    cfg = json.loads((out / "max_run_q2_0.json").read_text(encoding="utf-8"))
    sh = (out / "max_run_q2_0.sh").read_text(encoding="utf-8")
    assert cfg["tuned_by"]["verified"] is True
    assert cfg["tuned_by"]["metrics"]["gen_tok_s"] == float(expected["gen_tok_s"])
    assert sh.startswith("#!/bin/sh") and "serve/server.py" in sh
    assert "--max-context" in " ".join(cfg["args"])


def test_from_records_refuses_a_model_with_no_measurement():
    out = Path("/tmp/hc-optuna-records-empty")
    (out / "results").mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit):
        main(["--model", "q2_0", "--engine", str(STUB), "--pack", str(HERE / "fixtures"),
             "--native", str(HERE / "fixtures" / "tiny.gguf"),
             "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
             "--results-dir", str(out / "results"), "--from-records"])


def test_model_paths_resolve_to_the_account_that_owns_the_data(monkeypatch):
    """The pack, the GGUFs and the draft layer come from the data folder this user's
    installer recorded, never from a neighbour account's copy of the same model."""
    tmp_path = Path("/tmp/hc-data-roots")
    shutil.rmtree(tmp_path, ignore_errors=True)
    cfg_dir = tmp_path / "config"
    (cfg_dir / "hypercore").mkdir(parents=True)
    mine = tmp_path / "home" / "ai-user0" / "Strata-data"
    sibling = tmp_path / "home" / "ai-user0" / "HyperCore-data"
    other = tmp_path / "home" / "ai-agent" / "HyperCore-data"
    (cfg_dir / "hypercore" / "settings.json").write_text(
        json.dumps({"data_dir": str(mine)}), encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.delenv("HYPERCORE_DATA", raising=False)

    for base in (other, sibling, mine):          # the same model, three places
        (base / "packs" / "q2_0").mkdir(parents=True)
        (base / "packs" / "q2_0" / "experts.bin").write_text("x", encoding="utf-8")
        (base / "models" / "Q2_0").mkdir(parents=True)
        for n in ("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
                  "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"):
            (base / "models" / "Q2_0" / n).write_text("x", encoding="utf-8")
        (base / "mtp" / "rt").mkdir(parents=True)
    repo = tmp_path / "home" / "ai-user0" / "black_box"
    repo.mkdir(parents=True)

    roots = data_roots(repo)
    assert roots[0] == str(mine), roots          # the recorded data_dir wins
    assert str(sibling) in roots                 # a sibling is a fallback, not the first
    assert str(other) not in roots               # another account is not searched

    paths = resolve_model_paths(repo, "q2_0")
    assert paths.pack == str(mine / "packs" / "q2_0")
    assert paths.native.startswith(str(mine / "models" / "Q2_0"))
    assert "00002-of-00002" in paths.ple_gguf
    assert paths.mtp == str(mine / "mtp" / "rt")

    # --data-dir pins it, and an empty or missing folder is dropped rather than used
    pinned = resolve_model_paths(repo, "q2_0", str(other))
    assert pinned.pack == str(other / "packs" / "q2_0")
    (empty := tmp_path / "home" / "ai-user0" / "empty-data").mkdir()
    assert str(empty) not in data_roots(repo)
    with pytest.raises(FileNotFoundError):
        resolve_model_paths(repo, "q9_9")


def test_the_deliverable_tokenizer_matches_the_pack_in_use():
    """A base config naming another account's tokenizer must not survive the sweep."""
    pack = Path("/tmp/hc-token-pack")
    (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
    out = Path("/tmp/hc-optuna-tokenizer")
    shutil.rmtree(out, ignore_errors=True)
    (out / "results").mkdir(parents=True)
    code = main([
        "--model", "q2_0", "--engine", str(STUB), "--strata-root", str(FAKE_ROOT),
        "--pack", str(pack), "--native", str(HERE / "fixtures" / "tiny.gguf"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
        "--history-dir", str(out / "history"),
        "--max-workers", "8", "--trials", "3", "--seed", "7", "--gen",
    ])
    assert code == 0
    cfg = json.loads((out / "max_run_q2_0.json").read_text(encoding="utf-8"))
    args = cfg["args"]
    assert args[args.index("--pack") + 1] == str(pack)
    assert cfg["tokenizer"] == str(pack / "tokenizer")

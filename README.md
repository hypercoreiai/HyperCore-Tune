# HyperCore-Tune

<p align="center"><img src="docs/media/HyperCore-Tune.jpeg" width="420" alt="HyperCore-Tune"></p>

Bayesian (Optuna TPE) tuning of the Strata engine, as a standalone repository that
attaches to a Strata checkout. It has no other association with Strata than
compatibility: clone it inside a checkout, or next to one as its own app.

It searches the engine dimensions on a given machine, measures one request
per configuration by booting the engine, and writes back a run config and a launcher in
exactly the format `setup.py` writes for an installed model:

```
max_run_<model>.json    the winning run config (exe, args, cwd, tokenizer, port, ...)
max_run_<model>.sh      the launcher: serve/server.py with that config
max_run_<model>.bat     the same launcher for a Windows install
HyperCore-Tune/results/max_run_<model>.record.json   every trial and its metrics
HyperCore-Tune/results/bayesian-<machine>-<model>-<objective>-<stamp>.csv
```

Nothing in the sweep is hardcoded to a machine. The project root, the engine binary, the
server script and the venv are resolved from where this repository sits, and every path
written into a deliverable is absolute. Linux and Windows 10/11 are both supported.

## Install

Clone it inside a Strata checkout, or next to one (`~/HyperCore-CPP` or
`%USERPROFILE%\HyperCore-CPP` on Windows). Both layouts work: every path is absolute.

```bash
cd ~/Strata
git clone https://github.com/hypercoreiai/HyperCore-Tune.git
```

```bash
cd ~
git clone https://github.com/hypercoreiai/HyperCore-Tune.git HyperCore-CPP
```

Then install the sweep's own packages (`optuna`, `psutil`) into the project venv:

```bash
./HyperCore-Tune/max_tune.sh --install          # Linux
max_tune.bat --install                          # Windows 10/11
```

`--install` uses the Strata `.venv` if it exists, otherwise the interpreter on the PATH
(`python.exe` / `py -3` on Windows, including `.venv\Scripts\python.exe`).

## Run

```bash
./HyperCore-Tune/max_tune.sh --model q2_0 --gen      --cntx 131072
./HyperCore-Tune/max_tune.sh --model q2_0 --prefill  --cntx 131072 --trials 60
./HyperCore-Tune/max_tune.sh --model q2_0 --latency  --mode staged --dry-run
./HyperCore-Tune/max_tune.sh --show-project
./HyperCore-Tune/max_tune.sh --help
```

On Windows 10/11 the same flags go to `max_tune.bat` from this folder:

```bat
max_tune.bat --model q2_0 --gen --cntx 131072
max_tune.bat --show-project
```

Objectives: `--gen` maximises generated tokens/s, `--prefill` maximises prompt tokens/s,
`--latency` minimises `prompt_ms + decode_ms`.

`--trials` is the number of **unseen** configurations to evaluate. A suggestion that
matches a point already in history (or already measured in this sweep) is pruned, not
counted, and the sampler keeps drawing until that many new points have been booted.

## Search space

The dimensions searched are the columns every row is written with:

```
layer  vram  pcie  ple-io  kv  kvgrow  hostcore  affinity  worker  tail  draft-floor  status
```

`status` is not a parameter: it is what the engine did with the configuration
(`PASS`, `FAIL_BOOT`, `NO_TOKENS`, `TIMEOUT`).

| Tuner flag | Engine flag | Default search | Hold constant |
|---|---|---|---|
| `--vram-min` / `--vram-max` / `--vram-step` | `--vram-reserve-mib` | 256–1536 step 256 | (range flags, not a freeze) |
| `--pcie` | `--pcie-frac` | 0–1 | `--pcie 0.95` |
| `--ple-io` | `--ple-io` | `direct`, `mmap`, `ram` | `--ple-io ram` |
| `--kv` | `--kv` | `q4_0`, `k8v4`, `int8`, `fp16` | `--kv int8` |
| `--kvgrow` | `--kv-grow` | off / on | `--kvgrow 0` or `1` |
| `--hostcore` / `--host-core` | `--host-core` | `first`, `last` | `--hostcore first` |
| `--affinity` | `--pool-affinity` | `all`, `p-cores`, `auto` | `--affinity all` |
| `--worker` | `--pool-workers` | 0 … physical cores − 1 | `--worker 8` |
| `--tail` | `--prompt-cache-tail` | off / on | `--tail 0` or `1` |
| `--draft-floor` / `--spec-min-p` | `--spec-min-p` | 0.001–1 | `--draft-floor 0.5` |

A freeze flag holds that dimension at a constant instead of searching it. History rows
that do not match a freeze are not used as priors.

`--prompt-cache-tail` is measured when `tail` is searched, but it is **never** written
into the deliverable: the default Strata interface does not accept it. `--spec-min-p`
(from `draft-floor`) is a flag Strata does accept, so the winner's floor stays in
`max_run_<model>.json` / `.sh` / `.bat`.

`--pool-workers 0` is a legal arm: the engine treats it as "every physical core except
the host".

Example, pin KV and the draft floor, search the rest:

```bash
./HyperCore-Tune/max_tune.sh --model q2_0 --gen --cntx 131072 --kv int8 --draft-floor 0.5
```

## How the project is found

`strata_env.py` resolves the Strata root in this order, and every result is an absolute
path (`~`, `%USERPROFILE%` and `%VAR%` are expanded):

1. `--strata-root`, then `$STRATA_ROOT`, `$HYPERCORE_ROOT`, `$TUNE_ROOT`
2. the nearest ancestor of this repository that is a Strata checkout
   (`~/Strata/HyperCore-Tune` and `~/Strata/HyperCore-CPP` both walk up to `~/Strata`)
3. a sibling directory named `Strata` or `HyperCore`
   (`~/HyperCore-CPP` next to `~/Strata`)
4. any other sibling that looks like a Strata checkout

A directory counts as a root when it has `serve/server.py` and one of `setup.py`,
`CMakeLists.txt` or `src/program/generate.cpp`. HyperCore-Tune / HyperCore-CPP is this
app, not a Strata root. `--show-project` prints what was resolved:

```json
{
  "repo": "/home/you/Strata/HyperCore-Tune",
  "root": "/home/you/Strata",
  "engine": "/home/you/Strata/engine/strata",
  "server": "/home/you/Strata/serve/server.py",
  "venv": "/home/you/Strata/.venv/bin/python",
  "server_engine": "strata"
}
```

The engine binary is looked for at `engine/strata`, `engine/strata.exe`,
`engine/hypercore`, `engine-cuda12/strata`, `build/strata`, `dist/strata`, and can be
pinned with `--engine` or `$STRATA_ENGINE`. Model packs and GGUFs are found in the data
folder the installer recorded (`~/.config/strata/settings.json`, or `%APPDATA%\strata`
on Windows, or `$STRATA_DATA`), then in a data folder next to the checkout.

## Reusing previous sweeps

Run data is kept per machine in `HyperCore-Tune/history/`, named

```
<CPU>_<GPU>_<RAM>_<PCIe gen>x<width>_<model>_<objective>_<cntx>.json
```

A sweep on the same box loads it by default, never re-boots a configuration it already
measured, seeds the TPE sampler with those trials, and keeps drawing until `--trials`
unseen configurations have been evaluated. `--no-load` starts cold; a record from
different hardware is ignored unless `--allow-hardware-mismatch` is given. Pin the
fingerprint with `--hardware '<json>'` or `$TUNE_HARDWARE` for containers and test farms.

`--replay <record.json>` and `--from-records` rewrite the deliverables from measurements
that already exist, without booting the engine.

## Compatibility with Strata

- the deliverable is a run config in `setup.py`'s shape; keys the model's own config
  carries that setup does not own (`sampling`, `reasoning`, `mcp_servers`, ...) are kept
- a key is only written when this project's server can read it: `strata_env.server_flags()`
  reads the flags `serve/server.py` defines, so a flag from an older release (for example
  `turn_tail_check`, which Strata's server does not have) is left out rather than passed
  to argparse and rejected at startup
- `--serve`, `--max-new`, `--expert-profile-save-every` and `--prompt-cache-tail` are
  stripped from the production args: they exist only so the sweep can measure one request
- the launcher uses the backend name the local `serve/server.py` accepts, read from its
  own `--engine` choices
- the model's existing `strata-<model>.json` is used as the base for the new config
- the trial record and the CSV stay inside this repository; only the launcher pair is
  written at the project root
- a deliverable directory that cannot be written (another user's launcher, a read-only
  root) falls back to `HyperCore-Tune/results/` rather than losing a finished sweep

## Tests

The tests never touch a GPU: `stub_engine.py` speaks the real protocol (`INFO`,
`session is up`, `DONE`) and computes its numbers from the flags it was given.

```bash
python -m pip install -r HyperCore-Tune/requirements.txt
pytest HyperCore-Tune/test_bayesian_tune.py
```

## License

MIT. See `LICENSE`.

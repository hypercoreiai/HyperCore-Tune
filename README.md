# HyperCore-Tune

<p align="center"><img src="docs/media/HyperCore-Tune.jpeg" width="420" alt="HyperCore-Tune"></p>

Bayesian (Optuna TPE) tuning of the Strata engine, as a standalone repository that
attaches to a Strata checkout.

It searches the ten dimensions the engine exposes on a given machine, measures one request
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
written into a deliverable is absolute.

## Install

Clone it inside a Strata checkout:

```bash
cd ~/Strata
git clone https://github.com/hypercoreiai/HyperCore-Tune.git
```

or beside one; both work. Then install the sweep's own packages into the project venv:

```bash
./HyperCore-Tune/max_tune.sh --install
```

`--install` uses the Strata `.venv` if it exists, otherwise the interpreter on the PATH.

## Run

```bash
./HyperCore-Tune/max_tune.sh --model q2_0 --gen      --cntx 131072
./HyperCore-Tune/max_tune.sh --model q2_0 --prefill  --cntx 131072 --trials 60
./HyperCore-Tune/max_tune.sh --model q2_0 --latency  --mode staged --dry-run
./HyperCore-Tune/max_tune.sh --show-project
```

Objectives: `--gen` maximises generated tokens/s, `--prefill` maximises prompt tokens/s,
`--latency` minimises `prompt_ms + decode_ms`.

The dimensions searched are the columns every row is written with:

```
layer  vram  pcie  ple-io  kv  kvgrow  hostcore  affinity  worker  tail  status
```

`status` is not a parameter: it is what the engine did with the configuration
(`PASS`, `FAIL_BOOT`, `NO_TOKENS`, `TIMEOUT`).

## How the project is found

`strata_env.py` resolves the Strata root in this order, and every result is an absolute
path:

1. `--strata-root`, then `$STRATA_ROOT`, `$HYPERCORE_ROOT`, `$TUNE_ROOT`
2. the nearest ancestor of this repository that is a Strata checkout
3. a sibling directory named `Strata` or `HyperCore`

A directory counts as a root when it has `serve/server.py` and one of `setup.py`,
`CMakeLists.txt` or `src/program/generate.cpp`. `--show-project` prints what was resolved:

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

The engine binary is looked for at `engine/strata`, `engine/hypercore`,
`engine-cuda12/strata`, `build/strata`, `dist/strata`, and can be pinned with
`--engine` or `$STRATA_ENGINE`. Model packs and GGUFs are found in the data folder the
installer recorded (`~/.config/strata/settings.json`, or `$STRATA_DATA`), then in a data
folder next to the checkout.

## Reusing previous sweeps

Run data is kept per machine in `HyperCore-Tune/history/`, named

```
<CPU>_<GPU>_<RAM>_<PCIe gen>x<width>_<model>_<objective>_<cntx>.json
```

A sweep on the same box loads it, never re-boots a configuration it already measured, and
seeds the TPE sampler with those trials. `--no-load` starts cold; a record from different
hardware is ignored unless `--allow-hardware-mismatch` is given. Pin the fingerprint with
`--hardware '<json>'` or `$TUNE_HARDWARE` for containers and test farms.

`--replay <record.json>` and `--from-records` rewrite the deliverables from measurements
that already exist, without booting the engine.

## Compatibility with Strata

- the deliverable is a run config in `setup.py`'s shape; keys the model's own config
  carries that setup does not own (`sampling`, `reasoning`, `mcp_servers`, ...) are kept
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

MIT, the same as Strata.

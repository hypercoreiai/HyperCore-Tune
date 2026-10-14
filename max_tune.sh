#!/usr/bin/env bash
#
# HyperCore-Tune/max_tune.sh - entry point for the Bayesian (Optuna) sweep.
#
#   ./HyperCore-Tune/max_tune.sh --model q2_0 --gen      --cntx 131072
#   ./HyperCore-Tune/max_tune.sh --model q2_0 --prefill  --cntx 131072 --trials 60
#   ./HyperCore-Tune/max_tune.sh --model q2_0 --latency  --mode staged --dry-run
#   ./HyperCore-Tune/max_tune.sh --show-project           # print the resolved paths, then exit
#   ./HyperCore-Tune/max_tune.sh --install               # put optuna into the project venv, then exit
#
# The objective is one of three user-selected metrics:
#   --gen      maximise generation tokens/s
#   --prefill  maximise prompt tokens/s
#   --latency  minimise prompt_ms + decode_ms
#
# The dimensions searched are the columns every row is written with:
#   layer  vram  pcie  ple-io  kv  kvgrow  hostcore  affinity  worker  tail  draft-floor  status
#
# Hold a dimension constant instead of searching it:
#   --pcie --ple-io --kv --kvgrow --hostcore --affinity --worker --tail --draft-floor
# --draft-floor is --spec-min-p on the engine, searched in [0.001, 1] unless pinned.
# --prompt-cache-tail is measured when tail is searched, but never written into the
# deliverable (the default Strata interface does not accept it).
#
# --trials is unseen configurations: a repeat of a point already in history is redrawn
# and does not count.
#
# Deliverables, at the Strata project root, in the project's own run-config format:
#   max_run_<model>.sh / .bat   the startup script: serve/server.py with the winning config
#   max_run_<model>.json        that run config, with the winning engine arguments in "args"
#   HyperCore-Tune/results/max_run_<model>.record.json   every trial and its metrics
#   HyperCore-Tune/results/bayesian-<machine>-<model>-<objective>-<stamp>.csv
#
# Run data is kept per machine in HyperCore-Tune/history/, named
#   <CPU>_<GPU>_<RAM>_<PCIe>x<width>_<model>_<objective>_<cntx>.json
# so a sweep on the same box reuses what it already measured instead of booting again,
# and keeps drawing until --trials new points have been evaluated.
#   --no-load                   start the study cold
#   --allow-hardware-mismatch   load run data from a different machine too
#   --hardware '<json>'         pin the fingerprint instead of probing the host
#   --replay <record.json>      rewrite the deliverables from a finished sweep, no engine boot
#   --from-records              rebuild the deliverables from the best measured trial in the
#                             CSVs under HyperCore-Tune/results/, no engine boot
#
# The Strata root is resolved in the same order as the Python side: $STRATA_ROOT, then the
# nearest ancestor of this repository that is a Strata checkout, then a sibling named
# Strata / HyperCore. A value that starts with ~ is expanded. Every path the sweep writes
# is absolute, so this script works from ~/Strata/HyperCore-Tune and from ~/HyperCore-CPP.
#
# The Python side runs under the project venv: setup.sh/install.sh create .venv.
# If .venv is missing the system python3 is used and the script says so. optuna lives in
# this repository's requirements.txt (not Strata's); --install adds it to the venv.
# On Windows 10/11 the same flags go to max_tune.bat.
#
# Requires: bash 4.4+, a free GPU for a real run (stop the server with ./restart.sh first).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Turn a user path into an absolute one. ~ is not expanded once it lives in a variable.
abs_path() {
    local p="$1"
    case "$p" in
        ~) p="$HOME" ;;
        ~/*) p="$HOME/${p#~/}" ;;
    esac
    if [ -d "$p" ]; then
        (cd "$p" && pwd)
        return 0
    fi
    case "$p" in
        /*) printf '%s\n' "$p" ;;
        *) printf '%s/%s\n' "$(pwd)" "$p" ;;
    esac
}

is_strata_root() {
    local d="$1"
    [ -f "$d/serve/server.py" ] || return 1
    [ -f "$d/setup.py" ] || [ -f "$d/CMakeLists.txt" ] || [ -f "$d/src/program/generate.cpp" ]
}

find_root() {
    local cand d name
    if [ -n "${STRATA_ROOT:-}" ]; then abs_path "$STRATA_ROOT"; return 0; fi
    if [ -n "${HYPERCORE_ROOT:-}" ]; then abs_path "$HYPERCORE_ROOT"; return 0; fi
    if [ -n "${TUNE_ROOT:-}" ]; then abs_path "$TUNE_ROOT"; return 0; fi
    d="$HERE"
    while [ "$d" != "/" ]; do
        d="$(dirname "$d")"
        if is_strata_root "$d"; then
            printf '%s\n' "$d"
            return 0
        fi
    done
    for name in Strata HyperCore hypercore strata; do
        cand="$HERE/../$name"
        if is_strata_root "$cand"; then
            abs_path "$cand"
            return 0
        fi
    done
    for cand in "$HERE"/../*; do
        if is_strata_root "$cand"; then
            abs_path "$cand"
            return 0
        fi
    done
    cand="$HERE/../../Strata"
    if is_strata_root "$cand"; then
        abs_path "$cand"
        return 0
    fi
    return 1
}

ROOT="$(find_root)" || {
    echo "max_tune.sh: no Strata checkout found. Clone HyperCore-Tune inside one, or set" >&2
    echo "  STRATA_ROOT=/path/to/Strata  (or pass --strata-root)." >&2
    exit 4
}
VENV="$ROOT/.venv"

pick_python() {
    if [ -n "${TUNE_PYTHON:-}" ] && [ -x "$TUNE_PYTHON" ]; then
        printf '%s\n' "$TUNE_PYTHON"
        return 0
    fi
    for p in "$VENV/bin/python" "$VENV/bin/python3" "$VENV/Scripts/python.exe" \
            "$ROOT/venv/bin/python" "$ROOT/venv/Scripts/python.exe" \
            "$(command -v python3 2>/dev/null || true)" \
            "$(command -v python 2>/dev/null || true)"; do
        [ -n "$p" ] && [ -x "$p" ] || continue
        printf '%s\n' "$p"
        return 0
    done
    return 1
}

PY="$(pick_python)" || {
    echo "max_tune.sh: no python interpreter found. Create the venv: python3 -m venv .venv" >&2
    exit 1
}

if [ "${1:-}" = "--install" ]; then
    echo "[*] installing optuna into $PY"
    "$PY" -m pip install --quiet -r "$HERE/requirements.txt" || {
        echo "pip install failed" >&2; exit 1
    }
    "$PY" -c "import optuna; print('optuna', optuna.__version__)"
    exit 0
fi

if [ "${1:-}" = "--show-project" ]; then
    "$PY" "$HERE/strata_env.py"
    exit 0
fi

if ! "$PY" -c "import optuna" >/dev/null 2>&1; then
    echo "max_tune.sh: $PY cannot import optuna. Install it:" >&2
    echo "    $PY -m pip install -r HyperCore-Tune/requirements.txt" >&2
    echo "  or run: ./HyperCore-Tune/max_tune.sh --install" >&2
    exit 1
fi

engine_override=0
for a in "$@"; do [ "$a" = "--engine" ] && engine_override=1; done
if [ "$engine_override" = 0 ]; then
    engine="$("$PY" -c "import sys; sys.path.insert(0, '$HERE'); from strata_env import engine_binary; print(engine_binary())")"
    if [ -z "$engine" ]; then
        echo "max_tune.sh: no engine in $ROOT. Build it (./setup.sh or ./install.sh) or pass --engine." >&2
        exit 2
    fi
    echo "[*] engine: $engine"
fi

free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
is_dry=0
for a in "$@"; do [ "$a" = "--dry-run" ] && is_dry=1; done
is_help=0
for a in "$@"; do [ "$a" = "--help" ] || [ "$a" = "-h" ] && is_help=1; done
# A replay or a rebuild from stored records never boots the engine, so it needs no card.
is_offline=0
for a in "$@"; do [ "$a" = "--replay" ] || [ "$a" = "--from-records" ] && is_offline=1; done
if [ "$is_dry" = 0 ] && [ "$is_help" = 0 ] && [ "$is_offline" = 0 ] && [ "${free:-0}" -lt "${GPU_MIN_FREE:-12000}" ]; then
    echo "max_tune.sh: the GPU has ${free:-0} MiB free. Stop the running server first before tuning." >&2
    exit 3
fi

echo "[*] project root: $ROOT"
echo "[*] using $PY ($( "$PY" -V 2>&1 ))"
exec "$PY" "$HERE/bayesian_tune.py" "$@"

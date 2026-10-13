"""HyperCore-Tune/hardware.py - the machine a tuning record belongs to.

A configuration that wins on a 24-core box with a 4090 in a PCIe 4 slot is not a prior
for a 4-core laptop, so every record the sweep writes carries the hardware it was
measured on, and the loader refuses records from a different machine unless
--allow-hardware-mismatch is given.

The data file is named after the four things that decide whether a result transfers:

    <CPU>_<GPU>_<RAM>_<PCIe gen>x<width>
    Intel-Core-i9-14900KF_NVIDIA-GeForce-RTX-4090_31GB_PCIe4x16

Every probe is optional and none of them raise. A machine without nvidia-smi or lspci
still gets a fingerprint from /proc and /sys; a probe that cannot answer reports
"unknown"/0 rather than stopping the sweep.

TUNE_HARDWARE (or the older HC_TUNE_HARDWARE) may point at a JSON file (or an inline
JSON object) to pin the fingerprint, which is how the tests and a container with a
different host get a deterministic answer.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# GT/s is the PCIe signalling rate; the generation is the name for it.
PCI_GEN_FROM_GTS: dict[float, int] = {2.5: 1, 4.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
GPU_CLASS = "0x030000"  # /sys/bus/pci/devices/*/class of a display controller
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Hardware:
    """What the sweep actually ran on. All fields are safe to serialise."""

    cpu: str = UNKNOWN
    physical_cores: int = 0
    logical_cores: int = 0
    ram_gib: int = 0
    gpus: tuple[str, ...] = ()
    gpu_mem_mib: int = 0
    pci_gen: int = 0
    pci_width: int = 0
    probed: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu": self.cpu,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "ram_gib": self.ram_gib,
            "gpus": list(self.gpus),
            "gpu_mem_mib": self.gpu_mem_mib,
            "pci_gen": self.pci_gen,
            "pci_width": self.pci_width,
            "probed": list(self.probed),
        }

    def slug(self) -> str:
        """The machine key used in a data-file name. Deterministic, filesystem-safe."""
        gpu = "+".join(self.gpus) if self.gpus else "no-gpu"
        pci = f"PCIe{self.pci_gen}x{self.pci_width}" if self.pci_gen else "PCIe-unknown"
        parts = [_clean(self.cpu), _clean(gpu), f"{self.ram_gib}GB", pci]
        return "_".join(parts)[:160]

    def machine_key(self) -> tuple[str, tuple[str, ...], int, int]:
        """The fields that must match for a previous run to be a valid prior."""
        return (self.cpu.lower(), tuple(g.lower() for g in self.gpus), self.ram_gib, self.pci_gen)


def _clean(part: str) -> str:
    """Keep a slug readable and safe: no spaces, no vendor parentheticals, no doubled separators."""
    part = re.sub(r"\([^)]*\)", " ", str(part))
    out = re.sub(r"[^A-Za-z0-9]+", "-", part).strip("-")
    return out or UNKNOWN


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _run(cmd: list[str], timeout: int = 15) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout or ""


def slug(hw: Hardware) -> str:
    """The machine key of a fingerprint, as a standalone function for callers that only
    have a Hardware object at hand."""
    return hw.slug()


def detect_cpu() -> tuple[str, int, int]:
    """(model name, physical cores, logical cores)."""
    model = ""
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    if not model:
        model = os.uname().machine or UNKNOWN

    logical = os.cpu_count() or 0
    physical = 0
    try:
        import psutil  # in .venv; absent on a bare interpreter
        physical = int(psutil.cpu_count(logical=False) or 0)
        logical = int(psutil.cpu_count(logical=True) or logical)
    except Exception:  # noqa: BLE001 - a container that reports nothing must not stop the sweep
        physical = 0
    return model or UNKNOWN, physical or logical, logical


def detect_ram_gib() -> int:
    """Installed RAM as whole GiB, from MemTotal. 0 when the kernel says nothing."""
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            kb = int(re.sub(r"[^\d]", "", line.split(":", 1)[1]) or 0)
            return max(1, round(kb / 1048576))
    try:
        import psutil
        return max(1, int(psutil.virtual_memory().total // 2**30))
    except Exception:  # noqa: BLE001
        return 0


def detect_gpus() -> tuple[tuple[str, ...], int]:
    """(names, total VRAM in MiB). nvidia-smi first, then lspci, then sysfs."""
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    names: list[str] = []
    mem = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        names.append(parts[0])
        if len(parts) > 1:
            n = int(re.sub(r"[^\d]", "", parts[1]) or 0)
            mem += n
    if names:
        return tuple(names), mem

    out = _run(["lspci", "-vv"])
    names = [m.group(1).strip() for m in
             re.finditer(r"^\S+ VGA controller: (.+)$", out, re.MULTILINE)]
    if names:
        return tuple(names), 0

    found = []
    for dev in sorted(Path("/sys/bus/pci/devices").glob("*")):
        if _read(str(dev / "class")).strip() == GPU_CLASS:
            found.append(f"pci-{dev.name}")
    return tuple(found), 0


def pci_from_lspci() -> tuple[int, int]:
    """The negotiated link lspci reports, or (0, 0) when lspci is not installed.

    The display device's link is preferred: a PCIe 5 slot on the chipset does not make a
    PCIe 4 GPU link a PCIe 4.0-or-better transfer path.
    """
    out = _run(["lspci", "-vv"])
    if not out:
        return (0, 0)
    display = (0, 0)
    any_link = (0, 0)
    seen_display = False
    for line in out.splitlines():
        if re.search(r"VGA|Display controller|GPU", line):
            seen_display = True
        m = re.search(r"LnkSta:\s+([\d.]+)GT/s[^,]*,\s*Width x(\d+)", line)
        if not m:
            continue
        link = (PCI_GEN_FROM_GTS.get(float(m.group(1)), 0), int(m.group(2)))
        any_link = max(any_link, link)
        if seen_display:
            display = max(display, link)
    return display if display[0] else any_link


def pci_from_sysfs(devices_dir: Path | str = "/sys/bus/pci/devices") -> tuple[int, int]:
    """The slot capability the kernel exposes, read device by device.

    The display device's link wins; the fastest link on the board is the fallback.
    """
    root = Path(devices_dir)
    if not root.is_dir():
        return (0, 0)
    display = (0, 0)
    any_link = (0, 0)
    for dev in sorted(root.glob("*")):
        m = re.search(r"([\d.]+)\s*GT/s", _read(str(dev / "max_link_speed")))
        if not m:
            continue
        width = int(re.sub(r"[^\d]", "", _read(str(dev / "max_link_width"))) or 0)
        link = (PCI_GEN_FROM_GTS.get(float(m.group(1)), 0), width)
        any_link = max(any_link, link)
        if _read(str(dev / "class")).strip() == GPU_CLASS:
            display = max(display, link)
    return display if display[0] else any_link


def detect_pci(devices_dir: Path | str = "/sys/bus/pci/devices") -> tuple[int, int]:
    """(generation, lane width) of the PCIe link the accelerator sits on.

    lspci's LnkSta is the negotiated link; sysfs max_link_speed is the slot capability.
    Either is enough to tell a PCIe 3 host from a PCIe 4 one, which is what decides
    whether a pcie-frac result transfers.
    """
    gen, width = pci_from_lspci()
    if gen:
        return (gen, width)
    return pci_from_sysfs(devices_dir)


def detect(devices_dir: Path | str = "/sys/bus/pci/devices") -> Hardware:
    """The fingerprint of this machine, cached for the life of the process."""
    override = next((os.environ[name] for name in ("TUNE_HARDWARE", "HC_TUNE_HARDWARE")
                     if os.environ.get(name)), "")
    if override:
        try:
            spec = Path(override)
            raw = json.loads(spec.read_text(encoding="utf-8")) if spec.is_file() \
                else json.loads(override)
            if isinstance(raw, dict):
                return Hardware(
                    cpu=str(raw.get("cpu", UNKNOWN)),
                    physical_cores=int(raw.get("physical_cores", 0)),
                    logical_cores=int(raw.get("logical_cores", 0)),
                    ram_gib=int(raw.get("ram_gib", 0)),
                    gpus=tuple(str(g) for g in raw.get("gpus", ())),
                    gpu_mem_mib=int(raw.get("gpu_mem_mib", 0)),
                    pci_gen=int(raw.get("pci_gen", 0)),
                    pci_width=int(raw.get("pci_width", 0)),
                    probed=("TUNE_HARDWARE",),
                )
        except (OSError, ValueError, TypeError) as exc:
            print(f"note: TUNE_HARDWARE is not valid JSON ({exc}); probing the host",
                  file=__import__("sys").stderr)

    cpu, phys, logical = detect_cpu()
    gpus, mem = detect_gpus()
    gen, width = detect_pci(devices_dir)
    probed = []
    if cpu != UNKNOWN:
        probed.append("cpuinfo")
    if gpus:
        probed.append("nvidia-smi" if shutil.which("nvidia-smi") else "pci-class")
    if gen:
        probed.append("lspci" if shutil.which("lspci") else "sysfs")
    return Hardware(cpu=cpu, physical_cores=phys, logical_cores=logical,
                   ram_gib=detect_ram_gib(), gpus=gpus, gpu_mem_mib=mem,
                   pci_gen=gen, pci_width=width, probed=tuple(probed))


@lru_cache(maxsize=1)
def fingerprint() -> Hardware:
    """The one answer for this process, so a sweep cannot disagree with itself."""
    return detect()


def differences(a: Hardware, b: Hardware) -> list[str]:
    """Why two machines are not the same machine. Empty means a prior is transferable."""
    out: list[str] = []
    if a.cpu.lower() != b.cpu.lower():
        out.append(f"cpu {a.cpu} vs {b.cpu}")
    if tuple(g.lower() for g in a.gpus) != tuple(g.lower() for g in b.gpus):
        out.append(f"gpu {', '.join(a.gpus) or 'none'} vs {', '.join(b.gpus) or 'none'}")
    if a.ram_gib != b.ram_gib:
        out.append(f"ram {a.ram_gib} GiB vs {b.ram_gib} GiB")
    if a.pci_gen and b.pci_gen and a.pci_gen != b.pci_gen:
        out.append(f"PCIe gen {a.pci_gen} vs {b.pci_gen}")
    return out


def from_dict(raw: dict[str, Any]) -> Hardware:
    """A fingerprint from a dict: either the fields themselves, or a record's hardware block."""
    body: dict[str, Any] = raw["hardware"] if isinstance(raw.get("hardware"), dict) else raw
    return Hardware(
        cpu=str(body.get("cpu", UNKNOWN)),
        physical_cores=int(body.get("physical_cores", 0)),
        logical_cores=int(body.get("logical_cores", 0)),
        ram_gib=int(body.get("ram_gib", 0)),
        gpus=tuple(str(g) for g in body.get("gpus", ())),
        gpu_mem_mib=int(body.get("gpu_mem_mib", 0)),
        pci_gen=int(body.get("pci_gen", 0)),
        pci_width=int(body.get("pci_width", 0)),
        probed=tuple(str(p) for p in body.get("probed", ())),
    )


def hardware_from(record: dict[str, Any]) -> Hardware:
    """Rebuild a Hardware from a stored record; a record with no hardware is unmatched."""
    if not isinstance(record, dict):
        return Hardware()
    return from_dict(record)

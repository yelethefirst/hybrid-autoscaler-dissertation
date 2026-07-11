"""Capture and verify the measurement host specification (§3.10, §3.13).

Run once on the measurement host to generate host-spec.json, which is
committed alongside the final experiment data. This file is also included
in the Zenodo archive so reviewers can assess hardware confounds.

Usage:
    uv run python -m reproducibility.host_spec [--output host-spec.json]
    uv run python -m reproducibility.host_spec --verify host-spec.json

The dissertation specifies the following host spec (§3.10):
    CPU:         Apple M-series (arm64) for local dev; x86_64 cloud for measured runs
    RAM:         ≥ 8 GiB
    OS:          macOS (local) or Ubuntu 22.04 (cloud)
    kind:        v0.32.0
    Kubernetes:  v1.33.12
    Python:      3.11.x
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click


@dataclass
class HostSpec:
    captured_at_utc: str
    hostname: str
    platform: str
    architecture: str
    cpu_model: Optional[str]
    cpu_logical_cores: int
    ram_total_gb: float
    python_version: str
    python_implementation: str
    kind_version: Optional[str]
    kubectl_version: Optional[str]
    docker_version: Optional[str]
    uv_version: Optional[str]
    git_commit: Optional[str]
    git_dirty: bool


def _cmd(args: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        return -1.0


def _cpu_model() -> Optional[str]:
    pl = platform.system()
    if pl == "Darwin":
        return _cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    if pl == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            return None
    return None


def capture() -> HostSpec:
    import os

    git_commit = _cmd(["git", "rev-parse", "HEAD"])
    git_dirty_out = _cmd(["git", "status", "--porcelain"])
    git_dirty = bool(git_dirty_out)

    return HostSpec(
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),
        platform=platform.platform(),
        architecture=platform.machine(),
        cpu_model=_cpu_model(),
        cpu_logical_cores=os.cpu_count() or -1,
        ram_total_gb=_ram_gb(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        kind_version=_cmd(["kind", "version"]),
        kubectl_version=_cmd(["kubectl", "version", "--client", "--output=yaml"]),
        docker_version=_cmd(["docker", "version", "--format", "{{.Client.Version}}"]),
        uv_version=_cmd(["uv", "version"]),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


def verify(spec: HostSpec, path: str | Path) -> bool:
    """Check the captured spec against a previously saved spec file.

    Returns True if the specs match on key fields (architecture, Python version,
    kind version, git commit). Prints differences to stdout.
    """
    saved = HostSpec(**json.loads(Path(path).read_text()))
    fields_to_check = [
        "architecture",
        "python_version",
        "kind_version",
        "git_commit",
    ]
    ok = True
    for f in fields_to_check:
        current = getattr(spec, f)
        expected = getattr(saved, f)
        if current != expected:
            click.echo(f"MISMATCH  {f}: current={current!r}  expected={expected!r}")
            ok = False
        else:
            click.echo(f"OK        {f}: {current!r}")
    return ok


@click.command(name="host-spec")
@click.option("--output", default=None, type=click.Path(), help="Write spec to JSON file.")
@click.option("--verify", "verify_path", default=None, type=click.Path(exists=True),
              help="Verify current spec against a previously saved spec file.")
def main(output: Optional[str], verify_path: Optional[str]) -> None:
    spec = capture()

    if output:
        Path(output).write_text(json.dumps(asdict(spec), indent=2))
        click.echo(f"✅  host spec written to {output}")
    else:
        click.echo(json.dumps(asdict(spec), indent=2))

    if verify_path:
        ok = verify(spec, verify_path)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
